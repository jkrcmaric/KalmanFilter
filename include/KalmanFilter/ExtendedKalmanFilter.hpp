#ifndef EXTENDED_KALMAN_FILTER_HPP
#define EXTENDED_KALMAN_FILTER_HPP

#include "KalmanFilter.hpp"

/**
 * @brief Extended Kalman Filter (EKF).
 * Implements the standard EKF for non-linear systems via First-Order Linearization:
 * * Pred: x = f(x), P = FPF' + Q (where F is Jacobian of f)
 * * Upd:  x = x + K(z - h(x)), P = (I - KH)P(I - KH)' + KRK' (where H is Jacobian of h)
 * * @tparam StateDim Fixed size of the system state vector.
 */
template <int StateDim>
class ExtendedKalmanFilter : public KalmanFilter<ExtendedKalmanFilter<StateDim>, StateDim> {
    using Base = KalmanFilter<ExtendedKalmanFilter<StateDim>, StateDim>;
    friend class KalmanFilter<ExtendedKalmanFilter<StateDim>, StateDim>;

public:
    using typename Base::StateVector;
    using typename Base::StateMatrix;
    using typename Base::ProcessModel;
    
    template <int Dim> 
    using SensorModel = typename Base::template SensorModel<Dim>;

    ExtendedKalmanFilter(const StateVector& x, const StateMatrix& P) : Base(x, P) {}

protected:
    // Pre-allocated Identity for optimization in Joseph Form update
    StateMatrix I_{StateMatrix::Identity()};

    // =========================================================================
    // Interface Implementation (Called by Base Class)
    // =========================================================================

    /**
     * @brief Computes a priori state and covariance (Prediction).
     * Linearizes dynamics (updates F), propagates state non-linearly (f(x)),
     * and propagates covariance linearly.
     */
    void computePrediction(ProcessModel& model) {
        // Linearize: Update Jacobian F based on current state (old x)
        model.updateJacobian(this->x_);

        // Predict State (Non-Linear): x = f(x)
        // Note: model.fx() handles fallback to F*x if no function is set
        this->x_ = model.fx(this->x_);

        // Wrap angles if defined (e.g., heading)
        if (model.hasAngle()) {
            this->template normalizeAngles<StateDim>(this->x_, model.getAngleFlags());
        }

        // Predict Covariance (Linearized): P = F P F' + Q
        this->P_ = model.F() * this->P_ * model.F().transpose() + model.Q();

        // Force physical constraints (clamping/projection)
        model.enforceConstraints(this->x_);
    }

    /**
     * @brief Computes posterior state and covariance (Correction).
     * Linearizes measurement model (updates H), computes non-linear innovation,
     * performs 3-tier gating, and updates using Joseph Form.
     * @return true if measurement accepted, false if gated/rejected.
     */
    template <int MeasureDim>
    bool computeUpdate(SensorModel<MeasureDim>& model, const Eigen::Matrix<double, MeasureDim, 1>& z) {
        using MeasureMatrix = Eigen::Matrix<double, MeasureDim, MeasureDim>;
        using MeasureVector = Eigen::Matrix<double, MeasureDim, 1>;
        using MatrixK = Eigen::Matrix<double, StateDim, MeasureDim>;

        // 1. Linearize: Update Jacobian H based on predicted state
        model.updateJacobian(this->x_);

        // 2. Innovation: y = z - h(x) (Non-linear)
        MeasureVector y = z - model.hx(this->x_);

        // Wrap innovation angles (critical for shortest path error)
        if (model.hasAngle()) {
            this->template normalizeAngles<MeasureDim>(y, model.getAngleFlags());
        }

        // 3. Innovation Covariance: S = H P H' + R
        MeasureMatrix S = model.H() * this->P_ * model.H().transpose() + model.R();

        // 4. Gating (Outlier Rejection)
        if (!this->rectangularGate(model, S, y)) return false; // Tier 1: Fast Sigma check
        if (!model.domainGate(z, y, S)) return false;          // Tier 2: User Logic

        Eigen::LDLT<MeasureMatrix> S_ldlt(S);
        if (!this->mahalanobisGate(model, S_ldlt, y)) return false; // Tier 3: Statistical

        // 5. Kalman Gain: K = P H' S^-1
        // Uses LDLT for stable inversion of symmetric positive definite S
        MatrixK K = this->P_ * model.H().transpose() * S_ldlt.solve(MeasureMatrix::Identity());

        // 6. State Update: x = x + Ky
        this->x_ += K * y;

        // 7. Covariance Update (Joseph Form): P = (I-KH)P(I-KH)' + KRK'
        // Guarantees P remains Symmetric Positive Definite regardless of numerical noise
        StateMatrix I_KH = I_ - K * model.H();
        this->P_ = I_KH * this->P_ * I_KH.transpose() + K * model.R() * K.transpose();

        return true;
    }
};

#endif