#ifndef LINEAR_KALMAN_FILTER_HPP
#define LINEAR_KALMAN_FILTER_HPP

#include "KalmanFilter.hpp"

/**
 * @brief Standard Linear Kalman Filter (LKF).
 * Implements the classic Kalman equations for linear systems:
 * * Pred: x = Fx, P = FPF' + Q
 * * Upd:  x = x + K(z - Hx), P = (I - KH)P(I - KH)' + KRK' (Joseph Form)
 * * @tparam StateDim Fixed size of the system state vector.
 */
template <int StateDim>
class LinearKalmanFilter : public KalmanFilter<LinearKalmanFilter<StateDim>, StateDim> {
    using Base = KalmanFilter<LinearKalmanFilter<StateDim>, StateDim>;
    friend class KalmanFilter<LinearKalmanFilter<StateDim>, StateDim>;

public:
    using typename Base::StateVector;
    using typename Base::StateMatrix;
    using typename Base::ProcessModel;
    
    template <int Dim> 
    using SensorModel = typename Base::template SensorModel<Dim>;

    LinearKalmanFilter(const StateVector& x, const StateMatrix& P) : Base(x, P) {}

protected:
    // Pre-allocated Identity for optimization in Joseph Form update
    StateMatrix I_{StateMatrix::Identity()};

    // =========================================================================
    // Interface Implementation (Called by Base Class)
    // =========================================================================

    /**
     * @brief Computes a priori state and covariance (Prediction).
     * Applies F matrix, normalizes angles, adds Process Noise Q,
     * and enforces physical constraints.
     */
    void computePrediction(ProcessModel& model) {
        // x = F * x
        this->x_ = model.F() * this->x_;

        // Wrap angles if defined (e.g., heading)
        if (model.hasAngle()) {
            this->template normalizeAngles<StateDim>(this->x_, model.getAngleFlags());
        }

        // P = F * P * F' + Q
        this->P_ = model.F() * this->P_ * model.F().transpose() + model.Q();

        // Force physical constraints (clamping/projection)
        model.enforceConstraints(this->x_);
    }

    /**
     * @brief Computes posterior state and covariance (Correction).
     * Performs 3-tier gating before updating. Uses Joseph Form for numerical stability.
     * @return true if measurement accepted, false if gated/rejected.
     */
    template <int MeasureDim>
    bool computeUpdate(SensorModel<MeasureDim>& model, const Eigen::Matrix<double, MeasureDim, 1>& z) {
        using MeasureVector = Eigen::Matrix<double, MeasureDim, 1>;
        using MeasureMatrix = Eigen::Matrix<double, MeasureDim, MeasureDim>;
        using MatrixK = Eigen::Matrix<double, StateDim, MeasureDim>;
        
        // 1. Innovation: y = z - Hx
        MeasureVector y = z - model.H() * this->x_;

        // Wrap innovation angles (critical for shortest path error)
        if (model.hasAngle()) {
            this->template normalizeAngles<MeasureDim>(y, model.getAngleFlags());
        }

        // 2. Innovation Covariance: S = H P H' + R
        MeasureMatrix S = model.H() * this->P_ * model.H().transpose() + model.R();

        // 3. Gating (Outlier Rejection)
        if (!this->rectangularGate(model, S, y)) return false; // Tier 1: Fast Sigma check
        if (!model.domainGate(z, y, S)) return false;          // Tier 2: User Logic
        
        Eigen::LDLT<MeasureMatrix> S_ldlt(S);
        if (!this->mahalanobisGate(model, S_ldlt, y)) return false; // Tier 3: Statistical

        // 4. Kalman Gain: K = P H' S^-1
        // Uses LDLT for stable inversion of symmetric positive definite S
        MatrixK K = this->P_ * model.H().transpose() * S_ldlt.solve(MeasureMatrix::Identity());

        // 5. State Update: x = x + Ky
        this->x_ += K * y;

        // 6. Covariance Update (Joseph Form): P = (I-KH)P(I-KH)' + KRK'
        // Guarantees P remains Symmetric Positive Definite regardless of numerical noise
        StateMatrix I_KH = I_ - K * model.H();
        this->P_ = I_KH * this->P_ * I_KH.transpose() + K * model.R() * K.transpose();

        return true;
    }
};

#endif