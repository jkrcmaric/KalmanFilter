#ifndef EXTENDED_KALMAN_FILTER_HPP
#define EXTENDED_KALMAN_FILTER_HPP

#include "KalmanFilter.hpp"

/**
 * @brief Extended Kalman Filter (EKF).
 * * Implements the standard EKF for non-linear systems.
 * * Pred: x = f(x), P = FPF' + Q
 * * Upd:  K = PH'S^-1, x = x + K(z - h(x)), P = (I - KH)P(I - KH)' + KRK'
 * * @tparam StateDim Fixed size of the state vector.
 */
template <int StateDim>
class ExtendedKalmanFilter : public KalmanFilter<ExtendedKalmanFilter<StateDim>, StateDim> {
    using Base = KalmanFilter<ExtendedKalmanFilter<StateDim>, StateDim>;
    friend class KalmanFilter<ExtendedKalmanFilter<StateDim>, StateDim>;

public:
    using typename Base::StateVector;
    using typename Base::StateMatrix;
    using typename Base::ProcessModel;
    
    // Helper alias for generic Sensor Models
    template <int Dim> 
    using SensorModel = typename Base::template SensorModel<Dim>;

    /**
     * @brief Constructor.
     * @param x Initial State Vector
     * @param P Initial Covariance Matrix
     */
    ExtendedKalmanFilter(const StateVector& x, const StateMatrix& P)
        : Base(x, P) {}

protected:
    // Identity matrix cached for efficiency in updates
    StateMatrix I_{StateMatrix::Identity()};

    // =========================================================================
    // Interface Implementation (Called by Base Class)
    // =========================================================================

    /**
     * @brief Prediction Step.
     * 1. Updates Jacobian F (Numerical or Analytical).
     * 2. Propagates State x (Non-Linear f(x)).
     * 3. Propagates Covariance P (Linearized F).
     */
    void computePrediction(ProcessModel& model) {
        // 1. Compute Jacobian F
        model.updateJacobian(this->x_);

        // 2. Predict State (Non-Linear): x = f(x)
        // Note: The model.fx() handles the fallback to F*x if no function is set
        this->x_ = model.fx(this->x_);

        // Handle State Angle Wrapping
        if (model.hasAngle()) {
            this->template normalizeAngles<StateDim>(this->x_, model.getAngleFlags());
        }

        // 3. Predict Covariance (Linearized): P = F * P * F' + Q
        this->P_ = model.F() * this->P_ * model.F().transpose() + model.Q();

        // Force physical constraints (clamping/projection)
        model.enforceConstraints(this->x_);
    }

    /**
     * @brief Update Step.
     * 1. Updates Jacobian H (Numerical or Analytical).
     * 2. Computes Innovation (z - h(x)).
     * 3. Standard Kalman Gain and Covariance Update.
     */
    template <int MeasureDim>
    bool computeUpdate(SensorModel<MeasureDim>& model, const Eigen::Matrix<double, MeasureDim, 1>& z) {
        using MeasureMatrix = Eigen::Matrix<double, MeasureDim, MeasureDim>;
        using MeasureVector = Eigen::Matrix<double, MeasureDim, 1>;
        using MatrixH = Eigen::Matrix<double, MeasureDim, StateDim>;
        using MatrixK = Eigen::Matrix<double, StateDim, MeasureDim>;

        // 1. Compute Jacobian H
        model.updateJacobian(this->x_);

        // 2. Innovation: y = z - h(x)
        MeasureVector y = z - model.hx(this->x_);

        // Handle Angle Wrapping on Innovation
        if (model.hasAngle()) {
            this->template normalizeAngles<MeasureDim>(y, model.getAngleFlags());
        }

        // 3. Innovation Covariance: S = H * P * H' + R
        MeasureMatrix S = model.H() * this->P_ * model.H().transpose() + model.R();

        // 4. Gating (Outlier Rejection)
        if (!this->rectangularGate(model, S, y)) return false; // Tier 1: Fast Sigma check
        if (!model.domainGate(z, y, S)) return false;          // Tier 2: User Logic

        Eigen::LDLT<MeasureMatrix> S_ldlt(S);
        if (!this->mahalanobisGate(model, S_ldlt, y)) return false; // Tier 3: Statistical

        // 5. Kalman Gain: K = P * H' * S^-1
        // Use LDLT for stability
        MatrixK K = this->P_ * model.H().transpose() * S_ldlt.solve(MeasureMatrix::Identity());

        // 6. Update State: x = x + K * y
        this->x_ = this->x_ + K * y;

        // 7. Update Covariance (Joseph Form): P = (I - KH)P(I - KH)' + KRK'
        StateMatrix I_KH = I_ - K * model.H();
        this->P_ = I_KH * this->P_ * I_KH.transpose() + K * model.R() * K.transpose();

        return true;
    }
};

#endif