#ifndef LINEAR_KALMAN_FILTER_HPP
#define LINEAR_KALMAN_FILTER_HPP

#include "KalmanFilter.hpp"

/**
 * @brief Standard Linear Kalman Filter (LKF).
 * * Implements the standard Kalman equations for linear systems:
 * Pred: x = Fx, P = FPF' + Q
 * Upd:  x = x + K(z - Hx), P = (I - KH)P(I - KH)' + KRK'
 * * @note This class is STATELESS regarding the system model. 
 * You must pass the ProcessModel to predict() and SensorModel to update().
 * * @tparam StateDim The fixed size of the system state vector.
 */
template <int StateDim>
class LinearKalmanFilter : public KalmanFilter<LinearKalmanFilter<StateDim>, StateDim> {
    using Base = KalmanFilter<LinearKalmanFilter<StateDim>, StateDim>;
    friend class KalmanFilter<LinearKalmanFilter<StateDim>, StateDim>;

public:
    // Expose Base types for easier usage
    using typename Base::StateVector;
    using typename Base::StateMatrix;
    using typename Base::ProcessModel;
    
    // SensorModel is a template, so we use a template alias
    template <int Dim> 
    using SensorModel = typename Base::template SensorModel<Dim>;

    /**
     * @brief Constructor.
     * @param x Initial State Vector
     * @param P Initial Covariance Matrix
     */
    LinearKalmanFilter(const StateVector& x, const StateMatrix& P)
        : Base(x, P) {}

protected:
    // Pre-allocated Identity matrix for optimization in Joseph Form update
    StateMatrix I_{StateMatrix::Identity()};

    // =========================================================================
    // Interface Implementation (Called by Base Class)
    // =========================================================================

    /**
     * @brief Computes the a priori state and covariance.
     * Implements: x = Fx, P = FPF' + Q
     * * @param model Reference to the process model (F, Q)
     */
    void computePrediction(ProcessModel& model) {
        // 1. Predict State: x = F * x
        // We use the getter F() because the base ProcessModel encapsulates the matrix.
        this->x_ = model.F() * this->x_;

        // 2. Handle Angle Wrapping (e.g., if state includes heading)
        // Checks flags inside the model and wraps x_ to [-PI, PI] if needed.
        if (model.hasAngle()) {
            this->template normalizeAngles<StateDim>(this->x_, model.getAngleFlags());
        }

        // 3. Predict Covariance: P = F * P * F' + Q
        this->P_ = model.F() * this->P_ * model.F().transpose() + model.Q();
    }

    /**
     * @brief Computes the posterior state and covariance.
     * Implements: K = PH'S^-1, x = x + Ky, P_joseph
     * * @tparam MeasureDim, Automatically deduced from the sensor model!
     * * @param model Reference to the sensor model (H, R)
     * * @param z The actual measurement vector
     */
    template <int MeasureDim>
    void computeUpdate(SensorModel<MeasureDim>& model, const Eigen::Matrix<double, MeasureDim, 1>& z) {
        using MeasureVector = Eigen::Matrix<double, MeasureDim, 1>;
        using MeasureMatrix = Eigen::Matrix<double, MeasureDim, MeasureDim>;
        using MatrixK = Eigen::Matrix<double, StateDim, MeasureDim>;
        
        // 1. Compute Innovation: y = z - H * x
        // Use H() getter from the SensorModel
        MeasureVector y = z - model.H() * this->x_;

        // 2. Handle Angle Wrapping on Innovation
        // Critical for bearings/headings to ensure error is the "shortest path"
        if (model.hasAngle()) {
            this->template normalizeAngles<MeasureDim>(y, model.getAngleFlags());
        }

        // 3. Innovation Covariance: S = H * P * H' + R
        MeasureMatrix S = model.H() * this->P_ * model.H().transpose() + model.R();

        // 4. Kalman Gain: K = P * H' * S^-1
        // Uses LDLT decomposition for numerically stable inversion of S (symmetric positive definite)
        MatrixK K = this->P_ * model.H().transpose() * S.ldlt().solve(MeasureMatrix::Identity());

        // 5. Update State: x = x + K * y
        this->x_ = this->x_ + K * y;

        // 6. Update Covariance (Joseph Form): P = (I - KH)P(I - KH)' + KRK'
        // This form is computationally more expensive but guarantees that P remains 
        // Symmetric and Positive Definite, preventing filter divergence due to numerical error.
        StateMatrix I_KH = I_ - K * model.H();
        this->P_ = I_KH * this->P_ * I_KH.transpose() + K * model.R() * K.transpose();
    }
};

#endif