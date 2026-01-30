#ifndef EXTENDED_INFORMATION_FILTER_HPP
#define EXTENDED_INFORMATION_FILTER_HPP

#include <utility>
#include <vector>

#include "KalmanFilter.hpp"

/**
 * @brief Extended Information Filter (EIF).
 * * Implements the Information Form of the EKF.
 * * This filter tracks the Information State (y, Y) instead of (x, P).
 * * Y = P^-1  (Information Matrix)
 * * y = Y * x (Information Vector)
 * * Advantages:
 * 1. Multi-sensor fusion is purely additive (y += i, Y += I).
 * 2. Handling large sensor arrays is cheaper than EKF.
 * * Disadvantages:
 * 1. Prediction requires matrix inversion.
 * 2. Recovering state x requires matrix inversion.
 * * @tparam StateDim Fixed size of the state vector.
 */
template <int StateDim>
class ExtendedInformationFilter : public KalmanFilter<ExtendedInformationFilter<StateDim>, StateDim> {
    using Base = KalmanFilter<ExtendedInformationFilter<StateDim>, StateDim>;
    friend class KalmanFilter<ExtendedInformationFilter<StateDim>, StateDim>;

public:
    using typename Base::StateVector;
    using typename Base::StateMatrix;
    using typename Base::ProcessModel;
    
    // Helper alias for generic Sensor Models
    template <int Dim> 
    using SensorModel = typename Base::template SensorModel<Dim>;

    // Type for Information Space return values (i, I)
    using InformationPair = std::pair<StateVector, StateMatrix>;

    /**
     * @brief Constructor.
     * Initializes the filter in Information Space.
     * @param x Initial State Vector
     * @param P Initial Covariance Matrix
     */
    ExtendedInformationFilter(const StateVector& x, const StateMatrix& P)
        : Base(x, P) {
        // Initialize Information Matrix (Y) and Vector (y)
        Y_ = P.inverse();
        y_ = Y_ * this->x_;
    }

    /**
     * @brief Batch Update (Optimization).
     * Linearizes H *once* at the current estimate, then fuses all measurements.
     * Inverts Y -> P only once at the end.
     */
    template <int MeasureDim>
    void updateBatch(SensorModel<MeasureDim>& model, 
                     const std::vector<Eigen::Matrix<double, MeasureDim, 1>>& zs) {
        
        using MeasureVector = Eigen::Matrix<double, MeasureDim, 1>;
        using MatrixH = Eigen::Matrix<double, MeasureDim, StateDim>;

        // 1. Update Jacobian H (Once for the batch)
        // EKF Assumption: H is constant at the operating point x
        if (model.automaticJacobian()) {
            MatrixH H_num;
            this->template computeJacobian<MatrixH, MeasureVector>(
                this->x_, H_num,
                [&model](const StateVector& s) { return model.hx(s); }
            );
            model.setH(H_num);
        } else {
            model.computeAnalyticalJacobian(this->x_);
        }

        // 2. Cache R_inv
        auto R_inv = model.R().inverse();

        // 3. Accumulate Information
        for (const auto& z : zs) {
            InformationPair contrib = getInformationContribution<MeasureDim>(z, model, R_inv);
            y_ += contrib.first;
            Y_ += contrib.second;
        }

        // 4. Sync State (Single Inversion)
        syncState();
    }

protected:
    // Information Space State
    StateVector y_;
    StateMatrix Y_;

    // =========================================================================
    // Interface Implementation (Called by Base Class)
    // =========================================================================

    /**
     * @brief Prediction Step.
     * 1. Updates Jacobian F.
     * 2. Propagates State x (Non-Linear).
     * 3. Propagates Covariance P (Linearized).
     * 4. Updates Information State (Inversion).
     */
    void computePrediction(ProcessModel& model) {
        // 1. Compute Jacobian F
        if (model.automaticJacobian()) {
            StateMatrix F_num;
            this->template computeJacobian<StateMatrix, StateVector>(
                this->x_, F_num, 
                [&model](const StateVector& s) { return model.fx(s); }
            );
            model.setF(F_num);
        } else {
            model.computeAnalyticalJacobian(this->x_);
        }

        // 2. Predict State (Non-Linear): x = f(x)
        this->x_ = model.fx(this->x_);

        // Handle State Angle Wrapping
        if (model.hasAngle()) {
            this->template normalizeAngles<StateDim>(this->x_, model.getAngleFlags());
        }

        // 3. Predict Covariance (Linearized): P = F * P * F' + Q
        this->P_ = model.F() * this->P_ * model.F().transpose() + model.Q();

        // 4. Update Information Space (Expensive Inversion)
        // In EIF, prediction is the bottleneck.
        Y_ = this->P_.inverse();
        y_ = Y_ * this->x_;
    }

    /**
     * @brief Update Step.
     * 1. Updates Jacobian H.
     * 2. Computes Information contribution.
     * 3. Accumulates and Syncs.
     */
    template <int MeasureDim>
    void computeUpdate(SensorModel<MeasureDim>& model, const Eigen::Matrix<double, MeasureDim, 1>& z) {
        using MeasureVector = Eigen::Matrix<double, MeasureDim, 1>;
        using MatrixH = Eigen::Matrix<double, MeasureDim, StateDim>;

        // 1. Compute Jacobian H
        if (model.automaticJacobian()) {
            MatrixH H_num;
            this->template computeJacobian<MatrixH, MeasureVector>(
                this->x_, H_num,
                [&model](const StateVector& s) { return model.hx(s); }
            );
            model.setH(H_num);
        } else {
            model.computeAnalyticalJacobian(this->x_);
        }

        // 2. Compute Contribution
        auto R_inv = model.R().inverse();
        InformationPair contrib = getInformationContribution<MeasureDim>(z, model, R_inv);

        // 3. Accumulate
        y_ += contrib.first;
        Y_ += contrib.second;

        // 4. Sync State
        syncState();
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    /**
     * @brief Calculates EIF Information Contribution (i, I).
     * Uses Pseudo-Measurement logic for EKF consistency:
     * zeta = (z - h(x)) + H*x
     */
    template <int MeasureDim>
    InformationPair getInformationContribution(
        const Eigen::Matrix<double, MeasureDim, 1>& z,
        const SensorModel<MeasureDim>& model,
        const Eigen::Matrix<double, MeasureDim, MeasureDim>& R_inv) 
    {
        using MeasureVector = Eigen::Matrix<double, MeasureDim, 1>;

        // 1. Calculate Expected Measurement
        MeasureVector hx = model.hx(this->x_);
        
        // 2. Innovation: z - h(x)
        MeasureVector innovation = z - hx;

        // Handle Angle Wrapping on Innovation
        if (model.hasAngle()) {
            this->template normalizeAngles<MeasureDim>(innovation, model.getAngleFlags());
        }

        // 3. Pseudo-Measurement (Linearization point adjustment)
        // zeta = innovation + H * x
        MeasureVector zeta = innovation + model.H() * this->x_;

        // 4. Information Contribution
        // i = H' * R^-1 * zeta
        // I = H' * R^-1 * H
        StateVector i = model.H().transpose() * R_inv * zeta;
        StateMatrix I = model.H().transpose() * R_inv * model.H();

        return {i, I};
    }

    void syncState() {
        // P = Y^-1
        this->P_ = Y_.inverse();
        
        // x = P * y
        this->x_ = this->P_ * y_;

        // Note: Angle wrapping usually happens at the Prediction step (on State) 
        // or Update step (on Innovation). We rarely wrap 'x' here directly 
        // because we lack the angle flags from the ProcessModel in this scope.
    }
};

#endif