#ifndef INFORMATION_FILTER_HPP
#define INFORMATION_FILTER_HPP

#include <utility>
#include <vector>

#include "KalmanFilter.hpp"

/**
 * @brief Linear Information Filter (IF).
 * * The IF tracks the Information State (y, Y) instead of (x, P).
 * * Y = P^-1  (Information Matrix)
 * * y = Y * x (Information Vector)
 * * Advantages:
 * 1. Initialization with infinite uncertainty (P=inf) is easy (Y=0).
 * 2. Multi-sensor fusion is purely additive and very cheap (Y += I).
 * * Disadvantages:
 * 1. Prediction is computationally expensive (requires inversion).
 * 2. Recovering state x requires inversion (P = Y^-1).
 * * @tparam StateDim Fixed size of the state vector.
 */
template <int StateDim>
class InformationFilter : public KalmanFilter<InformationFilter<StateDim>, StateDim> {
    using Base = KalmanFilter<InformationFilter<StateDim>, StateDim>;
    friend class KalmanFilter<InformationFilter<StateDim>, StateDim>;

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
    InformationFilter(const StateVector& x, const StateMatrix& P)
        : Base(x, P) {
        // Initialize Information Matrix (Y) and Vector (y)
        // Note: If P is zero (perfect certainty), this inverse will fail. 
        // IF requires some uncertainty to start usually, or Y initialized manually.
        Y_ = P.inverse();
        y_ = Y_ * this->x_;
    }

    /**
     * @brief Specialized Batch Update.
     * Fuses a vector of measurements of the SAME type efficiently.
     * * Efficiency: Accumulates information (additive) and performs 
     * matrix inversion (syncState) ONLY ONCE at the end.
     * * @tparam MeasureDim Dimension of the sensor
     * @param model The sensor model (H, R)
     * @param zs Vector of measurements
     */
    template <int MeasureDim>
    void updateBatch(const SensorModel<MeasureDim>& model, 
                     const std::vector<Eigen::Matrix<double, MeasureDim, 1>>& zs) {
        
        // Cache Inverse of R once for the whole batch
        auto R_inv = model.R().inverse();

        for (const auto& z : zs) {
            InformationPair contrib = getInformationContribution<MeasureDim>(z, model, R_inv);
            y_ += contrib.first;
            Y_ += contrib.second;
        }
        
        // Invert Y -> P only once after fusing all data
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
     * The IF prediction is expensive because Y cannot be propagated linearly.
     * We convert to P, propagate P, and convert back to Y.
     * (Hybrid Information Filter approach).
     */
    void computePrediction(ProcessModel& model) {
        // 1. Predict State: x = F * x
        this->x_ = model.F() * this->x_;

        // Handle State Angle Wrapping
        if (model.hasAngle()) {
            this->template normalizeAngles<StateDim>(this->x_, model.getAngleFlags());
        }

        // 2. Predict Covariance: P = F * P * F' + Q
        this->P_ = model.F() * this->P_ * model.F().transpose() + model.Q();

        // 3. Update Information Space (The expensive part)
        Y_ = this->P_.inverse();
        y_ = Y_ * this->x_;
    }

    /**
     * @brief Update Step (Single Measurement).
     * Computes contributions i and I, adds them, and syncs P.
     */
    template <int MeasureDim>
    void computeUpdate(SensorModel<MeasureDim>& model, const Eigen::Matrix<double, MeasureDim, 1>& z) {
        
        // Calculate R_inv locally
        auto R_inv = model.R().inverse();

        // 1. Get Contribution
        InformationPair contrib = getInformationContribution<MeasureDim>(z, model, R_inv);

        // 2. Accumulate Information
        y_ += contrib.first;
        Y_ += contrib.second;

        // 3. Recover standard State (x, P)
        syncState();
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    /**
     * @brief Calculates the Information Contribution (i, I) for a measurement.
     * Handles angle wrapping via Pseudo-Measurement logic.
     */
    template <int MeasureDim>
    InformationPair getInformationContribution(
        const Eigen::Matrix<double, MeasureDim, 1>& z,
        const SensorModel<MeasureDim>& model,
        const Eigen::Matrix<double, MeasureDim, MeasureDim>& R_inv) 
    {
        using MeasureVector = Eigen::Matrix<double, MeasureDim, 1>;

        // Calculate Information Matrix Contribution: I = H' * R^-1 * H
        StateMatrix I = model.H().transpose() * R_inv * model.H();
        StateVector i;

        // Calculate Information Vector Contribution: i = H' * R^-1 * z
        // Note: If angles are involved, we cannot simply use z. 
        // We must compute a "Pseudo-measurement" that accounts for the wrap.
        if (model.hasAngle()) {
            // 1. Expected measurement
            MeasureVector Hx = model.H() * this->x_;
            
            // 2. Innovation with wrapping
            MeasureVector innovation = z - Hx;
            this->template normalizeAngles<MeasureDim>(innovation, model.getAngleFlags());
            
            // 3. Pseudo-Measurement (Linearized in the correct manifold)
            MeasureVector zeta = innovation + Hx;
            
            // 4. Compute 'i' using Pseudo-Measurement
            i = model.H().transpose() * R_inv * zeta;
        } else {
            // Standard Linear Case
            i = model.H().transpose() * R_inv * z;
        }

        return {i, I};
    }

    /**
     * @brief Synchronize P and x from Y and y.
     * Inverts the Information Matrix.
     */
    void syncState() {
        // P = Y^-1
        this->P_ = Y_.inverse();
        
        // x = P * y
        this->x_ = this->P_ * y_;
    }
};

#endif