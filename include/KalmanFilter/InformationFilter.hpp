#ifndef INFORMATION_FILTER_HPP
#define INFORMATION_FILTER_HPP

#include <utility>
#include <vector>

#include "KalmanFilter.hpp"

/**
 * @brief Information Filter (IF).
 * * The IF tracks the Information State (y, Y) instead of (x, P).
 * * Y = P^-1  (Information Matrix)
 * * y = Y * x (Information Vector)
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
     * @brief Fuses a measurement WITHOUT updating the covariance P.
     * Use this when you have multiple different sensors to process in one step.
     * 1. Call predict()
     * 2. Call fuse(radar, z1)
     * 3. Call fuse(lidar, z2)
     * 4. Call updateState() <- Expensive Inversion happens ONCE here.
     */
    template <int MeasureDim>
    void fuse(SensorModel<MeasureDim>& model, const Eigen::Matrix<double, MeasureDim, 1>& z) {

        model.updateJacobian(this->x_);

        auto R_inv = model.R().inverse();
        
        InformationPair contrib = getInformationContribution<MeasureDim>(z, model, R_inv);

        y_ += contrib.first;
        Y_ += contrib.second;
    }

    /**
     * @brief Fuses a batch of measurements of the SAME type.
     * * Optimized to compute R_inv only once.
     * * Does NOT update (x, P). You must call updateState() later.
     * * @tparam MeasureDim Dimension of the sensor
     * @param model The sensor model (H, R)
     * @param zs Vector of measurements
     */
    template <int MeasureDim>
    void fuseBatch(SensorModel<MeasureDim>& model, 
                     const std::vector<Eigen::Matrix<double, MeasureDim, 1>>& zs) {

        model.updateJacobian(this->x_);
        
        // Cache Inverse of R once for the whole batch
        auto R_inv = model.R().inverse();

        for (const auto& z : zs) {
            InformationPair contrib = getInformationContribution<MeasureDim>(z, model, R_inv);
            y_ += contrib.first;
            Y_ += contrib.second;
        }
    }

    /**
     * @brief Manually synchronizes P and x from Y and y.
     * Call this once after fusing all sensors.
     */
    void updateState() {
        syncState();
    }

    // Accessors
    void setState(const StateVector& x) { 
        this->x_ = x; 
        y_ = Y_ * this->x_;
    }
    void setP(const StateMatrix& P) { 
        this->P_ = P; 
        Y_ = this->P_.inverse();
        y_ = Y_ * this->x_;
    }
    void setY(const StateMatrix& Y) {
        Y_ = Y;
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

        // Compute Jacobian F (if non-linear)
        model.updateJacobian(this->x_);

        // Predict State: x = F * x (Linear) or fx(x) (Non-Linear)
        this->x_ = model.fx(this->x_);

        // Handle State Angle Wrapping
        if (model.hasAngle()) {
            this->template normalizeAngles<StateDim>(this->x_, model.getAngleFlags());
        }

        // Predict Covariance: P = F * P * F' + Q
        this->P_ = model.F() * this->P_ * model.F().transpose() + model.Q();

        // Update Information Space (The expensive part)
        Y_ = this->P_.inverse();
        y_ = Y_ * this->x_;
    }

    /**
     * @brief Update Step (Single Measurement).
     * Computes contributions i and I, adds them, and syncs P.
     */
    template <int MeasureDim>
    void computeUpdate(SensorModel<MeasureDim>& model, const Eigen::Matrix<double, MeasureDim, 1>& z) {

        // Compute Jacobian H (if non-linear)
        model.updateJacobian(this->x_);
        
        // Calculate R_inv locally
        auto R_inv = model.R().inverse();

        // Get Contribution
        InformationPair contrib = getInformationContribution<MeasureDim>(z, model, R_inv);

        // Accumulate Information
        y_ += contrib.first;
        Y_ += contrib.second;

        // Recover standard State (x, P)
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

        // Calculate Information Vector Contribution: i = H' * R^-1 * z
        // Note: If non-linear or angles are involved, we cannot simply use z. 
        // We must compute a "Pseudo-measurement" that accounts for the wrap.
        
        // 1. Expected measurement
        MeasureVector hx = model.hx(this->x_);

        MeasureVector innovation = z - hx;
            
        // 2. Innovation with wrapping
         if (model.hasAngle()) {
            this->template normalizeAngles<MeasureDim>(innovation, model.getAngleFlags());
         }
            
        // 3. Pseudo-Measurement (Linearized in the correct manifold)
        MeasureVector zeta = innovation + model.H() * this->x_;
            
        // 4. Compute 'i' using Pseudo-Measurement
        StateVector i = model.H().transpose() * R_inv * zeta;
        StateMatrix I = model.H().transpose() * R_inv * model.H();

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