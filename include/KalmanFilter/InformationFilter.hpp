#ifndef INFORMATION_FILTER_HPP
#define INFORMATION_FILTER_HPP

#include <utility>
#include <vector>
#include <cmath>

#include "KalmanFilter.hpp"

/**
 * @brief Information Filter (IF).
 * Implements the "Inverse Covariance" form of the Kalman Filter.
 * * Tracks Information State: y (Information Vector) and Y (Information Matrix).
 * * Relationship: Y = P^-1, y = Y * x.
 * * Architecture: Hybrid IF. Predicts in Covariance space (for non-linearities), 
 * Updates in Information space (for fast fusion).
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
    
    template <int Dim> 
    using SensorModel = typename Base::template SensorModel<Dim>;

    // Return type for contribution (i, I)
    using InformationPair = std::pair<StateVector, StateMatrix>;

    /**
     * @brief Constructor.
     * Initializes the filter in Information Space.
     * @param x Initial State Vector
     * @param P Initial Covariance Matrix (Must be invertible/non-singular)
     */
    InformationFilter(const StateVector& x, const StateMatrix& P)
        : Base(x, P) {
        // Initialize Information Matrix (Y) and Vector (y)
        Y_ = P.inverse();
        y_ = Y_ * this->x_;
    }

    /**
     * @brief Fuses a single measurement WITHOUT updating the covariance P.
     * * Lazy Update: Accumulates information (y+=i, Y+=I) but skips inversion (x=Y^-1*y).
     * * Optimization: Only calculates LDLT for Mahalanobis gating if threshold is finite.
     * * @return true if accepted, false if rejected by gate.
     */
    template <int MeasureDim>
    bool fuse(SensorModel<MeasureDim>& model, const Eigen::Matrix<double, MeasureDim, 1>& z) {
        using MeasureVector = Eigen::Matrix<double, MeasureDim, 1>;
        using MeasureMatrix = Eigen::Matrix<double, MeasureDim, MeasureDim>;

        // 1. Linearize H at current state
        model.updateJacobian(this->x_);

        // 2. Compute Innovation (for Gating)
        MeasureVector innovation = z - model.hx(this->x_);
        if (model.hasAngle()) {
            this->template normalizeAngles<MeasureDim>(innovation, model.getAngleFlags());
        }

        // 3. Compute Innovation Covariance S (Using Prior P_)
        MeasureMatrix S = model.H() * this->P_ * model.H().transpose() + model.R();

        // 4. Perform Gating (Lazy Evaluation of LDLT)
        if (!gating(model, z, innovation, S)) return false;

        // 5. Compute & Accumulate Information
        auto R_inv = model.R().inverse();
        InformationPair contrib = getInformationContribution<MeasureDim>(innovation, model, R_inv);

        y_ += contrib.first;
        Y_ += contrib.second;

        return true;
    }

    /**
     * @brief Fuses a batch of measurements of the SAME type.
     * * Optimization 1: Computes R^-1 only once.
     * * Optimization 2: Computes S and S_ldlt only once for the whole batch.
     * * @return int Number of measurements successfully fused (passed gating).
     */
    template <int MeasureDim>
    int fuseBatch(SensorModel<MeasureDim>& model, 
                     const std::vector<Eigen::Matrix<double, MeasureDim, 1>>& zs) {
        using MeasureVector = Eigen::Matrix<double, MeasureDim, 1>;
        using MeasureMatrix = Eigen::Matrix<double, MeasureDim, MeasureDim>;

        int fused_count{0};
        model.updateJacobian(this->x_);
        
        // 1. Pre-compute shared matrices
        auto R_inv = model.R().inverse();
        MeasureMatrix S = model.H() * this->P_ * model.H().transpose() + model.R();
        
        // 2. Conditional LDLT: Compute ONCE if Mahalanobis gating is active
        Eigen::LDLT<MeasureMatrix> S_ldlt;
        if (!std::isinf(model.getMahalanobisThreshold())) {
            S_ldlt.compute(S);
        }

        for (const auto& z : zs) {
            MeasureVector innovation = z - model.hx(this->x_);
            if (model.hasAngle()) {
                this->template normalizeAngles<MeasureDim>(innovation, model.getAngleFlags());
            }

            // 3. Gate using pre-computed decompositions
            if (!gating(model, z, innovation, S, S_ldlt)) continue;
                
            ++fused_count;
            InformationPair contrib = getInformationContribution<MeasureDim>(innovation, model, R_inv);
            y_ += contrib.first;
            Y_ += contrib.second;
        }

        return fused_count;
    }

    /**
     * @brief Manually synchronizes P and x from Y and y.
     * Performs the expensive matrix inversion (P = Y^-1). 
     * Call this once after a batch of fuse() calls.
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
     * Implementation: Hybrid Prediction.
     * 1. Propagates P (Covariance) using standard non-linear EKF math.
     * 2. Enforces constraints.
     * 3. Inverts P -> Y (Expensive Step).
     */
    void computePrediction(ProcessModel& model) {
        // 1. Linearize Dynamics
        model.updateJacobian(this->x_);

        // 2. Predict State (Non-Linear)
        this->x_ = model.fx(this->x_);
        if (model.hasAngle()) {
            this->template normalizeAngles<StateDim>(this->x_, model.getAngleFlags());
        }

        // 3. Predict Covariance
        this->P_ = model.F() * this->P_ * model.F().transpose() + model.Q();

        // 4. Constraints
        model.enforceConstraints(this->x_);

        // 5. Update Information Space (Inversion)
        Y_ = this->P_.inverse();
        y_ = Y_ * this->x_;
    }

    /**
     * @brief Standard Update Step (Single Measurement).
     * Wraps fuse() and immediately synchronizes state.
     */
    template <int MeasureDim>
    bool computeUpdate(SensorModel<MeasureDim>& model, const Eigen::Matrix<double, MeasureDim, 1>& z) {
        
        if (!fuse(model, z)) return false;

        // Recover standard State (x, P) immediately
        syncState();

        return true;
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    /**
     * @brief Calculates the Information Contribution (i, I).
     * Uses Pseudo-Measurement (zeta) to handle non-linearities/angles.
     * i = H' R^-1 (z - h(x) + Hx)
     * I = H' R^-1 H
     */
    template <int MeasureDim>
    InformationPair getInformationContribution(
        const Eigen::Matrix<double, MeasureDim, 1>& innovation,
        const SensorModel<MeasureDim>& model,
        const Eigen::Matrix<double, MeasureDim, MeasureDim>& R_inv) 
    {
        using MeasureVector = Eigen::Matrix<double, MeasureDim, 1>;

        // Pseudo-Measurement: zeta = innovation + Hx
        MeasureVector zeta = innovation + model.H() * this->x_;
            
        StateVector i = model.H().transpose() * R_inv * zeta;
        StateMatrix I = model.H().transpose() * R_inv * model.H();

        return {i, I};
    }

    /**
     * @brief Gating Helper (Lazy Computation).
     * Computes LDLT only if Rectangular/Domain gates pass AND threshold is finite.
     */
    template <int MeasureDim>
    bool gating(SensorModel<MeasureDim>& model,
                const Eigen::Matrix<double, MeasureDim, 1>& z,
                const Eigen::Matrix<double, MeasureDim, 1>& innovation,
                const Eigen::Matrix<double, MeasureDim, MeasureDim>& S) {
        using MeasureMatrix = Eigen::Matrix<double, MeasureDim, MeasureDim>;

        // 1. Fast Gates
        if (!this->rectangularGate(model, S, innovation)) return false;
        if (!model.domainGate(z, innovation, S)) return false;

        // 2. Expensive Gate (Conditional)
        if (std::isinf(model.getMahalanobisThreshold())) return true;
        
        Eigen::LDLT<MeasureMatrix> S_ldlt(S); // Compute decomposition here
        return this->mahalanobisGate(model, S_ldlt, innovation);
    }

    /**
     * @brief Gating Helper (Pre-Computed).
     * Uses an existing LDLT decomposition (optimized for batch processing).
     */
    template <int MeasureDim>
    bool gating(SensorModel<MeasureDim>& model,
                const Eigen::Matrix<double, MeasureDim, 1>& z,
                const Eigen::Matrix<double, MeasureDim, 1>& innovation,
                const Eigen::Matrix<double, MeasureDim, MeasureDim>& S,
                const Eigen::LDLT<Eigen::Matrix<double, MeasureDim, MeasureDim>>& S_ldlt) {
        using MeasureMatrix = Eigen::Matrix<double, MeasureDim, MeasureDim>;

        if (!this->rectangularGate(model, S, innovation)) return false;
        if (!model.domainGate(z, innovation, S)) return false;

        // Note: Base mahalanobisGate checks for isinf internally to avoid using S_ldlt if invalid
        return this->mahalanobisGate(model, S_ldlt, innovation);
    }

    /**
     * @brief Synchronize P and x from Y and y.
     * Inverts the Information Matrix (Expensive O(N^3)).
     */
    void syncState() {
        this->P_ = Y_.inverse();
        this->x_ = this->P_ * y_;
    }
};

#endif