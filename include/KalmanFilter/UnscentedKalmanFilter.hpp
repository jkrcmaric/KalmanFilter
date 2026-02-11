#ifndef UNSCENTED_KALMAN_FILTER_HPP
#define UNSCENTED_KALMAN_FILTER_HPP

#include "KalmanFilter.hpp"

/**
 * @brief Unscented Kalman Filter (UKF).
 * * Uses the Unscented Transform to handle highly non-linear process and measurement models
 * without requiring the computation of analytical or numerical Jacobians. It deterministically
 * extracts "Sigma Points" from the current Gaussian distribution, propagates them through
 * the non-linear functions, and reconstructs the new mean and covariance.
 * * @tparam StateDim Fixed size of the state vector.
 */
template <int StateDim>
class UnscentedKalmanFilter : public KalmanFilter<UnscentedKalmanFilter<StateDim>, StateDim> {
    using Base = KalmanFilter<UnscentedKalmanFilter<StateDim>, StateDim>;
    friend class KalmanFilter<UnscentedKalmanFilter<StateDim>, StateDim>;

public:
    using typename Base::StateVector;
    using typename Base::StateMatrix;
    using typename Base::ProcessModel;
    
    // Helper alias for generic Sensor Models
    template <int Dim>
    using SensorModel = typename Base::template SensorModel<Dim>;

    // Matrix to hold 2N + 1 sigma points as columns
    using SigmaMatrix = Eigen::Matrix<double, StateDim, 2*StateDim + 1>;
    // Vector to hold the weights for the 2N + 1 sigma points
    using WeightVector = Eigen::Matrix<double, 2*StateDim + 1, 1>;

    /**
     * @brief Generates sigma points and weights using the Merwe Scaled Unscented Transform.
     * * 
     * * Controls the spread of the sigma points to avoid sampling non-local non-linearities,
     * which is critical for highly non-linear systems.
     */
    class MerweScaledSigmaPoints {
    public:
        /**
         * @brief Constructor for Merwe Scaling parameters.
         * @param alpha Determines the spread of the sigma points around the mean. 
         * Typically a small positive value (e.g., 1e-3).
         * @param beta Incorporates prior knowledge of the distribution. 
         * For Gaussian distributions, beta = 2.0 is optimal.
         * @param kappa Secondary scaling parameter. Usually set to 0 or (3 - StateDim).
         */
        MerweScaledSigmaPoints(double alpha = 0.1, 
                               double beta = 2.0, 
                               double kappa = 1.0) {
            initialize(alpha, beta, kappa);
        }

        /**
         * @brief Calculates the deterministic sigma points around state x with covariance P.
         * Uses the Cholesky decomposition (LLT) to find the matrix square root of P.
         * @param x Current mean state vector.
         * @param P Current state covariance matrix.
         */
        void calculateSigmaPoints(const StateVector& x, const StateMatrix& P) {
            Eigen::LLT<StateMatrix> L_llt((n_ + lambda_) * P);
            StateMatrix L = L_llt.matrixL();

            sigmas_.col(0) = x;
            for (size_t i = 0; i < n_; ++i) {
                sigmas_.col(i+1) = x + L.col(i);
                sigmas_.col(n_+i+1) = x - L.col(i);
            }
        }

        // Accessors for Sigma Points and Weights
        auto getSigmas(int i) const { return sigmas_.col(i); }
        double getMeanWeights(int i) const { return Wm_(i); }
        double getCovarianceWeights(int i) const { return Wc_(i); }

    private:
        int n_{StateDim};
        double lambda_;

        WeightVector Wm_;
        WeightVector Wc_;
        SigmaMatrix sigmas_;

        void initialize(double alpha, double beta, double kappa) {
            lambda_ = alpha*alpha * (n_ + kappa) - n_;
            calculateWeights(alpha, beta);
        }

        void calculateWeights(double alpha, double beta) {
            Wm_.setConstant(1.0 / (2.0*n_ + 2.0*lambda_));
            Wc_.setConstant(1.0 / (2.0*n_ + 2.0*lambda_));

            Wm_(0) = lambda_ / (n_ + lambda_);
            Wc_(0) = lambda_ / (n_ + lambda_) + 1.0 - alpha*alpha + beta;
        }
    };

    /**
     * @brief Constructor.
     * Initializes the filter with a starting state and covariance.
     * @param x Initial State Vector
     * @param P Initial Covariance Matrix
     */
    UnscentedKalmanFilter(const StateVector& x, const StateMatrix& P) : Base(x, P) {}

    /**
     * @brief Configures the scaling parameters for the Unscented Transform.
     * @param alpha Spread of sigma points (e.g., 1e-3).
     * @param beta Distribution tuning (e.g., 2.0 for Gaussian).
     * @param kappa Secondary scaling (e.g., 0).
     */
    void setSigmaPointParameters(double alpha, double beta, double kappa) {
        sigma_points_ = MerweScaledSigmaPoints(alpha, beta, kappa);
    }

protected:
    MerweScaledSigmaPoints sigma_points_{MerweScaledSigmaPoints()};
    SigmaMatrix Y_;   // Propagated sigma points (Process space)
    SigmaMatrix Y_x_; // Deviations of propagated points from the mean

    // =========================================================================
    // Interface Implementation (Called by Base Class)
    // =========================================================================

    /**
     * @brief Prediction Step using the Unscented Transform.
     * 1. Extracts sigma points from the current state (x, P).
     * 2. Propagates each point through the non-linear process model (fx).
     * 3. Calculates the predicted mean using weighted deviations (safe for angles).
     * 4. Calculates the predicted covariance from the deviations.
     * @param model Process model defining system dynamics and constraints.
     */
    void computePrediction(ProcessModel& model) {
        // 1. Generate Sigma Points
        sigma_points_.calculateSigmaPoints(this->x_, this->P_);

        // 2 & 3. Propagate and Compute Mean via Deviations
        StateVector delta_x{StateVector::Zero()};
        for (size_t i = 0; i < Y_.cols(); ++i) {
            Y_.col(i) = model.fx(sigma_points_.getSigmas(i));
            
            // Wrap angles if defined (e.g., heading)
            if (model.hasAngle()) {
                this->template normalizeAngles<StateDim>(this->Y_.col(i), model.getAngleFlags());
            }

            StateVector diff = Y_.col(i) - Y_.col(0);
            
            // Wrap angular deviations
            if (model.hasAngle()) {
                this->template normalizeAngles<StateDim>(diff, model.getAngleFlags());
            }

            delta_x += sigma_points_.getMeanWeights(i)*diff;
        }
        
        // Recover new predicted state from anchor + deviation sum
        this->x_ = Y_.col(0) + delta_x;

        // Wrap final state angles
        if (model.hasAngle()) {
            this->template normalizeAngles<StateDim>(this->x_, model.getAngleFlags());
        }

        // 4. Compute Predicted Covariance
        StateMatrix Psum{StateMatrix::Zero()};
        for (size_t i = 0; i < Y_.cols(); ++i) {
            Y_x_.col(i) = Y_.col(i) - this->x_;
            
            // Wrap angular deviations from the new mean
            if (model.hasAngle()) {
                this->template normalizeAngles<StateDim>(Y_x_.col(i), model.getAngleFlags());
            }

            Psum += sigma_points_.getCovarianceWeights(i)*Y_x_.col(i)*Y_x_.col(i).transpose();
        }
        this->P_ = Psum + model.Q();

        // Force physical constraints (clamping/projection)
        model.enforceConstraints(this->x_);
    }

    /**
     * @brief Update Step using the Unscented Transform.
     * Maps the propagated sigma points into measurement space to compute the
     * expected measurement and cross-covariance, avoiding Jacobians entirely.
     * @tparam MeasureDim Dimension of the sensor data.
     * @param model Sensor model defining the measurement mapping (hx).
     * @param z The actual measurement vector received.
     * @return true if accepted by gating, false if rejected.
     */
    template <int MeasureDim>
    bool computeUpdate(SensorModel<MeasureDim>& model, const Eigen::Matrix<double, MeasureDim, 1>& z) {
        using MeasureMatrix = Eigen::Matrix<double, MeasureDim, MeasureDim>;
        using MeasureVector = Eigen::Matrix<double, MeasureDim, 1>;
        using MatrixK = Eigen::Matrix<double, StateDim, MeasureDim>;
        using MatrixZ = Eigen::Matrix<double, MeasureDim, 2*StateDim + 1>;

        MatrixZ Z{MatrixZ::Zero()};
        MeasureVector delta_z{MeasureVector::Zero()};
        
        // 1. Map Sigma Points to Measurement Space
        for (size_t i = 0; i < Y_.cols(); ++i) {
            Z.col(i) = model.hx(Y_.col(i));

            MeasureVector diff = Z.col(i) - Z.col(0);
            // Wrap angles if defined (e.g., bearing)
            if (model.hasAngle()) {
                this->template normalizeAngles<MeasureDim>(diff, model.getAngleFlags());
            }

            delta_z += sigma_points_.getMeanWeights(i)*diff;
        }
        
        // Compute predicted measurement (Anchor + Deviation)
        MeasureVector uz = Z.col(0) + delta_z;

        // 2. Compute Innovation
        MeasureVector y = z - uz;
        // Wrap innovation angles (critical for shortest path error)
        if (model.hasAngle()) {
            this->template normalizeAngles<MeasureDim>(y, model.getAngleFlags());
        }

        // 3. Compute Innovation Covariance (Pz) and Cross-Covariance (Pxz/K)
        MeasureMatrix Pz{MeasureMatrix::Zero()};
        MatrixK K{MatrixK::Zero()};
        for (size_t i = 0; i < Y_.cols(); ++i) {
            MeasureVector Z_uz = Z.col(i) - uz;
            
            // Wrap angles if defined (e.g., bearing)
            if (model.hasAngle()) {
                this->template normalizeAngles<MeasureDim>(Z_uz, model.getAngleFlags());
            }

            Pz += sigma_points_.getCovarianceWeights(i)*Z_uz*Z_uz.transpose();
            K += sigma_points_.getCovarianceWeights(i)*Y_x_.col(i)*Z_uz.transpose();
        }
        Pz += model.R();

        // 4. Perform 3-Tier Gating
        if (!this->rectangularGate(model, Pz, y)) return false; // Tier 1: Fast Sigma check
        if (!model.domainGate(z, y, Pz)) return false;          // Tier 2: User Logic

        Eigen::LDLT<MeasureMatrix> Pz_ldlt(Pz);
        if (!this->mahalanobisGate(model, Pz_ldlt, y)) return false; // Tier 3: Statistical
     
        // 5. Compute Kalman Gain and Update State
        K *= Pz_ldlt.solve(MeasureMatrix::Identity());

        this->x_ += K * y;
        this->P_ -= K * Pz * K.transpose();

        return true;
    }
};

#endif