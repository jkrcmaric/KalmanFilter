#ifndef KALMAN_FILTER_HPP
#define KALMAN_FILTER_HPP

#include <functional>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <limits>

#include <eigen3/Eigen/Dense>

/**
 * @brief Base class for Kalman Filter variants (LKF, EKF, UKF, Information Filter).
 * * Implements the Curiously Recurring Template Pattern (CRTP) to allow 
 * static polymorphism (no virtual function overhead) while sharing 
 * common state and math helpers.
 * * @tparam KalmanFilterType The derived class (e.g., ExtendedKalmanFilter)
 * * @tparam StateDim The fixed size of the state vector
 */
template <typename KalmanFilterType, int StateDim>
class KalmanFilter {
public:
    // Essential for Eigen fixed-size vectorization to prevent alignment crashes
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW 

    using StateVector = Eigen::Matrix<double, StateDim, 1>;
    using StateMatrix = Eigen::Matrix<double, StateDim, StateDim>;

    // =========================================================================
    // Nested Model Classes
    // =========================================================================

    /**
     * @brief Common base for Process and Sensor models.
     * Manages angle wrapping flags and Jacobian configuration.
     */
    template <int Dim>
    class SystemModel {
    public:
        SystemModel() = default;

        void setEpsilon(double eps) { std::fill(epsilon_.begin(), epsilon_.end(), eps); }
        void setEpsilon(int index, double eps) { epsilon_.at(index) = eps; }

        const std::vector<int>& getAngleFlags() const { return angle_flag_; }
        bool hasAngle() const { return has_angle_; }

        /**
         * @brief Mark a specific dimension as an angle (e.g., radians).
         * This triggers wrapping ( -PI to +PI ) during innovation.
         */
        void setAsAngle(int index, bool value = true) { 
            if (index >= 0 && index < Dim) {
                angle_flag_.at(index) = value;
                checkAngleFlags();
            }
        }

    protected:
        std::vector<double> epsilon_{std::vector<double>(StateDim, 1e-6)}; // for numerical differentiation

        std::vector<int> angle_flag_{std::vector<int>(Dim, 0)};
        bool has_angle_{false};

        ~SystemModel() = default;

        void checkAngleFlags() {
            auto is_non_zero = [](int i) { return i != 0; };
            has_angle_ = std::any_of(angle_flag_.begin(), angle_flag_.end(), is_non_zero);
        }

        /**
        * @brief Computes numerical Jacobian using Central Difference.
        */
        template <typename MatrixType, typename VectorType, typename Func>
        void computeNumericalJacobian(const StateVector& x, MatrixType& J, const Func& func) {
            StateVector x_temp{x};
            VectorType y_plus, y_minus;

            for (size_t i = 0; i < StateDim; ++i) {
                double original = x(i);
                double eps = epsilon_.at(i);

                x_temp(i) = original + eps;
                y_plus = func(x_temp);

                x_temp(i) = original - eps;
                y_minus = func(x_temp);

                x_temp(i) = original;
                J.col(i) = (y_plus - y_minus) / (2 * eps);
            }
        }
    };

    /**
     * @brief Defines the System Dynamics (Physics).
     * Supports both Linear (F, Q) and Non-Linear (f(x), jacob_f) models.
     */
    class ProcessModel : public SystemModel<StateDim> {
    public:
        using TransitionFunction = std::function<StateVector(const StateVector&)>;
        using JacobianFunction = std::function<StateMatrix(const StateVector&)>;

        ProcessModel() = default;
        ~ProcessModel() = default;

        // Accessors
        const StateMatrix& F() const { return F_; }
        const StateMatrix& Q() const { return Q_; }

        // Logic: Returns f(x) if available, otherwise falls back to Linear F*x
        StateVector fx(const StateVector& x) const { 
            if (fx_) return fx_(x); 
            return F_ * x; 
        }

        /**
         * @brief Updates F matrix.
         * Automatically chooses betwen Analytical (if provided) or Numerical.
         */
        void updateJacobian(const StateVector& x) {
            // return if non-linear transition function is not set
            if (!fx_) return;

            if (jacob_f_) {
                // compute analytical Jacobian
                F_ = jacob_f_(x);
            } else {
                this->template computeNumericalJacobian<StateMatrix, StateVector>(
                    x, F_, [&](const StateVector& s){ return this->fx(s); }
                );
            }
        }

        // Configuration
        void setF(const StateMatrix& F) { F_ = F; }
        void setQ(const StateMatrix& Q) { Q_ = Q; }
        
        void setTransitionFunction(const TransitionFunction& fx) { fx_ = fx; }
        
        void setAnalyticalJacobianF(const JacobianFunction& jacob_f) { 
            jacob_f_ = jacob_f; 
        }

    private:
        StateMatrix F_{StateMatrix::Identity()};
        StateMatrix Q_{StateMatrix::Zero()};
        TransitionFunction fx_;
        JacobianFunction jacob_f_;
    };

    /**
     * @brief Defines a Sensor Model.
     * Templated by MeasureDim to allow one filter to handle various sensor types.
     */
    template <int MeasureDim>
    class SensorModel : public SystemModel<MeasureDim> {
    public:
        using MeasureVector = Eigen::Matrix<double, MeasureDim, 1>;
        using MeasureMatrix = Eigen::Matrix<double, MeasureDim, MeasureDim>;
        using MatrixH = Eigen::Matrix<double, MeasureDim, StateDim>;

        using MeasurementFunction = std::function<MeasureVector(const StateVector&)>;
        using JacobianFunction = std::function<MatrixH(const StateVector&)>;

        SensorModel() = default;
        ~SensorModel() = default;

        const MatrixH& H() const { return H_; }
        const MeasureMatrix& R() const { return R_; }

        // Logic: Returns h(x) if available, otherwise Linear H*x
        MeasureVector hx(const StateVector& x) const { 
            if (hx_) return hx_(x); 
            return H_ * x; 
        }

        const std::vector<double>& getRectangularGateLimits() const {
            return rect_gate_limits_;
        }

        double getMahalanobisThreshold() const { return mahalanobis_threshold_; }

        /**
         * @brief Updates H matrix.
         * Automatically chooses between Analytical (if provided) or Numerical.
         */
        void updateJacobian(const StateVector& x) {
            // return if non-linear measurement function is not set
            if (!hx_) return;

            if (jacob_h_) {
                H_ = jacob_h_(x);
            } else {
                this->template computeNumericalJacobian<MatrixH, MeasureVector>(
                    x, H_, [&](const StateVector& s){ return this->hx(s); }
                );
            }
        }

        void setH(const MatrixH& H) { H_ = H; }
        void setR(const MeasureMatrix& R) { R_ = R; }
        
        void setMeasurementFunction(const MeasurementFunction& hx) { hx_ = hx; }
        
        void setAnalyticalJacobianH(const JacobianFunction& jacob_h) { 
            jacob_h_ = jacob_h; 
        }

        void setRectangularGateLimits(double sigma_limit) {
            std::fill(rect_gate_limits_.begin(), rect_gate_limits_.end(), sigma_limit);
        }

        void setRectangularGateLimits(int index, double sigma_limit) {
            rect_gate_limits_.at(index) = sigma_limit;
        }

        void setMahalanobisThreshold(double threshold) {mahalanobis_threshold_ = threshold; }

    private:
        MatrixH H_{MatrixH::Zero()};
        MeasureMatrix R_{MeasureMatrix::Identity()};
        MeasurementFunction hx_;
        JacobianFunction jacob_h_;

        // Initialize with Infinity: Effectively disables gating by default
        std::vector<double> rect_gate_limits_{
            std::vector<double>(MeasureDim, std::numeric_limits<double>::infinity())
        };
        double mahalanobis_threshold_{std::numeric_limits<double>::infinity()};
    };

    // =========================================================================
    // Core Filter Interface
    // =========================================================================

    /**
     * @brief Predicts the next state.
     * Delegates implementation to Derived::computePrediction via CRTP.
     * @param model Passed by non-const reference because the filter may update 
     * the model's internal Jacobian (F) during this step.
     */
    void predict(ProcessModel& model) {
        kf().computePrediction(model);
    }

    /**
     * @brief Updates state based on measurement z.
     * Delegates implementation to Derived::computeUpdate via CRTP.
     */
    template <int MeasureDim>
    void update(SensorModel<MeasureDim>& model, const Eigen::Matrix<double, MeasureDim, 1>& z) {
        kf().computeUpdate(model, z);
    }

    // Accessors
    void setState(const StateVector& x) { x_ = x; }
    void setP(const StateMatrix& P) { P_ = P; }
    const StateVector& getState() const { return x_; }
    const StateMatrix& getCov() const { return P_; }

protected:
    StateVector x_{StateVector::Zero()};
    StateMatrix P_{StateMatrix::Identity()};

    KalmanFilter() = default;
    
    KalmanFilter(const StateVector& x, const StateMatrix& P)
        : x_{x}, P_{P} {}

    ~KalmanFilter() = default;

    // Helper: Casts 'this' to Derived type for CRTP
    KalmanFilterType& kf() { return *static_cast<KalmanFilterType*>(this); }

    /**
     * @brief Normalize angles in a vector to range [-PI, PI].
     */
    template <int Dim>
    void normalizeAngles(Eigen::Matrix<double, Dim, 1>& y, const std::vector<int>& angle_flag) {
        // halt program immediately if false (Debug mode only)
        assert(static_cast<size_t>(y.size()) == angle_flag.size() && "Angle flag dimension mismatch");
        
        for (size_t i = 0; i < angle_flag.size(); ++i) {
            if (angle_flag[i]) {
                y(i) = std::remainder(y(i), 2.0 * M_PI);
            }
        }
    }

    /**
     * @brief Performs Rectangular Gating (Fast Element-wise Sigma Check).
     * Checks if the innovation of any single dimension exceeds the N-sigma limit 
     * defined in the sensor model.
     * * @param model The sensor model containing the gate limits.
     * @param S The innovation covariance matrix (S = HPH' + R).
     * @param y The innovation vector (y = z - hx).
     * @return true if the measurement is within bounds, false if it is an outlier.
     */
    template <int MeasureDim>
    bool rectangularGate(const SensorModel<MeasureDim>& model,
                         const Eigen::Matrix<double, MeasureDim, MeasureDim>& S,
                         const Eigen::Matrix<double, MeasureDim, 1>& y) {

        const std::vector<double>& limit = model.getRectangularGateLimits();
        
        for (size_t i = 0; i < MeasureDim; ++i) {
            double ysq = y(i) * y(i);
            double limitsq = limit[i] * limit[i] * S(i, i);

            // Reject if outside bounds
            if (ysq > limitsq) return false;
        }
        
        return true;
    }

    /**
     * @brief Performs Mahalanobis Gating (Statistical Outlier Rejection).
     * Calculates the squared Mahalanobis distance (D^2 = y^T * S^-1 * y) to determine
     * if a measurement is statistically valid given the current covariance.
     * * @note Uses a pre-computed LDLT decomposition for efficiency (O(N^2)).
     * * @param model The sensor model containing the chi-squared threshold.
     * @param S_ldlt The pre-computed LDLT decomposition of the Innovation Covariance S.
     * @param y The innovation vector (y = z - hx).
     * @return true if the measurement is within the threshold, false if it is an outlier.
     */
    template <int MeasureDim>
    bool mahalanobisGate(const SensorModel<MeasureDim>& model,
                         const Eigen::LDLT<Eigen::Matrix<double, MeasureDim, MeasureDim>>& S_ldlt,
                         const Eigen::Matrix<double, MeasureDim, 1>& y) {

        // Retrieve the chi-squared threshold (e.g., 5.99 for 95% confidence in 2D)
        const double threshold = model.getMahalanobisThreshold();

        // Optimization: If threshold is Infinity, gating is disabled -> Accept everything
        if (std::isinf(threshold)) return true;

        // Compute Mahalanobis Distance squared: D^2 = y^T * S^-1 * y
        // Uses the pre-computed LDLT solver to avoid explicit inversion
        double mahalanobis_sq = y.transpose() * S_ldlt.solve(y);

        // Accept if distance is within the confidence ellipsoid
        return mahalanobis_sq <= threshold;
    }

};

#endif