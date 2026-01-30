#ifndef KALMAN_FILTER_HPP
#define KALMAN_FILTER_HPP

#include <functional>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

#include <eigen3/Eigen/Dense>

/**
 * @brief Base class for Kalman Filter variants (LKF, EKF, UKF, Information Filter).
 * * Implements the Curiously Recurring Template Pattern (CRTP) to allow 
 * static polymorphism (no virtual function overhead) while sharing 
 * common state and math helpers.
 * * @tparam KalmanFilterType The derived class (e.g., ExtendedInformationFilter)
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

        // Getters
        const std::vector<int>& getAngleFlags() const { return angle_flag_; }
        bool hasAngle() const { return has_angle_; }
        bool automaticJacobian() const { return automaticJacobian_; }

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
        ~SystemModel() = default;

        std::vector<int> angle_flag_{std::vector<int>(Dim, 0)};
        bool has_angle_{false};
        bool automaticJacobian_{true};

        void checkAngleFlags() {
            auto is_non_zero = [](int i) { return i != 0; };
            has_angle_ = std::any_of(angle_flag_.begin(), angle_flag_.end(), is_non_zero);
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

        // Logic: Updates internal F matrix using the analytical function
        void computeAnalyticalJacobian(const StateVector& x) { 
            if (jacob_f_) {
                F_ = jacob_f_(x);
            } else {
                // Safety warning: User asked for analytical but didn't provide function
                std::cerr << "[KalmanFilter] Warning: Analytical Jacobian F requested but function not set." << std::endl;
            }
        }

        // Configuration
        void setF(const StateMatrix& F) { F_ = F; }
        void setQ(const StateMatrix& Q) { Q_ = Q; }
        
        void setTransitionFunction(const TransitionFunction& fx) { fx_ = fx; }
        
        void setAnalyticalJacobianF(const JacobianFunction& jacob_f) { 
            jacob_f_ = jacob_f; 
            this->automaticJacobian_ = false; // "this->" required for template inheritance
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

        // Logic: Updates internal H matrix
        void computeAnalyticalJacobian(const StateVector& x) { 
            if (jacob_h_) {
                H_ = jacob_h_(x);
            } else {
                 std::cerr << "[KalmanFilter] Warning: Analytical Jacobian H requested but function not set." << std::endl;
            }
        }

        void setH(const MatrixH& H) { H_ = H; }
        void setR(const MeasureMatrix& R) { R_ = R; }
        
        void setMeasurementFunction(const MeasurementFunction& hx) { hx_ = hx; }
        
        void setAnalyticalJacobianH(const JacobianFunction& jacob_h) { 
            jacob_h_ = jacob_h; 
            this->automaticJacobian_ = false;
        }

    private:
        MatrixH H_{MatrixH::Zero()};
        MeasureMatrix R_{MeasureMatrix::Identity()};
        MeasurementFunction hx_;
        JacobianFunction jacob_h_;
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
    std::vector<double> EPSILON_{std::vector<double>(StateDim, 1e-6)}; 

    KalmanFilter() = default;
    
    KalmanFilter(const StateVector& x, const StateMatrix& P)
        : x_{x}, P_{P} {}

    ~KalmanFilter() = default;

    // Helper: Casts 'this' to Derived type for CRTP
    KalmanFilterType& kf() { return *static_cast<KalmanFilterType*>(this); }

    /**
     * @brief Normalize angles in a vector to range [-PI, PI].
     * Includes bounds checking to prevent crashes.
     */
    template <int Dim>
    void normalizeAngles(Eigen::Matrix<double, Dim, 1>& y, const std::vector<int>& angle_flag) {
        // Safety: Ensure we don't read past the end of the vector
        size_t limit = std::min((size_t)y.size(), angle_flag.size());
        
        for (size_t i = 0; i < limit; ++i) {
            if (angle_flag[i]) {
                y(i) = std::remainder(y(i), 2.0 * M_PI);
            }
        }
    }

    /**
     * @brief Computes numerical Jacobian using Central Difference.
     */
    template <typename MatrixType, typename VectorType, typename Func>
    void computeJacobian(const StateVector& x, MatrixType& J, const Func& func) {
        StateVector x_temp{x};
        VectorType y_plus, y_minus;

        for (int i = 0; i < StateDim; ++i) {
            double original = x(i);
            double eps = EPSILON_.at(i);

            x_temp(i) = original + eps;
            y_plus = func(x_temp);

            x_temp(i) = original - eps;
            y_minus = func(x_temp);

            x_temp(i) = original;
            J.col(i) = (y_plus - y_minus) / (2 * eps);
        }
    }
};

#endif