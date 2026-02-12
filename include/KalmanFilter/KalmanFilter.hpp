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
 * @brief CRTP Base class for Kalman Filter variants (LKF, EKF, UKF, Information Filter).
 * * Provides shared state storage (x, P), math helpers (gating, angle normalization),
 * and model definitions. Uses Static Polymorphism to avoid virtual function overhead.
 * * @tparam KalmanFilterType The derived class implementing computePrediction/computeUpdate.
 * @tparam StateDim Fixed size of the state vector.
 */
template <typename KalmanFilterType, int StateDim>
class KalmanFilter {
public:
    // Required for Eigen fixed-size vectorization (avoids segfaults on 32-bit systems)
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW 

    using StateVector = Eigen::Matrix<double, StateDim, 1>;
    using StateMatrix = Eigen::Matrix<double, StateDim, StateDim>;

    // =========================================================================
    // Nested Model Classes
    // =========================================================================

    /**
     * @brief Base configuration for Process and Sensor models.
     * Handles numerical differentiation settings and angle wrapping logic.
     */
    template <int Dim>
    class SystemModel {
    public:
        SystemModel() = default;

        // Set step size for numerical differentiation (per dimension or global)
        void setEpsilon(double eps) { std::fill(epsilon_.begin(), epsilon_.end(), eps); }
        void setEpsilon(int index, double eps) { epsilon_.at(index) = eps; }

        const std::vector<int>& getAngleFlags() const { return angle_flag_; }
        bool hasAngle() const { return has_angle_; }

        /**
         * @brief Flags a dimension as a circular quantity (e.g., radians).
         * Ensures innovations wrap correctly between -PI and +PI.
         */
        void setAsAngle(int index, bool value = true) { 
            if (index >= 0 && index < Dim) {
                angle_flag_.at(index) = value;
                checkAngleFlags();
            }
        }

    protected:
        std::vector<double> epsilon_{std::vector<double>(StateDim, 1e-6)}; 
        std::vector<int> angle_flag_{std::vector<int>(Dim, 0)};
        bool has_angle_{false};

        ~SystemModel() = default;

        void checkAngleFlags() {
            auto is_non_zero = [](int i) { return i != 0; };
            has_angle_ = std::any_of(angle_flag_.begin(), angle_flag_.end(), is_non_zero);
        }

        // Helper: Central Difference Numerical Jacobian
        template <typename MatrixType, typename VectorType, typename Func>
        void computeNumericalJacobian(const Eigen::Ref<const StateVector> x, MatrixType& J, const Func& func) {
            StateVector x_temp{x};
            VectorType y_plus, y_minus;

            for (size_t i = 0; i < StateDim; ++i) {
                double original = x(i);
                double eps = epsilon_.at(i);

                x_temp(i) = original + eps; y_plus = func(x_temp);
                x_temp(i) = original - eps; y_minus = func(x_temp);
                x_temp(i) = original;       
                
                J.col(i) = (y_plus - y_minus) / (2 * eps);
            }
        }
    };

    /**
     * @brief Defines System Dynamics.
     * Can be Linear (F, Q) or Non-Linear (f(x) + Jacobian).
     * Supports state constraints (projection).
     */
    class ProcessModel : public SystemModel<StateDim> {
    public:
        using TransitionFunction = std::function<StateVector(const Eigen::Ref<const StateVector>)>;
        using JacobianFunction = std::function<StateMatrix(const Eigen::Ref<const StateVector>)>;
        using ConstraintFunction = std::function<void(Eigen::Ref<StateVector>)>;

        ProcessModel() = default;
        ~ProcessModel() = default;

        const StateMatrix& F() const { return F_; }
        const StateMatrix& Q() const { return Q_; }

        // Returns f(x) if non-linear function set, else F*x
        StateVector fx(const Eigen::Ref<const StateVector> x) const { 
            if (fx_) return fx_(x); 
            return F_ * x; 
        }

        // Apply physical constraints (clamping/projection) to state x
        void enforceConstraints(Eigen::Ref<StateVector> x) const {
            if (constraint_function_) constraint_function_(x);
        }

        /**
         * @brief Recomputes F based on current state.
         * Uses analytical Jacobian if provided, otherwise Numerical differentiation.
         */
        void updateJacobian(const Eigen::Ref<const StateVector> x) {
            if (!fx_) return; // Linear model, F is constant

            if (jacob_f_) {
                F_ = jacob_f_(x);
            } else {
                this->template computeNumericalJacobian<StateMatrix, StateVector>(
                    x, F_, [&](const StateVector& s){ return this->fx(s); }
                );
            }
        }

        void setF(const StateMatrix& F) { F_ = F; }
        void setQ(const StateMatrix& Q) { Q_ = Q; }
        void setTransitionFunction(const TransitionFunction& fx) { fx_ = fx; }
        void setAnalyticalJacobianF(const JacobianFunction& jacob_f) { jacob_f_ = jacob_f; }
        void setConstraintFunction(const ConstraintFunction& func) {constraint_function_ = func; }

    private:
        StateMatrix F_{StateMatrix::Identity()};
        StateMatrix Q_{StateMatrix::Zero()};
        TransitionFunction fx_;
        JacobianFunction jacob_f_;
        ConstraintFunction constraint_function_;
    };

    /**
     * @brief Defines Sensor Model.
     * Can be Linear (H, R) or Non-Linear (h(x) + Jacobian).
     * Supports 3-tier gating: Rectangular, Mahalanobis, and Custom Domain.
     */
    template <int MeasureDim>
    class SensorModel : public SystemModel<MeasureDim> {
    public:
        using MeasureVector = Eigen::Matrix<double, MeasureDim, 1>;
        using MeasureMatrix = Eigen::Matrix<double, MeasureDim, MeasureDim>;
        using MatrixH = Eigen::Matrix<double, MeasureDim, StateDim>;

        using MeasurementFunction = std::function<MeasureVector(const Eigen::Ref<const StateVector>)>;
        using JacobianFunction = std::function<MatrixH(const Eigen::Ref<const StateVector>)>;
        using DomainGateFunction = std::function<bool(const MeasureVector& z,
                                                      const MeasureVector& y,
                                                      const MeasureMatrix& S)>;

        SensorModel() = default;
        ~SensorModel() = default;

        const MatrixH& H() const { return H_; }
        const MeasureMatrix& R() const { return R_; }

        // Returns h(x) if non-linear function set, else H*x
        MeasureVector hx(const Eigen::Ref<const StateVector> x) const { 
            if (hx_) return hx_(x); 
            return H_ * x; 
        }

        const std::vector<double>& getRectangularGateLimits() const { return rect_gate_limits_; }
        double getMahalanobisThreshold() const { return mahalanobis_threshold_; }

        // Execute user-defined validation logic
        bool domainGate(const MeasureVector& z, 
                        const MeasureVector& y, 
                        const MeasureMatrix& S) const {
            if (!hasDomainGate()) return true;
            return domain_gate_(z, y, S);
        }

        // Recomputes H based on current state (Analytical or Numerical)
        void updateJacobian(const Eigen::Ref<const StateVector> x) {
            if (!hx_) return; // Linear model, H is constant

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
        void setAnalyticalJacobianH(const JacobianFunction& jacob_h) { jacob_h_ = jacob_h; }

        // Sets N-Sigma limits for element-wise gating
        void setRectangularGateLimits(double sigma_limit) {
            std::fill(rect_gate_limits_.begin(), rect_gate_limits_.end(), sigma_limit);
            use_rectangular_gate_ = true;
        }
        void setRectangularGateLimits(int index, double sigma_limit) {
            rect_gate_limits_.at(index) = sigma_limit;
            use_rectangular_gate_ = true;
        }
        bool useRectangularGate() const { return use_rectangular_gate_; }

        // Sets Chi-Squared threshold for statistical gating
        void setMahalanobisThreshold(double threshold) {mahalanobis_threshold_ = threshold; }

        void setDomainGate(const DomainGateFunction& func) { domain_gate_ = func; }
        bool hasDomainGate() const { return static_cast<bool>(domain_gate_); }

    private:
        MatrixH H_{MatrixH::Zero()};
        MeasureMatrix R_{MeasureMatrix::Identity()};
        MeasurementFunction hx_;
        JacobianFunction jacob_h_;

        // Default Infinity = Gating Disabled
        std::vector<double> rect_gate_limits_{
            std::vector<double>(MeasureDim, std::numeric_limits<double>::infinity())
        };
        bool use_rectangular_gate_{false}; // Default OFF
        double mahalanobis_threshold_{std::numeric_limits<double>::infinity()};

        DomainGateFunction domain_gate_;
    };

    // =========================================================================
    // Core Filter Interface
    // =========================================================================

    /**
     * @brief Steps the filter forward in time.
     * @param model Mutable reference allowed for Jacobian internal updates.
     */
    void predict(ProcessModel& model) {
        kf().computePrediction(model);
    }

    /**
     * @brief Integrates a new measurement.
     * Implementation (EKF/LKF) resides in derived class.
     */
    template <int MeasureDim>
    bool update(SensorModel<MeasureDim>& model, const Eigen::Matrix<double, MeasureDim, 1>& z) {
        return kf().computeUpdate(model, z);
    }

    void setState(const StateVector& x) { x_ = x; }
    void setP(const StateMatrix& P) { P_ = P; }
    const StateVector& getState() const { return x_; }
    const StateMatrix& getCov() const { return P_; }

protected:
    StateVector x_{StateVector::Zero()};
    StateMatrix P_{StateMatrix::Identity()};

    KalmanFilter() = default;
    KalmanFilter(const StateVector& x, const StateMatrix& P) : x_{x}, P_{P} {}
    ~KalmanFilter() = default;

    // CRTP Cast Helper
    KalmanFilterType& kf() { return *static_cast<KalmanFilterType*>(this); }

    /**
     * @brief Normalizes state/innovation angles to [-PI, PI].
     * Only applies to dimensions flagged via setAsAngle().
     */
    template <int Dim>
    void normalizeAngles(Eigen::Ref<Eigen::Matrix<double, Dim, 1>> y, const std::vector<int>& angle_flag) {
        assert(static_cast<size_t>(y.size()) == angle_flag.size() && "Angle flag dimension mismatch");
        
        for (size_t i = 0; i < angle_flag.size(); ++i) {
            if (angle_flag[i]) {
                y(i) = std::remainder(y(i), 2.0 * M_PI);
            }
        }
    }

    /**
     * @brief Fast, element-wise outlier rejection.
     * Checks if y(i)^2 > (Limit * S(i,i)). 
     * Cost: O(N). Use before decomposition.
     */
    template <int MeasureDim>
    bool rectangularGate(const SensorModel<MeasureDim>& model,
                         const Eigen::Matrix<double, MeasureDim, MeasureDim>& S,
                         const Eigen::Matrix<double, MeasureDim, 1>& y) {

        if (!model.useRectangularGate()) return true;

        const std::vector<double>& limit = model.getRectangularGateLimits();
        
        for (size_t i = 0; i < MeasureDim; ++i) {
            double ysq = y(i) * y(i);
            double limitsq = limit[i] * limit[i] * S(i, i);

            if (ysq > limitsq) return false;
        }
        return true;
    }

    /**
     * @brief Statistical outlier rejection using Covariance.
     * Checks if y^T * S^-1 * y > Threshold.
     * Cost: O(N^2) (assumes S_ldlt pre-computed).
     */
    template <int MeasureDim>
    bool mahalanobisGate(const SensorModel<MeasureDim>& model,
                         const Eigen::LDLT<Eigen::Matrix<double, MeasureDim, MeasureDim>>& S_ldlt,
                         const Eigen::Matrix<double, MeasureDim, 1>& y) {

        const double threshold = model.getMahalanobisThreshold();

        if (std::isinf(threshold)) return true; // Gate disabled

        double mahalanobis_sq = y.transpose() * S_ldlt.solve(y);
        return mahalanobis_sq <= threshold;
    }
};

#endif