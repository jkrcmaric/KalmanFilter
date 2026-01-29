#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include <iostream>
#include <functional>
#include <vector>
#include <cmath>
#include <algorithm>

#include <eigen3/Eigen/Dense>


template <typename KalmanFilterType, int StateDim, int MeasureDim>
class KalmanFilter {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // Ensures 'new' allocates aligned memory

    using StateVector = Eigen::Matrix<double, StateDim, 1>;
    using MeasureVector = Eigen::Matrix<double, MeasureDim, 1>;
    using StateMatrix = Eigen::Matrix<double, StateDim, StateDim>;
    using MeasureMatrix = Eigen::Matrix<double, MeasureDim, MeasureDim>;
    using MatrixH = Eigen::Matrix<double, MeasureDim, StateDim>;
    using MatrixK = Eigen::Matrix<double, StateDim, MeasureDim>;

    struct SystemModel {
        StateMatrix F{StateMatrix::Identity()};
        MatrixH H{MatrixH::Zero()};
        StateMatrix Q{StateMatrix::Zero()};
        MeasureMatrix R{MeasureMatrix::Identity()};
    };

    // Perform prediction step
    void predict() {
        kf().computePrediction();
    }

    // Perform update step
    void update(const MeasureVector& z) {
        kf().computeUpdate(z);
    }

    // Mark state index as an angle
    void setStateAsAngle(int index, bool value = true) { 
        state_angle_flag_.at(index) = value;
        checkAngleFlags(state_angle_flag_, has_state_angle_);
    }

    // Mark measurment index as an angle
    void setMeasurementAsAngle(int index, bool value = true) { 
        measurement_angle_flag_.at(index) = value;
        checkAngleFlags(measurement_angle_flag_, has_measurement_angle_);
    }

    void setState(const StateVector& x) { x_ = x; }
    void setP(const StateMatrix& P) { P_ = P; }

    const StateVector& getState() const { return x_; }
    const StateMatrix& getCov() const { return P_; }

protected:
    StateVector x_{StateVector::Zero()};
    StateMatrix P_{StateMatrix::Identity()};
    std::vector<int> state_angle_flag_{std::vector<int>(StateDim, 0)};
    std::vector<int> measurement_angle_flag_{std::vector<int>(MeasureDim, 0)};
    bool has_state_angle_{false};
    bool has_measurement_angle_{false};
    std::vector<double> EPSILON_{std::vector<double>(StateDim, 1e-6)}; // for numerical differentiation

    // Constructors
    KalmanFilter() = default;

    KalmanFilter(
        const StateVector& x, 
        const StateMatrix& P)
        : x_{x}, P_{P} {}

    // Destructors
    ~KalmanFilter() = default;

    // CRTP Accessor: Helper to cast 'this' to the Derived type
    KalmanFilterType& kf() { return *static_cast<KalmanFilterType*>(this); }

    template <typename T>
    void normalizeAngles(T& y, const std::vector<int>& angle_flag) {
        for (int i{0}; i < angle_flag.size(); ++i) {
            if (angle_flag[i]) y(i) = std::remainder(y(i), 2.0*M_PI);
        }
    }

    void checkAngleFlags(std::vector<int>& angle_flag, bool& has_angle) {
        auto is_non_zero = [](int i) { return i != 0; };

        if (std::any_of(angle_flag.begin(), angle_flag.end(), is_non_zero)) {
            has_angle = true;
        } else {
            has_angle = false;
        }
    }

    // compute Jacobian via numerical differentiation
    template <typename T1, typename T2>
    void computeJacobian(const StateVector& x, T1& J, const std::function<T2(const StateVector&)>& fx) {
        StateVector x_temp{x};
        T2 y_plus{};
        T2 y_minus{};

        for (int i{0}; i < StateDim; ++i) {
            double original_value{x(i)};
            double epsilon{EPSILON_.at(i)};

            x_temp(i) = original_value + epsilon;
            y_plus = fx(x_temp);

            x_temp(i) = original_value - epsilon;
            y_minus = fx(x_temp);

            x_temp(i) = original_value;
            J.col(i) = (y_plus - y_minus) / (2*epsilon);
        }
    }
};

#endif