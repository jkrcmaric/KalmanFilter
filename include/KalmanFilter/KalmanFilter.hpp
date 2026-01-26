#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include <functional>
#include <vector>
#include <cmath>

#include <eigen3/Eigen/Dense>


template <typename Derived, int StateDim, int MeasureDim>
class KalmanFilter {
public:
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

    // CRTP Accessor: Helper to cast 'this' to the Derived type
    Derived& derived() { return *static_cast<Derived*>(this); }

    // Perform prediction step
    void predict() {
        derived().computePrediction();
        this->template normalizeAngles<StateVector>(x_, state_angle_flag_);
        P_ = derived().model_.F * P_ * derived().model_.F.transpose() + derived().model_.Q;
    }

    // Perform update step
    void update(MeasureVector z) {
        MeasureVector y;
        derived().computeInnovation(z, y);
        this->template normalizeAngles<MeasureVector>(y, measurement_angle_flag_);
        MeasureMatrix S = derived().model_.H * P_ * derived().model_.H.transpose() + derived().model_.R;
        MatrixK K = P_ * derived().model_.H.transpose() * S.ldlt().solve(MeasureMatrix::Identity());
        x_ = x_ + K * y;
        P_ = (I_ - K * derived().model_.H) * P_ * (I_ - K * derived().model_.H).transpose() + K * derived().model_.R * K.transpose();
    }

    // Mark state index as an angle
    void setStateAsAngle(int index, bool value = true) { state_angle_flag_.at(index) = value; }

    // Mark measurment index as an angle
    void setMeasurementAsAngle(int index, bool value = true) { measurement_angle_flag_.at(index) = value; }

    void setState(const StateVector& x) { x_ = x; }
    void setF(const StateMatrix& F) { derived().model_.F = F; }
    void setH(const MatrixH& H) { derived().model_.H = H; }
    void setQ(const StateMatrix& Q) { derived().model_.Q = Q; }
    void setR(const MeasureMatrix& R) { derived().model_.R = R; }
    void setP(const StateMatrix& P) { P_ = P; }

    StateVector getState() const { return x_; }
    StateMatrix getCov() const { return P_; }

protected:
    StateVector x_{StateVector::Zero()};
    StateMatrix P_{StateMatrix::Identity()};
    StateMatrix I_{StateMatrix::Identity()};
    std::vector<int> state_angle_flag_{std::vector<int>(StateDim, 0)};
    std::vector<int> measurement_angle_flag_{std::vector<int>(MeasureDim, 0)};

    // Constructors
    KalmanFilter() = default;

    KalmanFilter(
        const StateVector& x, 
        const StateMatrix& P)
        : x_{x}, P_{P} {}

    // Destructors
    ~KalmanFilter() = default;

    template <typename T>
    void normalizeAngles(T& y, const std::vector<int>& angle_flag) {
        for (int i{0}; i < angle_flag.size(); ++i) {
            if (angle_flag[i]) y(i) = std::remainder(y(i), 2.0*M_PI);
        }
    }
};

#endif