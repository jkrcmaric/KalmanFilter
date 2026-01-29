#ifndef LINEAR_KALMAN_FILTER_H
#define LINEAR_KALMAN_FILTER_H

#include "KalmanFilter.hpp"


template <int StateDim, int MeasureDim>
class LinearKalmanFilter : public KalmanFilter<LinearKalmanFilter<StateDim, MeasureDim>, StateDim, MeasureDim> {
    using Base = KalmanFilter<LinearKalmanFilter<StateDim, MeasureDim>, StateDim, MeasureDim>;
    friend class KalmanFilter<LinearKalmanFilter<StateDim, MeasureDim>, StateDim, MeasureDim>;

public:
    using typename Base::StateVector;
    using typename Base::MeasureVector;
    using typename Base::StateMatrix;
    using typename Base::MeasureMatrix;
    using typename Base::MatrixH;
    using typename Base::MatrixK;

    using LKFSystemModel = typename Base::SystemModel;

    LinearKalmanFilter(
        const StateVector& x, 
        const LKFSystemModel& model, 
        const StateMatrix& P)
        : Base(x, P), model_{model} {}

    void setModel(const LKFSystemModel& model) { model_ = model; }

    void setF(const StateMatrix& F) { model_.F = F; }
    void setH(const MatrixH& H) { model_.H = H; }
    void setQ(const StateMatrix& Q) { model_.Q = Q; }
    void setR(const MeasureMatrix& R) { model_.R = R; }

private:
    StateMatrix I_{StateMatrix::Identity()};
    LKFSystemModel model_;

    void computePrediction() {
        this->x_ = model_.F * this->x_;
        if (this->has_state_angle_) {
            this->template normalizeAngles<StateVector>(this->x_, this->state_angle_flag_);
        }
        this->P_ = model_.F * this->P_ * model_.F.transpose() + model_.Q;
    }

    void computeUpdate(const MeasureVector& z) {
        MeasureVector y = z - model_.H * this->x_;
        if (this->has_measurement_angle_) {
            this->template normalizeAngles<MeasureVector>(y, this->measurement_angle_flag_);
        }
        MeasureMatrix S = model_.H * this->P_ * model_.H.transpose() + model_.R;
        MatrixK K = this->P_ * model_.H.transpose() * S.ldlt().solve(MeasureMatrix::Identity());
        this->x_ = this->x_ + K * y;
        this->P_ = (I_ - K*model_.H) * this->P_ * (I_ - K*model_.H).transpose() + K*model_.R*K.transpose();
    }
};

#endif