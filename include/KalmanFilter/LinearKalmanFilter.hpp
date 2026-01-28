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

    using LKFSystemModel = typename Base::SystemModel;

    LinearKalmanFilter(
        const StateVector& x, 
        const LKFSystemModel& model, 
        const StateMatrix& P)
        : Base(x, P), model_{model} {}

    void setModel(const LKFSystemModel& model) { model_ = model; }

private:
    LKFSystemModel model_;

    void computePrediction() {
        this->x_ = model_.F * this->x_;
    }

    void computeInnovation(MeasureVector& z, MeasureVector& y) {
        y = z - model_.H * this->x_;
    }
};

#endif