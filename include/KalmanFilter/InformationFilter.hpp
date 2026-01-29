#ifndef INFORMATION_FILTER_H
#define INFORMATION_FILTER_H

#include <utility>
#include <vector>

#include "KalmanFilter.hpp"


template <int StateDim, int MeasureDim>
class InformationFilter : public KalmanFilter<InformationFilter<StateDim, MeasureDim>, StateDim, MeasureDim> {
    using Base = KalmanFilter<InformationFilter<StateDim, MeasureDim>, StateDim, MeasureDim>;
    friend class KalmanFilter<InformationFilter<StateDim, MeasureDim>, StateDim, MeasureDim>;

public:
    using typename Base::StateVector;
    using typename Base::MeasureVector;
    using typename Base::StateMatrix;
    using typename Base::MeasureMatrix;
    using typename Base::MatrixH;
    using typename Base::MatrixK;

    using IFSystemModel = typename Base::SystemModel;
    using Information = std::pair<StateVector, StateMatrix>;

    InformationFilter(
        const StateVector& x,
        const IFSystemModel& model,
        const StateMatrix& P)
        : Base(x, P), model_{model} {
            Y_ = P.inverse();
            y_ = Y_ * this->x_;

            Rinv_ = model_.R.inverse();
        }

    void update(const std::vector<MeasureVector>& zs) {
        computeUpdate(zs);
    }

    void setF(const StateMatrix& F) { model_.F = F; }
    void setH(const MatrixH& H) { model_.H = H; }
    void setQ(const StateMatrix& Q) { model_.Q = Q; }
    void setR(const MeasureMatrix& R) { 
        model_.R = R;
        Rinv_ = model_.R.inverse(); 
    }

private:
    StateVector y_;
    StateMatrix Y_;
    MeasureMatrix Rinv_;
    IFSystemModel model_;

    void computePrediction() {
        this->x_ = model_.F * this->x_;
        if (this->has_state_angle_) {
            this->template normalizeAngles<StateVector>(this->x_, this->state_angle_flag_);
        }
        this->P_ = model_.F * this->P_ * model_.F.transpose() + model_.Q;

        Y_ = this->P_.inverse();
        y_ = Y_ * this->x_;
    }

    void computeUpdate(const MeasureVector& z) {
        Information contrib = getInformationContribution(z);
        y_ += contrib.first;
        Y_ += contrib.second;

        syncState();
    }

    void computeUpdate(const std::vector<MeasureVector>& zs) {
        for (const MeasureVector& z : zs) {
            Information contrib = getInformationContribution(z);
            y_ += contrib.first;
            Y_ += contrib.second;
        }
        syncState();
    }

    Information getInformationContribution(const MeasureVector& z) {
        StateMatrix I = model_.H.transpose() * Rinv_ * model_.H;
        StateVector i;
        if (this->has_measurement_angle_) {
            MeasureVector Hx = model_.H * this->x_; 
            MeasureVector innovation = z - Hx;
            this->template normalizeAngles<MeasureVector>(innovation, this->measurement_angle_flag_);
            i = model_.H.transpose() * Rinv_ * (innovation + Hx); // H^T * R^-1 * z_norm
        } else {
            i = model_.H.transpose() * Rinv_ * z;
        }

        return {i, I};
    }

    void syncState() {
        this->P_ = Y_.inverse();
        this->x_ = this->P_ * y_;

        if (this->has_state_angle_) {
            this->template normalizeAngles<StateVector>(this->x_, this->state_angle_flag_);
            y_ = Y_ * this->x_;
        }
    }
};

#endif