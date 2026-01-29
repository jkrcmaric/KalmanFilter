#ifndef EXTENDED_KALMAN_FILTER_H
#define EXTENDED_KALMAN_FILTER_H

#include <functional>
#include <vector>

#include "KalmanFilter.hpp"


template <int StateDim, int MeasureDim>
class ExtendedKalmanFilter : public KalmanFilter<ExtendedKalmanFilter<StateDim, MeasureDim>, StateDim, MeasureDim> {
    using Base = KalmanFilter<ExtendedKalmanFilter<StateDim, MeasureDim>, StateDim, MeasureDim>;
    friend class KalmanFilter<ExtendedKalmanFilter<StateDim, MeasureDim>, StateDim, MeasureDim>;

public:
    using typename Base::StateVector;
    using typename Base::MeasureVector;
    using typename Base::StateMatrix;
    using typename Base::MeasureMatrix;
    using typename Base::MatrixH;
    using typename Base::MatrixK;

    struct EKFSystemModel : Base::SystemModel {
        std::function<StateVector(const StateVector&)> fx;
        std::function<MeasureVector(const StateVector&)> hx;

        // Optional Analytical Jacobians
        std::function<StateMatrix(const StateVector&)> jacob_f;
        std::function<MatrixH(const StateVector&)> jacob_h;
    };

    // Constructors
    ExtendedKalmanFilter(
        const StateVector& x, 
        const EKFSystemModel& model,
        const StateMatrix& P)
        : Base(x, P), model_{model} {
            setAutomaticJacobianF(!model_.jacob_f);
            setAutomaticJacobianH(!model_.jacob_h);
        }

    // Allow tuning numerical differentiation step size per dimension
    void setEpsilon(int index, double value) { this->EPSILON_.at(index) = value; }
    const std::vector<double>& getEpsilon() const { return this->EPSILON_; }

    // Set whether automatic Jacobian is automatically computed via numerical differentiaiton
    void setAutomaticJacobianF(bool value) { autoJacobianF_ = value; }
    void setAutomaticJacobianH(bool value) { autoJacobianH_ = value; }

    void setModel(const EKFSystemModel& model) { 
        model_ = model; 

        setAutomaticJacobianF(!model_.jacob_f);
        setAutomaticJacobianH(!model_.jacob_h);
    }

    void setF(const StateMatrix& F) { model_.F = F; }
    void setH(const MatrixH& H) { model_.H = H; }
    void setQ(const StateMatrix& Q) { model_.Q = Q; }
    void setR(const MeasureMatrix& R) { model_.R = R; }

    void setTransitionFunction(const std::function<StateVector(const StateVector&)>& fx) { model_.fx = fx; }
    void setMeasurementFunction(const std::function<MeasureVector(const StateVector&)>& hx) { model_.hx = hx; }

    void setAnalyticalJacobianF(const std::function<StateMatrix(const StateVector&)>& jacob_f) { 
        model_.jacob_f = jacob_f; 
        setAutomaticJacobianF(false);
    }
    void setAnalyticalJacobianH(const std::function<MatrixH(const StateVector&)>& jacob_h) { 
        model_.jacob_h = jacob_h; 
        setAutomaticJacobianH(false);
    }

private:
    StateMatrix I_{StateMatrix::Identity()};
    bool autoJacobianF_{true};
    bool autoJacobianH_{true};
    EKFSystemModel model_;

    void computePrediction() {
        if (autoJacobianF_ ) {
            this->template computeJacobian<StateMatrix, StateVector>(this->x_, model_.F, model_.fx);
        } else if (model_.jacob_f) {
            model_.F = model_.jacob_f(this->x_);
        }

        this->x_ = model_.fx(this->x_);
        this->template normalizeAngles<StateVector>(this->x_, this->state_angle_flag_);
        this->P_ = model_.F * this->P_ * model_.F.transpose() + model_.Q;
    }

    void computeUpdate(const MeasureVector& z) {
        if (autoJacobianH_) {
            this->template computeJacobian<MatrixH, MeasureVector>(this->x_, model_.H, model_.hx);
        } else if (model_.jacob_h) {
            model_.H = model_.jacob_h(this->x_);
        }
        MeasureVector y = z - model_.hx(this->x_);
        this->template normalizeAngles<MeasureVector>(y, this->measurement_angle_flag_);
        MeasureMatrix S = model_.H * this->P_ * model_.H.transpose() + model_.R;
        MatrixK K = this->P_ * model_.H.transpose() * S.ldlt().solve(MeasureMatrix::Identity());
        this->x_ = this->x_ + K * y;
        this->P_ = (I_ - K*model_.H) * this->P_ * (I_ - K*model_.H).transpose() + K*model_.R*K.transpose();
    }
};

#endif