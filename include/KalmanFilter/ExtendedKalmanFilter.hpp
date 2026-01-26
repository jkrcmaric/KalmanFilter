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
    using typename Base::MatrixH;

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
            if (model_.jacob_f) setAutomaticJacobianF(false);
            if (model_.jacob_h) setAutomaticJacobianH(false);
        }

    // Allow tuning numerical differentiation step size per dimension
    void setEpsilon(int index, double value) { EPSILON_.at(index) = value; }

    // Set whether automatic Jacobian is automatically computed via numerical differentiaiton
    void setAutomaticJacobianF(bool value) { autoJacobianF_ = value; }
    void setAutomaticJacobianH(bool value) { autoJacobianH_ = value; }

    void setModel(const EKFSystemModel& model) { 
        model_ = model; 

        if (model_.jacob_f) setAutomaticJacobianF(false);
        if (model_.jacob_h) setAutomaticJacobianH(false);
    }
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
    std::vector<double> EPSILON_{std::vector<double>(StateDim, 1e-6)}; // for numerical differentiation
    bool autoJacobianF_{true};
    bool autoJacobianH_{true};
    EKFSystemModel model_;

    void computePrediction() {
        if (autoJacobianF_ ) {
            computeJacobian<StateMatrix, StateVector>(this->x_, model_.F, model_.fx);
        } else if (model_.jacob_f) {
            model_.F = model_.jacob_f(this->x_);
        }

        this->x_ = model_.fx(this->x_);
    }

    void computeInnovation(const MeasureVector& z, MeasureVector& y) {
        if (autoJacobianH_) {
            computeJacobian<MatrixH, MeasureVector>(this->x_, model_.H, model_.hx);
        } else if (model_.jacob_h) {
            model_.H = model_.jacob_h(this->x_);
        }
        y = z - model_.hx(this->x_);
    }

    // compute Jacobian via numerical differentiation
    template <typename T1, typename T2>
    void computeJacobian(const StateVector& x, T1& J, const std::function<T2(const StateVector&)>& fx) {
        StateVector x_temp{x};
        T2 y_plus{};
        T2 y_minus{};

        for (int i{0}; i < StateDim; ++i) {
            double original_value{x(i)};

            x_temp(i) = original_value + EPSILON_.at(i);
            y_plus = fx(x_temp);

            x_temp(i) = original_value - EPSILON_.at(i);
            y_minus = fx(x_temp);

            x_temp(i) = original_value;
            J.col(i) = (y_plus - y_minus) / (2*EPSILON_.at(i));
        }
    }
};

#endif