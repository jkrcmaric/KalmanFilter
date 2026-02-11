#ifndef UNSCENTED_KALMAN_FILTER_HPP
#define UNSCENTED_KALMAN_FILTER_HPP

#include "KalmanFilter.hpp"


template <int StateDim>
class UnscentedKalmanFilter : public KalmanFilter<UnscentedKalmanFilter<StateDim>, StateDim> {
    using Base = KalmanFilter<UnscentedKalmanFilter<StateDim>, StateDim>;
    friend class KalmanFilter<UnscentedKalmanFilter<StateDim>, StateDim>;

public:
    using typename Base::StateVector;
    using typename Base::StateMatrix;
    using typename Base::ProcessModel;
    
    template <int Dim>
    using SensorModel = typename Base::template SensorModel<Dim>;

    using SigmaMatrix = Eigen::Matrix<double, StateDim, 2*StateDim + 1>;
    using WeightVector = Eigen::Matrix<double, 2*StateDim + 1, 1>;

    class MerweScaledSigmaPoints {
    public:
        MerweScaledSigmaPoints(double alpha = 0.1, 
                               double beta = 2.0, 
                               double kappa = 1.0) {
            initialize(alpha, beta, kappa);
        }

        void calculateSigmaPoints(const StateVector& x, const StateMatrix& P) {
            Eigen::LLT<StateMatrix> L((n_ + lambda_) * P);

            sigmas_.col(0) = x;
            for (size_t i = 0; i < n_; ++i) {
                sigmas_.col(i+1) = x + L.matrixL().col(i);
                sigmas_.col(n_+i+1) = x - L.matrixL().col(i);
            }
        }

        auto getSigmas(int i) const { return sigmas_.col(i); }
        double getMeanWeights(int i) const { return Wm_(i); }
        double getCovarianceWeights(int i) const { return Wc_(i); }

    private:
        const int n_{StateDim};
        double lambda_;

        WeightVector Wm_;
        WeightVector Wc_;
        SigmaMatrix sigmas_;

        void initialize(double alpha, double beta, double kappa) {
            lambda_ = alpha*alpha * (n_ + kappa) - n_;
            calculateWeights(alpha, beta);
        }

        void calculateWeights(double alpha, double beta) {
            Wm_.setConstant(1.0 / (2.0*n_ + 2.0*lambda_));
            Wc_.setConstant(1.0 / (2.0*n_ + 2.0*lambda_));

            Wm_(0) = lambda_ / (n_ + lambda_);
            Wc_(0) = lambda_ / (n_ + lambda_) + 1.0 - alpha*alpha + beta;
        }
    };

    UnscentedKalmanFilter(const StateVector& x, const StateMatrix& P) : Base(x, P) {}

    void setSigmaPointParameters(double alpha, double beta, double kappa) {
        sigma_points_ = MerweScaledSigmaPoints(alpha, beta, kappa);
    }

protected:
    MerweScaledSigmaPoints sigma_points_{MerweScaledSigmaPoints()};
    SigmaMatrix Y_;
    SigmaMatrix Y_x_;

    void computePrediction(ProcessModel& model) {
        sigma_points_.calculateSigmaPoints(this->x_, this->P_);

        StateVector delta_x{StateVector::Zero()};
        for (size_t i = 0; i < Y_.cols(); ++i) {
            Y_.col(i) = model.fx(sigma_points_.getSigmas(i));
            // Wrap angles if defined (e.g., heading)
            if (model.hasAngle()) {
                this->template normalizeAngles<StateDim>(this->Y_.col(i), model.getAngleFlags());
            }

            StateVector diff = Y_.col(i) - Y_.col(0);
            // Wrap angles if defined (e.g., heading)
            if (model.hasAngle()) {
                this->template normalizeAngles<StateDim>(diff, model.getAngleFlags());
            }

            delta_x += sigma_points_.getMeanWeights(i)*diff;
        }
        this->x_ = Y_.col(0) + delta_x;

        // Wrap angles if defined (e.g., heading)
        if (model.hasAngle()) {
            this->template normalizeAngles<StateDim>(this->x_, model.getAngleFlags());
        }

        StateMatrix Psum{StateMatrix::Zero()};
        for (size_t i = 0; i < Y_.cols(); ++i) {
            Y_x_.col(i) = Y_.col(i) - this->x_;
            // Wrap angles if defined (e.g., heading)
            if (model.hasAngle()) {
                this->template normalizeAngles<StateDim>(Y_x_.col(i), model.getAngleFlags());
            }

            Psum += sigma_points_.getCovarianceWeights(i)*Y_x_.col(i)*Y_x_.col(i).transpose();
        }
        this->P_ = Psum + model.Q();

        // Force physical constraints (clamping/projection)
        model.enforceConstraints(this->x_);
    }

    template <int MeasureDim>
    bool computeUpdate(SensorModel<MeasureDim>& model, const Eigen::Matrix<double, MeasureDim, 1>& z) {
        using MeasureMatrix = Eigen::Matrix<double, MeasureDim, MeasureDim>;
        using MeasureVector = Eigen::Matrix<double, MeasureDim, 1>;
        using MatrixK = Eigen::Matrix<double, StateDim, MeasureDim>;
        using MatrixZ = Eigen::Matrix<double, MeasureDim, 2*StateDim + 1>;

        MatrixZ Z{MatrixZ::Zero()};
        MeasureVector delta_z{MeasureVector::Zero()};
        for (size_t i = 0; i < Y_.cols(); ++i) {
            Z.col(i) = model.hx(Y_.col(i));

            MeasureVector diff = Z.col(i) - Z.col(0);
            // Wrap angles if defined (e.g., heading)
            if (model.hasAngle()) {
                this->template normalizeAngles<MeasureDim>(diff, model.getAngleFlags());
            }

            delta_z += sigma_points_.getMeanWeights(i)*diff;
        }
        MeasureVector uz = Z.col(0) + delta_z;

        MeasureVector y = z - uz;
        // Wrap innovation angles (critical for shortest path error)
        if (model.hasAngle()) {
            this->template normalizeAngles<MeasureDim>(y, model.getAngleFlags());
        }

        MeasureMatrix Pz{MeasureMatrix::Zero()};
        MatrixK K{MatrixK::Zero()};
        for (size_t i = 0; i < Y_.cols(); ++i) {
            MeasureVector Z_uz = Z.col(i) - uz;
            // Wrap angles if defined (e.g., heading)
            if (model.hasAngle()) {
                this->template normalizeAngles<MeasureDim>(Z_uz, model.getAngleFlags());
            }

            Pz += sigma_points_.getCovarianceWeights(i)*Z_uz*Z_uz.transpose();
            K += sigma_points_.getCovarianceWeights(i)*Y_x_.col(i)*Z_uz.transpose();
        }
        Pz += model.R();

        if (!this->rectangularGate(model, Pz, y)) return false; // Tier 1: Fast Sigma check
        if (!model.domainGate(z, y, Pz)) return false;          // Tier 2: User Logic

        Eigen::LDLT<MeasureMatrix> Pz_ldlt(Pz);
        if (!this->mahalanobisGate(model, Pz_ldlt, y)) return false; // Tier 3: Statistical
     
        K *= Pz_ldlt.solve(MeasureMatrix::Identity());

        this->x_ += K * y;

        this->P_ -= K * Pz * K.transpose();

        return true;
    }
};

#endif