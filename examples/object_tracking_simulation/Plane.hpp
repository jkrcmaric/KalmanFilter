#include <random>
#include <chrono>
#include <cmath>

#include <eigen3/Eigen/Dense>

template <int ClimbRate, int MaxAltitude> 
class Plane {
public:
    using PositionVector = Eigen::Matrix<double, 2, 1>;
    using MeasurementVector = Eigen::Matrix<double, 3, 1>;

    Plane(PositionVector pos = PositionVector::Constant(2, 0.0), 
          PositionVector vel = PositionVector::Constant(2, 1.0), 
          double altitude = 0.0,
          double range_noise = 1, 
          double vel_noise = 0.08,
          double angle_noise = 0.01)
        : pos_{pos}, vel_{vel}, altitude_{altitude}, 
          range_noise_{range_noise}, vel_noise_{vel_noise}, angle_noise_{angle_noise} {}

    void move(double dt) {
        pos_ += dt*(vel_ + randomVector(vel_noise_));
        vel_ = vel_;

        if (altitude_ <= MaxAltitude) {
            double altitude = {altitude_ + dt*(climb_rate_ + randomNumber(vel_noise_))};
            altitude_ = (altitude <= MaxAltitude) ? altitude : MaxAltitude;
        }
        altitude_ += dt*(randomNumber(vel_noise_));
        climb_rate_ = climb_rate_;
    }

    MeasurementVector getMeasurment() {

        MeasurementVector measurement;
        measurement(0) = std::sqrt(pos_(0)*pos_(0) + pos_(1)*pos_(1) + altitude_*altitude_) + randomNumber(range_noise_);
        measurement(1) = std::atan2(pos_(1), pos_(0)) + randomNumber(angle_noise_);
        measurement(2) = std::atan2(altitude_, pos_.norm()) + randomNumber(angle_noise_);

        return measurement;
    }

    PositionVector getXYPosition() const {

        return pos_;
    }

    double getAltitude() const {

        return altitude_;
    }

private:
    PositionVector pos_{};
    PositionVector vel_{};
    double altitude_{};
    double climb_rate_{ClimbRate};
    double range_noise_{};
    double vel_noise_{};
    double angle_noise_{};

    long int seed_{std::chrono::system_clock::now().time_since_epoch().count()};
    std::mt19937 generator_{static_cast<long unsigned int>(seed_)};    

    PositionVector randomVector(double sd) {
        std::normal_distribution<double> distribution(0.0, sd);

        PositionVector random_vec;
        random_vec(0) = distribution(generator_);
        random_vec(1) = distribution(generator_);

        return random_vec;
    }

    double randomNumber(double sd) {
        std::normal_distribution<double> distribution(0.0, sd);

        return distribution(generator_);
    }
};