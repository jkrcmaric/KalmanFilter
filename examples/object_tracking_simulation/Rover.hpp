#include <eigen3/Eigen/Dense>

template <int PosDim>
class Rover {
    using PosVector = Eigen::Matrix<double, PosDim, 1>;

public:
    Rover(PosVector pos = PosVector::Constant(PosDim, 0), 
          PosVector vel = PosVector::Constant(PosDim, 1), 
          double pos_noise = 1, 
          double vel_noise = 0.05)
        : pos_{pos}, vel_{vel}, pos_noise_{pos_noise}, vel_noise_{vel_noise} {}

    void move(double dt) {
        pos_ = pos_ + dt*(vel_ + vel_noise_*PosVector::Random(2, 1));
        vel_ = vel_;
    }

    PosVector getMeasurment() const {

        return pos_ + pos_noise_*PosVector::Random(2, 1);
    }

    PosVector getPosition() const {

        return pos_;
    }

private:
    PosVector pos_{};
    PosVector vel_{};
    double pos_noise_{};
    double vel_noise_{};
};