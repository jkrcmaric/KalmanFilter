#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include "../include/KalmanFilter/ExtendedKalmanFilter.hpp" 

// --- Simulation Constants ---
const double DT           = 0.1;
const double SIM_DURATION = 60.0; 

// Physics (Target Motion)
const double RADIAL_SPEED = 20.0; 
const double ANGULAR_VEL  = 0.2;   
const double VERTICAL_VEL = 2.0;   
const double ALT_AMPLITUDE= 50.0; 

// Sensor Noise
const double NOISE_RANGE  = 5.0;
const double NOISE_AZIM   = 0.01; 
const double NOISE_ELEV   = 0.01;

// --- SPHERICAL STATE VECTOR ---
// State: [r, az, el, r_dot, az_dot, el_dot]
using EKF = ExtendedKalmanFilter<6, 3>;
using StateVector = EKF::StateVector;
using MeasureVector = EKF::MeasureVector;

struct DataPoint {
    double t;
    Eigen::Vector3d x_cart_true; // Cartesian Truth (for logging)
    MeasureVector z_meas;        // Spherical Measurement (for filter)
};

// --- Helper: Generate Data ---
DataPoint generateData(double t, std::mt19937& gen, std::normal_distribution<>& d) {
    DataPoint dp;
    dp.t = t;

    // 1. Physics in Cartesian
    double r_sim = 1000.0 + RADIAL_SPEED * t;
    double theta = ANGULAR_VEL * t;
    
    double px = r_sim * std::cos(theta);
    double py = r_sim * std::sin(theta);
    double pz = 500.0 + (VERTICAL_VEL * t) + (ALT_AMPLITUDE * std::sin(theta));

    // Store Cartesian Truth
    dp.x_cart_true << px, py, pz;

    // 2. Generate Noisy Measurement (Spherical)
    double range = std::sqrt(px*px + py*py + pz*pz);
    double gd    = std::sqrt(px*px + py*py);
    double az    = std::atan2(py, px);
    double el    = std::atan2(pz, gd);

    dp.z_meas << range + d(gen) * NOISE_RANGE,
                 az    + d(gen) * NOISE_AZIM,
                 el    + d(gen) * NOISE_ELEV;
    
    dp.z_meas(1) = std::remainder(dp.z_meas(1), 2.0*M_PI);

    return dp;
}

// --- Helper: Convert Spherical State to Cartesian Position ---
Eigen::Vector3d sphericalToCartesian(const StateVector& x_sph) {
    double r  = x_sph(0);
    double az = x_sph(1);
    double el = x_sph(2);

    // Geometry matching our generation logic:
    // Ground Dist = r * cos(el)
    // z = r * sin(el)
    // x = Ground Dist * cos(az)
    // y = Ground Dist * sin(az)
    
    double gd = r * std::cos(el);
    double px = gd * std::cos(az);
    double py = gd * std::sin(az);
    double pz = r * std::sin(el);

    return Eigen::Vector3d(px, py, pz);
}

int main() {
    EKF::SystemModel model;

    // --- 1. SPHERICAL PROCESS MODEL ---
    // Simple kinematic update on spherical states
    model.fx = [](const StateVector& x) {
        StateVector x_next = x;
        x_next(0) += x(3) * DT; // Range
        x_next(1) += x(4) * DT; // Azimuth
        x_next(2) += x(5) * DT; // Elevation
        return x_next;
    };

    model.jacob_f = [](const StateVector& x) {
        EKF::StateMatrix F = EKF::StateMatrix::Identity();
        F(0,3) = DT; F(1,4) = DT; F(2,5) = DT;
        return F;
    };

    // --- 2. MEASUREMENT MODEL (LINEAR) ---
    // We measure the first 3 states directly
    model.hx = [](const StateVector& x) {
        MeasureVector z;
        z << x(0), x(1), x(2); 
        return z;
    };

    model.jacob_h = [](const StateVector& x) {
        EKF::MatrixH H = EKF::MatrixH::Zero();
        H(0,0) = 1.0; H(1,1) = 1.0; H(2,2) = 1.0;
        return H;
    };

    // --- 3. TUNING ---
    model.Q.setIdentity();
    model.Q.block<3,3>(3,3) *= 0.5; // Allow velocity variance

    model.R.setIdentity();
    model.R.diagonal() << NOISE_RANGE*NOISE_RANGE, 
                          NOISE_AZIM*NOISE_AZIM, 
                          NOISE_ELEV*NOISE_ELEV;

    // --- Initialization ---
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);
    
    // Get initial truth to set starting state
    DataPoint init = generateData(0, gen, d);
    
    // Convert initial truth Cartesian -> Spherical for the state
    double r0 = init.x_cart_true.norm();
    double az0 = std::atan2(init.x_cart_true.y(), init.x_cart_true.x());
    double el0 = std::atan2(init.x_cart_true.z(), init.x_cart_true.head<2>().norm());

    StateVector x0;
    x0.setZero();
    x0(0) = r0; x0(1) = az0; x0(2) = el0; 
    // Leave velocities (indices 3,4,5) as 0 to test convergence

    EKF::StateMatrix P0 = EKF::StateMatrix::Identity();
    P0.diagonal() << 100, 0.1, 0.1, 100, 0.1, 0.1;

    EKF ekf(x0, model, P0);
    // Handle Azimuth Wrapping
    ekf.setStateAsAngle(1, true); 
    ekf.setMeasurementAsAngle(1, true);

    std::cout << "\nStarting Simulation (Spherical Tracking -> Cartesian Output)..." << std::endl;
    std::cout << std::left << std::setw(6) << "Time" 
              << " | " << std::setw(25) << "Truth (x, y, z)" 
              << " | " << std::setw(25) << "Est (x, y, z)" 
              << " | " << std::setw(10) << "3D Error" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (double t = DT; t <= SIM_DURATION; t += DT) {
        DataPoint dp = generateData(t, gen, d);
        
        ekf.predict();
        ekf.update(dp.z_meas);

        // 1. Get Spherical Estimate
        StateVector est_spherical = ekf.getState();
        
        // 2. Convert to Cartesian for Display
        Eigen::Vector3d est_cart = sphericalToCartesian(est_spherical);

        // 3. Compute Error
        double error_3d = (dp.x_cart_true - est_cart).norm();

        if (std::abs(std::remainder(t, 1.0)) < 1e-5) {
            std::cout << std::fixed << std::setprecision(1)
                      << std::setw(6) << t << " | "
                      // Truth
                      << std::setw(7) << dp.x_cart_true(0) << " "
                      << std::setw(7) << dp.x_cart_true(1) << " "
                      << std::setw(7) << dp.x_cart_true(2) << " | "
                      // Estimate (Converted)
                      << std::setw(7) << est_cart(0) << " "
                      << std::setw(7) << est_cart(1) << " "
                      << std::setw(7) << est_cart(2) << " | "
                      // Error
                      << std::setw(10) << std::setprecision(3) << error_3d
                      << std::endl;
        }
    }
    return 0;
}