#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

#include "../include/KalmanFilter/UnscentedKalmanFilter.hpp"

// ============================================================================
// System Dimensions
// State: [x, y, velocity, yaw, yaw_rate]
constexpr int STATE_DIM = 5; 
// Measurement: [range, bearing]
constexpr int MEASURE_DIM = 2; 

using UKF = UnscentedKalmanFilter<STATE_DIM>;
using ProcessModel = UKF::ProcessModel;
using SensorModel = UKF::SensorModel<MEASURE_DIM>;
using StateVector = Eigen::Matrix<double, STATE_DIM, 1>;
using MeasureVector = Eigen::Matrix<double, MEASURE_DIM, 1>;

int main() {
    std::cout << "========================================================\n";
    std::cout << "       UKF CTRV Radar Tracking Simulation\n";
    std::cout << "========================================================\n\n";

    // ========================================================================
    // 1. ADJUSTABLE SIMULATION PARAMETERS
    //    Feel free to modify these to see how the filter reacts!
    // ========================================================================
    
    // --- Timing ---
    const double dt = 0.1;               // Time step in seconds (10 Hz update)
    const int num_steps = 300;            // Total simulation steps

    // --- UKF Sigma Point Tuning (Merwe Scaled) ---
    // alpha: Spread of sigma points. Try 1e-1 (wide) vs 1e-4 (tight)
    const double alpha = 0.001;          
    const double beta = 2.0;             // 2.0 is optimal for Gaussian distributions
    const double kappa = 0.0;            // Secondary scaling (usually 0)

    // --- Noise Profiles (Standard Deviations) ---
    const double std_radar_range = 0.3;     // Radar distance error (meters)
    const double std_radar_bearing = 0.05;  // Radar angle error (radians ~ 2.8 deg)
    
    // --- Initial Conditions ---
    StateVector true_state;
    // Start at origin, moving 5m/s, turning at 0.1 rad/s
    true_state << 0.0, 0.0, 5.0, 0.0, 0.1; 

    StateVector filter_initial_state;
    // BAD INITIAL GUESS: Tell the filter it's at (2, -2), moving slower, facing the wrong way!
    filter_initial_state << 2.0, -2.0, 3.0, 0.5, 0.0; 

    // ========================================================================
    // 2. CONFIGURE PROCESS MODEL (CTRV Kinematics)
    // ========================================================================
    ProcessModel ctrv_model;
    
    // We only write the raw physics! No Jacobians needed.
    // Using 'auto' seamlessly captures the zero-copy Eigen::Ref wrapper.
    ctrv_model.setTransitionFunction([dt](auto state) {
        StateVector next_state = state;
        
        double v = state(2);
        double yaw = state(3);
        double yaw_rate = state(4);

        // Handle division by zero if driving perfectly straight
        if (std::abs(yaw_rate) > 1e-5) {
            next_state(0) += (v / yaw_rate) * (std::sin(yaw + yaw_rate * dt) - std::sin(yaw));
            next_state(1) += (v / yaw_rate) * (std::cos(yaw) - std::cos(yaw + yaw_rate * dt));
        } else {
            next_state(0) += v * std::cos(yaw) * dt;
            next_state(1) += v * std::sin(yaw) * dt;
        }
        
        next_state(3) += yaw_rate * dt; // Update yaw
        return next_state;
    });

    // CRITICAL: Tell the filter that Index 3 (yaw) is a circular angle
    ctrv_model.setAsAngle(3, true);

    // Set Process Noise Covariance (Q)
    Eigen::Matrix<double, STATE_DIM, STATE_DIM> Q = Eigen::Matrix<double, STATE_DIM, STATE_DIM>::Identity();
    Q(0, 0) = 0.1; Q(1, 1) = 0.1; // Small position uncertainty
    Q(2, 2) = 0.5;                // Velocity variance
    Q(3, 3) = 0.05;               // Yaw variance
    Q(4, 4) = 0.1;                // Yaw rate variance
    ctrv_model.setQ(Q);

    // ========================================================================
    // 3. CONFIGURE SENSOR MODEL (Radar)
    // ========================================================================
    SensorModel radar_model;

    // Raw Cartesian (x,y) to Polar (r, phi) conversion. 
    radar_model.setMeasurementFunction([](auto state) {
        MeasureVector z;
        double px = state(0);
        double py = state(1);
        
        double range = std::sqrt(px * px + py * py);
        if (range < 1e-4) range = 1e-4; // Prevent singularity at origin
        
        z(0) = range;
        z(1) = std::atan2(py, px); // Bearing
        return z;
    });

    // CRITICAL: Tell the filter that Index 1 (bearing) is a circular angle
    radar_model.setAsAngle(1, true);

    // Set Measurement Noise Covariance (R)
    Eigen::Matrix<double, MEASURE_DIM, MEASURE_DIM> R = Eigen::Matrix<double, MEASURE_DIM, MEASURE_DIM>::Zero();
    R(0, 0) = std_radar_range * std_radar_range;
    R(1, 1) = std_radar_bearing * std_radar_bearing;
    radar_model.setR(R);

    // ========================================================================
    // 4. INITIALIZE FILTER & RANDOM NOISE GENERATORS
    // ========================================================================
    Eigen::Matrix<double, STATE_DIM, STATE_DIM> initial_P = Eigen::Matrix<double, STATE_DIM, STATE_DIM>::Identity() * 2.0;
    
    UKF filter(filter_initial_state, initial_P);
    filter.setSigmaPointParameters(alpha, beta, kappa);
    
    std::default_random_engine gen;
    std::normal_distribution<double> noise_range(0.0, std_radar_range);
    std::normal_distribution<double> noise_bearing(0.0, std_radar_bearing);

    // ========================================================================
    // 5. RUN SIMULATION LOOP
    // ========================================================================
    std::cout << std::fixed << std::setprecision(2);
    std::cout << " Step | True (X, Y)     | Est (X, Y)      | True Yaw | Est Yaw " << std::endl;
    std::cout << "---------------------------------------------------------------" << std::endl;

    for (int step = 0; step <= num_steps; ++step) {
        
        // Print Current State
        StateVector est = filter.getState();
        std::cout << std::setw(5) << step << " | "
                  << std::setw(6) << true_state(0) << ", " << std::setw(6) << true_state(1) << " | "
                  << std::setw(6) << est(0) << ", " << std::setw(6) << est(1) << " | "
                  << std::setw(8) << true_state(3) << " | " << std::setw(7) << est(3) << std::endl;

        // 1. Simulate Reality (Move the true state forward)
        true_state = ctrv_model.fx(true_state);

        // 2. Simulate Radar Sensor (Generate noisy measurement)
        MeasureVector z = radar_model.hx(true_state);
        z(0) += noise_range(gen);
        z(1) += noise_bearing(gen);

        // 3. Filter Prediction (Extracts & Propagates Sigma Points!)
        filter.predict(ctrv_model);

        // 4. Filter Update (Maps Sigma Points to Radar Space!)
        filter.update(radar_model, z);
    }

    return 0;
}