#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>

#include "../include/KalmanFilter/LinearKalmanFilter.hpp"

// --- Constants ---
const double DT = 0.1;            // Time step
const double DURATION = 30.0;     // Run for 10 seconds
const double TRUE_VELOCITY = 5.0; // m/s
const double MEASURE_NOISE = 2.0; // meters (Standard Deviation)

// --- Define Types ---
// 1. The Filter: Defined ONLY by State Dimension (Pos, Vel)
using LKF = LinearKalmanFilter<2>;

// 2. The Models: Defined explicitly
using ProcessModel = LKF::ProcessModel;
using GPSSensor    = LKF::SensorModel<1>; // 1D Measurement (Position)

// 3. Data Types for convenience
using StateVector = LKF::StateVector;
using StateMatrix = LKF::StateMatrix;

int main() {
    // =========================================================================
    // 1. Configure the Process Model (Physics)
    // =========================================================================
    ProcessModel process_model;

    // A. Transition Matrix (F)
    // x_new = x + v*dt
    // v_new = v
    StateMatrix F;
    F << 1.0, DT,
         0.0, 1.0;
    process_model.setF(F);

    // B. Process Noise (Q)
    // Small uncertainty in velocity, very small in position
    StateMatrix Q;
    Q.setIdentity();
    Q(0,0) = 0.01; 
    Q(1,1) = 0.1;
    process_model.setQ(Q);

    // =========================================================================
    // 2. Configure the Sensor Model (GPS)
    // =========================================================================
    GPSSensor gps_model;

    // A. Measurement Matrix (H)
    // z = 1*x + 0*v  (We only measure position)
    GPSSensor::MatrixH H;
    H << 1.0, 0.0;
    gps_model.setH(H);

    // B. Measurement Noise (R)
    GPSSensor::MeasureMatrix R;
    R << MEASURE_NOISE * MEASURE_NOISE;
    gps_model.setR(R);

    // =========================================================================
    // 3. Initialize Filter
    // =========================================================================
    
    // Initial Estimate: Start at 0, assume 0 velocity
    StateVector x_est; 
    x_est << 0.0, 0.0;

    // Initial Covariance: High uncertainty in initial velocity
    StateMatrix P0;
    P0.setIdentity();
    P0(1,1) = 100.0;

    // Instantiate Filter (Note: No models passed here, just State & Covariance)
    LKF lkf(x_est, P0);

    // =========================================================================
    // 4. Simulation Loop
    // =========================================================================
    
    // Truth State: Start at 0, moving at 5 m/s
    StateVector x_true;
    x_true << 0.0, TRUE_VELOCITY;

    // Random Number Generator
    std::mt19937 gen(1234);
    std::normal_distribution<> noise(0.0, MEASURE_NOISE);

    std::cout << "Starting Linear Kalman Filter Test (1D Train)..." << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    std::cout << std::left << std::setw(6) << "Time" 
              << " | " << std::setw(10) << "True Pos" 
              << " | " << std::setw(10) << "Est Pos" 
              << " | " << std::setw(10) << "True Vel" 
              << " | " << std::setw(10) << "Est Vel" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (double t = 0; t <= DURATION; t += DT) {
        // --- A. Generate Data (Simulation) ---
        // Evolve Truth using the physics matrix directly
        x_true = F * x_true; 

        // Generate Noisy GPS Measurement
        Eigen::Matrix<double, 1, 1> z;
        z(0) = x_true(0) + noise(gen);

        // --- B. Filter Step ---
        // 1. Predict (Pass the Physics Model)
        lkf.predict(process_model);

        // 2. Update (Pass the specific Sensor Model and Measurement)
        lkf.update(gps_model, z);

        // --- C. Logging ---
        StateVector est = lkf.getState();
        
        // Print every 1.0 second
        if (std::abs(std::remainder(t, 1.0)) < 1e-5) {
            std::cout << std::fixed << std::setprecision(2)
                      << std::setw(6) << t << " | "
                      << std::setw(10) << x_true(0) << " | "
                      << std::setw(10) << est(0) << " | "
                      << std::setw(10) << x_true(1) << " | "
                      << std::setw(10) << est(1) 
                      << std::endl;
        }
    }

    return 0;
}