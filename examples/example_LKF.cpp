#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>

#include "../include/KalmanFilter/LinearKalmanFilter.hpp"

// --- Constants ---
const double DT = 0.1;        // Time step
const double DURATION = 60.0; // Run for 10 seconds
const double TRUE_VELOCITY = 5.0; // m/s
const double MEASURE_NOISE = 2.0; // meters (Standard Deviation)

// --- Define Types ---
// State: [Position, Velocity] (2 Dim)
// Meas:  [Position] (1 Dim)
using LKF = LinearKalmanFilter<2, 1>;
using StateVector = LKF::StateVector;
using MeasureVector = LKF::MeasureVector;

int main() {
    // 1. Configure the Linear System Model
    LKF::SystemModel model;

    // A. Transition Matrix (F)
    // x_new = x + v*dt
    // v_new = v
    model.F << 1.0, DT,
               0.0, 1.0;

    // B. Measurement Matrix (H)
    // z = 1*x + 0*v
    model.H << 1.0, 0.0;

    // C. Noise Matrices
    // Process Noise (Q): Small uncertainty in velocity
    model.Q.setIdentity();
    model.Q(0,0) = 0.01;
    model.Q(1,1) = 0.1;

    // Measurement Noise (R): Match sensor specs
    model.R << MEASURE_NOISE * MEASURE_NOISE;

    // 2. Initialize State
    // Truth: Start at 0, moving at 5 m/s
    StateVector x_true;
    x_true << 0.0, TRUE_VELOCITY;

    // Estimate: Start at 0, but assume 0 velocity (Let filter learn it)
    StateVector x_est; 
    x_est << 0.0, 0.0;

    // Initial Covariance (High uncertainty in velocity)
    LKF::StateMatrix P0;
    P0.setIdentity();
    P0(1,1) = 100.0;

    // Instantiate Filter
    LKF lkf(x_est, model, P0);

    // 3. Simulation Loop
    std::mt19937 gen(1234);
    std::normal_distribution<> noise(0.0, MEASURE_NOISE);

    std::cout << "Starting Linear Kalman Filter Test (1D Train)..." << std::endl;
    std::cout << std::left << std::setw(6) << "Time" 
              << " | " << std::setw(10) << "True Pos" 
              << " | " << std::setw(10) << "Est Pos" 
              << " | " << std::setw(10) << "True Vel" 
              << " | " << std::setw(10) << "Est Vel" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (double t = 0; t <= DURATION; t += DT) {
        // --- A. Generate Data ---
        // 1. Evolve Truth (Physics)
        x_true = model.F * x_true; 

        // 2. Generate Noisy Measurement (Sensor)
        MeasureVector z;
        z(0) = x_true(0) + noise(gen);

        // --- B. Filter Step ---
        lkf.predict();
        lkf.update(z);

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