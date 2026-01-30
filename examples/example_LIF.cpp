#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>

// Include the new header
#include "../include/KalmanFilter/InformationFilter.hpp"

// --- Simulation Constants ---
const double DT = 0.1;        
const double DURATION = 30.0; 
const double TRUE_VELOCITY = 5.0; // m/s

// Sensor A (High Precision)
const double NOISE_A = 1.0; 

// Sensor B (Low Precision)
const double NOISE_B = 3.0;

// --- TYPE DEFINITIONS ---
// 1. The Filter: Defined by State Dimension (Pos, Vel)
using IF = InformationFilter<2>;

// 2. The Models
using ProcessModel = IF::ProcessModel;
using PositionSensor = IF::SensorModel<1>; // 1D Measurement

// 3. Data Types
using StateVector = IF::StateVector;
using StateMatrix = IF::StateMatrix;

int main() {
    // =========================================================================
    // 1. Configure Process Model (Constant Velocity)
    // =========================================================================
    ProcessModel process_model;

    // Transition Matrix (F)
    StateMatrix F;
    F << 1.0, DT,
         0.0, 1.0;
    process_model.setF(F);

    // Process Noise (Q)
    StateMatrix Q = StateMatrix::Identity();
    Q(0,0) = 0.01; Q(1,1) = 0.1;
    process_model.setQ(Q);

    // =========================================================================
    // 2. Configure Sensor Models (Two different sensors)
    // =========================================================================
    
    // --- Sensor A (High Precision) ---
    PositionSensor sensor_A;
    PositionSensor::MatrixH H_A;
    H_A << 1.0, 0.0; // Measures Position
    sensor_A.setH(H_A);
    
    PositionSensor::MeasureMatrix R_A;
    R_A << NOISE_A * NOISE_A;
    sensor_A.setR(R_A);

    // --- Sensor B (Low Precision) ---
    PositionSensor sensor_B;
    // Identical H, different R
    sensor_B.setH(H_A); 
    
    PositionSensor::MeasureMatrix R_B;
    R_B << NOISE_B * NOISE_B;
    sensor_B.setR(R_B);

    // =========================================================================
    // 3. Initialize Filter
    // =========================================================================
    
    // Initial Estimate
    StateVector x_est; 
    x_est << 0.0, 0.0;

    // Initial Covariance
    StateMatrix P0 = StateMatrix::Identity();
    P0(1,1) = 100.0;

    // Instantiate Information Filter
    // Internally converts (x, P) -> (y, Y)
    IF info_filter(x_est, P0);

    // =========================================================================
    // 4. Simulation Loop
    // =========================================================================
    std::mt19937 gen(1234);
    std::normal_distribution<> noise_a(0.0, NOISE_A);
    std::normal_distribution<> noise_b(0.0, NOISE_B);
    
    StateVector x_true;
    x_true << 0.0, TRUE_VELOCITY;

    std::cout << "Starting Information Filter Test (Dual Sensor Fusion)..." << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    std::cout << std::left << std::setw(6) << "Time" 
              << " | " << std::setw(12) << "Truth Pos" 
              << " | " << std::setw(12) << "Est Pos" 
              << " | " << std::setw(12) << "Meas A"
              << " | " << std::setw(12) << "Meas B" << std::endl;
    std::cout << std::string(75, '-') << std::endl;

    for (double t = 0; t <= DURATION; t += DT) {
        // --- A. Generate Data ---
        x_true = F * x_true;

        // Generate two independent measurements
        Eigen::Matrix<double, 1, 1> z_a, z_b;
        z_a(0) = x_true(0) + noise_a(gen);
        z_b(0) = x_true(0) + noise_b(gen);

        // --- B. Filter Step ---
        
        // 1. Predict (Expensive in IF: Y -> P -> P_pred -> Y_pred)
        info_filter.predict(process_model);

        // 2. Update (Cheap in IF: Additive Information)
        // We can fuse as many sensors as we want simply by calling update multiple times
        // The expensive Matrix Inversion to recover State only happens when we ask for it (syncState)
        
        info_filter.update(sensor_A, z_a);
        info_filter.update(sensor_B, z_b);

        // --- C. Logging ---
        StateVector est = info_filter.getState(); // Auto-calls syncState() internally if needed

        if (std::abs(std::remainder(t, 1.0)) < 1e-5) {
            std::cout << std::fixed << std::setprecision(2)
                      << std::setw(6) << t << " | "
                      << std::setw(12) << x_true(0) << " | "
                      << std::setw(12) << est(0) << " | "
                      << std::setw(12) << z_a(0) << " | "
                      << std::setw(12) << z_b(0) 
                      << std::endl;
        }
    }

    return 0;
}