#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

#include "../include/KalmanFilter/InformationFilter.hpp" 

// --- Simulation Constants ---
const double DT           = 0.1;
const double SIM_DURATION = 60.0; 

// Physics (Target Motion)
const double RADIAL_SPEED = 20.0; 
const double ANGULAR_VEL  = 0.1;   
const double VERTICAL_VEL = 2.0;   

// --- SENSOR SPECS ---
// Sensor 1: RADAR (Good Range, Bad Angles)
const double RADAR_NOISE_R  = 2.0;  // 2m error
const double RADAR_NOISE_AZ = 0.1;  // ~5.7 degrees error (High!)
const double RADAR_NOISE_EL = 0.1; 

// Sensor 2: OPTICAL (No Range, Perfect Angles)
const double OPTIC_NOISE_AZ = 0.005; // ~0.2 degrees error (Precise!)
const double OPTIC_NOISE_EL = 0.005;

// --- TYPE DEFINITIONS ---
using EIF = InformationFilter<6>; // State: [r, az, el, r_dot, az_dot, el_dot]

// Define TWO different sensor models
using RadarModel   = EIF::SensorModel<3>; // Measures [r, az, el]
using OpticalModel = EIF::SensorModel<2>; // Measures [az, el] only

using StateVector = EIF::StateVector;

// Data Container
struct DataPoint {
    double t;
    Eigen::Vector3d x_cart_true; 
    
    // We hold two different measurements
    RadarModel::MeasureVector z_radar;
    OpticalModel::MeasureVector z_optical;
};

// --- Helper: Generate Data ---
DataPoint generateData(double t, std::mt19937& gen, std::normal_distribution<>& d) {
    DataPoint dp;
    dp.t = t;

    // 1. Physics (Spiral Upwards)
    double r_sim = 1000.0 + RADIAL_SPEED * t;
    double theta = ANGULAR_VEL * t;
    double px = r_sim * std::cos(theta);
    double py = r_sim * std::sin(theta);
    double pz = 500.0 + (VERTICAL_VEL * t);

    dp.x_cart_true << px, py, pz;

    // 2. Truth Spherical Coordinates
    double range = std::sqrt(px*px + py*py + pz*pz);
    double gd    = std::sqrt(px*px + py*py);
    double az    = std::atan2(py, px);
    double el    = std::atan2(pz, gd);

    // 3. Generate RADAR Data (Noisy Angles)
    dp.z_radar << range + d(gen) * RADAR_NOISE_R,
                  az    + d(gen) * RADAR_NOISE_AZ,
                  el    + d(gen) * RADAR_NOISE_EL;
    dp.z_radar(1) = std::remainder(dp.z_radar(1), 2.0*M_PI);

    // 4. Generate OPTICAL Data (Noisy Angles, No Range)
    dp.z_optical << az + d(gen) * OPTIC_NOISE_AZ,
                    el + d(gen) * OPTIC_NOISE_EL;
    dp.z_optical(0) = std::remainder(dp.z_optical(0), 2.0*M_PI);

    return dp;
}

// Helper: Cartesian Conversion
Eigen::Vector3d sphericalToCartesian(const StateVector& x) {
    double r = x(0); double az = x(1); double el = x(2);
    double gd = r * std::cos(el);
    return Eigen::Vector3d(gd*std::cos(az), gd*std::sin(az), r*std::sin(el));
}

int main() {
    // =========================================================================
    // 1. Process Model (Kinematics)
    // =========================================================================
    EIF::ProcessModel process;
    process.setTransitionFunction([](auto x) {
        StateVector next = x;
        next(0) += x(3) * DT; next(1) += x(4) * DT; next(2) += x(5) * DT;
        return next;
    });
    
    EIF::StateMatrix Q = EIF::StateMatrix::Identity();
    Q.block<3,3>(3,3) *= 0.1; // Low process noise
    process.setQ(Q);
    process.setAsAngle(1, true); // Azimuth wraps

    // =========================================================================
    // 2. Radar Model (3D, Noisy Angles)
    // =========================================================================
    RadarModel radar;
    radar.setMeasurementFunction([](auto x) {
        return x.head(3); // Returns [r, az, el]
    });
    
    // Analytical Jacobian for Radar (3x6)
    radar.setAnalyticalJacobianH([](auto x) {
        RadarModel::MatrixH H = RadarModel::MatrixH::Zero();
        H(0,0)=1; H(1,1)=1; H(2,2)=1;
        return H;
    });

    RadarModel::MeasureMatrix R_radar;
    R_radar.setIdentity();
    R_radar.diagonal() << RADAR_NOISE_R*RADAR_NOISE_R, 
                          RADAR_NOISE_AZ*RADAR_NOISE_AZ, 
                          RADAR_NOISE_EL*RADAR_NOISE_EL;
    radar.setR(R_radar);
    radar.setAsAngle(1, true); // Azimuth wraps

    // =========================================================================
    // 3. Optical Model (2D, Precise Angles)
    // =========================================================================
    OpticalModel optical;
    optical.setMeasurementFunction([](auto x) {
        OpticalModel::MeasureVector z;
        z << x(1), x(2); // Returns [az, el] only!
        return z;
    });

    // Analytical Jacobian for Optical (2x6)
    optical.setAnalyticalJacobianH([](auto x) {
        OpticalModel::MatrixH H = OpticalModel::MatrixH::Zero();
        H(0,1)=1; // az maps to state index 1
        H(1,2)=1; // el maps to state index 2
        return H;
    });

    OpticalModel::MeasureMatrix R_optic;
    R_optic.setIdentity();
    R_optic.diagonal() << OPTIC_NOISE_AZ*OPTIC_NOISE_AZ,
                          OPTIC_NOISE_EL*OPTIC_NOISE_EL;
    optical.setR(R_optic);
    optical.setAsAngle(0, true); // Optical index 0 is Azimuth

    // =========================================================================
    // 4. Setup Filter
    // =========================================================================
    std::mt19937 gen(1234);
    std::normal_distribution<> d(0, 1);
    
    // Initialize with a rough guess
    DataPoint init = generateData(0, gen, d);
    StateVector x0 = StateVector::Zero();
    x0(0) = 1000.0; // Rough range guess
    
    EIF::StateMatrix P0 = EIF::StateMatrix::Identity() * 100.0;
    
    EIF eif(x0, P0); // Initialize in Information Space

    std::cout << "Starting Multi-Sensor EIF Fusion (Radar + Optical)..." << std::endl;
    std::cout << "Demonstrating additive updates with different sensor dimensions." << std::endl;
    std::cout << std::string(85, '-') << std::endl;
    std::cout << std::left << std::setw(6) << "Time"
              << " | " << std::setw(20) << "Truth (x,y)"
              << " | " << std::setw(20) << "Est (x,y)"
              << " | " << std::setw(15) << "3D Error (m)" << std::endl;
    std::cout << std::string(85, '-') << std::endl;

    for (double t = DT; t <= SIM_DURATION; t += DT) {
        DataPoint dp = generateData(t, gen, d);

        // --- PREDICTION ---
        eif.predict(process);

        // --- FUSION STEP ---
        // The Information Filter shines here. 
        // We just add contributions from whatever sensors are available.
        
        // 1. FUSE RADAR (Provides coarse location + Range)
        eif.fuse(radar, dp.z_radar);

        // 2. FUSE OPTICAL (Tightens the Angle estimate significantly)
        eif.fuse(optical, dp.z_optical);

        eif.updateState();

        // --- LOGGING ---
        if (std::abs(std::remainder(t, 2.0)) < 1e-5) {
            Eigen::Vector3d est = sphericalToCartesian(eif.getState());
            double err = (dp.x_cart_true - est).norm();
            
            std::cout << std::fixed << std::setprecision(1)
                      << std::setw(6) << t << " | "
                      << std::setw(8) << dp.x_cart_true.x() << ", " << std::setw(8) << dp.x_cart_true.y() << " | "
                      << std::setw(8) << est.x() << ", " << std::setw(8) << est.y() << " | "
                      << std::setw(10) << std::setprecision(3) << err 
                      << std::endl;
        }
    }

    return 0;
}