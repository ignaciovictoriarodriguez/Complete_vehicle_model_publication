import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv, os
import math
from collections import deque
from scipy.interpolate import interp1d
import matplotlib as mpl
from dataclasses import dataclass, field

# ============================================================================
# DATA STRUCTURES
# ============================================================================


from scipy.optimize import minimize
# from pacejka_combined import PacejkaTire, create_truck_tire, create_trailer_tire
# MPC controller is defined in this file at line 334, no need to import
# from mpc_controller import MPCSteeringController
# from auto_dt_calculator import compute_optimal_dt, print_dt_analysis

# ============================================================================
# GLOBAL CONFIGURATION PARAMETERS - Easy tuning of solver behavior
# ============================================================================

# Initial velocities for all vehicles (m/s)
# Auto-sync with fixed speed value when fixed speed override is active
INITIAL_SPEED = 20  # Starting forward speed for tractor and all trailers
DEFAULT_TARGET_SPEED = 20  # Default desired speed [m/s] if waypoints don't specify

# Solver iteration limits
MAX_ITERATIONS_TRACTOR = 50  # Newton-Raphson iterations for tractor
MAX_ITERATIONS_TRAILER = MAX_ITERATIONS_TRACTOR # Newton-Raphson iterations for trailers/dolly

# Convergence thresholds (residual error limits)
RESIDUAL_THRESHOLD_STARTUP = 1e-4  # Relaxed threshold during startup (t < 5s) - increased for better low-speed convergence
RESIDUAL_THRESHOLD_STEADY = 1e-4   # Strict threshold for tractor in steady state
RESIDUAL_THRESHOLD_TRAILER_STEADY = 1e-4  # Relaxed threshold for trailers (they can have trickier dynamics)

# Run time
MAX_SIM_TIME = 15 # safety cap (seconds)

# ============================================================================
# TIRE RELAXATION LENGTH SETTINGS
# ============================================================================
# Tire forces don't develop instantly - they build up over a characteristic
# distance as the tire deforms while rolling (relaxation length).
# This adds transient response to steering/braking inputs.

ENABLE_TIRE_RELAXATION = True  # Set to False for steady-state tire model (instant response)
CAP_FORCES = True  # Caps forces Fx and Fy with TIRE RELAXATION to prevent numerical instability from large force jumps (set False to disable rate limiting)

# TIRE FORCE SATURATION MODE (For validation & debugging)
# 'PACEJKA_DIRECT' - Use Pacejka combined_slip() with its built-in sin/arctan saturation
#                    Matches RoadView behavior for validation
# 'SMOOTH_TANH'    - Use linear stiffness + smooth tanh saturation + friction ellipse
#                    Better NR solver gradient properties but does not match RoadView
TIRE_SATURATION_MODE = 'PACEJKA_DIRECT'  # Switch to 'SMOOTH_TANH' for alternative saturation

# Relaxation length parameters [meters]
# Typical values for truck tires (385/65R22.5): 0.3-0.6m
# Smaller = stiffer/faster response, Larger = softer/slower response

SIGMA_X_FRONT = 0.5    # Longitudinal relaxation, front axle [m]
SIGMA_Y_FRONT = 0.4    # Lateral relaxation, front axle [m]
SIGMA_X_REAR = 0.6     # Longitudinal relaxation, rear axle (softer compound) [m]
SIGMA_Y_REAR = 0.5     # Lateral relaxation, rear axle [m]
SIGMA_X_TRAILER = 0.4  # Trailer tires (stiffer compound) [m]
SIGMA_Y_TRAILER = 0.35 # Trailer lateral relaxation [m]

# ============================================================================
# GENERALIZED-ALPHA TIME INTEGRATION PARAMETERS
# ============================================================================
# The Generalized-α method (Chung & Hulbert, 1993) provides 2nd-order accuracy
# with tunable high-frequency numerical damping.
#
# RHO_INF (spectral radius at infinity) controls numerical damping:
#   RHO_INF = 1.0 : No damping (trapezoidal rule) - most accurate, may need smaller dt
#   RHO_INF = 0.6 : Moderate damping - good balance for vehicle dynamics (RECOMMENDED)
#   RHO_INF = 0.0 : Maximum damping (like backward Euler) - most robust but overdamped
#
# Set USE_GENERALIZED_ALPHA = False to use legacy backward Euler solver

USE_GENERALIZED_ALPHA = True   # Enable Generalized-α (True) or use legacy Backward Euler (False)
RHO_INF = 0.6                  # Spectral radius at infinity ∈ [0, 1]

# ============================================================================
# FIXED CONTROL OVERRIDES (Debug/Validation)
# ============================================================================
USE_FIXED_SPEED = False      # If True, overrides controller speed logic
FIXED_SPEED_VALUE = 10.0    # m/s
USE_FIXED_STEERING = False   # If True, overrides controller steering logic
FIXED_STEERING_VALUE = 3.0  # degrees (positive = left, negative = right)
USE_FIXED_SLOPE = False      # If True, overrides terrain pitch (affects gravity)
FIXED_SLOPE_DEG = 1.0        # degrees (positive = uphill, negative = downhill)
USE_FREE_PIVOT = False      # If True, disables rotational stiffness/damping at hitch (frictionless pivot)

# ============================================================================
# ABS (ANTI-LOCK BRAKING SYSTEM) PARAMETERS
# ============================================================================
ENABLE_ABS = True               # Enable Anti-lock Braking System
ABS_SLIP_TARGET = -0.12         # Target slip ratio for max braking (peak of μ-κ curve)
ABS_SLIP_THRESHOLD = -0.15      # κ below this triggers pressure reduction
ABS_RELEASE_THRESHOLD = -0.08   # κ above this allows pressure reapply
ABS_PRESSURE_INCREASE_RATE = 25.0  # Pressure ramp-up rate [1/s] (how fast to reapply)
ABS_PRESSURE_DECREASE_RATE = 40.0  # Pressure dump rate [1/s] (how fast to release)
ABS_MIN_SPEED = 3.0             # ABS inactive below this speed [m/s]

# ============================================================================
# TCS (TRACTION CONTROL SYSTEM) PARAMETERS
# ============================================================================
ENABLE_TCS = True                # Enable Traction Control System
TCS_SLIP_TARGET = 0.10           # Target positive κ for max traction
TCS_SLIP_THRESHOLD = 0.15        # κ above this → cut drive torque
TCS_RELEASE_THRESHOLD = 0.08     # κ below this → restore drive torque
TCS_TORQUE_REDUCTION_RATE = 30.0 # Rate of torque cut [1/s]
TCS_TORQUE_INCREASE_RATE = 20.0  # Rate of torque restore [1/s]
TCS_MIN_SPEED = 2.0              # TCS inactive below this speed [m/s]

# ============================================================================
# ESC (ELECTRONIC STABILITY CONTROL) PARAMETERS
# ============================================================================
ENABLE_ESC = True                # Enable Electronic Stability Control
ESC_YAW_RATE_DEADBAND = 0.02     # [rad/s] below this, no intervention
ESC_GAIN_UNDERSTEER = 0.3        # Corrective braking gain for understeer
ESC_GAIN_OVERSTEER = 0.5         # Corrective braking gain for oversteer
ESC_MAX_BRAKE_FORCE = 5000.0     # Max single-wheel corrective brake [N]
ESC_MIN_SPEED = 5.0              # ESC inactive below this speed [m/s]

# ============================================================================
# QUARTER-CAR VERTICAL DYNAMICS PARAMETERS
# ============================================================================
ENABLE_QUARTER_CAR = True        # Enable per-corner vertical dynamics
QC_UNSPRUNG_MASS = 120.0         # Per-corner unsprung mass [kg]
QC_K_SPRING_F = 250000.0         # Front spring rate [N/m]
QC_K_SPRING_R = 400000.0         # Rear spring rate [N/m] (stiffer for load)
QC_C_DAMPER_F = 15000.0          # Front damper rate [Ns/m]
QC_C_DAMPER_R = 20000.0          # Rear damper rate [Ns/m]
QC_K_TIRE = 1000000.0            # Tire vertical stiffness [N/m]

# ============================================================================
# VARIABLE ROAD FRICTION (μ-MAP) PARAMETERS
# ============================================================================
ENABLE_MU_MAP = True             # Enable position-dependent friction
DEFAULT_MU = 0.9                 # Dry asphalt baseline friction coefficient

# ============================================================================
# HYDROPLANING PARAMETERS
# ============================================================================
ENABLE_HYDROPLANING = True       # Enable hydroplaning model
HYDRO_WATER_DEPTH = 2.0          # Default water depth on wet zones [mm]
HYDRO_TIRE_PRESSURE = 800.0      # Tire inflation pressure [kPa]
HYDRO_TIRE_WIDTH = 315.0         # Tire width [mm] (315/80 R22.5 truck tire)

# ============================================================================
# DUAL-TIRE CONTACT PARAMETERS
# ============================================================================
ENABLE_DUAL_TIRES = True         # Enable dual-tire model on rear axles
DUAL_TIRE_SPACING = 0.30         # Center-to-center distance [m]
DUAL_TIRE_LOAD_BIAS = 0.02       # Inner/outer load bias from axle flex

# ============================================================================
# PNEUMATIC BRAKE DELAY PARAMETERS
# ============================================================================
ENABLE_BRAKE_DELAY = True        # Enable air brake delay model
BRAKE_DELAY_TRACTOR = 0.20       # Tractor brake response delay [s]
BRAKE_DELAY_TRAILER = 0.40       # Trailer brake response delay [s]
BRAKE_PRESSURE_RATE = 5.0        # Pressure build-up rate [1/s]
BRAKE_RELEASE_RATE = 8.0         # Pressure release rate [1/s]

# ============================================================================
# ROLL STABILITY CONTROL (RSC) PARAMETERS
# ============================================================================
ENABLE_RSC = True                # Enable Roll Stability Control
RSC_LAT_ACCEL_WARN = 0.35        # Warning threshold [g]
RSC_LAT_ACCEL_LIMIT = 0.50       # Hard intervention threshold [g]
RSC_ROLL_RATE_THRESHOLD = 0.15   # Roll rate threshold [rad/s]
RSC_THROTTLE_CUT_FACTOR = 0.5    # Reduce throttle to this fraction
RSC_MAX_BRAKE_FORCE = 8000.0     # Max corrective braking [N]

# ============================================================================
# TORQUE CONVERTER PARAMETERS
# ============================================================================
ENABLE_TORQUE_CONVERTER = True   # Enable torque converter model
TC_STALL_TORQUE_RATIO = 2.2      # Torque multiplication at stall
TC_COUPLING_POINT = 0.85         # Speed ratio for lockup
TC_K_FACTOR = 230.0              # Capacity factor [Nm/(rad/s)²]

# ============================================================================
# DIFFERENTIAL PARAMETERS
# ============================================================================
ENABLE_LSD = True                # Enable limited-slip differential
DIFF_TYPE = 'limited_slip'       # 'open', 'limited_slip', 'locking'
LSD_PRELOAD = 200.0              # Clutch preload torque [Nm]
LSD_RAMP_FACTOR = 2.5            # Torque bias ratio
LSD_LOCK_THRESHOLD = 0.1         # Speed diff for full lock [rad/s]

# Auto-sync: if using fixed speed, all vehicles must start at that speed
# to prevent catastrophic initial transient when tire relaxation is enabled
if USE_FIXED_SPEED:
    INITIAL_SPEED = FIXED_SPEED_VALUE

# ============================================================================
# MPC STEERING CONTROLLER PARAMETERS
# ============================================================================
USE_MPC_STEERING = True          # Enable MPC (False = Stanley controller)
MPC_PREDICTION_HORIZON = 30      # Number of future steps to predict (Increased for stability)
MPC_CONTROL_HORIZON = 5          # Number of control moves to optimize
MPC_WEIGHT_CTE = 20.0            # Cross-track error penalty (increased to prioritize tracking)
MPC_WEIGHT_HEADING = 5.0         # Heading error penalty
MPC_WEIGHT_EFFORT = 1.0          # Steering effort penalty (reduced to allow more steering)
MPC_WEIGHT_RATE = 5.0            # Steering rate penalty (reduced to allow faster corrections)
MPC_MAX_STEER_RATE = 35.0        # Maximum steering rate [deg/s] - Increased for agility

# ============================================================================
# EKF STATE ESTIMATOR PARAMETERS
# ============================================================================
ENABLE_EKF = True                    # Enable EKF state estimation (False = perfect state to MPC)

# --- Sensor noise standard deviations (simulated) ---
EKF_GPS_NOISE_XY    = 0.5            # GPS position noise std [m]
EKF_GPS_NOISE_YAW   = 0.02           # GPS/compass heading noise std [rad] (~1.1°)
EKF_IMU_NOISE_VX    = 0.1            # Wheel-odometry longitudinal velocity noise std [m/s]
EKF_IMU_NOISE_OMEGA = 0.01           # IMU gyroscope yaw-rate noise std [rad/s]

# --- Process noise (model uncertainty) ---
EKF_PROCESS_NOISE_POS   = 0.01       # Position states (x, y) [m²]
EKF_PROCESS_NOISE_YAW   = 0.005      # Heading state [rad²]
EKF_PROCESS_NOISE_VX    = 0.1        # Longitudinal velocity [m²/s²]
EKF_PROCESS_NOISE_VY    = 0.5        # Lateral velocity [m²/s²] (higher — hardest to model)
EKF_PROCESS_NOISE_OMEGA = 0.05       # Yaw rate [rad²/s²]

# --- Dynamic bicycle model parameters for EKF & upgraded MPC ---
# These must match TractorHead geometry; kept here for easy tuning
EKF_MASS     = 8000.0                # Vehicle mass [kg]
EKF_IZ       = 12000.0               # Yaw moment of inertia [kg·m²]
EKF_LF       = 1.48                  # CG to front axle [m]  (0.4 × L=3.7)
EKF_LR       = 2.22                  # CG to rear axle [m]   (0.6 × L=3.7)
EKF_CF       = 300000.0              # Front axle cornering stiffness [N/rad]
EKF_CR       = 320000.0              # Rear axle cornering stiffness [N/rad]

# ============================================================================


mpl.use('TkAgg')
# Sources:
# https://github.com/DongChen06/PathTrackingBicycle/tree/master
# https://publications.lib.chalmers.se/records/fulltext/192958/local_192958.pdf
# https://publications.lib.chalmers.se/records/fulltext/233463/233463.pdf

# F1 tracks:
# https://github.com/TUMFTM/racetrack-database

# waypoint file to load
# WAYPOINTS_FILENAME = 'racetrack_waypoints.txt'
WAYPOINTS_FILENAME = 'test_track.xlsx'
# WAYPOINTS_FILENAME = 'Nuerburgring.txt'
INTERP_DISTANCE_RES = 0.05  # distance between interpolated points
INTERP_LOOKAHEAD_DISTANCE = 20  # lookahead in meters
DIST_THRESHOLD_TO_LAST_WAYPOINT = 4.0  # some distance from last position before simulation ends#
DEFAULT_TARGET_SPEED = 20  # m/s (matched to INITIAL_SPEED for stability)

# ============================================================================
# AUTOMATIC TIMESTEP CALCULATION
# ============================================================================
AUTO_CALCULATE_DT = True  # Use eigenvalue analysis to find optimal dt_phys
# If False, uses the manual dt_phys value below

# dt = 0.00001
dt_phys=0.0001  # physics time step (only used if AUTO_CALCULATE_DT = False)
# Small & fixed dt_phys gives integration stability in simulation
dt_controller=0.02  # controller time step
# 20-50Hz matches real-life ECU loops

trajectory_path = "trajectory_non_linear.png"
speed_path = "speed_non_linear.png"
vy_path = "speed_non_linear_lateral.png"
show_animation = False

@dataclass
class HitchParams:
    """Parameters for implicit hitch coupling."""
    stiffness: float  # N/m
    damping: float    # Ns/m
    hitch_point_rel: float = 0.0 # Distance from leader CG to hitch point (m)
    target_pos: tuple = (0.0, 0.0) # (x, y) where follower wants to be
    target_vel: tuple = (0.0, 0.0) # (vx, vy) of follower target point
    mass_follower: float = 0.0   # Mass of follower (optional, for inertia coupling if needed)

def compute_optimal_dt(tractor, trailers=None, rho_inf=0.6, samples_per_period=10, 
                      safety_factor=0.5, enable_tire_relaxation=False, dt_controller=0.02):
    """
    Calculate optimal physics timestep based on system eigenvalues.
    
    The critical frequencies in vehicle dynamics come from:
    1. Cornering stiffness (lateral dynamics)
    2. Tire relaxation dynamics (if enabled)
    3. Articulation stiffness (hitch dynamics)
    4. Roll/pitch suspension dynamics
    5. Controller update rate (dt_phys must be submultiple of dt_controller)
    
    Theory:
    -------
    For a system dx/dt = A*x, eigenvalues λ of A determine dynamics:
    - Oscillatory modes: λ = α ± iω → frequency ω [rad/s]
    - Damped modes: λ = α (real) → time constant τ = 1/|α|
    
    Stability criterion (Generalized-α with RHO_INF):
    - High-frequency modes damped by spectral radius
    - Need: dt < T_min / samples_per_period
    - Where T_min = 2π / ω_max (period of fastest oscillation)
    
    Parameters:
    -----------
    tractor : TractorHead
        Main vehicle object with mass, stiffnesses, etc.
    trailers : list of ArticulatedSegment, optional
        Trailer chain (for articulation dynamics)
    rho_inf : float
        Generalized-α spectral radius (0.6 = moderate damping)
    samples_per_period : int
        Timesteps per oscillation period (10 = good accuracy)
    safety_factor : float
        Conservative multiplier (0.5 = 2× safety margin)
    enable_tire_relaxation : bool
        Whether tire relaxation is active
    dt_controller : float
        Controller update timestep [s] (dt_phys must be ≤ this)
        
    Returns:
    --------
    dt_optimal : float
        Recommended physics timestep [s]
    diagnostics : dict
        Eigenvalues and frequencies for analysis
    """
    
    # =========================================================================
    # 1. LATERAL DYNAMICS (Cornering Stiffness)
    # =========================================================================
    # Bicycle model state: [vy, omega] → 2×2 system
    # A = [[-Cf-Cr)/(m*v),  (-Cf*lf+Cr*lr)/(m*v) - v],
    #      [(-Cf*lf+Cr*lr)/Iz,  (-Cf*lf²-Cr*lr²)/(Iz*v)]]
    
    m = tractor.mass
    lf = tractor.lf  # Distance CG to front axle
    lr = tractor.lr  # Distance CG to rear axle
    Iz = tractor.Iz
    v = max(abs(tractor.vx), 1.0)  # Avoid division by zero
    
    # Get cornering stiffnesses (total per axle)
    Cf = tractor.Cf_axle  # Front axle cornering stiffness [N/rad]
    Cr = tractor.Cr_axle  # Rear axle cornering stiffness [N/rad]
    
    # Lateral dynamics state matrix (bicycle model)
    A_lateral = np.array([
        [-(Cf + Cr) / (m * v),  (-Cf * lf + Cr * lr) / (m * v) - v],
        [(-Cf * lf + Cr * lr) / Iz,  (-Cf * lf**2 - Cr * lr**2) / (Iz * v)]
    ])
    
    eig_lateral = np.linalg.eigvals(A_lateral)
    omega_lateral = np.max(np.abs(np.imag(eig_lateral)))  # Max oscillation freq
    
    # =========================================================================
    # 2. TIRE RELAXATION DYNAMICS
    # =========================================================================
    if enable_tire_relaxation:
        # Tire relaxation: dF/dt = (F_ss - F) / τ
        # Fastest mode: τ_min = σ / v
        # CRITICAL: Use maximum expected speed to be safe for entire simulation
        v_safe = max(v, 25.0) 
        sigma_min = 0.35  # Smallest relaxation length (trailer lateral)
        tau_relax = sigma_min / v_safe
        omega_relax = 1.0 / tau_relax  # Frequency of relaxation mode
    else:
        omega_relax = 0.0
    
    # =========================================================================
    # 3. ARTICULATION DYNAMICS (if trailers present)
    # =========================================================================
    if trailers and len(trailers) > 0:
        # Each articulation has stiffness k_psi and inertia Iz_trailer
        # Natural frequency: ω = sqrt(k_psi / Iz_trailer)
        omega_articulation = 0.0
        for trailer in trailers:
            # Rotational stiffness
            if hasattr(trailer, 'k_psi') and hasattr(trailer, 'Iz'):
                omega_art_rot = np.sqrt(trailer.k_psi / trailer.Iz)
                omega_articulation = max(omega_articulation, omega_art_rot)
            
            # Longitudinal Hitch Stiffness (New)
            # ω = sqrt(k_hitch / m_trailer)
            # Using m_trailer is conservative approximation for reduced mass
            if hasattr(trailer, 'k_hitch'):
                # Use reduced mass approximation if possible (assume leader similar mass)
                # m_reduced = m1*m2/(m1+m2) ~ m/2
                # But to be safe and simple, use m_trailer/2 (approx reduced mass against heavy tractor)
                m_eff = trailer.mass / 2.0 
                omega_art_long = np.sqrt(trailer.k_hitch / m_eff)
                omega_articulation = max(omega_articulation, omega_art_long)
    else:
        omega_articulation = 0.0
    
    # =========================================================================
    # 4. SUSPENSION DYNAMICS (Roll & Pitch)
    # =========================================================================
    # Roll: ω_roll = sqrt(K_phi / Ixx)
    if hasattr(tractor, 'K_phi') and hasattr(tractor, 'Ixx'):
        omega_roll = np.sqrt(tractor.K_phi / tractor.Ixx)
    else:
        omega_roll = 0.0
    
    # Pitch: ω_pitch = sqrt(K_theta / Iyy)
    if hasattr(tractor, 'K_theta') and hasattr(tractor, 'Iyy'):
        omega_pitch = np.sqrt(tractor.K_theta / tractor.Iyy)
    else:
        omega_pitch = 0.0
    
    # =========================================================================
    # 5. FIND MAXIMUM FREQUENCY (Fastest dynamics)
    # =========================================================================
    all_frequencies = [
        omega_lateral,
        omega_relax,
        omega_articulation,
        omega_roll,
        omega_pitch
    ]
    
    omega_max = max(all_frequencies)
    
    # =========================================================================
    # 6. COMPUTE OPTIMAL TIMESTEP
    # =========================================================================
    # Nyquist criterion: f_sample > 2 * f_max
    # For accuracy: f_sample = samples_per_period * f_max
    # Therefore: dt < T_max / samples_per_period
    #            dt < (2π / ω_max) / samples_per_period
    
    if omega_max > 0:
        T_min = 2 * np.pi / omega_max  # Period of fastest mode
        dt_nyquist = T_min / samples_per_period
        dt_optimal = dt_nyquist * safety_factor  # Add safety margin
    else:
        # No dynamics found, use conservative default
        dt_optimal = 0.001  # 1 kHz default
    
    # =========================================================================
    # 7. APPLY PRACTICAL LIMITS
    # =========================================================================
    dt_min = 1e-5   # 100 kHz max (prevents too-small timesteps)
    dt_max = 0.002  # 500 Hz min (Critical for segregated solver stability)
    # 0.01s (100Hz) is too slow for stiff hitch coupling (Explicit Ping-Pong instability)
    
    dt_optimal = np.clip(dt_optimal, dt_min, dt_max)
    
    # =========================================================================
    # 8. CONTROLLER TIMESTEP CONSTRAINT
    # =========================================================================
    # Physics dt must be <= controller dt (can't integrate faster than controls update)
    # Ideally, dt_phys should be a submultiple of dt_controller for clean synchronization
    
    if dt_optimal > dt_controller:
        # Physics dt larger than controller → force it to be equal or submultiple
        dt_optimal = dt_controller
        constraint_msg = f"Limited by dt_controller ({dt_controller*1000:.2f} ms)"
    else:
        # Find nearest submultiple: dt_controller / N where N is integer
        # This ensures controller updates happen at exact physics timesteps
        N = max(1, int(np.ceil(dt_controller / dt_optimal)))
        dt_submultiple = dt_controller / N
        
        # Only adjust if the submultiple is close to optimal (within 20%)
        if abs(dt_submultiple - dt_optimal) / dt_optimal < 0.2:
            dt_optimal = dt_submultiple
            constraint_msg = f"Adjusted to {N}× submultiple of dt_controller"
        else:
            constraint_msg = f"Physics dt is {dt_controller/dt_optimal:.1f}× faster than controller"
    
    # =========================================================================
    # 9. DIAGNOSTICS
    # =========================================================================
    diagnostics = {
        'omega_max': omega_max,
        'omega_lateral': omega_lateral,
        'omega_relax': omega_relax,
        'omega_articulation': omega_articulation,
        'omega_roll': omega_roll,
        'omega_pitch': omega_pitch,
        'eigenvalues_lateral': eig_lateral,
        'frequency_hz': omega_max / (2 * np.pi),
        'period_min': 2 * np.pi / omega_max if omega_max > 0 else np.inf,
        'samples_per_period': samples_per_period,
        'safety_factor': safety_factor,
        'dt_nyquist': dt_nyquist if omega_max > 0 else np.inf,
        'dt_optimal': dt_optimal,
        'dt_controller': dt_controller,
        'controller_constraint': constraint_msg,
        'physics_per_control': int(dt_controller / dt_optimal),
        'speedup_vs_0.0001': 0.0001 / dt_optimal
    }
    
    return dt_optimal, diagnostics


def print_dt_analysis(dt_optimal, diagnostics):
    """
    Pretty-print the timestep analysis results.
    """
    print("\n" + "="*80)
    print(" AUTOMATIC TIMESTEP CALCULATION (Eigenvalue-Based)")
    print("="*80)
    
    print("\n🔍 SYSTEM DYNAMICS ANALYSIS:")
    print(f"  Lateral dynamics:      ω = {diagnostics['omega_lateral']:.2f} rad/s  ({diagnostics['omega_lateral']/(2*np.pi):.2f} Hz)")
    if diagnostics['omega_relax'] > 0:
        print(f"  Tire relaxation:       ω = {diagnostics['omega_relax']:.2f} rad/s  ({diagnostics['omega_relax']/(2*np.pi):.2f} Hz)")
    if diagnostics['omega_articulation'] > 0:
        print(f"  Articulation:          ω = {diagnostics['omega_articulation']:.2f} rad/s  ({diagnostics['omega_articulation']/(2*np.pi):.2f} Hz)")
    if diagnostics['omega_roll'] > 0:
        print(f"  Roll dynamics:         ω = {diagnostics['omega_roll']:.2f} rad/s  ({diagnostics['omega_roll']/(2*np.pi):.2f} Hz)")
    if diagnostics['omega_pitch'] > 0:
        print(f"  Pitch dynamics:        ω = {diagnostics['omega_pitch']:.2f} rad/s  ({diagnostics['omega_pitch']/(2*np.pi):.2f} Hz)")
    
    print(f"\n⚡ CRITICAL FREQUENCY:")
    print(f"  Maximum:               ω_max = {diagnostics['omega_max']:.2f} rad/s")
    print(f"  Frequency:             f_max = {diagnostics['frequency_hz']:.2f} Hz")
    print(f"  Min period:            T_min = {diagnostics['period_min']*1000:.2f} ms")
    
    print(f"\n📊 TIMESTEP CALCULATION:")
    print(f"  Samples per period:    {diagnostics['samples_per_period']}")
    print(f"  Safety factor:         {diagnostics['safety_factor']}")
    print(f"  Nyquist limit:         dt < {diagnostics['dt_nyquist']*1000:.4f} ms")
    print(f"  → OPTIMAL dt:          dt = {dt_optimal*1000:.4f} ms  ({1/dt_optimal:.0f} Hz)")
    
    print(f"\n🎮 CONTROLLER SYNCHRONIZATION:")
    print(f"  Controller timestep:   dt_ctrl = {diagnostics['dt_controller']*1000:.2f} ms  ({1/diagnostics['dt_controller']:.0f} Hz)")
    print(f"  Physics per control:   {diagnostics['physics_per_control']} physics steps per control update")
    print(f"  Constraint status:     {diagnostics['controller_constraint']}")
    
    print(f"\n🚀 PERFORMANCE GAIN:")
    print(f"  vs dt=0.0001s:         {diagnostics['speedup_vs_0.0001']:.1f}× FASTER")
    print(f"  Simulation time saved: {(1 - 1/diagnostics['speedup_vs_0.0001'])*100:.1f}%")
    
    print("="*80 + "\n")


# Example usage (call at start of simulation):
if __name__ == "__main__":
    # This would be called from main.py after tractor initialization:
    # dt_optimal, diag = compute_optimal_dt(tractor, trailers=[trailer1, dolly, trailer2],
    #                                       rho_inf=RHO_INF, 
    #                                       enable_tire_relaxation=ENABLE_TIRE_RELAXATION)
    # print_dt_analysis(dt_optimal, diag)
    # dt_phys = dt_optimal  # Use calculated value
    pass


def load_path(filename):
    try:
        with open(filename) as f:
            waypoints = np.array(list(csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)), dtype=float)
    except:
        # Read Excel file and convert to numpy array (same structure as CSV)
        df = pd.read_excel(filename)
        waypoints = df.values.astype(float)  # Convert DataFrame to numpy array
    # handle empty or single-point files
    if waypoints.size == 0 or waypoints.shape[0] < 2:
        return waypoints

    # Add flat segment
    vec0 = waypoints[1] - waypoints[0]
    yaw0 = np.arctan2(vec0[1], vec0[0])
    pitch0 = np.arctan2(vec0[2], np.hypot(vec0[0], vec0[1]))
    segment_length = 50
    step_size = np.linalg.norm(vec0)
    if step_size <= 1e-8:
        n_steps = 0
    else:
        n_steps = int(segment_length / step_size)
    if n_steps > 0:
        flat_segment = [[
            waypoints[0, 0] - i * np.cos(yaw0) * step_size,
            waypoints[0, 1] - i * np.sin(yaw0) * step_size,
            waypoints[0, 2]
        ] for i in range(n_steps, 0, -1)]
        waypoints = np.vstack((flat_segment, waypoints))

    return waypoints


def get_terrain_elevation(x, y, waypoints):
    dists = np.linalg.norm(waypoints[:, :2] - np.array([x, y]), axis=1)
    idx = np.argmin(dists)
    return waypoints[idx][2]

def get_terrain_pitch(x, y, waypoints):
    dists = np.linalg.norm(waypoints[:, :2] - np.array([x, y]), axis=1)
    idx = np.argmin(dists)
    if idx == 0:
        vec = waypoints[1] - waypoints[0]
    else:
        vec = waypoints[idx] - waypoints[idx - 1]
    dz = vec[2]
    dx = np.linalg.norm(vec[:2])
    return np.arctan2(dz, dx)

# -----------------------------
# Tyre, aero and rolling helpers
# -----------------------------

def tyre_lateral_force(C_alpha, alpha, camber=0.0, C_gamma=30000.0, toe=0.0):
    """
    Simple lateral tyre model with linear cornering stiffness + camber thrust + toe offset.
    - C_alpha : cornering stiffness [N/rad] (per axle or per tyre as used)
    - alpha   : slip angle [rad] (already accounting for steer & toe)
    - camber  : camber angle [rad]
    - C_gamma : camber force coefficient [N/rad]
    - toe     : toe angle [rad] (used as an offset into alpha if needed)
    Returns lateral force in N.
    """
    # include toe as a small offset to alpha (toe positive reduces effective alpha)
    alpha_eff = alpha - toe
    # linear lateral force (safe, stable)
    Fy_lin = -C_alpha * alpha_eff
    # camber thrust (approx)
    F_camber = C_gamma * camber
    # total Fy (clamp if desired later)
    return Fy_lin + F_camber

# ============================================================================
# TIMESTEP STABILITY CHECK FUNCTIONS
# ============================================================================

def check_timestep_stability(dt, vehicle_states, solver_type='implicit'):
    """
    Check if timestep is stable for the current vehicle state.
    
    Args:
        dt: Current timestep [s]
        vehicle_states: dict with 'vx', 'vy', 'omega', 'L' (wheelbase)
        solver_type: 'explicit' or 'implicit'
    
    Returns:
        (is_stable, recommended_dt, warning_message)
    """
    vx = vehicle_states.get('vx', 0.0)
    vy = vehicle_states.get('vy', 0.0)
    omega = vehicle_states.get('omega', 0.0)
    L = vehicle_states.get('L', 3.7)  # wheelbase
    
    # Maximum velocities and rates
    v_max = np.sqrt(vx**2 + vy**2) + 1e-6  # Total velocity
    omega_max = abs(omega) + 1e-6
    
    # Estimate characteristic frequencies
    # 1. Spatial frequency: v/L
    freq_spatial = v_max / L
    
    # 2. Rotational frequency
    freq_rotation = omega_max
    
    # 3. Lateral dynamics frequency (slip angle rate)
    # For truck: ω_lat ≈ sqrt(C_alpha * L / (m * v * Iz))
    # Rough estimate: 2-10 rad/s for trucks at speed
    freq_lateral = 5.0  # Conservative estimate [rad/s]
    
    # Maximum characteristic frequency
    freq_max = max(freq_spatial, freq_rotation, freq_lateral)
    
    # Stability criteria
    if solver_type == 'explicit':
        # Explicit: dt < 2/λ_max ≈ 2/(2π*f_max) = 1/(π*f_max)
        dt_stable = 0.3 / (np.pi * freq_max)  # Safety factor 0.3
        factor = 'CFL'
    else:  # implicit
        # Implicit: more relaxed, but still need convergence
        dt_stable = 2.0 / (np.pi * freq_max)  # Safety factor 2.0
        factor = 'convergence'
    
    is_stable = dt <= dt_stable
    
    if not is_stable:
        warning = (
            f"⚠️  TIMESTEP TOO LARGE for {solver_type} solver!\n"
            f"   Current dt = {dt:.6f}s\n"
            f"   Recommended dt ≤ {dt_stable:.6f}s ({factor} criterion)\n"
            f"   Max frequency: {freq_max:.2f} rad/s\n"
            f"   State: vx={vx:.2f} m/s, omega={omega:.3f} rad/s"
        )
    else:
        warning = None
    
    return is_stable, dt_stable, warning


def estimate_spectral_radius_from_convergence(iterations, dt):
    """
    Estimate spectral radius from Newton-Raphson convergence behavior.
    
    If solver takes many iterations, dt is likely too large for the
    system's characteristic timescales.
    
    Returns warning if iterations suggest instability.
    """
    if iterations > 100:
        severity = "CRITICAL"
        recommended_dt = dt / 4.0
    elif iterations > 50:
        severity = "WARNING"
        recommended_dt = dt / 2.0
    elif iterations > 30:
        severity = "INFO"
        recommended_dt = dt / 1.5
    else:
        return None  # All good
    
    warning = (
        f"[{severity}] Solver convergence slow ({iterations} iterations)\n"
        f"   Current dt = {dt:.6f}s may be too large\n"
        f"   Recommended: reduce to dt ≤ {recommended_dt:.6f}s"
    )
    
    return warning

# ============================================================================
# GENERALIZED-ALPHA HELPER FUNCTIONS
# ============================================================================

def compute_genalpha_params(rho_inf):
    """
    Compute Generalized-α integration parameters from spectral radius at infinity.
    
    The Generalized-α method (Chung & Hulbert, 1993) is a 2nd-order accurate,
    unconditionally stable time integration scheme with controllable numerical
    damping for high-frequency modes.
    
    Reference: Chung, J., & Hulbert, G. M. (1993). "A Time Integration Algorithm
    for Structural Dynamics With Improved Numerical Dissipation: The Generalized-α
    Method", Journal of Applied Mechanics, 60(2), 371-375.
    
    Args:
        rho_inf: Spectral radius at infinity ∈ [0, 1]
                 - rho_inf = 1.0: No numerical damping (like trapezoidal rule)
                 - rho_inf = 0.5: Moderate damping (good for most applications)
                 - rho_inf = 0.0: Maximum damping (asymptotically like backward Euler)
    
    Returns:
        tuple: (alpha_m, alpha_f, beta, gamma)
            - alpha_m: Acceleration interpolation parameter
            - alpha_f: Force/state interpolation parameter  
            - beta: Newmark position update parameter
            - gamma: Newmark velocity update parameter
    
    The integration scheme is:
        State evaluation:  x_{n+1-αf} = (1-αf) x_{n+1} + αf x_n
        Accel evaluation:  a_{n+1-αm} = (1-αm) a_{n+1} + αm a_n
        Velocity update:   v_{n+1} = v_n + dt [(1-γ) a_n + γ a_{n+1}]
        Position update:   x_{n+1} = x_n + dt v_n + dt² [(0.5-β) a_n + β a_{n+1}]
    """
    # Clamp rho_inf to valid range
    rho_inf = np.clip(rho_inf, 0.0, 1.0)
    
    # Optimal parameters for 2nd-order accuracy and unconditional stability
    alpha_m = (2.0 * rho_inf - 1.0) / (rho_inf + 1.0)
    alpha_f = rho_inf / (rho_inf + 1.0)
    
    # Newmark parameters derived from alpha_m, alpha_f
    gamma = 0.5 - alpha_m + alpha_f
    beta = 0.25 * (1.0 - alpha_m + alpha_f)**2
    
    return alpha_m, alpha_f, beta, gamma


# ============================================================================

class MPCSteeringController:
    """
    Model Predictive Controller for vehicle steering using kinematic bicycle model.
    
    The controller predicts vehicle trajectory over N future timesteps and optimizes
    M control inputs (steering angles) to minimize:
        - Cross-track error from reference path
        - Heading error from reference path
        - Steering effort (smoothness)
        - Steering rate (avoid jerky movements)
    
    Vehicle Model (Kinematic Bicycle):
        x_dot = v * cos(yaw)
        y_dot = v * sin(yaw)
        yaw_dot = (v / L) * tan(delta)
    
    where:
        - (x, y): position in global frame [m]
        - yaw: heading angle [rad]
        - v: longitudinal velocity [m/s]
        - delta: steering angle [rad]
        - L: wheelbase [m]
    """
    
    def __init__(self, N=30, M=5, dt=0.02, wheelbase=3.7, 
                 max_steer=np.radians(30.0), max_steer_rate=np.radians(45.0),
                 weight_cte=10.0, weight_heading=5.0, 
                 weight_effort=0.1, weight_rate=0.5,
                 mass=8000.0, Iz=12000.0, lf=1.48, lr=2.22,
                 Cf=300000.0, Cr=320000.0):
        """
        Initialize MPC steering controller.
        
        Args:
            N: Prediction horizon (number of timesteps to predict) - Increased to 30 for better stability
            M: Control horizon (number of control moves to optimize)
            dt: Control timestep [s]
            wheelbase: Vehicle wheelbase [m]
            max_steer: Maximum steering angle [rad]
            max_steer_rate: Maximum steering rate [rad/s]
            weight_cte: Cross-track error weight in cost function
            weight_heading: Heading error weight in cost function
            weight_effort: Steering effort weight (penalize large angles)
            weight_rate: Steering rate weight (penalize rapid changes)
            mass: Vehicle mass [kg] (for dynamic bicycle model)
            Iz: Yaw moment of inertia [kg*m^2]
            lf: CG to front axle [m]
            lr: CG to rear axle [m]
            Cf: Front cornering stiffness [N/rad]
            Cr: Rear cornering stiffness [N/rad]
        """
        self.N = N
        self.M = M
        self.dt = dt
        self.L = wheelbase
        self.max_steer = max_steer
        self.max_steer_rate = max_steer_rate
        
        # Dynamic bicycle model parameters
        self.mass = mass
        self.Iz = Iz
        self.lf = lf
        self.lr = lr
        self.Cf = Cf
        self.Cr = Cr
        
        # Cost function weights
        self.w_cte = weight_cte
        self.w_heading = weight_heading
        self.w_effort = weight_effort
        self.w_rate = weight_rate
        
        # Previous solution for warm-starting
        self.u_prev = np.zeros(M)
        self.delta_prev = 0.0  # Last applied steering angle
        
        # Diagnostics
        self.solve_time = 0.0
        self.cost_history = []
        self.iterations = 0

        # Path caching
        self._cached_waypoints = None
        self._cached_s = None

    def compute_control(self, x, y, yaw, v, waypoints, vy=0.0, omega=0.0):
        """
        Compute optimal steering command using MPC.
        
        When coupled with EKF, receives filtered state estimates including
        lateral velocity and yaw rate for accurate dynamic prediction.
        
        Args:
            x, y: Current position [m]
            yaw: Current heading [rad]
            v: Current longitudinal velocity [m/s]
            waypoints: Reference path as (N_wp, 3) array [x, y, z]
            vy: Lateral velocity [m/s] (from EKF, 0 if no EKF)
            omega: Yaw rate [rad/s] (from EKF, 0 if no EKF)
        
        Returns:
            delta: Optimal steering angle [rad]
            predicted_trajectory: (N, 3) array of predicted [x, y, yaw]
        """
        import time
        t_start = time.time()
        
        # Update path cache if waypoints changed (or first run)
        if self._cached_waypoints is None or waypoints is not self._cached_waypoints:
            self._cached_waypoints = waypoints
            self._cached_s = self._compute_path_s(waypoints)

        # Current state: 6D for dynamic bicycle model
        x0 = np.array([x, y, yaw, v, vy, omega])
        
        # Get reference trajectory (N points ahead based on SPEED)
        x_ref = self._compute_reference_trajectory(x, y, yaw, v, waypoints, self._cached_s)
        
        # Initial guess: shift previous solution
        u0 = np.concatenate([self.u_prev[1:], [self.u_prev[-1]]])
        
        # Bounds: steering angle limits
        bounds = [(-self.max_steer, self.max_steer)] * self.M
        
        # Constraints: steering rate limits
        constraints = self._build_constraints()
        
        # TODO [OPTION B]: Incorporate EKF covariance into cost weights
        # If EKF reports high position uncertainty, reduce w_cte weight
        # to prevent the MPC from chasing noisy position estimates:
        # if self.ekf is not None:
        #     bounds_info = self.ekf.get_robust_bounds()
        #     dynamic_w_cte = self.w_cte * bounds_info['speed_factor']
        
        # TODO [OPTION C]: Add chance constraints from EKF covariance
        # Propagate uncertainty through prediction horizon and add
        # safety margins to tracking constraints:
        # if self.ekf is not None:
        #     P_sequence = self.ekf.propagate_uncertainty(u0, self.N)
        #     margins = [self.ekf.compute_chance_constraint_margin(P_k)
        #                for P_k in P_sequence]
        #     # Add margins to constraint bounds...
        
        # Solve optimization problem
        result = minimize(
            fun=self._cost_function,
            x0=u0,
            args=(x0, x_ref),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 50, 'ftol': 1e-6}
        )
        
        # Extract optimal control sequence
        u_opt = result.x
        self.u_prev = u_opt.copy()
        
        # First control input is applied
        delta = u_opt[0]
        self.delta_prev = delta
        
        # Predict trajectory with optimal control
        predicted_trajectory = self._predict_trajectory(x0, u_opt)
        
        # Diagnostics
        self.solve_time = time.time() - t_start
        self.cost_history.append(result.fun)
        self.iterations = result.nit
        
        return delta, predicted_trajectory

    def _compute_path_s(self, waypoints):
        """Precompute arc length s for the path."""
        # Calculate distances between consecutive points
        diffs = np.diff(waypoints[:, :2], axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        
        # Cumulative sum to get s at each waypoint
        s = np.zeros(len(waypoints))
        s[1:] = np.cumsum(dists)
        return s

    def _compute_reference_trajectory(self, x, y, yaw, v, waypoints, s_path):
        """
        Extract reference trajectory based on CURRENT SPEED and DISTANCE.
        
        CRITICAL FIX: Use distance-based interpolation instead of index-based.
        Former method (idx+k) failed with dense waypoints (delta_s=0.1m) because
        MPC horizon (v*dt*N) corresponds to much further distance than N indices.
        
        Args:
            x, y: Current position [m]
            yaw: Current heading [rad]
            v: Current velocity [m/s]
            waypoints: (N_wp, 3) array
            s_path: (N_wp,) array of cumulative distance
            
        Returns:
            x_ref: (N, 3) array of reference [x, y, yaw]
        """
        # 1. Find current position on path (closest point)
        distances_sq = (waypoints[:, 0] - x)**2 + (waypoints[:, 1] - y)**2
        nearest_idx = np.argmin(distances_sq)
        
        # Distance of closest point along path
        current_s = s_path[nearest_idx]
        
        # 2. Generate target distances
        # We need N reference points corresponding to t = dt, 2*dt, ..., N*dt
        # Distance to travel: d = v * t
        # If v is small (stopped), lookahead should be minimum distance to define heading
        v_ref = max(v, 2.0)  # Min lookahead speed 2 m/s
        
        t_predict = np.arange(1, self.N + 1) * self.dt
        target_s = current_s + v_ref * t_predict
        
        # Handle wrap-around/end-of-path
        max_s = s_path[-1]
        target_s_clamped = np.minimum(target_s, max_s)
        
        # 3. Interpolate reference states
        x_ref = np.zeros((self.N, 3))
        
        # Interpolate x, y
        x_ref[:, 0] = np.interp(target_s_clamped, s_path, waypoints[:, 0])
        x_ref[:, 1] = np.interp(target_s_clamped, s_path, waypoints[:, 1])
        
        # Compute reference heading from path tangent
        # Vectorized heading calculation:
        # Find indices where target_s falls
        # searchsorted returns index such that s_path[i-1] <= target < s_path[i]
        indices = np.searchsorted(s_path, target_s_clamped) - 1
        indices = np.clip(indices, 0, len(waypoints) - 2)
        
        for k in range(self.N):
            idx = indices[k]
            # Segment vector
            dx = waypoints[idx+1, 0] - waypoints[idx, 0]
            dy = waypoints[idx+1, 1] - waypoints[idx, 1]
            x_ref[k, 2] = np.arctan2(dy, dx)
            
        return x_ref
    
    def _predict_trajectory(self, x0, u_sequence):
        """
        Forward integrate DYNAMIC bicycle model with given control sequence.
        
        Uses the full dynamic bicycle model (lateral force balance) instead
        of the simplified kinematic model. This provides more accurate
        predictions at higher speeds and during aggressive maneuvers.
        
        The dynamic model accounts for:
        - Lateral tire forces (linear cornering stiffness)
        - Centripetal coupling between vx and vy
        - Yaw moment balance from front/rear tire forces
        
        Args:
            x0: Initial state [x, y, yaw, vx, vy, omega]
            u_sequence: Control inputs [delta_0, ..., delta_M-1]
        
        Returns:
            trajectory: (N, 3) array of [x, y, yaw] predictions
        """
        trajectory = np.zeros((self.N, 3))
        
        # Unpack initial 6D state
        x, y, yaw, vx, vy, omega = x0
        
        for k in range(self.N):
            # Determine control input (hold last value after M steps)
            if k < self.M:
                delta = u_sequence[k]
            else:
                delta = u_sequence[-1]
            
            # Clamp steering for numerical stability
            delta = np.clip(delta, -self.max_steer, self.max_steer)
            
            # Store current state (position + heading for tracking)
            trajectory[k] = [x, y, yaw]
            
            # Dynamic bicycle model (Euler forward integration)
            # Protect against division by zero at low speed
            vx_safe = max(abs(vx), 1.0)  # Minimum 1 m/s for stability
            
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            
            # Position kinematics (global frame)
            x_dot = vx * cos_yaw - vy * sin_yaw
            y_dot = vx * sin_yaw + vy * cos_yaw
            yaw_dot = omega
            
            # Velocity dynamics (body frame)
            # Centripetal coupling
            vx_dot = omega * vy
            
            # Lateral force balance: vy_dot from linear tire model
            vy_dot = (-(self.Cf + self.Cr) / (self.mass * vx_safe) * vy
                      - (vx_safe + (self.Cf * self.lf - self.Cr * self.lr) / (self.mass * vx_safe)) * omega
                      + self.Cf / self.mass * delta)
            
            # Yaw moment balance
            omega_dot = (-(self.Cf * self.lf - self.Cr * self.lr) / (self.Iz * vx_safe) * vy
                         - (self.Cf * self.lf**2 + self.Cr * self.lr**2) / (self.Iz * vx_safe) * omega
                         + self.Cf * self.lf / self.Iz * delta)
            
            # Update state (Euler integration)
            x += x_dot * self.dt
            y += y_dot * self.dt
            yaw += yaw_dot * self.dt
            vx += vx_dot * self.dt
            vy += vy_dot * self.dt
            omega += omega_dot * self.dt
            
            # Normalize yaw to [-pi, pi]
            yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
        
        return trajectory
    

    
    def _cost_function(self, u_sequence, x0, x_ref):
        """
        Compute cost function for MPC optimization.
        
        Cost = sum over horizon of:
            w_cte * (cross_track_error)^2
            + w_heading * (heading_error)^2
            + w_effort * (steering)^2
            + w_rate * (steering_rate)^2
        
        Args:
            u_sequence: Control inputs [delta_0, ..., delta_M-1]
            x0: Initial state [x, y, yaw, v]
            x_ref: Reference trajectory (N, 3) [x_ref, y_ref, yaw_ref]
        
        Returns:
            cost: Scalar cost value
        """
        # Predict trajectory
        traj = self._predict_trajectory(x0, u_sequence)
        
        cost = 0.0
        
        # Tracking error cost
        for k in range(self.N):
            x_pred, y_pred, yaw_pred = traj[k]
            x_r, y_r, yaw_r = x_ref[k]
            
            # Cross-track error (Euclidean distance to reference)
            cte = np.sqrt((x_pred - x_r)**2 + (y_pred - y_r)**2)
            
            # Heading error
            heading_error = yaw_pred - yaw_r
            # Normalize to [-pi, pi]
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
            
            # Accumulate cost
            cost += self.w_cte * cte**2
            cost += self.w_heading * heading_error**2
        
        # Control effort cost
        for k in range(self.M):
            delta = u_sequence[k]
            cost += self.w_effort * delta**2
        
        # Steering rate cost (smoothness)
        for k in range(self.M - 1):
            delta_rate = (u_sequence[k+1] - u_sequence[k]) / self.dt
            cost += self.w_rate * delta_rate**2
        
        # Also penalize difference from previous commanded steering
        delta_rate_init = (u_sequence[0] - self.delta_prev) / self.dt
        cost += self.w_rate * delta_rate_init**2
        
        return cost
    
    def _build_constraints(self):
        """
        Build steering rate constraints for optimizer.
        
        Returns:
            list of constraint dictionaries for scipy.optimize.minimize
        """
        constraints = []
        
        max_delta_per_step = self.max_steer_rate * self.dt
        
        # Constraint: |delta[k+1] - delta[k]| <= max_delta_per_step
        for k in range(self.M - 1):
            # Upper bound: delta[k+1] - delta[k] <= max_delta_per_step
            constraints.append({
                'type': 'ineq',
                'fun': lambda u, k=k: max_delta_per_step - (u[k+1] - u[k])
            })
            
            # Lower bound: delta[k+1] - delta[k] >= -max_delta_per_step
            constraints.append({
                'type': 'ineq',
                'fun': lambda u, k=k: max_delta_per_step + (u[k+1] - u[k])
            })
        
        # Constraint from previous steering to first control
        constraints.append({
            'type': 'ineq',
            'fun': lambda u: max_delta_per_step - (u[0] - self.delta_prev)
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda u: max_delta_per_step + (u[0] - self.delta_prev)
        })
        
        return constraints
    
    def get_diagnostics(self):
        """
        Get MPC diagnostics for debugging/tuning.
        
        Returns:
            dict with solve_time, iterations, recent_cost
        """
        return {
            'solve_time_ms': self.solve_time * 1000,
            'iterations': self.iterations,
            'recent_cost': self.cost_history[-1] if self.cost_history else 0.0,
            'avg_cost_10': np.mean(self.cost_history[-10:]) if len(self.cost_history) >= 10 else 0.0
        }


# ============================================================================
# EXTENDED KALMAN FILTER (EKF) FOR MPC STATE ESTIMATION
# ============================================================================

class VehicleEKF:
    """
    Extended Kalman Filter for vehicle state estimation using a dynamic bicycle model.
    
    Provides filtered state estimates to the MPC controller, replacing direct
    access to perfect simulation state. In a real vehicle, sensors (GPS, IMU,
    wheel encoders) are noisy and some states (lateral velocity) are not
    directly measurable.
    
    State vector (6D):
        x     = [x, y, yaw, vx, vy, omega]
        
        x, y   : Global position [m]         — from GPS
        yaw    : Heading angle [rad]          — from GPS/compass
        vx     : Longitudinal velocity [m/s]  — from wheel odometry
        vy     : Lateral velocity [m/s]       — NOT directly measured, estimated
        omega  : Yaw rate [rad/s]             — from IMU gyroscope
    
    Measurement vector (5D):
        z = [x_gps, y_gps, yaw_gps, vx_odom, omega_imu]
    
    Process model: Linearized dynamic bicycle model
        x_dot   = vx * cos(yaw) - vy * sin(yaw)
        y_dot   = vx * sin(yaw) + vy * cos(yaw)
        yaw_dot = omega
        vx_dot  = omega * vy  (centripetal coupling, no drive model)
        vy_dot  = -(Cf + Cr) / (m * vx) * vy
                  - (vx + (Cf*lf - Cr*lr) / (m * vx)) * omega
                  + Cf / m * delta
        omega_dot = -(Cf*lf - Cr*lr) / (Iz * vx) * vy
                    - (Cf*lf^2 + Cr*lr^2) / (Iz * vx) * omega
                    + Cf * lf / Iz * delta
    
    Coupling:
        - MPC provides delta (steering) → fed into EKF predict step
        - EKF provides x_hat → fed into MPC as initial state
    """
    
    def __init__(self, dt, L=3.7, lf=1.48, lr=2.22,
                 mass=8000.0, Iz=12000.0, Cf=300000.0, Cr=320000.0,
                 Q_diag=None, R_diag=None):
        """
        Initialize EKF for vehicle state estimation.
        
        Args:
            dt: Estimation timestep [s] (should match controller dt)
            L: Wheelbase [m]
            lf: CG to front axle [m]
            lr: CG to rear axle [m]
            mass: Vehicle mass [kg]
            Iz: Yaw moment of inertia [kg*m^2]
            Cf: Front cornering stiffness [N/rad]
            Cr: Rear cornering stiffness [N/rad]
            Q_diag: Process noise diagonal (6,) — if None, uses global params
            R_diag: Measurement noise diagonal (5,) — if None, uses global params
        """
        self.dt = dt
        self.L = L
        self.lf = lf
        self.lr = lr
        self.mass = mass
        self.Iz = Iz
        self.Cf = Cf
        self.Cr = Cr
        
        # State vector: [x, y, yaw, vx, vy, omega]
        self.n = 6
        self.x_hat = np.zeros(self.n)
        
        # State covariance matrix
        self.P = np.eye(self.n) * 1.0  # Initial uncertainty
        
        # Process noise covariance
        if Q_diag is not None:
            self.Q = np.diag(Q_diag)
        else:
            self.Q = np.diag([
                EKF_PROCESS_NOISE_POS,    # x
                EKF_PROCESS_NOISE_POS,    # y
                EKF_PROCESS_NOISE_YAW,    # yaw
                EKF_PROCESS_NOISE_VX,     # vx
                EKF_PROCESS_NOISE_VY,     # vy
                EKF_PROCESS_NOISE_OMEGA   # omega
            ])
        
        # Measurement noise covariance
        # Measurement: [x_gps, y_gps, yaw_gps, vx_odom, omega_imu]
        self.m = 5
        if R_diag is not None:
            self.R = np.diag(R_diag)
        else:
            self.R = np.diag([
                EKF_GPS_NOISE_XY**2,      # x variance
                EKF_GPS_NOISE_XY**2,      # y variance
                EKF_GPS_NOISE_YAW**2,     # yaw variance
                EKF_IMU_NOISE_VX**2,      # vx variance
                EKF_IMU_NOISE_OMEGA**2    # omega variance
            ])
        
        # Measurement matrix H (5x6): z = H @ x
        # Measures [x, y, yaw, vx, omega] — NOT vy (unobservable directly)
        self.H = np.zeros((self.m, self.n))
        self.H[0, 0] = 1.0  # x
        self.H[1, 1] = 1.0  # y
        self.H[2, 2] = 1.0  # yaw
        self.H[3, 3] = 1.0  # vx
        self.H[4, 5] = 1.0  # omega
        
        # Diagnostics
        self.innovation = np.zeros(self.m)
        self.innovation_covariance = np.eye(self.m)
        self._last_delta = 0.0
        self._initialized = False
        
    def initialize(self, x0):
        """
        Set initial state estimate.
        
        Args:
            x0: Initial state [x, y, yaw, vx, vy, omega]
        """
        self.x_hat = np.array(x0, dtype=float)
        self.P = np.eye(self.n) * 0.01  # Small initial uncertainty (close to truth)
        self._initialized = True
    
    def _f(self, x, delta):
        """
        Nonlinear process model: x_{k+1} = f(x_k, delta_k).
        
        Dynamic bicycle model (continuous-time, then discretized with Euler).
        
        Args:
            x: State [x, y, yaw, vx, vy, omega]
            delta: Steering angle [rad]
            
        Returns:
            x_next: Predicted next state (6,)
        """
        px, py, yaw, vx, vy, omega = x
        
        # Protect against division by zero at low speed
        vx_safe = max(abs(vx), 1.0)  # Minimum 1 m/s for linearization
        
        # Dynamic bicycle model derivatives
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # Position kinematics (global frame)
        x_dot = vx * cos_yaw - vy * sin_yaw
        y_dot = vx * sin_yaw + vy * cos_yaw
        yaw_dot = omega
        
        # Velocity dynamics (body frame)
        # vx_dot: centripetal coupling (no engine model in EKF)
        vx_dot = omega * vy
        
        # vy_dot: lateral force balance from linear tire model
        #   Fy_f = Cf * alpha_f,  alpha_f ≈ delta - (vy + lf*omega)/vx
        #   Fy_r = Cr * alpha_r,  alpha_r ≈ -(vy - lr*omega)/vx
        #   m * vy_dot = Fy_f + Fy_r - m * vx * omega
        vy_dot = (-(self.Cf + self.Cr) / (self.mass * vx_safe) * vy
                  - (vx_safe + (self.Cf * self.lf - self.Cr * self.lr) / (self.mass * vx_safe)) * omega
                  + self.Cf / self.mass * delta)
        
        # omega_dot: yaw moment balance
        #   Iz * omega_dot = lf * Fy_f - lr * Fy_r
        omega_dot = (-(self.Cf * self.lf - self.Cr * self.lr) / (self.Iz * vx_safe) * vy
                     - (self.Cf * self.lf**2 + self.Cr * self.lr**2) / (self.Iz * vx_safe) * omega
                     + self.Cf * self.lf / self.Iz * delta)
        
        # Euler discretization
        x_next = np.array([
            px + x_dot * self.dt,
            py + y_dot * self.dt,
            yaw + yaw_dot * self.dt,
            vx + vx_dot * self.dt,
            vy + vy_dot * self.dt,
            omega + omega_dot * self.dt
        ])
        
        # Normalize yaw to [-pi, pi]
        x_next[2] = np.arctan2(np.sin(x_next[2]), np.cos(x_next[2]))
        
        return x_next
    
    def _compute_F(self, x, delta):
        """
        Compute Jacobian of process model F = ∂f/∂x using central differences.
        
        More robust than analytic Jacobian for complex models.
        
        Args:
            x: Current state (6,)
            delta: Steering angle [rad]
            
        Returns:
            F: State transition Jacobian (6x6)
        """
        F = np.zeros((self.n, self.n))
        eps = 1e-6
        
        f0 = self._f(x, delta)
        
        for j in range(self.n):
            x_plus = x.copy()
            x_plus[j] += eps
            x_minus = x.copy()
            x_minus[j] -= eps
            
            f_plus = self._f(x_plus, delta)
            f_minus = self._f(x_minus, delta)
            
            F[:, j] = (f_plus - f_minus) / (2 * eps)
        
        return F
    
    def predict(self, delta):
        """
        EKF Prediction (Time Update) step.
        
        Propagates state estimate and covariance forward using the
        dynamic bicycle model with the MPC's last steering command.
        
        This is the MPC → EKF coupling: the control input delta from MPC
        is used to predict how the vehicle state evolves.
        
        Args:
            delta: Steering angle from MPC [rad]
        """
        self._last_delta = delta
        
        # State prediction: x̂_{k+1|k} = f(x̂_k, δ_k)
        self.x_hat = self._f(self.x_hat, delta)
        
        # Jacobian of process model
        F = self._compute_F(self.x_hat, delta)
        
        # Covariance prediction: P_{k+1|k} = F * P_k * F^T + Q
        self.P = F @ self.P @ F.T + self.Q
        
        # Ensure symmetry (numerical stability)
        self.P = 0.5 * (self.P + self.P.T)
    
    def update(self, z):
        """
        EKF Update (Measurement Correction) step.
        
        Corrects the predicted state using noisy sensor measurements.
        
        This is the Sensor → EKF coupling: GPS and IMU readings
        pull the estimate toward reality.
        
        Args:
            z: Measurement vector [x_gps, y_gps, yaw_gps, vx_odom, omega_imu]
        """
        # Innovation: y = z - H @ x̂
        z_pred = self.H @ self.x_hat
        
        # Handle yaw angle wrapping in innovation
        y = z - z_pred
        y[2] = np.arctan2(np.sin(y[2]), np.cos(y[2]))  # Wrap yaw innovation
        
        self.innovation = y
        
        # Innovation covariance: S = H @ P @ H^T + R
        S = self.H @ self.P @ self.H.T + self.R
        self.innovation_covariance = S
        
        # Kalman gain: K = P @ H^T @ S^{-1}
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Fallback: pseudo-inverse if S is singular
            K = self.P @ self.H.T @ np.linalg.pinv(S)
        
        # State correction: x̂ = x̂ + K @ y
        self.x_hat = self.x_hat + K @ y
        
        # Normalize yaw
        self.x_hat[2] = np.arctan2(np.sin(self.x_hat[2]), np.cos(self.x_hat[2]))
        
        # Covariance correction: P = (I - K @ H) @ P
        # Joseph form for numerical stability: P = (I-KH)P(I-KH)^T + K R K^T
        I_KH = np.eye(self.n) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        # Ensure symmetry
        self.P = 0.5 * (self.P + self.P.T)
    
    def add_noise(self, true_state):
        """
        Simulate noisy sensor measurements from true simulation state.
        
        In a real vehicle, sensors would provide these noisy readings.
        In simulation, we add synthetic noise to study EKF robustness.
        
        Args:
            true_state: dict or tuple with keys/indices for
                        (x, y, yaw, vx, vy, omega)
                        
        Returns:
            z: Noisy measurement vector (5,) = [x, y, yaw, vx, omega]
               Note: vy is NOT measured (key EKF estimation challenge)
        """
        if isinstance(true_state, dict):
            x_true = true_state['x']
            y_true = true_state['y']
            yaw_true = true_state['yaw']
            vx_true = true_state['vx']
            omega_true = true_state['omega']
        else:
            # Assume tuple/list: (x, y, yaw, vx, vy, omega)
            x_true, y_true, yaw_true, vx_true = true_state[0], true_state[1], true_state[2], true_state[3]
            omega_true = true_state[5] if len(true_state) > 5 else 0.0
        
        z = np.array([
            x_true   + np.random.randn() * EKF_GPS_NOISE_XY,
            y_true   + np.random.randn() * EKF_GPS_NOISE_XY,
            yaw_true + np.random.randn() * EKF_GPS_NOISE_YAW,
            vx_true  + np.random.randn() * EKF_IMU_NOISE_VX,
            omega_true + np.random.randn() * EKF_IMU_NOISE_OMEGA
        ])
        
        return z
    
    def get_state(self):
        """
        Return current filtered state estimate.
        
        This is the EKF → MPC coupling: the MPC uses this as its
        initial state for trajectory prediction.
        
        Returns:
            tuple: (x, y, yaw, vx, vy, omega)
        """
        return tuple(self.x_hat)
    
    def get_covariance(self):
        """
        Return current state covariance matrix.
        
        Returns:
            P: (6, 6) covariance matrix
        """
        return self.P.copy()
    
    def get_diagnostics(self):
        """
        Return EKF diagnostics for debugging and logging.
        
        Returns:
            dict with estimation metrics
        """
        P_diag = np.diag(self.P)
        return {
            'trace_P': np.trace(self.P),
            'std_x': np.sqrt(max(0, P_diag[0])),
            'std_y': np.sqrt(max(0, P_diag[1])),
            'std_yaw': np.sqrt(max(0, P_diag[2])),
            'std_vx': np.sqrt(max(0, P_diag[3])),
            'std_vy': np.sqrt(max(0, P_diag[4])),
            'std_omega': np.sqrt(max(0, P_diag[5])),
            'innovation_norm': np.linalg.norm(self.innovation),
            'NIS': float(self.innovation @ np.linalg.solve(
                self.innovation_covariance, self.innovation))
                if np.linalg.det(self.innovation_covariance) > 1e-12 else 0.0,
            'initialized': self._initialized
        }
    
    # =====================================================================
    # TODO [OPTION B]: Covariance-based MPC constraint tightening
    # =====================================================================
    # When implementing Option B, add a method here that returns
    # uncertainty-adjusted constraints for the MPC:
    #
    # def get_robust_bounds(self, confidence=0.95):
    #     """
    #     Compute widened tracking corridor based on state uncertainty.
    #     
    #     At high uncertainty, the MPC should be more conservative
    #     (wider cross-track tolerance, lower speed limit).
    #     
    #     Args:
    #         confidence: Confidence level (0.95 = 2σ bounds)
    #         
    #     Returns:
    #         dict with:
    #             'cte_margin': Additional cross-track error margin [m]
    #             'heading_margin': Additional heading error margin [rad]
    #             'speed_factor': Speed reduction factor [0-1]
    #     """
    #     from scipy.stats import chi2
    #     k = chi2.ppf(confidence, df=2)  # 2-DOF for position
    #     
    #     # Position uncertainty ellipse semi-axes
    #     pos_cov = self.P[:2, :2]
    #     eigvals = np.linalg.eigvalsh(pos_cov)
    #     cte_margin = np.sqrt(k * max(eigvals))
    #     
    #     # Heading uncertainty
    #     heading_margin = np.sqrt(k * self.P[2, 2])
    #     
    #     # Speed reduction when uncertain
    #     vel_uncertainty = np.sqrt(self.P[3, 3] + self.P[4, 4])
    #     speed_factor = 1.0 / (1.0 + 0.1 * vel_uncertainty)
    #     
    #     return {
    #         'cte_margin': cte_margin,
    #         'heading_margin': heading_margin,
    #         'speed_factor': np.clip(speed_factor, 0.5, 1.0)
    #     }
    
    # =====================================================================
    # TODO [OPTION C]: Full chance-constrained robust MPC interface
    # =====================================================================
    # When implementing Option C, add methods here that propagate the
    # covariance through the MPC prediction horizon for chance constraints:
    #
    # def propagate_uncertainty(self, u_sequence, N):
    #     """
    #     Propagate covariance through MPC prediction horizon.
    #     
    #     For each step k in [0, N]:
    #         P_{k+1} = F_k @ P_k @ F_k^T + Q
    #     
    #     This gives the predicted uncertainty at each future timestep,
    #     enabling chance constraints of the form:
    #         Pr(x_k ∈ X_safe) >= 1 - epsilon
    #     
    #     Args:
    #         u_sequence: MPC control sequence [delta_0, ..., delta_{M-1}]
    #         N: Prediction horizon
    #         
    #     Returns:
    #         P_sequence: List of (6,6) covariance matrices for each step
    #     """
    #     P_sequence = [self.P.copy()]
    #     x_pred = self.x_hat.copy()
    #     P_pred = self.P.copy()
    #     
    #     for k in range(N):
    #         delta = u_sequence[min(k, len(u_sequence)-1)]
    #         F = self._compute_F(x_pred, delta)
    #         x_pred = self._f(x_pred, delta)
    #         P_pred = F @ P_pred @ F.T + self.Q
    #         P_sequence.append(P_pred.copy())
    #     
    #     return P_sequence
    #
    # def compute_chance_constraint_margin(self, P_k, epsilon=0.05):
    #     """
    #     Compute position margin for chance constraint satisfaction.
    #     
    #     Pr(||pos_error|| <= margin) >= 1 - epsilon
    #     
    #     For 2D Gaussian: margin = sqrt(chi2_inv(1-epsilon, 2) * max_eigenvalue(P_pos))
    #     
    #     Args:
    #         P_k: Covariance at prediction step k (6x6)
    #         epsilon: Violation probability (default 5%)
    #         
    #     Returns:
    #         margin: Position safety margin [m]
    #     """
    #     from scipy.stats import chi2
    #     chi2_val = chi2.ppf(1 - epsilon, df=2)
    #     pos_cov = P_k[:2, :2]
    #     max_eig = np.max(np.linalg.eigvalsh(pos_cov))
    #     return np.sqrt(chi2_val * max_eig)


class PacejkaTire:
    """
    Complete Pacejka Magic Formula tire model with combined slip.
    
    The Magic Formula for pure slip:
        F = D * sin(C * arctan(B*x - E*(B*x - arctan(B*x))))
    
    Where:
        B = Stiffness factor
        C = Shape factor  
        D = Peak value (D = μ * Fz for normalized)
        E = Curvature factor
        x = slip (α for lateral, κ for longitudinal)
    """
    
    def __init__(self,
                 # Lateral (pure slip) coefficients
                 B_lat=12.0, C_lat=1.4, D_lat=1.0, E_lat=-0.1,
                 # Longitudinal (pure slip) coefficients  
                 B_lon=10.0, C_lon=1.3, D_lon=1.0, E_lon=-0.1,
                 # Combined slip weighting - lateral reduction due to kappa
                 r_BY=10.0, r_CY=1.0,
                 # Combined slip weighting - longitudinal reduction due to alpha
                 r_BX=12.0, r_CX=1.0,
                 # Camber thrust coefficient
                 C_gamma=35000.0,
                 # Load sensitivity exponent (cornering stiffness ~ Fz^n)
                 n_load=0.9,
                 # Reference load for stiffness scaling
                 Fz_nom=20000.0):
        """
        Initialize tire with Pacejka coefficients.
        
        Args:
            B_lat, C_lat, D_lat, E_lat: Lateral MF coefficients
            B_lon, C_lon, D_lon, E_lon: Longitudinal MF coefficients
            r_BY, r_CY: Combined slip - lateral weighting parameters
            r_BX, r_CX: Combined slip - longitudinal weighting parameters
            C_gamma: Camber thrust coefficient [N/rad]
            n_load: Load sensitivity exponent
            Fz_nom: Nominal vertical load [N] for coefficient scaling
        """
        # Lateral coefficients
        self.B_lat = B_lat
        self.C_lat = C_lat
        self.D_lat = D_lat
        self.E_lat = E_lat
        
        # Longitudinal coefficients
        self.B_lon = B_lon
        self.C_lon = C_lon
        self.D_lon = D_lon
        self.E_lon = E_lon
        
        # Combined slip weighting
        self.r_BY = r_BY
        self.r_CY = r_CY
        self.r_BX = r_BX
        self.r_CX = r_CX
        
        # Camber and load sensitivity
        self.C_gamma = C_gamma
        self.n_load = n_load
        self.Fz_nom = Fz_nom
    
    def _magic_formula(self, x, B, C, D, E):
        """
        Core Magic Formula calculation.
        
        F = D * sin(C * arctan(B*x - E*(B*x - arctan(B*x))))
        
        Args:
            x: Slip input (α or κ)
            B: Stiffness factor
            C: Shape factor
            D: Peak value
            E: Curvature factor
            
        Returns:
            Force (normalized by D if D=μ*Fz)
        """
        Bx = B * x
        return D * np.sin(C * np.arctan(Bx - E * (Bx - np.arctan(Bx))))
    
    def pure_lateral(self, alpha, Fz, gamma=0.0, mu=0.9):
        """
        Pure lateral slip force (κ = 0).
        
        Args:
            alpha: Slip angle [rad]
            Fz: Normal load [N]
            gamma: Camber angle [rad]
            mu: Friction coefficient
            
        Returns:
            Fy0: Pure lateral force [N]
        """
        Fz_eff = max(1.0, float(Fz))
        
        # Load-sensitive peak value
        D_y = mu * Fz_eff * self.D_lat * (Fz_eff / self.Fz_nom) ** (self.n_load - 1.0)
        
        # Pure lateral from Magic Formula
        Fy0 = self._magic_formula(alpha, self.B_lat, self.C_lat, D_y, self.E_lat)
        
        # Add camber thrust (linear with camber angle)
        Fy_camber = self.C_gamma * gamma * (Fz_eff / self.Fz_nom) ** self.n_load
        
        return Fy0 + Fy_camber
    
    def pure_longitudinal(self, kappa, Fz, mu=0.9):
        """
        Pure longitudinal slip force (α = 0).
        
        Args:
            kappa: Slip ratio [-]
            Fz: Normal load [N]
            mu: Friction coefficient
            
        Returns:
            Fx0: Pure longitudinal force [N]
        """
        Fz_eff = max(1.0, float(Fz))
        
        # Load-sensitive peak value
        D_x = mu * Fz_eff * self.D_lon * (Fz_eff / self.Fz_nom) ** (self.n_load - 1.0)
        
        # Pure longitudinal from Magic Formula
        Fx0 = self._magic_formula(kappa, self.B_lon, self.C_lon, D_x, self.E_lon)
        
        return Fx0
    
    def _weighting_Gyk(self, alpha, kappa, Fz):
        """
        Lateral force weighting due to longitudinal slip.
        
        Fy = Fy0 * Gyk(κ)
        
        When κ ≠ 0, some of the tire's friction capacity is used for
        longitudinal force, reducing available lateral force.
        
        Uses simplified cosine-based weighting:
            Gyk = cos(arctan(r_BY * κ))^r_CY
        """
        if abs(kappa) < 1e-6:
            return 1.0
        
        # Weighting function (reduces Fy when κ is large)
        Gyk = np.cos(np.arctan(self.r_BY * kappa)) ** self.r_CY
        return max(0.0, min(1.0, Gyk))
    
    def _weighting_Gxa(self, alpha, kappa, Fz):
        """
        Longitudinal force weighting due to lateral slip.
        
        Fx = Fx0 * Gxa(α)
        
        When α ≠ 0, some of the tire's friction capacity is used for
        lateral force, reducing available longitudinal force.
        
        Uses simplified cosine-based weighting:
            Gxa = cos(arctan(r_BX * α))^r_CX
        """
        if abs(alpha) < 1e-6:
            return 1.0
        
        # Weighting function (reduces Fx when α is large)
        Gxa = np.cos(np.arctan(self.r_BX * alpha)) ** self.r_CX
        return max(0.0, min(1.0, Gxa))
    
    def smooth_friction_ellipse(self, Fx, Fy, Fz, mu, sharpness=3.0):
        """
        Smooth friction ellipse saturation using tanh.
        
        Maintains C¹ continuity (continuous first derivatives) for 
        Newton-Raphson solver stability.
        
        Args:
            Fx: Longitudinal force [N]
            Fy: Lateral force [N]
            Fz: Normal load [N]
            mu: Friction coefficient
            sharpness: Saturation transition steepness
            
        Returns:
            (Fx_sat, Fy_sat): Saturated forces on friction ellipse
        """
        F_lim = mu * max(Fz, 1.0)
        F_total = np.sqrt(Fx**2 + Fy**2)
        
        if F_total < 1e-6:
            return Fx, Fy
        
        # Smooth saturation: scale factor transitions from 1.0 (inside) to F_lim/F_total (outside)
        ratio = F_total / F_lim
        
        if ratio <= 1.0:
            # Inside friction circle - no saturation needed
            return Fx, Fy
        
        # Smooth transition using tanh
        # scale approaches F_lim/F_total as ratio gets large
        overshoot = ratio - 1.0
        scale = 1.0 - (1.0 - 1.0/ratio) * np.tanh(sharpness * overshoot)
        scale = max(F_lim / F_total, min(1.0, scale))
        
        return Fx * scale, Fy * scale
    
    def combined_slip(self, alpha, kappa, Fz, gamma=0.0, mu=0.9):
        """
        Combined slip tire forces with friction ellipse saturation.
        
        This is the main interface for the tire model.
        
        Args:
            alpha: Slip angle [rad]
            kappa: Slip ratio [-] (κ = (ωr - vx)/|vx|)
            Fz: Normal load [N]
            gamma: Camber angle [rad]
            mu: Friction coefficient
            
        Returns:
            (Fx, Fy): Longitudinal and lateral forces [N]
        """
        # Clamp inputs to reasonable ranges
        alpha = np.clip(float(alpha), -np.pi/2, np.pi/2)
        kappa = np.clip(float(kappa), -1.0, 1.0)
        Fz = max(1.0, float(Fz))
        
        # Pure slip forces
        Fx0 = self.pure_longitudinal(kappa, Fz, mu)
        Fy0 = self.pure_lateral(alpha, Fz, gamma, mu)
        
        # Apply combined slip weighting
        Gxa = self._weighting_Gxa(alpha, kappa, Fz)
        Gyk = self._weighting_Gyk(alpha, kappa, Fz)
        
        Fx_combined = Fx0 * Gxa
        Fy_combined = Fy0 * Gyk
        
        # Apply friction ellipse constraint (smooth)
        Fx_sat, Fy_sat = self.smooth_friction_ellipse(Fx_combined, Fy_combined, Fz, mu)
        
        return float(Fx_sat), float(Fy_sat)
    
    def get_cornering_stiffness(self, Fz, mu=0.9):
        """
        Get effective cornering stiffness at small slip angles.
        
        Useful for stability analysis and linear approximations.
        C_alpha = dFy/dα at α=0
        
        For Magic Formula: C_alpha ≈ B * C * D (at small slip)
        """
        Fz_eff = max(1.0, float(Fz))
        D_y = mu * Fz_eff * self.D_lat * (Fz_eff / self.Fz_nom) ** (self.n_load - 1.0)
        return self.B_lat * self.C_lat * D_y
    
    def get_longitudinal_stiffness(self, Fz, mu=0.9):
        """
        Get effective longitudinal stiffness at small slip ratios.
        
        C_kappa = dFx/dκ at κ=0
        
        For Magic Formula: C_kappa ≈ B * C * D (at small slip)
        """
        Fz_eff = max(1.0, float(Fz))
        D_x = mu * Fz_eff * self.D_lon * (Fz_eff / self.Fz_nom) ** (self.n_load - 1.0)
        return self.B_lon * self.C_lon * D_x


# Default tire configurations for different vehicle types
def create_truck_tire(axle='front'):
    """
    Create tire with typical heavy truck parameters.
    
    Args:
        axle: 'front' or 'rear' (affects stiffness)
    """
    if axle == 'front':
        return PacejkaTire(
            B_lat=10.0, C_lat=1.3, D_lat=1.0, E_lat=-0.2,
            B_lon=8.0, C_lon=1.2, D_lon=1.0, E_lon=-0.1,
            r_BY=8.0, r_CY=1.0,
            r_BX=10.0, r_CX=1.0,
            C_gamma=35000.0,
            n_load=0.85,
            Fz_nom=25000.0
        )
    else:  # rear
        return PacejkaTire(
            B_lat=12.0, C_lat=1.4, D_lat=1.0, E_lat=-0.15,
            B_lon=10.0, C_lon=1.3, D_lon=1.0, E_lon=-0.1,
            r_BY=10.0, r_CY=1.0,
            r_BX=12.0, r_CX=1.0,
            C_gamma=25000.0,
            n_load=0.9,
            Fz_nom=35000.0
        )


def create_trailer_tire():
    """
    Create tire with typical trailer parameters.
    """
    return PacejkaTire(
        B_lat=11.0, C_lat=1.35, D_lat=1.0, E_lat=-0.15,
        B_lon=9.0, C_lon=1.25, D_lon=1.0, E_lon=-0.1,
        r_BY=9.0, r_CY=1.0,
        r_BX=11.0, r_CX=1.0,
        C_gamma=20000.0,
        n_load=0.88,
        Fz_nom=30000.0
    )


# ============================================================================
# TIRE RELAXATION LENGTH MODEL
# ============================================================================

class TireRelaxationDynamics:
    """
    First-order lag model for transient tire force development.
    
    Physical background:
    - Tire forces don't appear instantly when slip angle/ratio changes
    - Contact patch deforms gradually as tire rolls
    - Relaxation length (sigma) = distance to reach 63% of steady-state force
    - Typical values: 0.3-0.6m for truck tires
    
    Mathematical model:
        dF/ds = (F_steady_state - F_transient) / sigma
        
    where s = distance traveled. Converting to time domain:
        dF/dt = v * (F_ss - F_transient) / sigma
              = (F_ss - F_transient) / tau
        
    where tau = sigma / v (relaxation time constant)
    
    Stability:
    - Uses implicit Euler integration (unconditionally stable)
    - Clamps tau to prevent numerical issues at low speeds
    - Rate limiting prevents force spikes
    """
    
    def __init__(self, sigma_x, sigma_y):
        """
        Initialize relaxation dynamics.
        
        Args:
            sigma_x: Longitudinal relaxation length [m]
            sigma_y: Lateral relaxation length [m]
        """
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        
        # Stability limiters
        self.TAU_MIN = 0.001  # 1ms minimum time constant [s]
        self.TAU_MAX = 0.5    # 500ms maximum time constant [s]
        self.MAX_FORCE_RATE = 2000000.0  # Maximum force rate [N/s] (Increased for heavy truck dynamics)
        
        # Telemetry (for debugging)
        self.tau_x_last = 0.0
        self.tau_y_last = 0.0
    
    def compute_transient_forces(self, Fx_ss, Fy_ss, Fx_current, Fy_current, vx, dt):
        """
        Update transient tire forces using first-order lag with implicit integration.
        
        Uses implicit Euler for unconditional stability:
            F_new = (F_old + dt/tau * F_ss) / (1 + dt/tau)
        
        This is A-stable (stable for any dt > 0, regardless of tau).
        
        Args:
            Fx_ss, Fy_ss: Steady-state forces from Pacejka [N]
            Fx_current, Fy_current: Current transient forces [N]
            vx: Longitudinal velocity [m/s]
            dt: Timestep [s]
        
        Returns:
            Fx_new, Fy_new: Updated transient forces [N]
        """
        # Regularize velocity (avoid division by zero at standstill)
        v_safe = max(abs(vx), 0.1)
        
        # Compute time constants: tau = sigma / v
        tau_x = self.sigma_x / v_safe
        tau_y = self.sigma_y / v_safe
        
        # Clamp to stable range (prevents extreme values at low speed)
        tau_x = np.clip(tau_x, self.TAU_MIN, self.TAU_MAX)
        tau_y = np.clip(tau_y, self.TAU_MIN, self.TAU_MAX)
        
        # Store for telemetry
        self.tau_x_last = tau_x
        self.tau_y_last = tau_y
        
        # Implicit Euler integration
        # Rearranging: F_new * (1 + dt/tau) = F_old + dt/tau * F_ss
        alpha_x = dt / tau_x
        alpha_y = dt / tau_y
        
        Fx_new = (Fx_current + alpha_x * Fx_ss) / (1.0 + alpha_x)
        Fy_new = (Fy_current + alpha_y * Fy_ss) / (1.0 + alpha_y)
        
        # Rate limiting (prevents spikes from numerical errors or extreme inputs)
        dFx_dt = (Fx_new - Fx_current) / dt
        dFy_dt = (Fy_new - Fy_current) / dt
        
        if CAP_FORCES:
            if abs(dFx_dt) > self.MAX_FORCE_RATE:
                Fx_new = Fx_current + np.sign(dFx_dt) * self.MAX_FORCE_RATE * dt
                # Warn if rate limiting is active (indicates potential issues)
                if abs(dFx_dt) > 2.0 * self.MAX_FORCE_RATE:
                    print(f"[RELAX WARNING] Extreme Fx rate: {dFx_dt:.0f} N/s (limited to ±{self.MAX_FORCE_RATE:.0f})")
       

            if abs(dFy_dt) > self.MAX_FORCE_RATE:
                Fy_new = Fy_current + np.sign(dFy_dt) * self.MAX_FORCE_RATE * dt
                if abs(dFy_dt) > 2.0 * self.MAX_FORCE_RATE:
                    print(f"[RELAX WARNING] Extreme Fy rate: {dFy_dt:.0f} N/s (limited to ±{self.MAX_FORCE_RATE:.0f})")
        
        return Fx_new, Fy_new


# ============================================================================
# ABS (ANTI-LOCK BRAKING SYSTEM) CONTROLLER
# ============================================================================

class ABSController:
    """
    Per-wheel Anti-lock Braking System controller.
    
    Uses a 3-phase cycle based on slip ratio κ:
      - APPLY:   κ > release_threshold  → ramp up brake pressure
      - HOLD:    between thresholds     → maintain current pressure
      - RELEASE: κ < slip_threshold     → ramp down brake pressure
    
    Each wheel is independently controlled. The controller outputs
    a pressure_factor ∈ [0, 1] that scales the requested brake force.
    
    Physical background:
    - Peak braking force occurs at κ ≈ -0.10 to -0.15
    - Beyond this (κ → -1), tire transitions from rolling to sliding
    - Sliding friction < peak friction, so locked wheels = longer stopping
    - ABS keeps tires near the peak by modulating brake pressure
    """
    
    def __init__(self, slip_target=-0.12, slip_threshold=-0.15,
                 release_threshold=-0.08, pressure_increase_rate=25.0,
                 pressure_decrease_rate=40.0, min_speed=3.0):
        """
        Args:
            slip_target: Optimal braking κ (peak of μ-κ curve) [−]
            slip_threshold: κ below which pressure is released [−]
            release_threshold: κ above which pressure is reapplied [−]
            pressure_increase_rate: Rate to ramp up pressure factor [1/s]
            pressure_decrease_rate: Rate to ramp down pressure factor [1/s]
            min_speed: Disable ABS below this speed [m/s]
        """
        self.slip_target = slip_target
        self.slip_threshold = slip_threshold
        self.release_threshold = release_threshold
        self.pressure_increase_rate = pressure_increase_rate
        self.pressure_decrease_rate = pressure_decrease_rate
        self.min_speed = min_speed
        
        # Per-wheel state
        self.phase = {}            # 'apply', 'hold', or 'release'
        self.pressure_factor = {}  # 0.0 to 1.0 (scales brake force)
        
        for key in ['fl', 'fr', 'rl', 'rr']:
            self.phase[key] = 'apply'
            self.pressure_factor[key] = 1.0
        
        # Diagnostics
        self.abs_active = {k: False for k in ['fl', 'fr', 'rl', 'rr']}
        self._cycle_count = {k: 0 for k in ['fl', 'fr', 'rl', 'rr']}
    
    def modulate(self, brake_forces, wheel_kappas, vx, dt):
        """
        Modulate per-wheel brake forces to prevent wheel lockup.
        
        Args:
            brake_forces: dict {'fl','fr','rl','rr'} of requested brake forces [N]
            wheel_kappas: dict {'fl','fr','rl','rr'} of current slip ratios [-]
            vx: Vehicle longitudinal speed [m/s]
            dt: Timestep [s]
            
        Returns:
            modulated_forces: dict with ABS-limited brake forces [N]
            abs_active: dict of bool indicating ABS intervention per wheel
        """
        modulated = {}
        
        for key in ['fl', 'fr', 'rl', 'rr']:
            F_req = brake_forces.get(key, 0.0)
            kappa = wheel_kappas.get(key, 0.0)
            
            # No braking requested or speed too low → pass through
            if F_req < 1.0 or vx < self.min_speed:
                self.phase[key] = 'apply'
                self.pressure_factor[key] = 1.0
                self.abs_active[key] = False
                modulated[key] = F_req
                continue
            
            # ---- 3-phase slip ratio control ----
            # κ is negative during braking (wheel slower than ground)
            # More negative = closer to lockup
            
            if kappa < self.slip_threshold:
                # Wheel approaching lockup → RELEASE brake pressure
                if self.phase[key] != 'release':
                    self._cycle_count[key] += 1
                self.phase[key] = 'release'
                self.pressure_factor[key] -= self.pressure_decrease_rate * dt
                self.pressure_factor[key] = max(0.05, self.pressure_factor[key])
                self.abs_active[key] = True
                
            elif kappa > self.release_threshold:
                # Wheel recovered → APPLY brake pressure
                self.phase[key] = 'apply'
                self.pressure_factor[key] += self.pressure_increase_rate * dt
                self.pressure_factor[key] = min(1.0, self.pressure_factor[key])
                # ABS is still "active" if pressure_factor < 1 (still recovering)
                self.abs_active[key] = self.pressure_factor[key] < 0.99
                
            else:
                # In the optimal zone → HOLD current pressure
                self.phase[key] = 'hold'
                # Small correction toward target slip
                if kappa < self.slip_target:
                    # Slightly too much slip, gently reduce
                    self.pressure_factor[key] -= 0.3 * self.pressure_decrease_rate * dt
                    self.pressure_factor[key] = max(0.05, self.pressure_factor[key])
                else:
                    # Can accept slightly more brake
                    self.pressure_factor[key] += 0.3 * self.pressure_increase_rate * dt
                    self.pressure_factor[key] = min(1.0, self.pressure_factor[key])
                self.abs_active[key] = True
            
            modulated[key] = F_req * self.pressure_factor[key]
        
        return modulated, self.abs_active.copy()
    
    def reset(self):
        """Reset all wheels to full pressure (e.g., after brake release)."""
        for key in ['fl', 'fr', 'rl', 'rr']:
            self.phase[key] = 'apply'
            self.pressure_factor[key] = 1.0
            self.abs_active[key] = False
    
    def get_diagnostics(self):
        """Return diagnostic info for logging."""
        return {
            'phase': self.phase.copy(),
            'pressure_factor': self.pressure_factor.copy(),
            'abs_active': self.abs_active.copy(),
            'cycle_count': self._cycle_count.copy()
        }


# ============================================================================
# QUARTER-CAR SUSPENSION MODEL
# ============================================================================
class QuarterCarSuspension:
    """
    Per-corner 2-DOF vertical dynamics model.
    
    Two masses connected by spring-damper, tire modeled as vertical spring:
    
        [Sprung mass (m_s)]
             |
        [Spring (k_s) + Damper (c_s)]
             |
        [Unsprung mass (m_u)]
             |
        [Tire spring (k_t)]
             |
        ~~~ Road surface ~~~
    
    States: z_s, vz_s (sprung), z_u, vz_u (unsprung)
    All displacements relative to static equilibrium.
    """
    
    def __init__(self, m_s, m_u, k_s, c_s, k_t, static_load):
        """
        Args:
            m_s: Sprung mass at this corner [kg]
            m_u: Unsprung mass [kg] (wheel+brake+hub)
            k_s: Spring rate [N/m]
            c_s: Damper rate [Ns/m]
            k_t: Tire vertical stiffness [N/m]
            static_load: Static vertical load [N] (for equilibrium)
        """
        self.m_s = m_s
        self.m_u = m_u
        self.k_s = k_s
        self.c_s = c_s
        self.k_t = k_t
        self.static_load = static_load
        
        # States (displacements from static equilibrium)
        self.z_s = 0.0    # Sprung mass displacement [m]
        self.vz_s = 0.0   # Sprung mass velocity [m/s]
        self.z_u = 0.0    # Unsprung mass displacement [m]
        self.vz_u = 0.0   # Unsprung mass velocity [m/s]
        
        # Static deflection under gravity
        self.z_s_static = static_load / k_s
        self.z_u_static = static_load / k_t
    
    def update(self, z_road, F_inertial, dt):
        """
        Integrate one timestep.
        
        Args:
            z_road: Road surface displacement at this corner [m]
                    (from roll × py + pitch × px)
            F_inertial: Additional inertial force on sprung mass [N]
                        (from lateral/longitudinal acceleration load transfer)
            dt: Timestep [s]
        
        Returns:
            (suspension_deflection, tire_contact_force, bump_displacement)
        """
        # Spring-damper force between sprung and unsprung
        delta_z = self.z_s - self.z_u
        delta_v = self.vz_s - self.vz_u
        F_spring_damper = self.k_s * delta_z + self.c_s * delta_v
        
        # Tire contact force (tire spring between unsprung mass and road)
        tire_deflection = self.z_u - z_road
        F_tire = self.k_t * tire_deflection
        
        # Equations of motion
        # Sprung: m_s * az_s = -F_spring_damper + F_inertial
        az_s = (-F_spring_damper + F_inertial) / self.m_s
        
        # Unsprung: m_u * az_u = F_spring_damper - F_tire
        az_u = (F_spring_damper - F_tire) / self.m_u
        
        # Semi-implicit Euler integration (velocity first, then position)
        self.vz_s += az_s * dt
        self.vz_u += az_u * dt
        self.z_s += self.vz_s * dt
        self.z_u += self.vz_u * dt
        
        # Clamp to prevent numerical divergence
        self.z_s = np.clip(self.z_s, -0.15, 0.15)
        self.z_u = np.clip(self.z_u, -0.08, 0.08)
        self.vz_s = np.clip(self.vz_s, -2.0, 2.0)
        self.vz_u = np.clip(self.vz_u, -3.0, 3.0)
        
        # Outputs
        suspension_deflection = self.z_s - self.z_u  # positive = compression
        # Tire contact force: static load + dynamic component
        # Ensure tire doesn't leave ground (Fz >= 0)
        Fz_contact = self.static_load + F_tire
        Fz_contact = max(50.0, Fz_contact)  # Minimum load floor
        
        bump_displacement = self.z_u - z_road  # For alignment calculations
        
        return suspension_deflection, Fz_contact, bump_displacement
    
    def reset(self):
        """Reset to static equilibrium."""
        self.z_s = 0.0
        self.vz_s = 0.0
        self.z_u = 0.0
        self.vz_u = 0.0


# ============================================================================
# TRACTION CONTROL SYSTEM (TCS / ASR)
# ============================================================================
class TractionControlSystem:
    """
    Per-wheel traction control that prevents wheel spin during acceleration.
    Mirrors ABS logic but for positive slip ratio κ.
    
    Modulates a torque_factor ∈ [0.05, 1.0] per driven wheel.
    """
    
    def __init__(self, slip_target=0.10, slip_threshold=0.15,
                 release_threshold=0.08, torque_reduction_rate=30.0,
                 torque_increase_rate=20.0, min_speed=2.0):
        self.slip_target = slip_target
        self.slip_threshold = slip_threshold
        self.release_threshold = release_threshold
        self.torque_reduction_rate = torque_reduction_rate
        self.torque_increase_rate = torque_increase_rate
        self.min_speed = min_speed
        
        # Per-wheel state (only for driven wheels, but keep all 4 for generality)
        self.torque_factor = {k: 1.0 for k in ['fl', 'fr', 'rl', 'rr']}
        self.tcs_active = {k: False for k in ['fl', 'fr', 'rl', 'rr']}
        self.phase = {k: 'APPLY' for k in ['fl', 'fr', 'rl', 'rr']}
    
    def modulate(self, drive_forces, wheel_kappas, vx, dt, driven_wheels=None):
        """
        Modulate per-wheel drive forces based on slip ratio.
        
        Args:
            drive_forces: dict {wheel_key: force [N]}
            wheel_kappas: dict {wheel_key: κ [-]}
            vx: longitudinal velocity [m/s]
            dt: timestep [s]
            driven_wheels: list of driven wheel keys (default: ['rl', 'rr'])
        
        Returns:
            (modulated_forces, tcs_active_dict)
        """
        if driven_wheels is None:
            driven_wheels = ['rl', 'rr']
        
        modulated = dict(drive_forces)
        
        for key in driven_wheels:
            kappa = wheel_kappas.get(key, 0.0)
            
            # TCS inactive at low speed or if not spinning
            if abs(vx) < self.min_speed or kappa <= self.release_threshold:
                self.phase[key] = 'APPLY'
                self.torque_factor[key] = min(1.0, self.torque_factor[key] + self.torque_increase_rate * dt)
                self.tcs_active[key] = False
                continue
            
            self.tcs_active[key] = True
            
            if kappa > self.slip_threshold:
                # RELEASE phase: wheel spinning too much, cut torque
                self.phase[key] = 'RELEASE'
                self.torque_factor[key] -= self.torque_reduction_rate * dt
            elif kappa > self.release_threshold:
                # HOLD phase: fine-tune toward target
                self.phase[key] = 'HOLD'
                error = kappa - self.slip_target
                if error > 0:
                    self.torque_factor[key] -= 0.5 * self.torque_reduction_rate * dt
                else:
                    self.torque_factor[key] += 0.5 * self.torque_increase_rate * dt
            
            # Clamp
            self.torque_factor[key] = np.clip(self.torque_factor[key], 0.05, 1.0)
            modulated[key] = drive_forces.get(key, 0.0) * self.torque_factor[key]
        
        return modulated, self.tcs_active.copy()
    
    def reset(self):
        """Reset all wheels to full torque."""
        for k in self.torque_factor:
            self.torque_factor[k] = 1.0
            self.tcs_active[k] = False
            self.phase[k] = 'APPLY'
    
    def get_diagnostics(self):
        return {
            'torque_factor': self.torque_factor.copy(),
            'phase': self.phase.copy(),
            'tcs_active': self.tcs_active.copy()
        }


# ============================================================================
# ELECTRONIC STABILITY CONTROL (ESC)
# ============================================================================
class ESController:
    """
    Electronic Stability Control.
    
    Compares actual yaw rate to a reference linear bicycle model.
    Applies corrective asymmetric braking:
      - Oversteer (|ω| > |ω_ref|): brake outer front wheel
      - Understeer (|ω| < |ω_ref|): brake inner rear wheel
    """
    
    def __init__(self, wheelbase, understeer_gradient=0.0025,
                 yaw_rate_deadband=0.02, gain_understeer=0.3,
                 gain_oversteer=0.5, max_brake_force=5000.0,
                 min_speed=5.0):
        """
        Args:
            wheelbase: L = lf + lr [m]
            understeer_gradient: K_us [rad/(m/s²)] for reference model
            yaw_rate_deadband: minimum yaw rate error before intervention [rad/s]
            gain_understeer: braking force proportional gain for understeer
            gain_oversteer: braking force proportional gain for oversteer
            max_brake_force: maximum corrective brake force per wheel [N]
            min_speed: ESC inactive below this [m/s]
        """
        self.L = wheelbase
        self.K_us = understeer_gradient
        self.deadband = yaw_rate_deadband
        self.gain_understeer = gain_understeer
        self.gain_oversteer = gain_oversteer
        self.max_brake_force = max_brake_force
        self.min_speed = min_speed
        
        # Diagnostics
        self.omega_ref = 0.0
        self.omega_error = 0.0
        self.mode = 'INACTIVE'  # 'INACTIVE', 'OVERSTEER', 'UNDERSTEER'
        self.corrective_forces = {k: 0.0 for k in ['fl', 'fr', 'rl', 'rr']}
    
    def compute(self, vx, omega_actual, delta, dt):
        """
        Compute corrective brake forces.
        
        Args:
            vx: longitudinal velocity [m/s]
            omega_actual: actual yaw rate [rad/s]
            delta: average steering angle [rad]
            dt: timestep [s]
        
        Returns:
            dict of per-wheel corrective brake forces [N]
        """
        self.corrective_forces = {k: 0.0 for k in ['fl', 'fr', 'rl', 'rr']}
        
        # Inactive at low speed
        if abs(vx) < self.min_speed:
            self.mode = 'INACTIVE'
            self.omega_ref = 0.0
            self.omega_error = 0.0
            return self.corrective_forces.copy()
        
        # Reference yaw rate from linear bicycle model
        # ω_ref = (vx × δ) / (L + K_us × vx²)
        self.omega_ref = (vx * delta) / (self.L + self.K_us * vx * vx)
        
        # Yaw rate error
        self.omega_error = omega_actual - self.omega_ref
        
        if abs(self.omega_error) < self.deadband:
            self.mode = 'INACTIVE'
            return self.corrective_forces.copy()
        
        # Determine if turning left (positive delta/omega) or right
        turning_left = delta > 0 or omega_actual > 0
        
        if abs(omega_actual) > abs(self.omega_ref) + self.deadband:
            # OVERSTEER: vehicle rotating too much
            # Brake the OUTER FRONT wheel to create corrective yaw moment
            self.mode = 'OVERSTEER'
            F_correct = min(
                self.gain_oversteer * self.mass_for_scaling * abs(self.omega_error),
                self.max_brake_force
            ) if hasattr(self, 'mass_for_scaling') else min(
                abs(self.omega_error) * self.max_brake_force / 0.5,
                self.max_brake_force
            )
            
            if turning_left:
                # Turning left, oversteer → brake front RIGHT (outer)
                self.corrective_forces['fr'] = F_correct
            else:
                # Turning right, oversteer → brake front LEFT (outer)
                self.corrective_forces['fl'] = F_correct
                
        elif abs(omega_actual) < abs(self.omega_ref) - self.deadband and abs(delta) > 0.01:
            # UNDERSTEER: vehicle not rotating enough despite steering input
            # Brake the INNER REAR wheel to tighten the turn
            self.mode = 'UNDERSTEER'
            F_correct = min(
                self.gain_understeer * abs(self.omega_error) * self.max_brake_force / 0.3,
                self.max_brake_force
            )
            
            if turning_left:
                # Turning left, understeer → brake rear LEFT (inner)
                self.corrective_forces['rl'] = F_correct
            else:
                # Turning right, understeer → brake rear RIGHT (inner)
                self.corrective_forces['rr'] = F_correct
        else:
            self.mode = 'INACTIVE'
        
        return self.corrective_forces.copy()
    
    def get_diagnostics(self):
        return {
            'mode': self.mode,
            'omega_ref': self.omega_ref,
            'omega_error': self.omega_error,
            'corrective_forces': self.corrective_forces.copy()
        }


# ============================================================================
# ROAD FRICTION MAP
# ============================================================================

class RoadFrictionMap:
    """
    Position-dependent friction coefficient.
    Supports circular wet/ice/gravel patches and split-μ roads.
    """
    def __init__(self, default_mu=DEFAULT_MU):
        self.default_mu = default_mu
        self.patches = []        # List of (x, y, radius, mu, water_depth)
        self.split_mu_zones = [] # List of (x_start, x_end, y_split, mu_left, mu_right)
    
    def add_patch(self, x, y, radius, mu, water_depth=0.0):
        """Add a circular friction zone (wet, ice, gravel)."""
        self.patches.append({'x': x, 'y': y, 'radius': radius, 
                             'mu': mu, 'water_depth': water_depth})
    
    def add_split_mu(self, x_start, x_end, y_split, mu_left, mu_right):
        """Split-μ road: different grip left vs right of y_split."""
        self.split_mu_zones.append({
            'x_start': x_start, 'x_end': x_end, 'y_split': y_split,
            'mu_left': mu_left, 'mu_right': mu_right
        })
    
    def get_mu(self, x, y):
        """Return friction coefficient at world position (x, y)."""
        # Check split-μ zones first
        for zone in self.split_mu_zones:
            if zone['x_start'] <= x <= zone['x_end']:
                if y >= zone['y_split']:
                    return zone['mu_left']
                else:
                    return zone['mu_right']
        
        # Check circular patches
        for p in self.patches:
            dx = x - p['x']
            dy = y - p['y']
            if dx * dx + dy * dy < p['radius'] * p['radius']:
                return p['mu']
        
        return self.default_mu
    
    def get_water_depth(self, x, y):
        """Return water depth in mm at position (x, y). 0 = dry."""
        for p in self.patches:
            dx = x - p['x']
            dy = y - p['y']
            if dx * dx + dy * dy < p['radius'] * p['radius']:
                return p.get('water_depth', 0.0)
        return 0.0


# ============================================================================
# HYDROPLANING MODEL
# ============================================================================

class HydroplaningModel:
    """
    Speed-dependent μ reduction on wet surfaces.
    Based on NASA hydroplaning formula: v_hydro = 6.36 × √(tire_pressure_kPa) [km/h]
    """
    def __init__(self, tire_pressure_kPa=HYDRO_TIRE_PRESSURE, 
                 tire_width_mm=HYDRO_TIRE_WIDTH):
        self.tire_pressure = tire_pressure_kPa
        self.tire_width = tire_width_mm
        # Critical hydroplaning speed [m/s]
        self.v_hydro = 6.36 * math.sqrt(tire_pressure_kPa) / 3.6
    
    def compute_mu_factor(self, speed_ms, water_depth_mm):
        """
        Compute grip reduction factor ∈ [0.05, 1.0].
        Multiply road μ by this factor.
        
        Args:
            speed_ms: wheel ground speed [m/s]
            water_depth_mm: water depth [mm] (0 = dry)
        
        Returns:
            mu_factor: grip multiplier (1.0 = full grip, 0.05 = near-zero)
        """
        if water_depth_mm < 0.1:
            return 1.0  # Dry road
        
        speed_ratio = abs(speed_ms) / max(self.v_hydro, 1.0)
        depth_factor = min(water_depth_mm / 5.0, 1.0)  # Normalized depth severity
        
        if speed_ratio < 0.3:
            # Below 30% of hydroplaning speed — minimal effect
            # Just wet road friction reduction (about 10-20%)
            return 1.0 - 0.2 * depth_factor
        elif speed_ratio < 0.7:
            # Progressive grip loss zone
            reduction = depth_factor * (speed_ratio - 0.3) * 1.25
            return max(0.3, 1.0 - 0.2 * depth_factor - reduction)
        elif speed_ratio < 1.0:
            # Near-hydroplaning: severe grip loss
            reduction = depth_factor * 0.5 + depth_factor * (speed_ratio - 0.7) * 1.5
            return max(0.1, 1.0 - reduction)
        else:
            # Full hydroplaning: near-zero grip
            return max(0.05, 0.15 * (1.0 - depth_factor))
    
    def get_diagnostics(self):
        return {
            'v_hydro_kmh': self.v_hydro * 3.6,
            'v_hydro_ms': self.v_hydro,
            'tire_pressure': self.tire_pressure
        }


# ============================================================================
# DUAL-TIRE CONTACT MODEL
# ============================================================================

class DualTireContact:
    """
    Models twin (dual) tires on a single hub position.
    Truck rear axles typically have two tires per side sharing load.
    Each tire in the pair has slightly different load and camber.
    """
    def __init__(self, tire_model, spacing=DUAL_TIRE_SPACING, 
                 load_bias=DUAL_TIRE_LOAD_BIAS):
        self.tire_model = tire_model
        self.spacing = spacing       # Center-to-center [m]
        self.load_bias = load_bias   # Inner/outer load imbalance
        self.Fx_inner = 0.0  # Diagnostics
        self.Fx_outer = 0.0
        self.Fy_inner = 0.0
        self.Fy_outer = 0.0
    
    def combined_forces(self, alpha, kappa, Fz_total, gamma, mu):
        """
        Compute total force from dual tire pair.
        
        Returns:
            (Fx_total, Fy_total): Combined forces from both tires
        """
        # Split load between inner and outer tire
        Fz_inner = Fz_total * (0.5 + self.load_bias)
        Fz_outer = Fz_total * (0.5 - self.load_bias)
        
        # Slight camber difference from dual-tire geometry
        # Inner tire leans slightly inward, outer slightly outward
        gamma_inner = gamma + 0.005   # ~0.3° inward lean
        gamma_outer = gamma - 0.005
        
        # Compute forces for each tire independently
        Fx_i, Fy_i = self.tire_model.combined_slip(
            alpha, kappa, max(Fz_inner, 50.0), gamma_inner, mu
        )
        Fx_o, Fy_o = self.tire_model.combined_slip(
            alpha, kappa, max(Fz_outer, 50.0), gamma_outer, mu
        )
        
        # Store for diagnostics
        self.Fx_inner, self.Fy_inner = Fx_i, Fy_i
        self.Fx_outer, self.Fy_outer = Fx_o, Fy_o
        
        return Fx_i + Fx_o, Fy_i + Fy_o
    
    def get_diagnostics(self):
        return {
            'Fx_inner': self.Fx_inner, 'Fy_inner': self.Fy_inner,
            'Fx_outer': self.Fx_outer, 'Fy_outer': self.Fy_outer,
        }


# ============================================================================
# PNEUMATIC BRAKE SYSTEM (AIR BRAKE DELAY)
# ============================================================================

class PneumaticBrakeSystem:
    """
    Models air brake pressure dynamics for trucks.
    - Pure time delay (air propagation through lines)
    - First-order pressure build/release dynamics
    
    Tractor: ~200ms delay
    Trailer: ~400ms delay (longer air lines)
    """
    def __init__(self, delay=BRAKE_DELAY_TRACTOR, 
                 pressure_rate=BRAKE_PRESSURE_RATE,
                 release_rate=BRAKE_RELEASE_RATE):
        self.delay = delay
        self.pressure_rate = pressure_rate
        self.release_rate = release_rate
        self.actual_pressure = 0.0    # Current brake pressure [0-1]
        self.command_buffer = []      # Ring buffer for delay: (time, cmd)
        self.buffer_time = 0.0
    
    def update(self, commanded_pressure, dt):
        """
        Apply pure delay + first-order pressure dynamics.
        
        Args:
            commanded_pressure: Driver pedal command [0-1]
            dt: Timestep [s]
        
        Returns:
            actual_pressure: Effective brake pressure [0-1]
        """
        # Append to delay buffer
        self.buffer_time += dt
        self.command_buffer.append((self.buffer_time, commanded_pressure))
        
        # Read delayed command
        target_time = self.buffer_time - self.delay
        delayed_cmd = 0.0
        for t, cmd in self.command_buffer:
            if t <= target_time:
                delayed_cmd = cmd
        
        # First-order pressure dynamics (build is slower than release)
        if delayed_cmd > self.actual_pressure:
            rate = self.pressure_rate
        else:
            rate = self.release_rate
        
        delta = (delayed_cmd - self.actual_pressure) * min(rate * dt, 1.0)
        self.actual_pressure += delta
        self.actual_pressure = np.clip(self.actual_pressure, 0.0, 1.0)
        
        # Prune old buffer entries (keep last 2 seconds)
        cutoff = self.buffer_time - 2.0
        self.command_buffer = [(t, c) for t, c in self.command_buffer if t > cutoff]
        
        return self.actual_pressure
    
    def reset(self):
        """Reset brake system state."""
        self.actual_pressure = 0.0
        self.command_buffer.clear()
        self.buffer_time = 0.0
    
    def get_diagnostics(self):
        return {
            'actual_pressure': self.actual_pressure,
            'delay': self.delay,
            'buffer_size': len(self.command_buffer)
        }


# ============================================================================
# ROLL STABILITY CONTROL (RSC)
# ============================================================================

class RollStabilityControl:
    """
    Detects and prevents rollover risk.
    
    Strategy:
    1. ay > WARN threshold → throttle cut
    2. ay > LIMIT threshold → apply brakes on high-side wheels  
    3. roll_rate > threshold → emergency braking on all wheels
    """
    def __init__(self, mass, h_cg, track_width,
                 ay_warn=RSC_LAT_ACCEL_WARN,
                 ay_limit=RSC_LAT_ACCEL_LIMIT,
                 roll_rate_threshold=RSC_ROLL_RATE_THRESHOLD,
                 throttle_cut_factor=RSC_THROTTLE_CUT_FACTOR,
                 max_brake_force=RSC_MAX_BRAKE_FORCE):
        self.mass = mass
        self.h_cg = h_cg
        self.track_width = track_width
        self.ay_warn = ay_warn * 9.81       # Convert g → m/s²
        self.ay_limit = ay_limit * 9.81
        self.roll_rate_threshold = roll_rate_threshold
        self.throttle_cut_factor = throttle_cut_factor
        self.max_brake_force = max_brake_force
        
        # Static rollover threshold [m/s²]
        self.rollover_ay = 0.5 * track_width * 9.81 / max(h_cg, 0.1)
        
        # State
        self.mode = 'INACTIVE'  # INACTIVE, WARNING, INTERVENTION, EMERGENCY
        self.throttle_factor = 1.0
        self.corrective_forces = {'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0}
    
    def compute(self, ay, roll, roll_rate, vx, dt):
        """
        Compute RSC intervention.
        
        Args:
            ay: Lateral acceleration [m/s²]
            roll: Roll angle [rad]
            roll_rate: Roll rate [rad/s]
            vx: Longitudinal speed [m/s]
            dt: Timestep [s]
        
        Returns:
            (throttle_factor, corrective_brake_forces)
        """
        self.corrective_forces = {'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0}
        self.throttle_factor = 1.0
        
        abs_ay = abs(ay)
        abs_roll_rate = abs(roll_rate)
        
        # Emergency: high roll rate → immediate braking
        if abs_roll_rate > self.roll_rate_threshold:
            self.mode = 'EMERGENCY'
            self.throttle_factor = 0.0
            # Brake all wheels heavily
            emergency_force = self.max_brake_force * 1.5
            self.corrective_forces = {
                'fl': emergency_force, 'fr': emergency_force,
                'rl': emergency_force, 'rr': emergency_force
            }
            return self.throttle_factor, self.corrective_forces
        
        # Hard intervention: ay above limit
        if abs_ay > self.ay_limit:
            self.mode = 'INTERVENTION'
            self.throttle_factor = 0.0  # Full throttle cut
            
            # Brake intensity proportional to how far above limit
            severity = min((abs_ay - self.ay_limit) / (self.rollover_ay - self.ay_limit + 1e-3), 1.0)
            brake_force = self.max_brake_force * severity
            
            # Apply brakes on the HIGH side (loaded wheels — opposing the turn)
            if ay > 0:  # Turning left → right side loaded
                self.corrective_forces['fl'] = brake_force * 0.6
                self.corrective_forces['rl'] = brake_force * 0.4
            else:       # Turning right → left side loaded
                self.corrective_forces['fr'] = brake_force * 0.6
                self.corrective_forces['rr'] = brake_force * 0.4
            
            return self.throttle_factor, self.corrective_forces
        
        # Warning: ay nearing limit → throttle cut
        if abs_ay > self.ay_warn:
            self.mode = 'WARNING'
            # Progressive throttle reduction
            progress = (abs_ay - self.ay_warn) / (self.ay_limit - self.ay_warn + 1e-3)
            self.throttle_factor = max(self.throttle_cut_factor, 1.0 - progress * (1.0 - self.throttle_cut_factor))
            return self.throttle_factor, self.corrective_forces
        
        self.mode = 'INACTIVE'
        return self.throttle_factor, self.corrective_forces
    
    def get_diagnostics(self):
        return {
            'mode': self.mode,
            'throttle_factor': self.throttle_factor,
            'corrective_forces': self.corrective_forces.copy(),
            'rollover_ay': self.rollover_ay
        }


# ============================================================================
# TORQUE CONVERTER
# ============================================================================

class TorqueConverter:
    """
    Hydrodynamic torque converter with lockup clutch.
    
    Behavior:
    - At stall (ω_turbine = 0): torque multiplication = stall_ratio (2.2×)
    - As speed_ratio → coupling_point: torque ratio → 1.0
    - Above coupling_point: lockup clutch engages (direct drive)
    """
    def __init__(self, stall_ratio=TC_STALL_TORQUE_RATIO,
                 coupling_point=TC_COUPLING_POINT,
                 k_factor=TC_K_FACTOR):
        self.stall_ratio = stall_ratio
        self.coupling_point = coupling_point
        self.k_factor = k_factor
        self.locked = False
        self.speed_ratio = 0.0
        self.torque_ratio = stall_ratio
        self.efficiency = 0.0
        self.T_output = 0.0
    
    def compute(self, T_engine, omega_engine, omega_gearbox_input):
        """
        Compute output torque through torque converter.
        
        Args:
            T_engine: Engine torque [Nm]
            omega_engine: Engine crankshaft speed [rad/s]
            omega_gearbox_input: Gearbox input shaft speed [rad/s]
        
        Returns:
            T_output: Torque delivered to gearbox [Nm]
        """
        # Speed ratio
        if abs(omega_engine) < 1.0:
            self.speed_ratio = 0.0
        else:
            self.speed_ratio = abs(omega_gearbox_input) / abs(omega_engine)
        
        # Lockup check
        if self.speed_ratio > self.coupling_point:
            self.locked = True
            self.torque_ratio = 1.0
            self.efficiency = 1.0
            self.T_output = T_engine
            return self.T_output
        
        self.locked = False
        
        # Torque ratio curve (linear from stall_ratio to 1.0 at coupling point)
        self.torque_ratio = self.stall_ratio - \
            (self.stall_ratio - 1.0) * (self.speed_ratio / max(self.coupling_point, 0.01))
        
        # Pump torque limited by capacity factor
        # T_pump = K × (N_pump / 1000)² where N is in RPM
        omega_rps = abs(omega_engine) / (2.0 * np.pi)
        T_capacity = self.k_factor * omega_rps * omega_rps
        T_pump = min(abs(T_engine), T_capacity)
        
        # Output torque = pump torque × torque ratio
        self.T_output = T_pump * self.torque_ratio * np.sign(T_engine + 1e-10)
        
        # Efficiency = speed_ratio × torque_ratio
        self.efficiency = self.speed_ratio * self.torque_ratio
        
        return self.T_output
    
    def get_diagnostics(self):
        return {
            'locked': self.locked,
            'speed_ratio': self.speed_ratio,
            'torque_ratio': self.torque_ratio,
            'efficiency': self.efficiency,
            'T_output': self.T_output
        }


# ============================================================================
# DIFFERENTIAL MODEL
# ============================================================================

class Differential:
    """
    Models open, limited-slip, or locking differential.
    
    Open: equal torque split (50/50), no speed coupling
    LSD: clutch-pack transfers torque from fast to slow wheel
    Locking: forces both wheels to same speed (off-road / low-speed)
    """
    def __init__(self, diff_type=DIFF_TYPE, preload=LSD_PRELOAD,
                 ramp_factor=LSD_RAMP_FACTOR, lock_threshold=LSD_LOCK_THRESHOLD):
        self.type = diff_type
        self.preload = preload
        self.ramp_factor = ramp_factor
        self.lock_threshold = lock_threshold
        self.T_clutch = 0.0   # Diagnostic: clutch transfer torque
        self.locked = False
    
    def distribute(self, T_input, omega_left, omega_right):
        """
        Distribute input torque between left and right wheels.
        
        Args:
            T_input: Total drive torque to differential [N or Nm depending on context]
            omega_left: Left wheel rotational speed [rad/s]
            omega_right: Right wheel rotational speed [rad/s]
        
        Returns:
            (T_left, T_right): Torque/force to each wheel
        """
        if self.type == 'open':
            self.T_clutch = 0.0
            self.locked = False
            return T_input / 2.0, T_input / 2.0
        
        delta_omega = omega_left - omega_right
        
        if self.type == 'locking':
            # Force equal speed — effectively locks both wheels
            if abs(delta_omega) < self.lock_threshold:
                self.locked = True
                self.T_clutch = 0.0
                return T_input / 2.0, T_input / 2.0
            else:
                self.locked = True
                # Transfer torque proportional to speed difference
                T_transfer = np.clip(1000.0 * delta_omega, -abs(T_input), abs(T_input))
                T_left = T_input / 2.0 - T_transfer / 2.0
                T_right = T_input / 2.0 + T_transfer / 2.0
                self.T_clutch = abs(T_transfer)
                return T_left, T_right
        
        # Limited-slip differential (clutch-pack model)
        # Clutch torque = preload + ramp_factor × input_torque
        self.T_clutch = self.preload + self.ramp_factor * abs(T_input / 2.0)
        
        # Transfer from faster to slower wheel
        # Positive delta_omega means left is faster → transfer left→right
        if abs(delta_omega) < 0.01:
            # Negligible speed difference — equal split
            self.locked = True
            return T_input / 2.0, T_input / 2.0
        
        self.locked = False
        T_transfer = self.T_clutch * np.sign(delta_omega)
        T_transfer = np.clip(T_transfer, -abs(T_input * 0.8), abs(T_input * 0.8))
        
        T_left = T_input / 2.0 - T_transfer / 2.0
        T_right = T_input / 2.0 + T_transfer / 2.0
        
        # Ensure no wheel gets negative torque when driving
        if T_input > 0:
            T_left = max(0.0, T_left)
            T_right = max(0.0, T_right)
        
        return T_left, T_right
    
    def get_diagnostics(self):
        return {
            'type': self.type,
            'T_clutch': self.T_clutch,
            'locked': self.locked
        }


# ============================================================================
# AERODYNAMIC AND ROLLING RESISTANCE HELPERS

# ============================================================================

def compute_aero_and_rolling_resistance(vx, vehicle_components):
    """
    vehicle_components: list of dicts with keys {'Cd','A','mass','Crr'}.
    returns F_aero + F_rr
    """
    rho = 1.225
    F_aero_total = 0.0
    F_rr_total = 0.0
    for comp in vehicle_components:
        Cd = comp.get('Cd', 0.6)
        A  = comp.get('A', 10.0)
        m  = comp.get('mass', 0.0)
        Crr = comp.get('Crr', 0.008)
        F_aero_total += 0.5 * rho * Cd * A * vx**2
        F_rr_total  += Crr * m * 9.81
    return F_aero_total + F_rr_total

def pacejka_lat(alpha, Fz, mu, B=12.0, C=1.4):
    """Simple MF-like lateral curve (alpha in rad)"""
    D = mu * Fz
    return D * np.sin(C * np.arctan(B * alpha))

def pacejka_lon(kappa, Fz, mu, B=10.0, C=1.3):
    """Simple MF-like longitudinal curve (kappa = slip ratio)"""
    D = mu * Fz
    return D * np.sin(C * np.arctan(B * kappa))

def friction_ellipse_limit(Fx_raw, Fy_raw, Fz, mu):
    """Apply friction ellipse saturation: scale forces to lie on ellipse boundary if needed."""
    denom = mu * Fz
    if denom <= 0:
        return 0.0, 0.0
    n = math.hypot(Fx_raw/denom, Fy_raw/denom)
    if n > 1.0:
        return Fx_raw / n, Fy_raw / n
    return Fx_raw, Fy_raw

# ============================================================================
# ONLINE KOOPMAN PREDICTOR (Data-Driven Linear Dynamics)
# ============================================================================
# Learns a linear approximation K of the nonlinear dynamics from simulation data.
# Uses Dynamic Mode Decomposition (DMD) with sliding window for online learning.
# Koopman eigenvalues relate to Lyapunov exponents and entropy production rate.

class OnlineKoopman:
    """
    Online Koopman operator for fast linear prediction.
    
    Learns from simulation data and provides fast one-step predictions
    to improve Newton-Raphson initial guesses.
    """
    
    def __init__(self, n_states, window_size=100, min_samples=20, update_interval=10):
        """
        Args:
            n_states: Dimension of state vector
            window_size: Number of recent samples to keep for training
            min_samples: Minimum samples before Koopman is active
            update_interval: Recompute K every N timesteps
        """
        self.n_states = n_states
        self.window_size = window_size
        self.min_samples = min_samples
        self.update_interval = update_interval
        
        self.X_history = []  # Past states
        self.Y_history = []  # Future states
        self.K = None        # Koopman operator matrix
        self.timestep_count = 0
        
        # Diagnostics
        self.prediction_error_history = []
        self.eigenvalues = None  # Koopman eigenvalues (stability info)
    
    def add_sample(self, x_old, x_new):
        """Add a state transition pair to the training buffer."""
        self.X_history.append(x_old.copy())
        self.Y_history.append(x_new.copy())
        
        # Maintain sliding window
        if len(self.X_history) > self.window_size:
            self.X_history.pop(0)
            self.Y_history.pop(0)
        
        self.timestep_count += 1
        
        # Update K periodically
        if self.timestep_count % self.update_interval == 0 and len(self.X_history) >= self.min_samples:
            self.update_operator()
    
    def update_operator(self):
        """Recompute Koopman operator from recent data using DMD."""
        if len(self.X_history) < self.min_samples:
            return
        
        X = np.array(self.X_history).T  # [n_states × n_samples]
        Y = np.array(self.Y_history).T
        
        # Regularized pseudo-inverse for robustness
        try:
            # SVD-based pseudo-inverse with regularization
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            
            # Regularization: truncate small singular values
            threshold = 1e-10 * s[0]
            s_inv = np.where(s > threshold, 1.0 / s, 0.0)
            
            X_pinv = Vt.T @ np.diag(s_inv) @ U.T
            self.K = Y @ X_pinv
            
            # Compute Koopman eigenvalues for stability analysis
            self.eigenvalues = np.linalg.eigvals(self.K)
            
        except np.linalg.LinAlgError:
            # If SVD fails, skip this update
            pass
    
    def predict(self, x):
        """
        One-step prediction using learned Koopman operator.
        
        Returns:
            Predicted next state, or None if not trained yet
        """
        if self.K is None:
            return None
        return self.K @ x
    
    def is_ready(self):
        """Check if Koopman predictor has enough data."""
        return self.K is not None and len(self.X_history) >= self.min_samples
    
    def get_stability_info(self):
        """
        Get stability information from Koopman eigenvalues.
        
        Returns:
            dict with stability metrics:
            - max_eigenvalue: Largest |λ| (>1 = unstable)
            - entropy_rate: Sum of log|λ| for |λ|>1 (Lyapunov-like)
            - is_stable: All |λ| < 1
        """
        if self.eigenvalues is None:
            return {'max_eigenvalue': None, 'entropy_rate': 0.0, 'is_stable': True}
        
        abs_eig = np.abs(self.eigenvalues)
        max_eig = np.max(abs_eig)
        
        # Entropy rate from positive Lyapunov exponents (Kolmogorov-Sinai entropy)
        # Only count eigenvalues > 1 (expanding modes)
        expanding = abs_eig[abs_eig > 1.0]
        entropy_rate = np.sum(np.log(expanding)) if len(expanding) > 0 else 0.0
        
        return {
            'max_eigenvalue': max_eig,
            'entropy_rate': entropy_rate,
            'is_stable': max_eig < 1.0 + 1e-6,
            'eigenvalues': self.eigenvalues
        }


# ============================================================================
# PROBABILISTIC UNCERTAINTY TRACKER
# ============================================================================
# Tracks solver uncertainty as a covariance matrix Σ.
# Uncertainty grows during prediction and shrinks during Newton-Raphson updates.
# Information entropy = log|det(Σ)| - connects to thermodynamic entropy.

class UncertaintyTracker:
    """
    Probabilistic uncertainty quantification for the solver.
    
    Tracks state covariance and provides:
    - Uncertainty-based convergence criterion
    - Adaptive timestep suggestions
    - Information entropy computation
    """
    
    def __init__(self, n_states, initial_variance=1e-6):
        """
        Args:
            n_states: Dimension of state vector
            initial_variance: Initial diagonal variance
        """
        self.n_states = n_states
        self.Sigma = initial_variance * np.eye(n_states)  # Covariance matrix
        
        # Process noise (model uncertainty per timestep)
        self.Q = 1e-8 * np.eye(n_states)  # Small process noise
        
        # Observation noise (residual uncertainty)
        self.R = 1e-6 * np.eye(n_states)
        
        # History for tracking
        self.entropy_history = []
        self.trace_history = []
    
    def predict_uncertainty(self, J_dynamics):
        """
        Propagate uncertainty through dynamics: Σ' = F·Σ·F^T + Q
        
        Args:
            J_dynamics: Jacobian of dynamics (∂f/∂x)
        """
        self.Sigma = J_dynamics @ self.Sigma @ J_dynamics.T + self.Q
        
        # Ensure symmetry and positive definiteness
        self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)
        eigvals = np.linalg.eigvalsh(self.Sigma)
        if np.min(eigvals) < 1e-15:
            self.Sigma += 1e-15 * np.eye(self.n_states)
    
    def update_from_solver(self, J_solver, residual_norm):
        """
        Update uncertainty after Newton-Raphson iteration.
        
        The solver step provides information that reduces uncertainty.
        Uses Gauss-Newton approximation: Σ_posterior ≈ (J^T R^{-1} J + Σ_prior^{-1})^{-1}
        
        Args:
            J_solver: Jacobian from Newton-Raphson
            residual_norm: Current residual norm (informs observation noise)
        """
        # Scale observation noise by residual
        R_scaled = self.R * (1.0 + residual_norm)
        
        try:
            # Information form update
            R_inv = np.linalg.inv(R_scaled)
            Sigma_inv = np.linalg.inv(self.Sigma)
            
            # Posterior precision = prior precision + observation precision
            Sigma_posterior_inv = Sigma_inv + J_solver.T @ R_inv @ J_solver
            self.Sigma = np.linalg.inv(Sigma_posterior_inv)
            
            # Ensure symmetry
            self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)
            
        except np.linalg.LinAlgError:
            # If inversion fails, just reduce uncertainty slightly
            self.Sigma *= 0.9
    
    def compute_entropy(self):
        """
        Compute information entropy: S = 0.5 * log|det(2πe·Σ)|
        
        For Gaussian, this simplifies to: S = 0.5 * (n + n*log(2π) + log|det(Σ)|)
        We return just log|det(Σ)| as the key quantity.
        """
        try:
            sign, logdet = np.linalg.slogdet(self.Sigma)
            entropy = logdet  # log|det(Σ)|
        except np.linalg.LinAlgError:
            entropy = float('inf')
        
        self.entropy_history.append(entropy)
        return entropy
    
    def get_trace(self):
        """Total variance (trace of covariance matrix)."""
        trace = np.trace(self.Sigma)
        self.trace_history.append(trace)
        return trace
    
    def is_converged(self, threshold=1e-8):
        """Check if uncertainty is below threshold."""
        return self.get_trace() < threshold
    
    def suggest_timestep_factor(self, base_factor=1.0):
        """
        Suggest timestep scaling based on uncertainty.
        
        High uncertainty → reduce timestep
        Low uncertainty → can increase timestep
        """
        trace = self.get_trace()
        
        if trace > 1e-4:
            return base_factor * 0.5  # Reduce timestep
        elif trace < 1e-10:
            return base_factor * 1.2  # Can increase timestep
        else:
            return base_factor  # Keep current
    
    def reset(self, variance=1e-6):
        """Reset uncertainty to initial state."""
        self.Sigma = variance * np.eye(self.n_states)


class TractorHead:
    def __init__(self, x, y, z, yaw, pitch, vx, vy, omega, waypoints):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.pitch = pitch
        self.vx = vx
        self.vy = vy
        self.omega = omega
        
        # Store previous accelerations for explicit Euler predictor
        # Provides better initial guess than kinematic model
        self.ax_prev = 0.0
        self.ay_prev = 0.0
        self.alpha_prev = 0.0  # Angular acceleration
        
        # ============================================================================
        # LATERAL LOAD TRANSFER FIX: Cached lateral acceleration from force balance
        # ============================================================================
        # Cache the lateral acceleration computed from tire forces (ay = sum_Fy/m - vx*omega)
        # instead of approximating with kinematics alone (ay = vx*omega).
        # Lag by one iteration for implicit solver self-consistency.
        self._ay_for_load_transfer = 0.0  # [m/s²] body-frame lateral acceleration from forces
        
        # ============================================================================
        # GENERALIZED-ALPHA STATE VARIABLES
        # ============================================================================
        # The Generalized-α method requires storing accelerations as state variables
        # (not just for predictor, but as part of the solved system)
        self.ax = 0.0       # Longitudinal acceleration [m/s²]
        self.ay = 0.0       # Lateral acceleration [m/s²] (body frame)
        self.alpha = 0.0    # Yaw angular acceleration [rad/s²]
        
        # Precompute Generalized-α integration parameters from RHO_INF
        if USE_GENERALIZED_ALPHA:
            self._ga_alpha_m, self._ga_alpha_f, self._ga_beta, self._ga_gamma = compute_genalpha_params(RHO_INF)
            print(f"[TRACTOR] Generalized-alpha enabled: rho_inf={RHO_INF}, alpha_m={self._ga_alpha_m:.3f}, alpha_f={self._ga_alpha_f:.3f}, beta={self._ga_beta:.4f}, gamma={self._ga_gamma:.3f}")
        else:
            self._ga_alpha_m = self._ga_alpha_f = self._ga_beta = self._ga_gamma = None
            print("[TRACTOR] Using legacy Backward Euler solver")
        
        # ============================================================================
        # KOOPMAN + PROBABILISTIC NUMERICS
        # ============================================================================
        # Online Koopman predictor: learns linear dynamics from simulation
        # State vector is 9D (x, y, yaw, vx, vy, omega, ax, ay, alpha)
        self.koopman = OnlineKoopman(n_states=9, window_size=100, min_samples=20, update_interval=10)
        
        # Uncertainty tracker: probabilistic quantification of solver error
        self.uncertainty = UncertaintyTracker(n_states=9, initial_variance=1e-6)
        
        print("[TRACTOR] Koopman + Probabilistic numerics enabled")
        
        self.max_steer = np.radians(30.0)  # [rad] max steering angle
        self.waypoints = waypoints
        self.c_a = 1.36
        self.c_r1 = 0.01
        self.gear_ratios = [14.94, 12.3, 9.57, 7.47, 5.77, 4.67, 3.64, 2.84, 2.21, 1.74, 1.36, 1.0] # 12-speed I shift gearbox
        self.final_drive = 3.5  # CRITICAL FIX: Typical truck final drive ratio (was 1.0!)
        self.efficiency = 0.95
        self.current_gear = 0  # start in 1st gear
        # CRITICAL FIX: Upshift speeds were absurdly high (10-160 m/s = 36-576 km/h!)
        # Realistic truck shift points: ~3-22 m/s (10-80 km/h) across 12 gears
        self.upshift_speeds = [3, 5, 7, 9, 11, 13, 15, 17, 19, 20, 22]  # m/s (realistic truck speeds)
        self.downshift_speeds = [2, 4, 6, 8, 10, 12, 14, 16, 18, 19, 21]  # m/s (with hysteresis)
        self.r_w = 0.5  # wheel radius in meters

        # Hybrid electric motor parameters
        self.electric_torque_max = 2000.0  # Nm, peak electric torque
        self.electric_speed_limit = 10.0  # m/s, speed below which electric motor assists
        self.electric_efficiency = 0.95  # efficiency of electric motor
        self.mass = 8000.0  # vehicle mass in kg
        self.Iz = 12000.0  # yaw moment of inertia in kg/m2
        self.L=3.7 # wheelbase in meters
        self.lf = 0.4 * self.L  # distance from CG to front axle
        self.lr = self.L - self.lf  # distance from CG to rear axle
        self.track_width = 2.05  # track width in meters
        self.h_cg = 1.15  # height of center of gravity in meters
        self.mu = DEFAULT_MU  # tire-road friction coefficient (references global for sensitivity studies)
        self.hitch_length = 0.7  # length to trailer hitch point
        self.trailer_drag = 0.0  # CRITICAL: Initialize to prevent undefined behavior

        # ---------- geometry / tyre / suspension offsets ----------
        self.toe = np.radians(0.1)       # static toe [rad] (front axle)
        self.static_camber_front = np.radians(-0.5)  # front static camber [rad] (negative typically)
        self.static_camber_rear  = np.radians(-0.2)  # rear static camber [rad]
        self.caster = np.radians(3.0)     # caster angle [rad]
        self.camber_gain = 0.08           # camber gain [rad camber] per [rad roll] from suspension geometry

        # tyre/cornering stiffness (per axle total). Use realistic truck values.
        # You can later split per wheel if needed.
        self.Cf_axle = 300000.0    # front axle cornering stiffness [N/rad]
        self.Cr_axle = 320000.0    # rear axle cornering stiffness [N/rad]
        self.Cgamma_f = 35000.0    # camber coefficient front [N/rad]
        self.Cgamma_r = 25000.0    # camber coefficient rear [N/rad]

        # rolling / aero
        self.Crr = 0.008           # rolling resistance coefficient (typical truck)
        self.Cd  = 0.60            # tractor Cd
        self.A_front = 8.5         # frontal area tractor [m^2]

        # Braking system parameters
        self.max_brake_decel = 0.7 * 9.81  # maximum deceleration from braking [m/s^2] (~0.7g)
        self.brake_force_distribution_front = 0.65  # 65% of brake force to front axle
        self.brake_force_distribution_rear = 0.35   # 35% of brake force to rear axle
        
        # Regenerative braking parameters
        self.regen_max_decel = 0.3 * 9.81  # max deceleration from e-machine regen (~0.3g = 2.94 m/s^2)
        self.regen_efficiency = 0.90       # regenerative braking efficiency (90%)
        self.regen_speed_limit = 5.0       # minimum speed for regen operation [m/s]

        # suspension / roll dynamics
        self.roll = 0.0            # roll angle [rad]
        self.roll_rate = 0.0       # roll rate [rad/s]
        self.Ixx = 3000.0          # roll inertia [kg m^2] (approx)
        self.K_phi = 300000.0      # roll stiffness [Nm/rad]
        self.C_phi = 40000.0       # roll damping [Nms/rad]

        # pitch dynamics
        self.pitch_dyn = 0.0       # dynamic pitch angle [rad] (from suspension response)
        self.pitch_terrain = 0.0   # terrain pitch [rad] (from ground slope)
        self.pitch_rate = 0.0      # pitch angular velocity [rad/s]
        self.Iyy = 15000.0         # pitch moment of inertia [kg m^2] (larger than roll for trucks)
        self.K_theta = 200000.0    # pitch stiffness [Nm/rad] (suspension longitudinal stiffness)
        self.C_theta = 30000.0     # pitch damping [Nms/rad]
        self.anti_dive = 0.15      # anti-dive percentage (0-1): suspension geometry resists brake dive
        self.anti_squat = 0.25     # anti-squat percentage (0-1): suspension geometry resists accel squat


        # Simplified tire stiffness parameters (for now, could link to Pacejka later)
        self.tyre_B = 10.0
        self.tyre_C = 1.9
        self.tyre_Bx = 10.0
        self.tyre_Cx = 1.9

        # Tire positions (4 corners)

        self.n_front_tires = 2   # single wheels left/right
        self.n_rear_tires  = 4   # dual wheels left/right (typical 4x2 tractor head)

        self.tire_positions = {
            "front_left":  np.array([ self.lf,  self.track_width / 2 ]),
            "front_right": np.array([ self.lf, -self.track_width / 2 ]),
            "rear_left":   np.array([-self.lr,  self.track_width / 2 ]),
            "rear_right":  np.array([-self.lr, -self.track_width / 2 ]),
        }

        # drivetrain / wheel spin
        self.wheel_omega = { 'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0 }   # rad/s
        self.wheel_inertia = 40.0      # kg*m2 (per wheel approximate)
        self.driven_wheels = ['rl', 'rr']   # rear drive

        # Combined-slip Pacejka tire models (replaces old linear tire models)
        self.tire_front = create_truck_tire('front')
        self.tire_rear = create_truck_tire('rear')
        self.mu_static = DEFAULT_MU  # friction coefficient (references global for sensitivity studies)

        # Legacy parameters (kept for compatibility but not used in main solver)
        self.B_lat = 12.0
        self.C_lat = 1.4
        self.B_lon = 10.0
        self.C_lon = 1.3

        # scale factors for camber effect
        self.K_camber = 0.0   # how camber shifts effective slip angle (rad -> rad). Set 0.02..0.1 if desired

        # per-wheel Fz holders (updated each step)
        self.Fz = {'fl': self.mass*9.81/4.0, 'fr': self.mass*9.81/4.0, 'rl': self.mass*9.81/4.0, 'rr': self.mass*9.81/4.0}

        
        # --- Steering axis & trail geometry (for self-centering / aligning torque) ---
        self.SAI = np.radians(8.0)      # Steering Axis Inclination (aka KPI) [rad]
        self.mechanical_trail = 0.05    # Mechanical trail at ground [m] (caster trail projection)

        # --- Static alignment baselines (per axle; you can split per wheel later) ---
        self.static_toe_front = np.radians(0.10)  # small toe-in front [rad]
        self.static_toe_rear  = np.radians(0.00)  # typically ~0 on rear [rad]

        self.static_caster_front = np.radians(3.0)  # front caster [rad]
        self.static_caster_rear  = np.radians(0.0)  # non-steered rear caster [rad]

        # --- Kinematic & compliance gains (tunable, small-signal mappings) ---
        # Camber changes
        self.k_sai_camber   = 0.02      # camber change per rad of steer due to SAI [rad/rad]
        self.k_camber_Fz    = 5e-6      # camber change per N of vertical load [rad/N]
        self.k_camber_bump  = 2e-3      # camber change per m of wheel bump/jounce [rad/m]

        # Toe changes (compliance steer, bump-steer, roll-steer)
        self.k_toe_roll     = 0.05      # toe change per rad of body roll [rad/rad]
        self.k_toe_Fy       = 1e-7      # toe change per N of lateral force (per wheel) [rad/N]
        self.k_toe_bump     = 1e-3      # toe change per m of wheel bump/jounce [rad/m]

        # Caster variation with bump/jounce (usually small)
        self.k_caster_bump  = 1e-3      # caster change per m of wheel bump/jounce [rad/m]

        # --- Pneumatic trail model (for self-aligning torque) ---
        self.t_p0           = 0.06      # nominal pneumatic trail at small slip [m]
        self.t_p_decay      = 12.0      # trail decay factor with |alpha| [-]

        # --- Optional: steering DOF parameters (enable if you want delta to be dynamic) ---
        # If you keep delta as input, these can stay unused. If you add a steering state,
        # use them in a simple rotary mass-spring-damper for the steering column.
        self.delta          = 0.0       # steering angle state [rad]
        self.delta_rate     = 0.0       # steering rate state [rad/s]
        self.I_delta        = 0.5       # equivalent steering inertia [kg·m^2]
        self.K_delta        = 200.0     # steering stiffness (column + assist) [N·m/rad]
        self.C_delta        = 10.0      # steering damping [Nms/rad]

        # ============================================================================
        # TIRE RELAXATION LENGTH - Transient Force States
        # ============================================================================
        # If ENABLE_TIRE_RELAXATION is True, tire forces lag behind steady-state
        # values due to contact patch deformation dynamics
        
        if ENABLE_TIRE_RELAXATION:
            # Transient tire forces (actual forces applied to vehicle)
            # Initialize to zero - will build up during first few timesteps
            # HISTORY STATE (fixed during solver iteration)
            self.Fx_fl_transient_hist = 0.0
            self.Fy_fl_transient_hist = 0.0
            self.Fx_fr_transient_hist = 0.0
            self.Fy_fr_transient_hist = 0.0
            self.Fx_rl_transient_hist = 0.0
            self.Fy_rl_transient_hist = 0.0
            self.Fx_rr_transient_hist = 0.0
            self.Fy_rr_transient_hist = 0.0
            
            # Relaxation dynamics objects
            self.tire_relaxation_front = TireRelaxationDynamics(
                sigma_x=SIGMA_X_FRONT,
                sigma_y=SIGMA_Y_FRONT
            )
            self.tire_relaxation_rear = TireRelaxationDynamics(
                sigma_x=SIGMA_X_REAR,
                sigma_y=SIGMA_Y_REAR
            )
            
            print("[TIRE RELAXATION] Enabled for TractorHead")
            print(f"  Front: sigma_x={SIGMA_X_FRONT}m, sigma_y={SIGMA_Y_FRONT}m")
            print(f"  Rear:  sigma_x={SIGMA_X_REAR}m, sigma_y={SIGMA_Y_REAR}m")
        else:
            print("[TIRE RELAXATION] Disabled - using steady-state tire model")

        # ============================================================================
        # ABS (ANTI-LOCK BRAKING SYSTEM)
        # ============================================================================
        if ENABLE_ABS:
            self.abs_controller = ABSController(
                slip_target=ABS_SLIP_TARGET,
                slip_threshold=ABS_SLIP_THRESHOLD,
                release_threshold=ABS_RELEASE_THRESHOLD,
                pressure_increase_rate=ABS_PRESSURE_INCREASE_RATE,
                pressure_decrease_rate=ABS_PRESSURE_DECREASE_RATE,
                min_speed=ABS_MIN_SPEED
            )
            print("[ABS] Enabled for TractorHead")
            print(f"  Target κ={ABS_SLIP_TARGET}, threshold={ABS_SLIP_THRESHOLD}, release={ABS_RELEASE_THRESHOLD}")
        else:
            self.abs_controller = None

        # ============================================================================
        # TCS (TRACTION CONTROL SYSTEM)
        # ============================================================================
        if ENABLE_TCS:
            self.tcs_controller = TractionControlSystem(
                slip_target=TCS_SLIP_TARGET,
                slip_threshold=TCS_SLIP_THRESHOLD,
                release_threshold=TCS_RELEASE_THRESHOLD,
                torque_reduction_rate=TCS_TORQUE_REDUCTION_RATE,
                torque_increase_rate=TCS_TORQUE_INCREASE_RATE,
                min_speed=TCS_MIN_SPEED
            )
            print("[TCS] Enabled for TractorHead")
        else:
            self.tcs_controller = None

        # ============================================================================
        # ESC (ELECTRONIC STABILITY CONTROL)
        # ============================================================================
        if ENABLE_ESC:
            self.esc_controller = ESController(
                wheelbase=self.lf + self.lr,
                yaw_rate_deadband=ESC_YAW_RATE_DEADBAND,
                gain_understeer=ESC_GAIN_UNDERSTEER,
                gain_oversteer=ESC_GAIN_OVERSTEER,
                max_brake_force=ESC_MAX_BRAKE_FORCE,
                min_speed=ESC_MIN_SPEED
            )
            self.esc_controller.mass_for_scaling = self.mass
            print("[ESC] Enabled for TractorHead")
        else:
            self.esc_controller = None

        # ============================================================================
        # QUARTER-CAR VERTICAL DYNAMICS
        # ============================================================================
        if ENABLE_QUARTER_CAR:
            # Static loads per corner
            g = 9.81
            L = self.lf + self.lr
            Fz_front_static = self.mass * g * self.lr / L / 2.0
            Fz_rear_static  = self.mass * g * self.lf / L / 2.0
            
            # Sprung mass per corner (total mass - 4 unsprung)
            m_sprung_total = self.mass - 4 * QC_UNSPRUNG_MASS
            m_s_front = m_sprung_total * (self.lr / L) / 2.0
            m_s_rear  = m_sprung_total * (self.lf / L) / 2.0
            
            self.quarter_car = {
                'fl': QuarterCarSuspension(m_s_front, QC_UNSPRUNG_MASS, QC_K_SPRING_F, QC_C_DAMPER_F, QC_K_TIRE, Fz_front_static),
                'fr': QuarterCarSuspension(m_s_front, QC_UNSPRUNG_MASS, QC_K_SPRING_F, QC_C_DAMPER_F, QC_K_TIRE, Fz_front_static),
                'rl': QuarterCarSuspension(m_s_rear, QC_UNSPRUNG_MASS, QC_K_SPRING_R, QC_C_DAMPER_R, QC_K_TIRE, Fz_rear_static),
                'rr': QuarterCarSuspension(m_s_rear, QC_UNSPRUNG_MASS, QC_K_SPRING_R, QC_C_DAMPER_R, QC_K_TIRE, Fz_rear_static),
            }
            self._qc_Fz = {'fl': Fz_front_static, 'fr': Fz_front_static,
                           'rl': Fz_rear_static, 'rr': Fz_rear_static}
            self._qc_bump = {'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0}
            print(f"[QUARTER-CAR] Enabled: m_s_f={m_s_front:.0f}kg, m_s_r={m_s_rear:.0f}kg")
        else:
            self.quarter_car = None

        # ============================================================================
        # ROAD FRICTION MAP
        # ============================================================================
        if ENABLE_MU_MAP:
            self.road_friction = RoadFrictionMap(default_mu=DEFAULT_MU)
            print(f"[MU-MAP] Enabled: default μ={DEFAULT_MU}")
        else:
            self.road_friction = None

        # ============================================================================
        # HYDROPLANING MODEL
        # ============================================================================
        if ENABLE_HYDROPLANING:
            self.hydroplaning = HydroplaningModel(
                tire_pressure_kPa=HYDRO_TIRE_PRESSURE,
                tire_width_mm=HYDRO_TIRE_WIDTH
            )
            print(f"[HYDROPLANING] Enabled: v_hydro={self.hydroplaning.v_hydro*3.6:.1f} km/h")
        else:
            self.hydroplaning = None

        # ============================================================================
        # DUAL-TIRE CONTACT (rear axles)
        # ============================================================================
        if ENABLE_DUAL_TIRES:
            self.dual_tire_rl = DualTireContact(self.tire_rear, DUAL_TIRE_SPACING, DUAL_TIRE_LOAD_BIAS)
            self.dual_tire_rr = DualTireContact(self.tire_rear, DUAL_TIRE_SPACING, DUAL_TIRE_LOAD_BIAS)
            print(f"[DUAL-TIRES] Enabled: spacing={DUAL_TIRE_SPACING}m, bias={DUAL_TIRE_LOAD_BIAS}")
        else:
            self.dual_tire_rl = None
            self.dual_tire_rr = None

        # ============================================================================
        # PNEUMATIC BRAKE SYSTEM
        # ============================================================================
        if ENABLE_BRAKE_DELAY:
            self.brake_system = PneumaticBrakeSystem(
                delay=BRAKE_DELAY_TRACTOR,
                pressure_rate=BRAKE_PRESSURE_RATE,
                release_rate=BRAKE_RELEASE_RATE
            )
            print(f"[BRAKE-DELAY] Enabled: delay={BRAKE_DELAY_TRACTOR*1000:.0f}ms")
        else:
            self.brake_system = None

        # ============================================================================
        # ROLL STABILITY CONTROL (RSC)
        # ============================================================================
        if ENABLE_RSC:
            self.rsc_controller = RollStabilityControl(
                mass=self.mass,
                h_cg=self.h_cg,
                track_width=self.track_width
            )
            print(f"[RSC] Enabled: warn={RSC_LAT_ACCEL_WARN}g, limit={RSC_LAT_ACCEL_LIMIT}g")
        else:
            self.rsc_controller = None

        # ============================================================================
        # TORQUE CONVERTER
        # ============================================================================
        if ENABLE_TORQUE_CONVERTER:
            self.torque_converter = TorqueConverter(
                stall_ratio=TC_STALL_TORQUE_RATIO,
                coupling_point=TC_COUPLING_POINT,
                k_factor=TC_K_FACTOR
            )
            print(f"[TORQUE-CONV] Enabled: stall={TC_STALL_TORQUE_RATIO}x, lockup={TC_COUPLING_POINT}")
        else:
            self.torque_converter = None

        # ============================================================================
        # DIFFERENTIAL (LSD)
        # ============================================================================
        if ENABLE_LSD:
            self.differential = Differential(
                diff_type=DIFF_TYPE,
                preload=LSD_PRELOAD,
                ramp_factor=LSD_RAMP_FACTOR,
                lock_threshold=LSD_LOCK_THRESHOLD
            )
            print(f"[DIFF] Enabled: type={DIFF_TYPE}, preload={LSD_PRELOAD}Nm")
        else:
            self.differential = None

    def warm_start_tire_relaxation(self):
        """
        Initialize transient force history to steady-state Pacejka forces
        at the current vehicle state. Must be called AFTER __init__ and
        initialize_wheel_states().
        
        Without this, history starts at 0 and the implicit Euler lag
        leaves tires with ~5% grip for the first ~0.1s, causing
        uncontrolled lateral drift.
        """
        if not ENABLE_TIRE_RELAXATION:
            return
        
        vx = max(self.vx, 0.1)
        vy = self.vy
        omega = self.omega
        delta = 0.0  # Assume straight-line initial condition
        
        Fz_local = self.compute_wheel_loads_local(vx, vy, omega)
        wheels = self.compute_wheel_local_states(vx, vy, omega, delta, Fz_local)
        
        for key in ['fl', 'fr', 'rl', 'rr']:
            w = wheels[key]
            tire = self.tire_front if key in ['fl', 'fr'] else self.tire_rear
            
            Fx_ss, Fy_ss = tire.combined_slip(
                alpha=w['alpha'], kappa=w.get('kappa', 0.0),
                Fz=Fz_local[key], gamma=w['camber'], mu=self.mu_static
            )
            Fy_ss = -Fy_ss  # Negate (same convention as compute_residual)
            
            if key == 'fl':
                self.Fx_fl_transient_hist = Fx_ss
                self.Fy_fl_transient_hist = Fy_ss
            elif key == 'fr':
                self.Fx_fr_transient_hist = Fx_ss
                self.Fy_fr_transient_hist = Fy_ss
            elif key == 'rl':
                self.Fx_rl_transient_hist = Fx_ss
                self.Fy_rl_transient_hist = Fy_ss
            else:
                self.Fx_rr_transient_hist = Fx_ss
                self.Fy_rr_transient_hist = Fy_ss
        
        print(f"[TIRE RELAXATION] Warm-started tractor: "
              f"FL=({self.Fx_fl_transient_hist:.0f},{self.Fy_fl_transient_hist:.0f})N "
              f"RR=({self.Fx_rr_transient_hist:.0f},{self.Fy_rr_transient_hist:.0f})N")

    def initialize_wheel_states(self):
        """
        Initialize wheel rotational speeds (omega) based on current vehicle velocity.
        Prevents initial skidding (locked wheels) at startup.
        
        Calculates: omega = v_longitudinal / r_w
        where v_longitudinal is projected onto the wheel's heating (steer + toe).
        """
        print(f"[INIT] Initializing Tractor wheel speeds for vx={self.vx:.2f} m/s")
        
        # Ackermann steer angles (if any steering applied at start)
        delta_fl, delta_fr = self.ackermann_steering_angles(0.0)
        
        positions = {
            'fl': self.tire_positions['front_left'],
            'fr': self.tire_positions['front_right'],
            'rl': self.tire_positions['rear_left'],
            'rr': self.tire_positions['rear_right'],
        }
        
        for key, pos in positions.items():
            px, py = float(pos[0]), float(pos[1])
            
            # 1. Wheel center velocity
            vwx = self.vx - self.omega * py
            vwy = self.vy + self.omega * px
            
            # 2. Wheel heading (Steer + Toe)
            if key == 'fl':
                steer = delta_fl
                toe = self.static_toe_front
            elif key == 'fr':
                steer = delta_fr
                toe = self.static_toe_front
            else:
                steer = 0.0
                toe = self.static_toe_rear
                
            heading = steer + toe
            
            # 3. Project velocity onto wheel plane
            # v_long = v_x_local * cos(heading) + v_y_local * sin(heading)
            v_long = vwx * np.cos(heading) + vwy * np.sin(heading)
            
            # 4. Set rotational speed (v = omega * r)
            if self.r_w > 0:
                self.wheel_omega[key] = v_long / self.r_w
                
            print(f"  Wheel {key}: v_long={v_long:.2f} m/s -> omega={self.wheel_omega[key]:.2f} rad/s")

    def integrate_wheel_speeds(self, throttle, brake, dt):
        """
        Explicit time integration of wheel rotational speeds:
        I_w * dw/dt = T_drive - T_brake - Fx_tire * r_w
        
        Updates self.wheel_omega based on torques and tire forces.
        Calculated AFTER solver convergence using the final state.
        """
        # 1. Get current vehicle state (already updated by unpack_and_update)
        vx, vy, omega = self.vx, self.vy, self.omega
        
        # 2. Get local wheel states (slip, etc.)
        Fz_local = self.compute_wheel_loads_local(vx, vy, omega)
        
        # Steering (assume constant during step or use last command if available)
        # We don't have 'delta' here easily unless we store it.
        # For now assume small steering change or store delta in update.
        # Better: Assume Ackermann from self.delta (need to store self.delta in update)
        delta_fl, delta_fr = self.ackermann_steering_angles(getattr(self, 'delta_prev', 0.0))
        
        # 3. Compute Forces & Torques
        for key in ['fl', 'fr', 'rl', 'rr']:
            # Wheel position
            pos = self.tire_positions[{'fl':'front_left', 'fr':'front_right', 'rl':'rear_left', 'rr':'rear_right'}[key]]
            px, py = float(pos[0]), float(pos[1])
            
            # Local velocities
            vwx = vx - omega * py
            vwy = vy + omega * px
            
            # Steering
            if key == 'fl': steer = delta_fl
            elif key == 'fr': steer = delta_fr
            else: steer = 0.0
            
            # Heading & Slip
            # (Simplified: assume small angles/linearization is okay for torque calculation)
            # Actually we need full Pacejka to get Fx correctly
            toe = self.static_toe_front if key in ['fl','fr'] else self.static_toe_rear
            heading = steer + toe
            
            v_long = vwx * np.cos(heading) + vwy * np.sin(heading) # Approx longitudinal speed
            
            # Slip Ratio kappa
            omega_w = self.wheel_omega.get(key, 0.0)
            v_wheel = omega_w * self.r_w
            
            if abs(v_long) > 0.1:
                kappa = (v_wheel - v_long) / abs(v_long)
            else:
                kappa = 0.0
            
            # Check limits
            kappa = np.clip(kappa, -1.0, 1.0)
            
            # Loads & Camber
            Fz = Fz_local[key]
            camber = self.camber_front_total if key in ['fl','fr'] else self.camber_rear_total
            
            # Tire Force Fx
            # We only need Fx for the torque balance
            if key in ['fl', 'fr']:
                tire = self.tire_front
            else:
                tire = self.tire_rear
            
            # Get Fx from tire model
            # Note: We assume alpha is small or use simplified alpha calculation
            alpha = math.atan2(vwy, max(abs(vwx), 0.1)) - heading
            
            Fx, _ = tire.combined_slip(alpha, kappa, Fz, camber, self.mu_static)
            
            # Limit Fx by friction circle (simplified)
            Fx = np.clip(Fx, -self.mu_static*Fz, self.mu_static*Fz)
            
            # 4. Drive & Brake Torques
            # Drive
            T_drive = 0.0
            if key in self.driven_wheels:
                # Distribute drive torque
                # Re-calculate total drive force roughly
                rpm = self.compute_engine_rpm(vx, self.r_w)
                T_engine = throttle * self.engine_torque_curve(rpm) + (throttle * self.electric_torque(vx) if hasattr(self, 'electric_torque') else 0)
                # Gear ratio torque at wheel: T_wheel = T_engine * gear_ratio * diff_ratio * efficiency
                # Simplified: F_drive * r_w
                F_drive_total = self.compute_drive_force(T_engine, self.r_w, vx)
                F_drive_per_wheel = F_drive_total / len(self.driven_wheels)
                T_drive = F_drive_per_wheel * self.r_w
            
            # Brake
            # Simplified brake distribution
            F_brake_total = self.compute_brake_force(brake, vx)
            brake_dict, _, _ = self.distribute_braking_forces(F_brake_total, vx)
            
            # ABS modulation for wheel-speed integration
            if ENABLE_ABS and self.abs_controller is not None and brake > 0.01:
                wheel_kappas = getattr(self, '_wheel_kappas', {k: 0.0 for k in ['fl','fr','rl','rr']})
                brake_dict, _ = self.abs_controller.modulate(brake_dict, wheel_kappas, vx, dt)
            
            F_brake = brake_dict.get(key, 0.0)
            T_brake = F_brake * self.r_w * np.sign(omega_w) # Opposes rotation
            
            # 5. Integration: I * dw/dt = T_drive - T_brake - Fx * r
            # Note: Fx is force BY ROAD ON TIRE. Positive Fx pushes vehicle forward.
            # Reaction on wheel opposes rotation?
            # Free body diagram of wheel:
            # Torque from axle (Drive) -> Positive Omega
            # Torque from brake -> Negative Omega
            # Force from road Fx (static friction pushing car forward) implies road pushes BACK on tire calculation?
            # Wait, Fx in Pacejka is Force ON TIRE. 
            # If Kappa > 0 (Wheel spinning fast), Fx is Positive (Thrust).
            # This Fx creates a torque Fx * r opposing the spin? No.
            # Road pushes forward. Wheel pushes backward.
            # Reaction torque on wheel = -Fx * r.
            # If Fx is positive (Forward thrust), road pushes forward.
            # This torque *speeds up* the vehicle, but *slows down* the wheel (reaction)?
            # Correct equation: I * dw/dt = T_drive - T_brake - Fx * r
            
            torque_net = T_drive - T_brake - Fx * self.r_w
            
            dw_dt = torque_net / self.wheel_inertia
            
            self.wheel_omega[key] += dw_dt * dt
            
            # Prevent reversal from braking if stopped? 
            # Optional: if abs(omega) < small and T_brake > T_drive, clamp to 0
            if abs(self.wheel_omega[key]) < 0.1 and abs(T_brake) > abs(T_drive) and abs(vx) < 0.1:
                self.wheel_omega[key] = 0.0


    
    def compute_alignment_per_wheel(self, key, steer_w, roll, pitch, Fz_i, bump_i):
        # base static
        if key in ('fl','fr'):
            toe0    = self.static_toe_front
            camber0 = self.static_camber_front
            caster0 = self.static_caster_front
        else:
            toe0    = self.static_toe_rear
            camber0 = self.static_camber_rear
            caster0 = self.static_caster_rear

        # kinematic deltas
        dcamber_roll  = self.camber_gain * roll
        dcamber_steer = self.k_sai_camber * steer_w     # SAI-induced camber with steer
        dcamber_load  = self.k_camber_Fz * (Fz_i - self.mass*9.81/4.0)
        dcamber_bump  = self.k_camber_bump * bump_i

        dtoe_roll     = self.k_toe_roll * roll
        # add compliance steer once Fy_i is known (done later; here we add kinematic/bump)
        dtoe_bump     = self.k_toe_bump * bump_i

        dcaster_bump  = self.k_caster_bump * bump_i

        camber = camber0 + dcamber_roll + dcamber_steer + dcamber_load + dcamber_bump
        toe    = toe0    + dtoe_roll    + dtoe_bump
        caster = caster0 + dcaster_bump

        return toe, camber, caster
    
    
    def estimate_wheel_bump(self, px, py, roll, pitch):
        """
        Approximate vertical deflection (bump/jounce) at a wheel due to small-angle
        body rotations (roll, pitch). Signs depend on your axis conventions.
        For small angles: z_defl ≈ roll * py + pitch * px
        """
        return roll * py + pitch * px


    def compute_alignment_per_wheel(self, key, steer_w, roll, pitch, Fz_i, bump_i):
        """
        Returns dynamic (toe, camber, caster) for each wheel:
        - base static values per axle
        - plus kinematic/compliance deltas from roll, steer (via SAI), load, and bump
        """
        # Base static per axle
        if key in ('fl', 'fr'):
            toe0    = self.static_toe_front
            camber0 = self.static_camber_front
            caster0 = self.static_caster_front
        else:
            toe0    = self.static_toe_rear
            camber0 = self.static_camber_rear
            caster0 = self.static_caster_rear

        # Camber deltas: roll, steer (via SAI), load sensitivity, bump/jounce
        dcamber_roll  = self.camber_gain * roll
        dcamber_steer = self.k_sai_camber * steer_w
        dcamber_load  = self.k_camber_Fz * (Fz_i - (self.mass * 9.81 / 4.0))
        dcamber_bump  = self.k_camber_bump * bump_i

        # Toe deltas: roll-steer, bump-steer (compliance steer due to Fy is added later if desired)
        dtoe_roll     = self.k_toe_roll * roll
        dtoe_bump     = self.k_toe_bump * bump_i

        # Caster delta: small change with bump/jounce
        dcaster_bump  = self.k_caster_bump * bump_i

        camber = camber0 + dcamber_roll + dcamber_steer + dcamber_load + dcamber_bump
        toe    = toe0    + dtoe_roll    + dtoe_bump
        caster = caster0 + dcaster_bump

        return toe, camber, caster


    def compute_wheel_local_states(self, vx, vy, yaw_rate, steer):
        """
        Per-wheel local states with dynamic alignment:
        Returns dict per wheel:
            vwx, vwy      : wheel center velocities in body frame
            alpha         : effective slip angle including dynamic toe and steer
            camber        : dynamic camber (static + roll/steer/load/bump deltas)
            caster        : dynamic caster (static + bump delta)
            steer         : steering angle at the wheel (front wheels only)
            toe           : dynamic toe (static + roll/bump [+ optional compliance])
        """
        wheels = {}

        # Positions relative to CG (use same keys as self.Fz)
        positions = {
            'fl': self.tire_positions['front_left'],
            'fr': self.tire_positions['front_right'],
            'rl': self.tire_positions['rear_left'],
            'rr': self.tire_positions['rear_right'],
        }

        # Update loads first so Fz is fresh for alignment calculations
        self.update_wheel_loads(vx, vy, yaw_rate)

        for key, pos in positions.items():
            px = pos[0]  # longitudinal offset from CG (+ forward)
            py = pos[1]  # lateral offset from CG (+ left)

            # Wheel linear velocities in body frame (rigid-body kinematics)
            vwx = vx - yaw_rate * py
            vwy = vy + yaw_rate * px

            # Steering angle: front wheels steer; rears remain 0 unless you add rear-steer
            steer_w = steer if key in ('fl', 'fr') else 0.0

            # Estimate bump/jounce at the wheel from small-angle body rotations
            bump_i = self.estimate_wheel_bump(px, py, self.roll, self.pitch)

            # Dynamic alignment (toe, camber, caster)
            toe_i, camber_i, caster_i = self.compute_alignment_per_wheel(
                key=key,
                steer_w=steer_w,
                roll=self.roll,
                pitch=self.pitch,
                Fz_i=self.Fz[key],
                bump_i=bump_i
            )

            # Effective slip angle includes toe as a bias (preload) to steer
            # alpha = atan2(v_lat, v_long) - (steer + toe)
            alpha = math.atan2(vwy, max(1e-6, vwx)) - (steer_w + toe_i)

            wheels[key] = {
                'vwx': vwx,
                'vwy': vwy,
                'alpha': alpha,
                'camber': camber_i,
                'caster': caster_i,
                'steer': steer_w,
                'toe': toe_i
            }
        return wheels



    def compute_equivalent_stiffness(self):
        """
        Compute per-axle vertical loads (static + pitch + roll approx) and
        return effective cornering stiffness per axle (Cf_eq, Cr_eq, Cx_eq).
        """
        g = 9.81
        pitch_angle = self.pitch

        # 1) static axle loads (pitch only)
        Fz_front_static = self.mass * g * self.lr / (self.lf + self.lr)
        Fz_rear_static  = self.mass * g * self.lf / (self.lf + self.lr)

        # 2) pitch-induced dynamic load transfer (approx)
        delta_Fz_front_pitch = -self.mass * g * self.h_cg * math.sin(pitch_angle) * (self.lr / (self.lf + self.lr))
        delta_Fz_rear_pitch  = +self.mass * g * self.h_cg * math.sin(pitch_angle) * (self.lf / (self.lf + self.lr))

        # 3) roll-induced lateral load transfer (approx)
        # use current roll and lateral acceleration estimate (small-angle)
        # we will approximate ay here with vy * yaw_rate if available; if not, 0
        ay_est = getattr(self, 'vy', 0.0) * getattr(self, 'omega', 0.0)
        delta_Fz_roll = (self.mass * ay_est * self.h_cg) / max(self.track_width, 0.1)

        # Left/right distribution (useful if splitting per wheel later)
        # For coarse per-axle effective load, assume half of roll transfer affects each axle equally:
        delta_Fz_front = delta_Fz_front_pitch - 0.5 * delta_Fz_roll
        delta_Fz_rear  = delta_Fz_rear_pitch + 0.5 * delta_Fz_roll

        # total per-axle Fz
        Fz_front = Fz_front_static + delta_Fz_front
        Fz_rear  = Fz_rear_static  + delta_Fz_rear

        # Ensure non-negative
        Fz_front = max(1.0, Fz_front)
        Fz_rear  = max(1.0, Fz_rear)

        # 4) convert to equivalent cornering stiffness (axle-level)
        # Use Cf_axle/Cr_axle as base values scaled with load (rough linear scaling)
        Cf_eq = self.Cf_axle * (Fz_front / (self.mass * g / 2.0))  # normalized by nominal half-mass
        Cr_eq = self.Cr_axle * (Fz_rear  / (self.mass * g / 2.0))

        # longitudinal stiffness (approx)
        Cx_eq = (self.tyre_Bx * self.tyre_Cx) * (Fz_front + Fz_rear) / 2.0

        return Cf_eq, Cr_eq, Cx_eq



    def select_gear(self, v):
        """Automatic gear selection with hysteresis for I-Shift 12-speed."""
        # Upshift logic (v is in m/s, compare directly)
        if self.current_gear < len(self.upshift_speeds) and v >= self.upshift_speeds[self.current_gear]:
            self.current_gear += 1

        # Downshift logic
        elif self.current_gear > 0 and v <= self.downshift_speeds[self.current_gear - 1]:
            self.current_gear -= 1

        # Clamp gear index
        self.current_gear = max(0, min(self.current_gear, len(self.gear_ratios) - 1))


    def get_total_ratio(self):
        return self.gear_ratios[self.current_gear] * self.final_drive * self.efficiency

    def compute_drive_force(self, engine_torque, wheel_radius, speed):
        self.select_gear(speed)
        total_ratio = self.get_total_ratio()
        F_drive = (engine_torque * total_ratio) / wheel_radius
        return F_drive

    def compute_brake_force(self, brake_input, vx):
        """
        Compute total brake force from brake input signal.
        brake_input: 0-1 (0 = no braking, 1 = max braking)
        vx: longitudinal velocity [m/s]
        Returns total brake force [N] (positive value)
        """
        if vx < 1e-3:
            return 0.0
        # Size tractor brakes to be capable of 0.85g deceleration on its own mass
        brake_design_mu = 0.85
        max_brake_force = self.mass * 9.81 * brake_design_mu
        return brake_input * max_brake_force

    def distribute_brake_force(self, total_brake_force):
        """
        Distribute brake force across axles (front-rear split).
        Returns dict with brake forces per wheel: {'fl', 'fr', 'rl', 'rr'}
        """
        F_brake_front_axle = total_brake_force * self.brake_force_distribution_front
        F_brake_rear_axle = total_brake_force * self.brake_force_distribution_rear
        
        # Equal distribution left-right on each axle
        brake_forces = {
            'fl': F_brake_front_axle / 2.0,
            'fr': F_brake_front_axle / 2.0,
            'rl': F_brake_rear_axle / 2.0,
            'rr': F_brake_rear_axle / 2.0,
        }
        return brake_forces

    def compute_regenerative_braking_force(self, brake_input, vx):
        """
        Compute regenerative braking force from electric motor.
        Returns the braking force that the e-machine can provide.
        If speed is below regen_speed_limit, returns 0 (e-machine cannot regen at low speeds).
        """
        if vx < self.regen_speed_limit or brake_input < 1e-3:
            return 0.0
        
        # Maximum regenerative force available from e-machine
        max_regen_force = self.mass * self.regen_max_decel
        
        # Apply brake input proportionally (0-1)
        regen_force = brake_input * max_regen_force
        
        return regen_force

    def compute_mechanical_braking_force(self, total_brake_required, regen_force, vx):
        """
        Compute mechanical brake force needed after accounting for regenerative braking.
        If regen provides enough braking, mechanical brakes don't engage.
        Otherwise, mechanical brakes provide the difference.
        """
        # Mechanical brakes only engage if regen cannot provide all needed braking
        mechanical_brake_force = max(0.0, total_brake_required - regen_force)
        return mechanical_brake_force

    def distribute_braking_forces(self, total_brake_required, vx):
        """
        Smart braking distribution: regenerative braking first, then mechanical.
        Returns dict with per-wheel forces and regen/mechanical split info.
        """
        # The input 'total_brake_required' is already a force in Newtons
        # We need the original brake input (0-1) to pass to regen calculator
        # Compute brake input from force: brake_input = force / (mass * max_decel)
        if self.mass * self.max_brake_decel > 1e-6:
            brake_input_normalized = min(1.0, total_brake_required / (self.mass * self.max_brake_decel))
        else:
            brake_input_normalized = 0.0
        
        # Calculate regenerative braking (acts on driven wheels - rear axle)
        regen_force = self.compute_regenerative_braking_force(brake_input_normalized, vx)
        
        # Calculate mechanical brake force needed
        mechanical_force = self.compute_mechanical_braking_force(total_brake_required, regen_force, vx)
        
        # Distribute mechanical brake force across axles (front-rear split for mechanical brakes)
        F_mech_front = mechanical_force * self.brake_force_distribution_front
        F_mech_rear = mechanical_force * self.brake_force_distribution_rear
        
        # Regenerative force goes to rear axle (where drive wheels are)
        # Split regenerative force equally across rear wheels
        F_rear_total = (F_mech_rear + regen_force)
        
        brake_forces = {
            'fl': F_mech_front / 2.0,
            'fr': F_mech_front / 2.0,
            'rl': F_rear_total / 2.0,
            'rr': F_rear_total / 2.0,
        }
        
        return brake_forces, regen_force, mechanical_force

    def engine_torque_curve(self, rpm):
        """
        Volvo D13-inspired torque curve (tractor head).
        Peak ~2600 Nm near 1400 rpm, broad plateau 1100–1500 rpm,
        gentle rise from idle, and fall-off toward rated speed.
        """
        if rpm <= 0:
            return 0.0
        # Key waypoints (rpm : torque Nm)
        # Idle region (truck idle ~600-700 rpm; keep low torque)
        rpm_idle = 650
        T_idle  = 200.0
        # Build-up toward plateau
        rpm_rise_start = 700
        T_rise_start   = 300.0
        rpm_plateau_start = 1000
        T_plateau_low     = 2400.0  # conservative lower plateau
        rpm_peak = 1400
        T_peak   = 2600.0           # documented peak near 1400 rpm
        rpm_plateau_end = 1500
        T_plateau_high  = 2550.0    # slight dip after peak, still strong
        # Fall-off region toward rated speed (~2100–2300 rpm)
        rpm_fall_start = 1600
        T_fall_start   = 2400.0
        rpm_rated = 2200
        T_rated   = 1600.0
        rpm_cutoff = 2500
        T_cutoff   = 800.0
        r = float(rpm)
        # 1) Idle to rise start: interpolate idle torque
        if r < rpm_rise_start:
            # linear from (0, 0) to (idle, T_idle), then to rise_start
            if r <= rpm_idle:
                return T_idle * (r / rpm_idle)
            else:
                return T_idle + (T_rise_start - T_idle) * ((r - rpm_idle) / (rpm_rise_start - rpm_idle))

        # 2) Rise to plateau start
        if r < rpm_plateau_start:
            return T_rise_start + (T_plateau_low - T_rise_start) * ((r - rpm_rise_start) / (rpm_plateau_start - rpm_rise_start))

        # 3) Plateau start to peak
        if r < rpm_peak:
            return T_plateau_low + (T_peak - T_plateau_low) * ((r - rpm_plateau_start) / (rpm_peak - rpm_plateau_start))

        # 4) Peak to plateau end (slight dip)
        if r < rpm_plateau_end:
            return T_peak + (T_plateau_high - T_peak) * ((r - rpm_peak) / (rpm_plateau_end - rpm_peak))

        # 5) Plateau end to fall start
        if r < rpm_fall_start:
            return T_plateau_high + (T_fall_start - T_plateau_high) * ((r - rpm_plateau_end) / (rpm_fall_start - rpm_plateau_end))

        # 6) Fall to rated
        if r < rpm_rated:
            return T_fall_start + (T_rated - T_fall_start) * ((r - rpm_fall_start) / (rpm_rated - rpm_fall_start))

        # 7) Rated to cutoff
        if r < rpm_cutoff:
            return T_rated + (T_cutoff - T_rated) * ((r - rpm_rated) / (rpm_cutoff - rpm_rated))

        # 8) Beyond cutoff
        return max(0.0, T_cutoff - 2.0 * (r - rpm_cutoff))  # fade out


    def electric_torque(self, vx):
        """
        Electric motor torque assistance with smooth cutoff.
        Provides maximum torque at low speeds (good for startup), ramps down at high speed.
        
        FIXED: Now provides sufficient torque at vx=0 to prevent stalling.
        """
        import math
        limit = self.electric_speed_limit  # 10 m/s
        
        # CORRECTED: Provide MAXIMUM torque at low speeds (good for electric motors)
        # then ramp down as speed increases
        # Use exponential decay centered at the speed limit
        factor = 1.0 / (1.0 + math.exp((vx - limit) / 3.0))
        # At vx=0: factor ≈ 0.95 (nearly full!)
        # At vx=10: factor ≈ 0.5 (half power)
        # At vx=20: factor ≈ 0.05 (minimal)
        
        # Torque value: 800 Nm nominal (reduced from original 2000 to prevent acceleration explosion)
        electric_torque_max_limited = 800.0  # Nm
        return electric_torque_max_limited * factor * self.electric_efficiency


    def compute_engine_rpm(self, v, wheel_radius):
        omega_wheel = v / wheel_radius
        omega_engine = omega_wheel * self.get_total_ratio()
        rpm = omega_engine * 60 / (2 * np.pi)
        return rpm

    def compute_turn_radius(self, delta):
        if abs(delta) < 1e-3:
            return float('inf')
        return self.L / np.tan(delta)

    def ackermann_steering_angles(self, delta):
        R = self.compute_turn_radius(delta)
        if R == float('inf'):
            return delta, delta
        delta_inner = np.arctan(self.L / (R - self.track_width / 2))
        delta_outer = np.arctan(self.L / (R + self.track_width / 2))
        return (delta_inner, delta_outer) if delta > 0 else (delta_outer, delta_inner)
    
    def update_wheel_loads(self, vx, vy, yaw_rate):
        g = 9.81
        # static per-wheel (simple split)
        Fz_total = self.mass * g
        Fz_front_static = Fz_total * (self.lr / (self.lf + self.lr))
        Fz_rear_static  = Fz_total * (self.lf / (self.lf + self.lr))
        # per-wheel before transfer
        Fz_fl = Fz_fr = Fz_front_static / 2.0
        Fz_rl = Fz_rr = Fz_rear_static / 2.0
        # lateral load transfer approx
        ay = (vy * yaw_rate) if abs(vx) > 1e-3 else 0.0
        deltaFz_roll = (self.mass * ay * self.h_cg) / max(self.track_width, 0.1)
        # apply: unload left, load right (sign convention depending on your axes; adapt if needed)
        Fz_fl -= 0.5 * deltaFz_roll
        Fz_fr += 0.5 * deltaFz_roll
        Fz_rl -= 0.5 * deltaFz_roll
        Fz_rr += 0.5 * deltaFz_roll
        # clamp
        for k,v in [('fl',Fz_fl),('fr',Fz_fr),('rl',Fz_rl),('rr',Fz_rr)]:
            self.Fz[k] = max(50.0, v)   # avoid zero
    
    def tyre_lateral_force_enhanced(self, alpha, camber, Fz, mu):
        # load-sensitive cornering stiffness
        Fz_ref = (self.mass * 9.81) / 4.0
        n_load = 0.9
        C_alpha = (self.Cf_axle/2.0 if Fz>0 else 0.0) * (Fz / Fz_ref)**n_load  # per front wheel; for rear use self.Cr_axle/2

        # linear + camber thrust
        Fy_lin = C_alpha * alpha + (self.Cgamma_f * camber if C_alpha==self.Cf_axle/2.0 else self.Cgamma_r * camber)

        # friction saturation (pure lateral here; combine with Fx later)
        Fy_max = mu * Fz
        Fy = np.clip(Fy_lin, -Fy_max, Fy_max)
        return Fy

    def advance_tire_states(self, state_new, state_old, dt):
        """
        Explicitly update tire relaxation state after solver convergence.
        """
        if not ENABLE_TIRE_RELAXATION:
            return

        # Unpack states
        rx, ry, ryaw, vx, vy, omega = state_new[0:6]
        
        # Re-compute wheel states at converged solution
        Fz_local = self.compute_wheel_loads_local(vx, vy, omega)
        wheels = self.compute_wheel_local_states(vx, vy, omega, self.delta, Fz_local)
        
        for key, st in wheels.items():
            # Get Steady State forces
            tire_model = self.tire_front if key in ('fl', 'fr') else self.tire_rear
            Fx_drive_req = 0.0 # Simplified for update
            
            # Simple kappa est
            if abs(Fx_drive_req) > 1.0:
                 C_kappa_est = tire_model.get_longitudinal_stiffness(Fz_local[key], self.mu)
                 kappa_drive = np.clip(Fx_drive_req / max(C_kappa_est, 1.0), -0.3, 0.3)
            else:
                 kappa_drive = 0.0
            
            # Use stored kappa or estimate
            kappa_eff = st['kappa'] if abs(st['kappa']) > abs(kappa_drive) else kappa_drive
            
            Fx_ss, Fy_ss = tire_model.combined_slip(
                alpha=st['alpha'], kappa=kappa_eff, Fz=Fz_local[key], 
                gamma=st['camber'], mu=self.mu
            )
            # Negate Fy: Pacejka outputs in slip direction, but history stores
            # restoring force convention (matching compute_residual / genalpha)
            Fy_ss = -Fy_ss
            
            # Update history using relaxation dynamics
            if key == 'fl':
                self.Fx_fl_transient_hist, self.Fy_fl_transient_hist = \
                    self.tire_relaxation_front.compute_transient_forces(
                        Fx_ss, Fy_ss, self.Fx_fl_transient_hist, self.Fy_fl_transient_hist, vx, dt)
            elif key == 'fr':
                self.Fx_fr_transient_hist, self.Fy_fr_transient_hist = \
                    self.tire_relaxation_front.compute_transient_forces(
                        Fx_ss, Fy_ss, self.Fx_fr_transient_hist, self.Fy_fr_transient_hist, vx, dt)
            elif key == 'rl':
                self.Fx_rl_transient_hist, self.Fy_rl_transient_hist = \
                    self.tire_relaxation_rear.compute_transient_forces(
                        Fx_ss, Fy_ss, self.Fx_rl_transient_hist, self.Fy_rl_transient_hist, vx, dt)
            elif key == 'rr':
                self.Fx_rr_transient_hist, self.Fy_rr_transient_hist = \
                    self.tire_relaxation_rear.compute_transient_forces(
                        Fx_ss, Fy_ss, self.Fx_rr_transient_hist, self.Fy_rr_transient_hist, vx, dt)

    def pneumatic_trail(self, alpha):
        # decays with |alpha|
        return self.t_p0 / (1.0 + self.t_p_decay * abs(alpha))

    def aligning_moment(self, Fy, alpha, Fz, caster, sai):
        # pneumatic trail contribution
        t_p = self.pneumatic_trail(alpha)
        Mz_pneu = -t_p * Fy

        # mechanical trail + caster/SAI self-centering (very simplified)
        # projection of trail along ground wrt steer axis gives restoring torque
        t_mech = self.mechanical_trail
        Mz_mech = -t_mech * Fz * np.sin(caster + sai)

        return Mz_pneu + Mz_mech

    def compute_energy_balance(self, state_new, state_old, throttle, brake, delta, dt):
        """
        Compute complete energy balance for convergence checking.
        
        Energy balance: ΔE_kinetic + ΔE_potential = W_engine - W_dissipation
        
        Returns:
            dict with all energy components and the balance residual
        """
        # Unpack states
        x_old, y_old, yaw_old = state_old[0], state_old[1], state_old[2]
        vx_old, vy_old, omega_old = state_old[3], state_old[4], state_old[5]
        
        x_new, y_new, yaw_new = state_new[0], state_new[1], state_new[2]
        vx_new, vy_new, omega_new = state_new[3], state_new[4], state_new[5]
        
        # ========================================================================
        # KINETIC ENERGY: E_k = 0.5*m*(vx² + vy²) + 0.5*Izz*ω²
        # ========================================================================
        E_k_old = 0.5 * self.mass * (vx_old**2 + vy_old**2) + 0.5 * self.Iz * omega_old**2
        E_k_new = 0.5 * self.mass * (vx_new**2 + vy_new**2) + 0.5 * self.Iz * omega_new**2
        dE_kinetic = E_k_new - E_k_old
        
        # ========================================================================
        # POTENTIAL ENERGY (Springs): E_p = 0.5*K_φ*φ² + 0.5*K_θ*θ²
        # ========================================================================
        E_p_roll_old = 0.5 * self.K_phi * self.roll**2 if hasattr(self, 'roll') else 0.0
        E_p_pitch_old = 0.5 * self.K_theta * self.pitch_dyn**2 if hasattr(self, 'pitch_dyn') else 0.0
        E_p_spring_old = E_p_roll_old + E_p_pitch_old
        
        # Note: roll/pitch are updated before the Newton-Raphson loop,
        # so for convergence checking within the loop, we use the current values
        E_p_spring_new = E_p_spring_old  # Springs updated outside NR loop
        dE_spring = E_p_spring_new - E_p_spring_old
        
        # ========================================================================
        # POTENTIAL ENERGY (Gravity): E_g = m*g*z
        # ========================================================================
        z_old = get_terrain_elevation(x_old, y_old, self.waypoints)
        z_new = get_terrain_elevation(x_new, y_new, self.waypoints)
        E_g_old = self.mass * 9.81 * z_old
        E_g_new = self.mass * 9.81 * z_new
        dE_gravity = E_g_new - E_g_old
        
        # Total potential energy change
        dE_potential = dE_spring + dE_gravity
        
        # ========================================================================
        # WORK INPUT (Engine): W_engine = F_drive * vx * dt
        # ========================================================================
        # Compute drive force using engine model
        rpm = self.compute_engine_rpm(max(vx_new, 0.1), self.r_w)
        T_engine = throttle * self.engine_torque_curve(rpm)
        T_electric = throttle * self.electric_torque(vx_new)
        T_total = T_engine + T_electric
        F_drive = self.compute_drive_force(T_total, self.r_w, max(vx_new, 0.1))
        
        # Work = Force * displacement = Force * velocity * dt (power * dt)
        W_engine = F_drive * vx_new * dt
        
        # ========================================================================
        # DISSIPATION: Braking, aerodynamic drag, roll/pitch damping
        # ========================================================================
        # Braking dissipation
        F_brake = brake * self.mass * self.max_brake_decel
        W_brake = F_brake * abs(vx_new) * dt
        
        # Aerodynamic drag dissipation: F_drag = 0.5 * rho * Cd * A * v²
        rho = 1.225  # Air density
        F_drag = 0.5 * rho * self.Cd * self.A_front * vx_new**2
        W_drag = F_drag * abs(vx_new) * dt
        
        # Roll damping: P = C_φ * φ̇²
        roll_rate = self.roll_rate if hasattr(self, 'roll_rate') else 0.0
        W_roll_damping = self.C_phi * roll_rate**2 * dt
        
        # Pitch damping: P = C_θ * θ̇²
        pitch_rate = self.pitch_rate if hasattr(self, 'pitch_rate') else 0.0
        W_pitch_damping = self.C_theta * pitch_rate**2 * dt
        
        # Tire friction dissipation (simplified: lateral slip dissipation)
        # P_tire ≈ F_y * v_slip
        v_lateral = abs(vy_new)
        F_lateral_approx = self.mass * abs(vy_new * omega_new) if abs(vx_new) > 0.1 else 0.0
        W_tire_friction = F_lateral_approx * v_lateral * dt * 0.1  # Empirical factor
        
        # Total dissipation
        W_dissipation = W_brake + W_drag + W_roll_damping + W_pitch_damping + W_tire_friction
        
        # ========================================================================
        # ENERGY BALANCE RESIDUAL
        # ========================================================================
        # Energy balance: ΔE_kinetic + ΔE_potential = W_engine - W_dissipation
        energy_balance = dE_kinetic + dE_potential - W_engine + W_dissipation
        
        # Normalize by total energy scale for relative error
        E_total = abs(E_k_new) + abs(W_engine) + abs(W_dissipation) + 1e-10
        energy_balance_normalized = abs(energy_balance) / E_total
        
        return {
            'dE_kinetic': dE_kinetic,
            'dE_potential': dE_potential,
            'dE_gravity': dE_gravity,
            'dE_spring': dE_spring,
            'W_engine': W_engine,
            'W_dissipation': W_dissipation,
            'W_brake': W_brake,
            'W_drag': W_drag,
            'W_roll_damping': W_roll_damping,
            'W_pitch_damping': W_pitch_damping,
            'W_tire_friction': W_tire_friction,
            'energy_balance': energy_balance,
            'energy_balance_normalized': energy_balance_normalized,
            'E_kinetic': E_k_new
        }

    def compute_jacobian_diagnostics(self, J):
        """
        Compute Jacobian diagnostics for convergence and stability monitoring.
        
        Uses single SVD decomposition to efficiently compute:
        - Determinant (phase space volume change)
        - Condition number (numerical stability)
        - Information entropy (log|det J|)
        
        Returns:
            dict with all diagnostic metrics
        """
        # SVD decomposition: J = U @ diag(s) @ V^T
        # Computing singular values only is O(n³) for n×n matrix
        s = np.linalg.svd(J, compute_uv=False)
        
        # Avoid log(0) and division by zero
        s_safe = np.maximum(s, 1e-15)
        
        # Determinant = product of singular values
        # For dissipative systems: |det(J)| < 1 (phase space contracts)
        det_J = np.prod(s)
        
        # Condition number = ratio of largest to smallest singular value
        # High condition number indicates ill-conditioning (numerical instability)
        cond_J = s[0] / s_safe[-1]
        
        # Information entropy = log|det(J)| = sum of log(singular values)
        # Negative entropy → phase space contraction (dissipative)
        # Positive entropy → phase space expansion (potentially unstable)
        entropy_J = np.sum(np.log(s_safe))
        
        # Additional diagnostic: smallest singular value (nearness to singularity)
        min_sv = s[-1]
        
        # Entropy rate (per degree of freedom)
        n_dof = len(s)
        entropy_rate = entropy_J / n_dof
        
        return {
            'det': det_J,              # Phase space volume change
            'cond': cond_J,            # Numerical stability (lower is better)
            'entropy': entropy_J,      # Information entropy (log|det|)
            'entropy_rate': entropy_rate,  # Entropy per DOF
            'min_singular': min_sv,    # Nearness to singularity
            'max_singular': s[0],      # Largest singular value
            'is_stable': cond_J < 1e10 and min_sv > 1e-12,  # Quick stability check
        }

        
    def update(self, throttle, brake, delta, dt, follower_info=None):
        """Update vehicle state using implicit Euler with Newton-Raphson solver.
        
        CRITICAL FIX: Roll and alignment angles MUST be computed BEFORE the solver loop
        to ensure consistency during residual/Jacobian evaluations.
        
        Args:
            follower_info: HitchParams object with follower's target state (for implicit coupling)
                          or None if no trailer attached.
        """
        delta = np.clip(delta, -self.max_steer, self.max_steer)
        
        # Store follower info for use in compute_residual
        self.follower_info = follower_info
        
        # ========================================================================
        # STEP 1: Update terrain, roll, and alignment BEFORE solver iterations
        # ========================================================================
        # This ensures the residual/Jacobian use CONSISTENT roll values during
        # the entire Newton-Raphson solve, not stale values from previous timestep
        self.z = get_terrain_elevation(self.x, self.y, self.waypoints)
        
        # Update roll dynamics (use current velocity for lateral accel estimate)
        # Centripetal acceleration in body frame: ay = vx * omega (NOT vy*omega which is Coriolis)
        ay_est = (self.vx * self.omega) if abs(self.vx) > 1e-3 else 0.0
        M_roll = self.mass * ay_est * self.h_cg - self.K_phi * self.roll - self.C_phi * self.roll_rate
        self.roll_rate += (M_roll / self.Ixx) * dt
        self.roll += self.roll_rate * dt
        
        # Clamp roll to prevent numerical instability
        self.roll = np.clip(self.roll, -np.radians(15), np.radians(15))
        self.roll_rate = np.clip(self.roll_rate, -np.radians(60), np.radians(60))
        
        # Update total camber (includes roll effects)
        self.camber_front_total = self.static_camber_front + self.camber_gain * self.roll
        self.camber_rear_total  = self.static_camber_rear  + self.camber_gain * self.roll
        
        # Update equivalent stiffness
        Cf_eq, Cr_eq, Cx_eq = self.compute_equivalent_stiffness()
        self.c_f, self.c_r, self.c_x = Cf_eq, Cr_eq, Cx_eq

        # ========================================================================
        # Pitch Dynamics (brake dive / acceleration squat)
        # ========================================================================
        # Estimate longitudinal acceleration from velocity change
        # (Better approach: use force-based once available, but this gives reasonable results)
        if hasattr(self, 'vx_prev'):
            ax_est = (self.vx - self.vx_prev) / dt if dt > 1e-6 else 0.0
        else:
            ax_est = 0.0
        
        # Store for next timestep
        self.vx_prev = self.vx

        # Pitch moment from longitudinal acceleration
        M_pitch = self.mass * ax_est * self.h_cg

        # Anti-dive / Anti-squat geometry compensation
        # These suspension features resist pitch during braking/acceleration
        if ax_est < 0:  # Braking (negative acceleration)
            # Anti-dive reduces nose-down moment during braking
            M_anti = self.anti_dive * abs(ax_est) * self.mass * self.h_cg
            M_pitch -= M_anti
        else:  # Accelerating (positive acceleration)
            # Anti-squat reduces tail-down moment during acceleration
            M_anti = self.anti_squat * ax_est * self.mass * self.h_cg
            M_pitch -= M_anti

        # Suspension restoring moment (pitch stiffness and damping)
        M_pitch -= self.K_theta * self.pitch_dyn + self.C_theta * self.pitch_rate

        # Integrate pitch dynamics (Euler forward, same as roll)
        self.pitch_rate += (M_pitch / self.Iyy) * dt
        self.pitch_dyn += self.pitch_rate * dt

        # Clamp pitch to prevent numerical divergence
        self.pitch_dyn = np.clip(self.pitch_dyn, -np.radians(8), np.radians(8))
        self.pitch_rate = np.clip(self.pitch_rate, -np.radians(30), np.radians(30))

        # Total pitch = terrain slope + dynamic suspension response
        self.pitch_terrain = get_terrain_pitch(self.x, self.y, self.waypoints)
        self.pitch = self.pitch_terrain + self.pitch_dyn

        # ========================================================================
        # QUARTER-CAR VERTICAL DYNAMICS (per-corner spring-damper integration)
        # ========================================================================
        if ENABLE_QUARTER_CAR and self.quarter_car is not None:
            # Road input per corner from roll and pitch geometry
            pos = {
                'fl': ( self.lf,  self.track_width / 2),
                'fr': ( self.lf, -self.track_width / 2),
                'rl': (-self.lr,  self.track_width / 2),
                'rr': (-self.lr, -self.track_width / 2),
            }
            
            # Inertial load from lateral + longitudinal acceleration
            ay_qc = (self.vx * self.omega) if abs(self.vx) > 1e-3 else 0.0
            ax_qc = getattr(self, 'ax_prev', 0.0)
            
            for key, (px, py) in pos.items():
                # Road displacement at this corner from body roll and pitch
                z_road = self.roll * py + self.pitch_dyn * px
                
                # Inertial force on this corner's sprung mass
                # Lateral: increases outer load, decreases inner
                # Longitudinal: increases rear during accel, front during braking
                m_s = self.quarter_car[key].m_s
                F_inertial_lat = -m_s * ay_qc * (py / max(self.track_width, 0.1))
                F_inertial_lon = -m_s * ax_qc * (px / (self.lf + self.lr))
                F_inertial = F_inertial_lat + F_inertial_lon
                
                # Update quarter-car model
                susp_defl, Fz_contact, bump = self.quarter_car[key].update(z_road, F_inertial, dt)
                self._qc_Fz[key] = Fz_contact
                self._qc_bump[key] = bump

        # ========================================================================
        # STEP 2: Solve for new velocities/positions with FIXED roll
        # ========================================================================
        x_old = np.array([self.x, self.y, self.yaw, self.vx, self.vy, self.omega], dtype=float)
        
        # *** TRUST REGION METHOD: Use rectilinear motion as hard limit + starting reference ***
        # This provides a physically reasonable initial guess and prevents teleporting
        
        # Store old values
        vx_old = self.vx
        vy_old = self.vy
        omega_old = self.omega
        yaw_old = self.yaw
        
        # ========================================================================
        # HYBRID PREDICTOR: Koopman (if trained) or Newmark (fallback)
        # ========================================================================
        dt2 = dt * dt
        
        # Build state_old for Koopman (9D state vector)
        state_old = np.array([
            x_old[0], x_old[1], yaw_old,
            vx_old, vy_old, omega_old,
            self.ax, self.ay, self.alpha
        ], dtype=float)
        
        # Try Koopman prediction first (if trained)
        koopman_prediction = None
        if self.koopman.is_ready():
            koopman_prediction = self.koopman.predict(state_old)
        
        if koopman_prediction is not None:
            # Use Koopman prediction (fast linear approximation)
            x_guess = koopman_prediction[0]
            y_guess = koopman_prediction[1]
            yaw_guess = np.arctan2(np.sin(koopman_prediction[2]), np.cos(koopman_prediction[2]))
            vx_guess = max(0.0, koopman_prediction[3])
            vy_guess = koopman_prediction[4]
            omega_guess = koopman_prediction[5]
            ax_guess = koopman_prediction[6]
            ay_guess = koopman_prediction[7]
            alpha_guess = koopman_prediction[8]
        else:
            # Fallback to Newmark predictor (2nd-order accurate)
            vx_guess = max(0.0, vx_old + dt * self.ax)
            vy_guess = vy_old + dt * self.ay
            omega_guess = omega_old + dt * self.alpha
            
            yaw_guess = yaw_old + dt * omega_old + 0.5 * dt2 * self.alpha
            yaw_guess = np.arctan2(np.sin(yaw_guess), np.cos(yaw_guess))
            
            cos_yaw = np.cos(yaw_old)
            sin_yaw = np.sin(yaw_old)
            vx_global = vx_old * cos_yaw - vy_old * sin_yaw
            vy_global = vx_old * sin_yaw + vy_old * cos_yaw
            ax_global = self.ax * cos_yaw - self.ay * sin_yaw
            ay_global = self.ax * sin_yaw + self.ay * cos_yaw
            
            x_guess = x_old[0] + dt * vx_global + 0.5 * dt2 * ax_global
            y_guess = x_old[1] + dt * vy_global + 0.5 * dt2 * ay_global
            
            ax_guess = self.ax
            ay_guess = self.ay
            alpha_guess = self.alpha
        
        # Newton-Raphson settings
        max_iterations = MAX_ITERATIONS_TRACTOR
        
        # ========================================================================
        # GENERALIZED-ALPHA SOLVER (Pure implementation - no legacy fallback)
        # ========================================================================
        # State vector: [x, y, yaw, vx, vy, omega, ax, ay, alpha]
        
        if not USE_GENERALIZED_ALPHA:
            raise RuntimeError("Legacy Backward Euler solver has been removed. Set USE_GENERALIZED_ALPHA = True")
        
        if USE_GENERALIZED_ALPHA:
            # ================================================================
            # GENERALIZED-ALPHA SOLVER (2nd-order accurate)
            # ================================================================
            # State vector: [x, y, yaw, vx, vy, omega, ax, ay, alpha]
            
            # Build old state with accelerations
            state_old = np.array([
                self.x, self.y, self.yaw,
                vx_old, vy_old, omega_old,
                self.ax, self.ay, self.alpha
            ], dtype=float)
            
            # Initial guess: use predictor for positions/velocities, zero acceleration change
            ax_guess = self.ax
            ay_guess = self.ay
            alpha_guess = self.alpha
            
            state_new = np.array([
                x_guess, y_guess, yaw_guess,
                vx_guess, vy_guess, omega_guess,
                ax_guess, ay_guess, alpha_guess
            ], dtype=float)
            
            # RELAXED CONVERGENCE during startup
            if hasattr(self, '_sim_time') and self._sim_time < 5.0:
                residual_threshold = RESIDUAL_THRESHOLD_STARTUP
            else:
                residual_threshold = RESIDUAL_THRESHOLD_STEADY
            
            state_change_threshold = 1e-7
            energy_balance_threshold = 1e-4  # Full energy balance convergence
            it = 0
            residual_norm = float('inf')
            solver_converged = False
            
            # Energy balance tracking for convergence
            energy_balance_prev = 0.0
            
            while it < max_iterations:
                f = self.compute_residual_genalpha(state_new, state_old, throttle, brake, delta, dt)
                residual_norm = np.linalg.norm(f)
                
                # Compute full energy balance
                energy_data = self.compute_energy_balance(state_new, state_old, throttle, brake, delta, dt)
                energy_balance_norm = energy_data['energy_balance_normalized']
                
                # Check residual convergence (primary)
                if residual_norm < residual_threshold:
                    solver_converged = True
                    break
                
                J = self.compute_jacobian_genalpha(state_new, state_old, throttle, brake, delta, dt)
                
                # Compute Jacobian diagnostics (information entropy, condition, determinant)
                jac_diag = self.compute_jacobian_diagnostics(J)
                
                try:
                    dx = np.linalg.solve(J, -f)
                except np.linalg.LinAlgError:
                    dx = np.linalg.lstsq(J, -f, rcond=None)[0]
                
                # Update uncertainty from solver step (Bayesian update)
                self.uncertainty.update_from_solver(J, residual_norm)
                uncertainty_trace = self.uncertainty.get_trace()
                uncertainty_entropy = self.uncertainty.compute_entropy()
                
                # Check state change convergence (secondary)
                state_change = np.linalg.norm(dx) / (np.linalg.norm(state_new) + 1e-6)
                if state_change < state_change_threshold and residual_norm < residual_threshold * 10:
                    solver_converged = True
                    break
                
                # Combined convergence check (tertiary): energy + entropy + uncertainty
                # Convergence when:
                # 1. Energy balance is small (physics satisfied)
                # 2. Jacobian is well-conditioned (numerically stable)
                # 3. Jacobian entropy is bounded (phase space not exploding)
                # 4. Uncertainty is low (probabilistic numerics)
                if it > 0:
                    combined_ok = (
                        energy_balance_norm < energy_balance_threshold and
                        jac_diag['is_stable'] and
                        jac_diag['entropy'] < 10.0 and  # Bounded Jacobian entropy
                        uncertainty_trace < 1e-6        # Low uncertainty (converged)
                    )
                    if combined_ok and residual_norm < residual_threshold * 20:
                        solver_converged = True
                        break
                
                # Limit step size to prevent divergence
                dx_norm = np.linalg.norm(dx)
                if dx_norm > 30.0:
                    dx = dx * (30.0 / dx_norm)
                
                # Simple line search
                step = 1.0
                for ls_iter in range(10):
                    state_trial = state_new + step * dx
                    f_trial = self.compute_residual_genalpha(state_trial, state_old, throttle, brake, delta, dt)
                    if np.linalg.norm(f_trial) < 0.95 * residual_norm or step < 0.01:
                        state_new = state_trial
                        break
                    step *= 0.5
                
                it += 1
            
            # Commit new state
            self.x, self.y, self.yaw = float(state_new[0]), float(state_new[1]), float(state_new[2])
            self.vx, self.vy, self.omega = float(state_new[3]), float(state_new[4]), float(state_new[5])
            self.ax, self.ay, self.alpha = float(state_new[6]), float(state_new[7]), float(state_new[8])
            
            # Update ax_prev/ay_prev/alpha_prev for compatibility with other code
            self.ax_prev = self.ax
            self.ay_prev = self.ay
            self.alpha_prev = self.alpha
            
            # ================================================================
            # KOOPMAN TRAINING: Add converged (state_old, state_new) pair
            # ================================================================
            # This allows Koopman to learn the nonlinear dynamics online
            self.koopman.add_sample(state_old, state_new)

        # ==================================================================
        # POST-PROCESSING
        # ==================================================================
        # guards
        self.vx = max(self.vx, 0.0)
        
        # Limit extreme yaw rates only as last resort (vehicle in uncontrolled spin)
        # Normal vehicles: ω < 3 rad/s; Race cars in drift: ω < 6 rad/s; Extreme spin: ω < 10 rad/s
        # Allow up to 10 rad/s (≈57°/s) for realistic simulations
        max_yaw_rate = 10.0  # rad/s - realistic upper bound even for extreme maneuvers
        if abs(self.omega) > max_yaw_rate:
            self.omega = np.sign(self.omega) * max_yaw_rate
        
        # CRITICAL FIX: Progressive low-speed stabilization to prevent spin
        # At low speeds, lateral dynamics should be heavily damped to prevent instability
        if self.vx < 1.0:  # Increased from 1e-3 to 1.0 m/s for better stability
            # Progressive damping: stronger at lower speeds
            # At vx=0: 100% damping (vy=0, omega=0)
            # At vx=1: 0% damping (full dynamics)
            damping_factor = self.vx  # 0 to 1 range
            
            # Apply progressive damping to lateral velocity and yaw rate
            self.vy = self.vy * damping_factor
            self.omega = self.omega * damping_factor
            
            # Hard limits at very low speeds to prevent numerical explosion
            if self.vx < 0.01:
                self.vy = 0.0
                self.omega = 0.0
        
        # Output convergence error at every timestep
        print(f"[SOLVER] iter={it} converged={solver_converged} max_error={residual_norm:.2e} vx={self.vx:.2f} ax_implicit={(self.vx-vx_old)/dt:.2f}")
        
        # CRITICAL: If solver fails to converge, this state is INVALID
        # Return convergence status so caller can decide whether to abort
        if not solver_converged:
            print(f"[SOLVER_FAIL] WARNING: Solver did not converge! Error={residual_norm:.2e} > {residual_threshold:.2e}")
            print(f"[SOLVER_FAIL] State may be invalid: vx={self.vx:.2f} vy={self.vy:.2f} omega={self.omega:.3f}")
        
        # DIAGNOSTIC: Final state summary at startup (every 0.1s)
        if hasattr(self, '_sim_time') and self._sim_time < 3.0:
            if hasattr(self, '_diagnostic_counter') and self._diagnostic_counter % 1000 == 0:  # Every 1000 calls = 0.1s
                print(f"[STATE_SUMMARY] t={self._sim_time:.3f}s converged={solver_converged} vx={self.vx:.2f} vy={self.vy:.3f} omega={self.omega:.3f} ax={self.ax_prev:.2f} ay={self.ay_prev:.2f}")
        
        return solver_converged


    
    def compute_wheel_local_states(self, vx, vy, yaw_rate, delta, Fz_local):
        """
        Compute per-wheel local states including slip angle (α) and slip ratio (κ).
        
        Returns dict per wheel with:
            vwx, vwy  : Wheel center velocities in body frame
            alpha     : Slip angle [rad]
            kappa     : Longitudinal slip ratio [-] (for combined-slip Pacejka)
            camber    : Dynamic camber angle [rad]
            caster    : Dynamic caster angle [rad]
            steer     : Steering angle at the wheel [rad]
            toe       : Dynamic toe angle [rad]
        """
        wheels = {}
        positions = {
            'fl': self.tire_positions['front_left'],
            'fr': self.tire_positions['front_right'],
            'rl': self.tire_positions['rear_left'],
            'rr': self.tire_positions['rear_right'],
        }

        # Ackermann split for front wheels
        delta_fl, delta_fr = self.ackermann_steering_angles(delta)

        for key, pos in positions.items():
            px, py = float(pos[0]), float(pos[1])
            vwx = vx - yaw_rate * py
            vwy = vy + yaw_rate * px

            if key == 'fl':
                steer_w = delta_fl
            elif key == 'fr':
                steer_w = delta_fr
            else:
                steer_w = 0.0

            bump_i = self.estimate_wheel_bump(px, py, self.roll, self.pitch)
            toe_i, camber_i, caster_i = self.compute_alignment_per_wheel(
                key=key, steer_w=steer_w, roll=self.roll, pitch=self.pitch,
                Fz_i=Fz_local[key], bump_i=bump_i
            )

            # Slip angle calculation with low-speed regularization
            vwx_safe = max(abs(vwx), 0.1)  # Prevent division by zero
            alpha = math.atan2(vwy, vwx_safe) - (steer_w + toe_i)
            
            # =================================================================
            # SLIP RATIO (kappa) CALCULATION - Required for combined-slip Pacejka
            # =================================================================
            # κ = (ω*r - vx) / |vx|  where ω is wheel angular velocity
            # Positive κ: driving (wheel spinning faster than ground speed)
            # Negative κ: braking (wheel spinning slower than ground speed)
            omega_wheel = self.wheel_omega.get(key, 0.0)  # Wheel angular velocity [rad/s]
            v_wheel_ground = omega_wheel * self.r_w       # Wheel surface velocity [m/s]
            
            if abs(vwx) > 0.1:  # Only compute kappa when moving
                kappa = (v_wheel_ground - vwx) / abs(vwx)
            else:
                # At very low speeds, assume no slip (quasi-static)
                kappa = 0.0
            
            # Clamp kappa to physical limits (-1 = locked wheel, +1 = full spin)
            kappa = np.clip(kappa, -1.0, 1.0)
            
            # Store kappa on self for ABS controller access
            if not hasattr(self, '_wheel_kappas'):
                self._wheel_kappas = {}
            self._wheel_kappas[key] = kappa
            
            wheels[key] = {
                'vwx': vwx, 'vwy': vwy,
                'alpha': alpha,
                'kappa': kappa,  # NEW: slip ratio for combined-slip Pacejka
                'camber': camber_i,
                'caster': caster_i,
                'steer':  steer_w,
                'toe':    toe_i
            }
        return wheels

        
    def smooth_saturate_force(self, force_linear, force_max, saturation_sharpness=2.0):
        """
        UNIFIED SATURATION FUNCTION: Used for both Fx and Fy
        
        Applies smooth saturation to any force using tanh function.
        Ensures: force ∈ [-force_max, force_max] with continuous derivatives everywhere.
        
        Args:
            force_linear: Unsaturated force value (can be ±∞)
            force_max: Maximum allowable force (friction limit, typically mu*Fz)
            saturation_sharpness: Controls how quickly saturation occurs
                                 (higher = sharper transition, but still smooth)
        Returns:
            Saturated force ∈ [-force_max, force_max], smooth and differentiable
        """
        if force_max < 1e-6:
            return 0.0
        
        # Map force to [-1, 1] range, then scale back to [-force_max, force_max]
        # tanh ensures smooth saturation with continuous derivatives
        normalized = float(force_linear) / force_max
        saturated_normalized = np.tanh(normalized * saturation_sharpness)
        return float(force_max * saturated_normalized)

    def tyre_lateral_force(self, C_alpha_nom, alpha, camber, Fz, mu, Fz_ref, n_load, C_gamma, sharpness=2.0):
        """
        Tire lateral force with smooth saturation at friction limit.
        
        Two-stage smooth saturation:
        1. Input compression: Slip angle saturated to prevent extreme angles
        2. Output clamping: Force saturated to friction limit
        
        Both stages smooth with continuous derivatives → Newton-Raphson convergence.
        """
        Fz_eff = max(1.0, float(Fz))
        C_alpha = C_alpha_nom * (Fz_eff / Fz_ref)**n_load
        
        # ===== STAGE 1: Compress extreme slip angles smoothly =====
        # Real tires show nonlinear response at extreme slip angles (>20°)
        # Compress large angles progressively to prevent model saturation
        alpha_float = float(alpha)
        k_slip = 0.3  # Compression rate (tunable)
        
        # Smooth compression: multiply by tanh(k*|alpha|) to compress extreme angles
        # Example: |alpha| = 10 rad → compressed to ~1.5 rad (smoother response)
        if abs(alpha_float) > 1e-6:
            alpha_compressed = alpha_float * np.tanh(k_slip * abs(alpha_float)) / abs(alpha_float)
        else:
            alpha_compressed = alpha_float
        
        # ===== STAGE 2: Compute force with compressed slip angle =====
        Fy_linear = C_alpha * alpha_compressed + C_gamma * float(camber)
        
        # ===== STAGE 3: Apply unified smooth saturation =====
        Fy_max = mu * Fz_eff
        Fy = self.smooth_saturate_force(Fy_linear, Fy_max, sharpness)
        
        return float(Fy)


    def smooth_saturate_force(self, F_req, F_max, sharpness=2.0):
        """
        Smoothly saturate force to max limit using tanh.
        
        Parameters:
        -----------
        F_req : float
            Requested force
        F_max : float
            Maximum allowable force
        sharpness : float
            Controls transition steepness (higher = sharper)
            
        Returns:
        --------
        float : Saturated force
        """
        if abs(F_max) < 1e-6:
            return 0.0
        
        # Normalize request by limit
        ratio = F_req / F_max
        
        # Apply smooth saturation: F = F_max * tanh(sharpness * ratio) / tanh(sharpness)
        # This gives smooth approach to ±F_max while maintaining sign
        saturated_ratio = np.tanh(sharpness * ratio) / np.tanh(sharpness)
        
        return F_max * saturated_ratio


    def apply_friction_ellipse(self, Fx_req, Fy_req, Fz, mu):
        """
        Apply friction ellipse constraint to BOTH Fx and Fy together (coupled saturation).
        
        CRITICAL FIX: Previous implementation only limited Fx based on Fy, but LEFT Fy UNLIMITED!
        This caused massive lateral forces (379 kN) that broke solver convergence.
        
        PHYSICS: Tire forces must satisfy: sqrt(Fx² + Fy²) ≤ mu*Fz
                 When combined force exceeds friction limit, scale BOTH proportionally.
        
        Args:
            Fx_req: Requested longitudinal force [N]
            Fy_req: Requested lateral force [N] (from tyre_lateral_force, may be huge!)
            Fz: Normal load [N]
            mu: Friction coefficient
            
        Returns:
            (Fx_saturated, Fy_saturated): Both forces limited to friction circle
        """
        F_lim = mu * max(1.0, float(Fz))
        Fx_float = float(Fx_req)
        Fy_float = float(Fy_req)
        
        # Compute total force magnitude
        F_total = np.sqrt(Fx_float**2 + Fy_float**2)
        
        # If within friction circle, return as-is
        if F_total <= F_lim:
            return Fx_float, Fy_float
        
        # Outside friction circle: scale both forces proportionally to lie ON the circle
        # This maintains the direction but limits magnitude
        scale = F_lim / max(F_total, 1e-6)
        
        Fx_saturated = Fx_float * scale
        Fy_saturated = Fy_float * scale
        
        return float(Fx_saturated), float(Fy_saturated)


    
    def compute_wheel_loads_local(self, vx, vy, yaw_rate):
        g = 9.81
        Fz_total = self.mass * g
        L = self.lf + self.lr
        Fz_front_static = Fz_total * (self.lr / L)
        Fz_rear_static  = Fz_total * (self.lf / L)
        Fz_fl = Fz_fr = Fz_front_static / 2.0
        Fz_rl = Fz_rr = Fz_rear_static  / 2.0

        # Lateral load transfer (from body-frame lateral acceleration)
        # FIX: Use cached ay from force balance (computed in residual) instead of kinematic approximation.
        # On first iteration, ay_cache will be 0 (use kinematic). On subsequent iterations, use force-based ay.
        # This breaks the circular dependency while maintaining accuracy as solver converges.
        ay_cached = getattr(self, '_ay_for_load_transfer', 0.0)
        # Fallback to kinematic approximation if cache is zero or undefined
        ay = ay_cached if abs(ay_cached) > 1e-6 else ((vx * yaw_rate) if abs(vx) > 1e-3 else 0.0)
        deltaFz_roll = (self.mass * ay * self.h_cg) / max(self.track_width, 0.1)

        Fz_fl -= 0.5 * deltaFz_roll
        Fz_fr += 0.5 * deltaFz_roll
        Fz_rl -= 0.5 * deltaFz_roll
        Fz_rr += 0.5 * deltaFz_roll

        # ===============================================================
        # LONGITUDINAL LOAD TRANSFER: ΔFz = m·ax·h_cg / L
        # ===============================================================
        # Braking (ax < 0) → front loads up, rear unloads
        # Acceleration (ax > 0) → rear loads up, front unloads
        ax = getattr(self, 'ax_prev', 0.0)
        deltaFz_pitch = (self.mass * ax * self.h_cg) / L
        
        Fz_fl -= 0.5 * deltaFz_pitch  # front unloads during accel
        Fz_fr -= 0.5 * deltaFz_pitch
        Fz_rl += 0.5 * deltaFz_pitch  # rear loads up during accel
        Fz_rr += 0.5 * deltaFz_pitch

        # If quarter-car is enabled, override with dynamic contact forces
        if ENABLE_QUARTER_CAR and self.quarter_car is not None:
            fz = {
                'fl': max(50.0, self._qc_Fz.get('fl', Fz_fl)),
                'fr': max(50.0, self._qc_Fz.get('fr', Fz_fr)),
                'rl': max(50.0, self._qc_Fz.get('rl', Fz_rl)),
                'rr': max(50.0, self._qc_Fz.get('rr', Fz_rr)),
            }
            return fz

        # clamp
        fz = {
            'fl': max(50.0, Fz_fl),
            'fr': max(50.0, Fz_fr),
            'rl': max(50.0, Fz_rl),
            'rr': max(50.0, Fz_rr)
        }
        return fz


    
    def compute_residual(self, x_new, x_old, throttle, brake, delta, dt):
        """
        Implicit residuals for position, heading, and velocities using
        two-track tire forces, dynamic alignment (toe/camber/caster), and
        self-aligning torque (pneumatic + mechanical trail).
        """
        # -------------------------
        # Unpack states (candidate)
        # -------------------------
        rx, ry, ryaw, vx, vy, omega = x_new
        rx_old, ry_old, ryaw_old, vx_old, vy_old, omega_old = x_old

        # -------------------------
        # 1) Kinematic residuals
        # -------------------------
        # Implicit Euler with candidate velocities/orientation
        res_x   = rx   - rx_old   - dt * (vx * np.cos(ryaw) - vy * np.sin(ryaw))
        res_y   = ry   - ry_old   - dt * (vx * np.sin(ryaw) + vy * np.cos(ryaw))
        res_yaw = ryaw - ryaw_old - dt * omega

        # -------------------------
        # 2) Tire/wheel states
        # -------------------------
        # Fresh wheel loads for this candidate (keeps consistency)
        Fz_local = self.compute_wheel_loads_local(vx, vy, omega)

        # Per-wheel local states (includes dynamic toe/camber/caster)
        wheels = self.compute_wheel_local_states(vx, vy, omega, delta, Fz_local)

        # -------------------------
        # 3) Longitudinal resistances and drive
        # -------------------------
        vehicle_components = [{
            'Cd':   self.Cd,
            'A':    self.A_front,
            'mass': self.mass,
            'Crr':  self.Crr
        }]

        # FIX #1: Use regularized velocity for resistance calculations to avoid singularities
        # At very low speeds, aero/rolling resistance should be near zero anyway
        vx_safe = max(abs(vx), 1e-2)  # Regularize velocity for resistance calculations
        
        # Aerodynamic + rolling resistance (positive against motion)
        F_load = compute_aero_and_rolling_resistance(vx_safe, vehicle_components)
        # Engine & driveline
        rpm      = self.compute_engine_rpm(vx, self.r_w)
        # Only apply electric torque when throttle is active, not during braking
        T_engine = throttle * self.engine_torque_curve(rpm) + throttle * self.electric_torque(vx)
        
        # TORQUE CONVERTER: modify engine torque before drive force
        if ENABLE_TORQUE_CONVERTER and self.torque_converter is not None:
            omega_engine = rpm * 2.0 * np.pi / 60.0
            total_ratio = self.get_total_ratio()
            omega_gearbox = abs(vx) / max(self.r_w, 0.01) * total_ratio
            T_engine = self.torque_converter.compute(T_engine, omega_engine, omega_gearbox)
        
        F_drive  = self.compute_drive_force(T_engine, self.r_w, vx)
        
        # Debug: check throttle and drive force
        if brake > 0.01 or throttle > 0.01:
            print(f"[DRIVE] throttle={throttle:.3f} T_engine={T_engine:.1f} F_drive={F_drive:.1f}")

        # PNEUMATIC BRAKE DELAY: filter brake input
        brake_effective = brake
        if ENABLE_BRAKE_DELAY and self.brake_system is not None:
            brake_effective = self.brake_system.update(brake, dt)
        
        # Braking system with regenerative braking (using delayed brake)
        F_brake_total = self.compute_brake_force(brake_effective, vx)
        brake_forces, regen_force, mechanical_force = self.distribute_braking_forces(F_brake_total, vx)
        
        # ABS: modulate per-wheel brake forces based on slip ratios
        abs_active_any = False
        if ENABLE_ABS and self.abs_controller is not None and brake > 0.01:
            wheel_kappas = getattr(self, '_wheel_kappas', {k: 0.0 for k in ['fl','fr','rl','rr']})
            brake_forces, abs_active = self.abs_controller.modulate(brake_forces, wheel_kappas, vx, dt)
            abs_active_any = any(abs_active.values())
            if abs_active_any:
                diag = self.abs_controller.get_diagnostics()
                pf = diag['pressure_factor']
                print(f"[ABS] ACTIVE pf_fl={pf['fl']:.2f} pf_fr={pf['fr']:.2f} pf_rl={pf['rl']:.2f} pf_rr={pf['rr']:.2f}")
        elif ENABLE_ABS and self.abs_controller is not None and brake <= 0.01:
            # No braking → reset ABS state for clean re-engagement
            self.abs_controller.reset()
        
        # Debug: trace brake signal path
        if brake > 0.01 or F_brake_total > 100:
            print(f"[BRAKE] input={brake:.3f} total_force={F_brake_total:.1f} regen={regen_force:.1f} mech={mechanical_force:.1f} abs={abs_active_any}")

        # Distribute tractive effort across driven wheels
        # DIFFERENTIAL: distribute torque based on wheel speed difference
        if ENABLE_LSD and self.differential is not None:
            omega_rl = self.wheel_omega.get('rl', 0.0)
            omega_rr = self.wheel_omega.get('rr', 0.0)
            F_left, F_right = self.differential.distribute(F_drive, omega_rl, omega_rr)
            drive_forces_dict = {'fl': 0.0, 'fr': 0.0, 'rl': F_left, 'rr': F_right}
            Fx_drive_per_wheel = F_drive / max(len(self.driven_wheels), 1)
        else:
            Fx_drive_per_wheel = 0.0
            if len(self.driven_wheels) > 0:
                Fx_drive_per_wheel = F_drive / float(len(self.driven_wheels))
            drive_forces_dict = {k: 0.0 for k in ['fl', 'fr', 'rl', 'rr']}
            for dw in self.driven_wheels:
                drive_forces_dict[dw] = Fx_drive_per_wheel

        
        tcs_active_any = False
        if ENABLE_TCS and self.tcs_controller is not None and throttle > 0.01:
            wheel_kappas = getattr(self, '_wheel_kappas', {k: 0.0 for k in ['fl','fr','rl','rr']})
            drive_forces_dict, tcs_active = self.tcs_controller.modulate(
                drive_forces_dict, wheel_kappas, vx, dt, self.driven_wheels
            )
            tcs_active_any = any(tcs_active.values())
            if tcs_active_any:
                diag = self.tcs_controller.get_diagnostics()
                tf = diag['torque_factor']
                print(f"[TCS] ACTIVE tf_rl={tf['rl']:.2f} tf_rr={tf['rr']:.2f}")
        elif ENABLE_TCS and self.tcs_controller is not None and throttle <= 0.01:
            self.tcs_controller.reset()
        
        # ESC: compute corrective braking forces
        esc_forces = {k: 0.0 for k in ['fl', 'fr', 'rl', 'rr']}
        if ENABLE_ESC and self.esc_controller is not None:
            esc_forces = self.esc_controller.compute(vx, omega, delta, dt)
            if self.esc_controller.mode != 'INACTIVE':
                diag = self.esc_controller.get_diagnostics()
                print(f"[ESC] {diag['mode']} omega_err={diag['omega_error']:.3f} forces={diag['corrective_forces']}")

        # RSC: Roll Stability Control (overrides ESC when rollover risk is higher)
        rsc_throttle_factor = 1.0
        rsc_forces = {k: 0.0 for k in ['fl', 'fr', 'rl', 'rr']}
        if ENABLE_RSC and self.rsc_controller is not None:
            ay_rsc = (vx * omega) if abs(vx) > 1e-3 else 0.0
            roll_rsc = getattr(self, 'roll', 0.0)
            roll_rate_rsc = getattr(self, 'roll_rate', 0.0)
            rsc_throttle_factor, rsc_forces = self.rsc_controller.compute(
                ay_rsc, roll_rsc, roll_rate_rsc, vx, dt
            )
            if self.rsc_controller.mode != 'INACTIVE':
                # Apply RSC throttle cut to drive forces
                for k in drive_forces_dict:
                    drive_forces_dict[k] *= rsc_throttle_factor
                # Add RSC corrective braking
                for k in rsc_forces:
                    esc_forces[k] = esc_forces.get(k, 0.0) + rsc_forces.get(k, 0.0)
                diag_rsc = self.rsc_controller.get_diagnostics()
                print(f"[RSC] {diag_rsc['mode']} throttle_factor={diag_rsc['throttle_factor']:.2f}")

        # -------------------------
        # 4) Per-wheel forces/moments (two-track)
        # -------------------------
        # Reference load (per wheel) for stiffness scaling
        Fz_ref = (self.mass * 9.81) / 4.0
        n_load = 0.9  # cornering stiffness load exponent (tunable)

        sum_Fx = 0.0
        sum_Fy = 0.0
        sum_Mz = 0.0

        # Access wheel lever arms
        pos_map = {
            'fl': self.tire_positions['front_left'],
            'fr': self.tire_positions['front_right'],
            'rl': self.tire_positions['rear_left'],
            'rr': self.tire_positions['rear_right'],
        }

        # DIAGNOSTIC: Log per-wheel data during startup (first 3 seconds)
        diagnostic_startup = hasattr(self, '_sim_time') and self._sim_time < 3.0
        
        for key, st in wheels.items():
            # Stiffness & camber-thrust coefficient per axle
            
            # ... inside the wheel loop in compute_residual
            # make stiffness & camber coeff
            if key in ('fl', 'fr'):
                C_alpha_nom = self.Cf_axle / 2.0
                C_gamma     = self.Cgamma_f
                sai_i       = self.SAI
            else:
                C_alpha_nom = self.Cr_axle / 2.0
                C_gamma     = self.Cgamma_r
                sai_i       = 0.0

            alpha_i  = st['alpha']
            kappa_i  = st['kappa']  # NEW: slip ratio for combined-slip Pacejka
            camber_i = st['camber']
            caster_i = st['caster']
            steer_i  = st['steer']
            Fz_i     = Fz_local[key]

            # =================================================================
            # COMBINED-SLIP PACEJKA TIRE MODEL
            # =================================================================
            # Select tire model based on axle
            tire_model = self.tire_front if key in ('fl', 'fr') else self.tire_rear
            sai_i = self.SAI if key in ('fl', 'fr') else 0.0
            
            # MU-MAP + HYDROPLANING: compute per-wheel friction coefficient
            mu_wheel = self.mu
            if ENABLE_MU_MAP and self.road_friction is not None:
                # Compute wheel world position
                cos_yaw = np.cos(x_new[2])
                sin_yaw = np.sin(x_new[2])
                wp = pos_map[key]
                wx = x_new[0] + wp[0] * cos_yaw - wp[1] * sin_yaw
                wy = x_new[1] + wp[0] * sin_yaw + wp[1] * cos_yaw
                mu_wheel = self.road_friction.get_mu(wx, wy)
                
                # HYDROPLANING: further reduce μ based on speed and water depth
                if ENABLE_HYDROPLANING and self.hydroplaning is not None:
                    water_depth = self.road_friction.get_water_depth(wx, wy)
                    if water_depth > 0.1:
                        wheel_speed = abs(st['vwx'])
                        hydro_factor = self.hydroplaning.compute_mu_factor(wheel_speed, water_depth)
                        mu_wheel *= hydro_factor
            
            # Get drive force request for driven wheels (TCS-modulated)
            Fx_drive_req = drive_forces_dict.get(key, 0.0)
            
            # Add ESC/RSC corrective brake force to wheel brake forces
            brake_forces[key] = brake_forces.get(key, 0.0) + esc_forces.get(key, 0.0)
            
            # Estimate kappa from drive force request (for traction control)
            # During braking/driving, kappa is determined by wheel spin dynamics
            # For now, use a simplified approach: kappa from diff between requested and tire capability
            if abs(Fx_drive_req) > 1.0:
                # Rough estimate: kappa ≈ Fx_req / (tire stiffness)
                C_kappa_est = tire_model.get_longitudinal_stiffness(Fz_i, mu_wheel)
                kappa_drive = np.clip(Fx_drive_req / max(C_kappa_est, 1.0), -0.3, 0.3)
            else:
                kappa_drive = 0.0
            
            # Combine kappa from wheel spin and drive request
            # kappa_i from wheel spin, kappa_drive from force request
            kappa_eff = kappa_i if abs(kappa_i) > abs(kappa_drive) else kappa_drive
            
            # DUAL-TIRE CONTACT: use dual tire model for rear wheels
            if ENABLE_DUAL_TIRES and key in ('rl', 'rr'):
                dual_tire = self.dual_tire_rl if key == 'rl' else self.dual_tire_rr
                Fx_tire, Fy_tire = dual_tire.combined_forces(
                    alpha=alpha_i, kappa=kappa_eff, Fz_total=Fz_i,
                    gamma=camber_i, mu=mu_wheel
                )
            else:
                # ===================================================================
                # TIRE SATURATION MODE SELECTOR
                # ===================================================================
                if TIRE_SATURATION_MODE == 'PACEJKA_DIRECT':
                    # Direct Pacejka saturation (like RoadView)
                    # Uses model's built-in sin/arctan clipping
                    Fx_tire, Fy_tire = tire_model.combined_slip(
                        alpha=alpha_i, kappa=kappa_eff, Fz=Fz_i,
                        gamma=camber_i, mu=mu_wheel
                    )
                else:  # SMOOTH_TANH mode
                    # Compute longitudinal from kappa using Pacejka stiffness
                    C_kappa = tire_model.get_longitudinal_stiffness(Fz_i, mu_wheel)
                    Fx_linear = C_kappa * kappa_eff
                    Fx_max = mu_wheel * max(1.0, Fz_i)
                    Fx_tire = self.smooth_saturate_force(Fx_linear, Fx_max, sharpness=2.0)
                    
                    # Compute lateral with smooth saturation
                    C_alpha_nom = self.Cf_axle / 2.0 if key in ('fl', 'fr') else self.Cr_axle / 2.0
                    Fy_linear = C_alpha_nom * alpha_i
                    
                    # Apply smooth lateral force saturation
                    Fy_max = mu_wheel * max(1.0, Fz_i)
                    Fy_tire = self.smooth_saturate_force(Fy_linear, Fy_max, sharpness=2.0)
                    
                    # Apply friction ellipse to couple Fx and Fy
                    Fx_tire, Fy_tire = self.apply_friction_ellipse(Fx_tire, Fy_tire, Fz_i, mu_wheel)
            # FIX: Negate lateral force for stability (positive slip -> restoring force)
            Fy_tire = -Fy_tire
            
            # Low-speed lateral force reduction: tires less effective at generating lateral force
            # when vehicle is nearly stationary (physical behavior)
            if vx < 1.0:
                # Progressive reduction: at vx=0 → 0% lateral force, at vx=1 → 100%
                speed_factor = vx  # 0 to 1
                Fy_tire = Fy_tire * speed_factor
                Fx_tire = Fx_tire * speed_factor

            # DIAGNOSTIC: Log tire state at startup
            if diagnostic_startup and hasattr(self, '_diagnostic_counter'):
                if self._diagnostic_counter % 500 == 0:  # Every 500 calls (0.05s at dt=0.0001)
                    print(f"[TIRE_{key.upper()}] alpha={np.degrees(alpha_i):.2f}° kappa={kappa_eff:.3f} Fz={Fz_i:.1f}N Fx={Fx_tire:.1f}N Fy={Fy_tire:.1f}N")

            # apply compliance steer (toe change due to Fy)
            toe_i_corr = st['toe'] + self.k_toe_Fy * Fy_tire
            alpha_i_corr = math.atan2(st['vwy'], max(1e-6, st['vwx'])) - (steer_i + toe_i_corr)

            # Recompute lateral with corrected alpha (optional but more consistent)
            Fx_ss, Fy_ss = tire_model.combined_slip(
                alpha=alpha_i_corr,
                kappa=kappa_eff,
                Fz=Fz_i,
                gamma=camber_i,
                mu=self.mu
            )
            # FIX: Negate steady-state lateral force as well
            Fy_ss = -Fy_ss
            
            # Re-apply low-speed reduction to corrected force (steady-state)
            if vx < 1.0:
                Fy_ss = Fy_ss * vx
                Fx_ss = Fx_ss * vx

            # =================================================================
            # TIRE RELAXATION LENGTH - Transient Force Dynamics
            # =================================================================
            # If enabled, tire forces lag behind steady-state values due to
            # contact patch deformation as tire rolls
            
            if ENABLE_TIRE_RELAXATION:
                # Get current transient forces for this wheel using HISTORY
                if key == 'fl':
                    Fx_current = self.Fx_fl_transient_hist
                    Fy_current = self.Fy_fl_transient_hist
                    relaxation = self.tire_relaxation_front
                elif key == 'fr':
                    Fx_current = self.Fx_fr_transient_hist
                    Fy_current = self.Fy_fr_transient_hist
                    relaxation = self.tire_relaxation_front
                elif key == 'rl':
                    Fx_current = self.Fx_rl_transient_hist
                    Fy_current = self.Fy_rl_transient_hist
                    relaxation = self.tire_relaxation_rear
                else:  # 'rr'
                    Fx_current = self.Fx_rr_transient_hist
                    Fy_current = self.Fy_rr_transient_hist
                    relaxation = self.tire_relaxation_rear
                
                # Update transient forces using first-order lag (implicit Euler)
                # Returns CANDIDATE forces, does NOT mutate self
                Fx_i, Fy_i = relaxation.compute_transient_forces(
                    Fx_ss, Fy_ss, Fx_current, Fy_current, vx, dt
                )
                
                # DO NOT update self.Fx_..._hist here!
            else:
                # Steady-state model (instant force response)
                Fx_i = Fx_ss
                Fy_i = Fy_ss
            
            # Debug: check if brake is being applied
            if brake_forces[key] > 10:
                print(f"[WHEEL {key}] brake_force={brake_forces[key]:.1f} Fx_pacejka={Fx_i:.1f} Fy_pacejka={Fy_i:.1f}")

            # Rotate driven+lateral forces to body frame (steering affects these)
            c, s = np.cos(steer_i), np.sin(steer_i)
            Fx_body = c * Fx_i - s * Fy_i  # Longitudinal in body frame
            Fy_body = s * Fx_i + c * Fy_i  # Lateral in body frame
            
            # Brake force acts purely longitudinally in body frame (NOT affected by steering)
            Fx_body -= brake_forces[key]

            dtoe_compliance = self.k_toe_Fy * Fy_i
            # Self-aligning torque (pneumatic + mechanical trail)
            Mz_i = self.aligning_moment(Fy=Fy_i, alpha=alpha_i, Fz=Fz_i,
                                        caster=caster_i, sai=sai_i)

            # Lever arms to yaw (body frame)
            rx_w, ry_w = pos_map[key][0], pos_map[key][1]
            # Moment from forces (right-hand rule): Mz = r_x*F_y - r_y*F_x
            Mz_force = rx_w * Fy_body - ry_w * Fx_body

            # Debug: detailed per-wheel force breakdown
            if brake > 0.01:
                print(f"[WHEEL_DETAIL {key}] Fx_i={Fx_i:.1f} Fy_i={Fy_i:.1f} steer={steer_i:.3f} -> Fx_body={Fx_body:.1f} Fy_body={Fy_body:.1f}")

            # Accumulate totals
            sum_Fx += Fx_body
            sum_Fy += Fy_body
            sum_Mz += (Mz_force + Mz_i)

        # Total longitudinal force: tire forces minus resistances (aero, rolling)
        # Trailer drag is now handled IMPLICITLY via hitch force, not as explicit drag
        sum_Fx_before = sum_Fx  # For debug output
        sum_Fx = sum_Fx - F_load 

        # ====================================================================
        # IMPLICIT HITCH FORCE (if trailer attached)
        # ====================================================================
        if self.follower_info:
            # 1. Calculate leader's hitch point (pin joint) in global frame
            # self.hitch_length is distance from CG to hitch point
            hitch_x_leader = self.x - self.hitch_length * np.cos(self.yaw)
            hitch_y_leader = self.y - self.hitch_length * np.sin(self.yaw)
            
            # 2. Follower target position (where follower wants to be)
            target_x, target_y = self.follower_info.target_pos
            
            # 3. Position error (vector from target to current hitch point)
            # F = -k * (current - target) -> Force pulling leader BACK towards follower
            dx = hitch_x_leader - target_x
            dy = hitch_y_leader - target_y
            
            # 4. Velocity error
            vx_hitch = self.vx * np.cos(self.yaw) - self.vy * np.sin(self.yaw) - self.hitch_length * (-np.sin(self.yaw) * self.omega)
            vy_hitch = self.vx * np.sin(self.yaw) + self.vy * np.cos(self.yaw) + self.hitch_length * (np.cos(self.yaw) * self.omega)
            
            target_vx, target_vy = self.follower_info.target_vel
            dvx = vx_hitch - target_vx
            dvy = vy_hitch - target_vy
            
            # 5. Global hitch force on LEADER
            Fx_hitch_global = -self.follower_info.stiffness * dx - self.follower_info.damping * dvx
            Fy_hitch_global = -self.follower_info.stiffness * dy - self.follower_info.damping * dvy
            
            # 6. Transform to body frame
            Fx_hitch_body = Fx_hitch_global * np.cos(self.yaw) + Fy_hitch_global * np.sin(self.yaw)
            Fy_hitch_body = -Fx_hitch_global * np.sin(self.yaw) + Fy_hitch_global * np.cos(self.yaw)
            
            # 7. Add to sums
            sum_Fx += Fx_hitch_body
            sum_Fy += Fy_hitch_body # Tractor now feels lateral yank!
            
            # 8. Moment: hitch is at (-hitch_length, 0) in body frame
            # Mz = rx*Fy - ry*Fx = (-hitch_len)*Fy - (0)*Fx
            sum_Mz += (-self.hitch_length) * Fy_hitch_body

        # Debug: show force accumulation with trailer drag
        if brake > 0.01 or throttle > 0.01:
            print(f"[FORCE_ACCUM] sum_Fx_tires={sum_Fx_before:.1f} F_load={F_load:.1f} trailer_drag={self.trailer_drag:.1f} sum_Fx_net={sum_Fx:.1f} sum_Fy={sum_Fy:.1f}")


        # -------------------------
        # 5) Rigid-body dynamics
        # -------------------------
        # Body-frame equations with centrifugal acceleration terms
        # ax = sum_Fx/m - vy*omega (centrifugal effect on longitudinal)
        # ay = sum_Fy/m + vx*omega (centrifugal effect on lateral)
        coriolis_term = vy * omega * self.mass
        ax_without_coriolis = sum_Fx / self.mass
        ax = sum_Fx / self.mass + vy * omega
        ay = sum_Fy / self.mass - vx * omega
        
        # Debug: force balance with Coriolis breakdown
        if brake > 0.01:
            print(f"[DYNAMICS_DETAIL] sum_Fx={sum_Fx:.1f} coriolis_vy*omega*m={coriolis_term:.1f} vy={vy:.2f} omega={omega:.3f} ax_brake_only={ax_without_coriolis:.2f} ax_total={ax:.2f}")
        alpha_ang_a =  sum_Mz / self.Iz

        # ===================================================================
        # FIX: Cache the computed lateral acceleration for load transfer
        # ===================================================================
        # After computing ay from forces, store it for use in next residual eval.
        # This ensures load transfer uses actual tire force contribution, not just kinematics.
        self._ay_for_load_transfer = ay

        # -------------------------
        # 6) Dynamic residuals (Implicit Euler)
        # -------------------------
        res_vx    = vx    - vx_old    - dt * ax
        res_vy    = vy    - vy_old    - dt * ay
        res_omega = omega - omega_old - dt * alpha_ang_a

        # -------------------------
        # Return residual vector
        # -------------------------
        
        return np.array([
            res_x,
            res_y,
            res_yaw,
            res_vx,
            res_vy,
            res_omega])

    
    
    def tyre_lateral_force_enhanced(self, C_alpha_nom, alpha, camber, Fz, mu, Fz_ref, n_load, C_gamma):
        """
        Load-sensitive lateral force with camber thrust and friction capping.
        C_alpha scales ~ Fz^n_load; linear region + camber-thrust; clipped by mu*Fz.
        """
        # Cornering stiffness vs load (smooth, positive)
        Fz_eff = max(1.0, Fz)
        C_alpha = C_alpha_nom * (Fz_eff / Fz_ref)**n_load

        # Linear lateral + camber thrust
        # FIX: Negate linear term so positive slip -> negative force (restoring)
        Fy_lin = -C_alpha * alpha + C_gamma * camber

        # Friction cap
        Fy_max = mu * Fz_eff
        return float(np.clip(Fy_lin, -Fy_max, Fy_max))


    def pneumatic_trail(self, alpha):
        """
        Simple pneumatic trail model: decays with |alpha|.
        """
        return self.t_p0 / (1.0 + self.t_p_decay * abs(alpha))


    def aligning_moment(self, Fy, alpha, Fz, caster, sai):
        """
        Self-aligning torque from pneumatic trail + mechanical trail (caster/SAI).
        Sign convention: restoring torque opposes slip (negative with positive Fy).
        """
        # Pneumatic trail contribution
        t_p = self.pneumatic_trail(alpha)
        Mz_pneu = -t_p * Fy

        # Mechanical trail / caster & SAI projection
        t_mech = self.mechanical_trail
        # Small-angle projection of normal load along steering axis inclination
        Mz_mech = -t_mech * Fz * np.sin(caster + sai)

        return Mz_pneu + Mz_mech

    
    def compute_jacobian(self, x_new, x_old, throttle, brake, delta, dt):
        # FIX #3: Adaptive perturbation size based on state magnitude
        # Fixed eps=1e-6 causes numerical noise when state values are small (e.g., vx≈0)
        # Use relative perturbation: eps = max(1e-6, |state| * 1e-4)
        dim = 6
        J = np.zeros((dim, dim), dtype=float)
        f0 = self.compute_residual(x_new, x_old, throttle, brake, delta, dt)
        
        for i in range(dim):
            # Adaptive eps: larger for small states to avoid numerical cancellation
            # For vx=0.01: eps=1e-6, for vx=1.0: eps=1e-4, for vx=10: eps=1e-3
            eps = max(1e-6, abs(x_new[i]) * 1e-4)
            
            x_plus = x_new.copy();  x_plus[i] += eps
            x_minus = x_new.copy(); x_minus[i] -= eps
            f_plus  = self.compute_residual(x_plus,  x_old, throttle, brake, delta, dt)
            f_minus = self.compute_residual(x_minus, x_old, throttle, brake, delta, dt)
            J[:, i] = (f_plus - f_minus) / (2.0 * eps)
        return J

    # ========================================================================
    # GENERALIZED-ALPHA TIME INTEGRATION METHODS
    # ========================================================================
    
    def compute_residual_genalpha(self, state_new, state_old, throttle, brake, delta, dt):
        """
        Generalized-α residual with 9-state vector for 2nd-order accurate integration.
        
        State vector: [x, y, yaw, vx, vy, omega, ax, ay, alpha]
            - x, y: Global positions [m]
            - yaw: Heading angle [rad]
            - vx, vy: Body-frame velocities [m/s]
            - omega: Yaw rate [rad/s]
            - ax, ay: Body-frame accelerations [m/s²]
            - alpha: Yaw angular acceleration [rad/s²]
        
        The residual has 9 equations:
            Equations 0-2: Newmark position constraints (x, y, yaw)
            Equations 3-5: Newmark velocity constraints (vx, vy, omega)
            Equations 6-8: Dynamic equilibrium (F = ma)
        
        Returns:
            9-element residual vector
        """
        # Unpack new state
        x_new, y_new, yaw_new = state_new[0], state_new[1], state_new[2]
        vx_new, vy_new, omega_new = state_new[3], state_new[4], state_new[5]
        ax_new, ay_new, alpha_new = state_new[6], state_new[7], state_new[8]
        
        # Unpack old state
        x_old, y_old, yaw_old = state_old[0], state_old[1], state_old[2]
        vx_old, vy_old, omega_old = state_old[3], state_old[4], state_old[5]
        ax_old, ay_old, alpha_old = state_old[6], state_old[7], state_old[8]
        
        # Get Generalized-α parameters
        am = self._ga_alpha_m
        af = self._ga_alpha_f
        beta = self._ga_beta
        gamma = self._ga_gamma
        dt2 = dt * dt
        
        # =====================================================================
        # NEWMARK KINEMATIC CONSTRAINTS
        # =====================================================================
        # These ensure 2nd-order accuracy in time integration
        
        # Velocity update: v_{n+1} = v_n + dt * [(1-γ)*a_n + γ*a_{n+1}]
        res_vx = vx_new - vx_old - dt * ((1.0 - gamma) * ax_old + gamma * ax_new)
        res_vy = vy_new - vy_old - dt * ((1.0 - gamma) * ay_old + gamma * ay_new)
        res_omega = omega_new - omega_old - dt * ((1.0 - gamma) * alpha_old + gamma * alpha_new)
        
        # Position update in global frame
        # For planar motion: dx/dt = vx*cos(yaw) - vy*sin(yaw)
        # Using midpoint velocities for position integration
        cos_yaw_old = np.cos(yaw_old)
        sin_yaw_old = np.sin(yaw_old)
        
        # Global velocity at old timestep
        vx_global_old = vx_old * cos_yaw_old - vy_old * sin_yaw_old
        vy_global_old = vx_old * sin_yaw_old + vy_old * cos_yaw_old
        
        # Approximate global acceleration (using body-frame acceleration transformed)
        # This is a simplification - full coupling would require yaw rate terms
        ax_global_old = ax_old * cos_yaw_old - ay_old * sin_yaw_old
        ay_global_old = ax_old * sin_yaw_old + ay_old * cos_yaw_old
        ax_global_new = ax_new * np.cos(yaw_new) - ay_new * np.sin(yaw_new)
        ay_global_new = ax_new * np.sin(yaw_new) + ay_new * np.cos(yaw_new)
        
        # Position: x_{n+1} = x_n + dt*v_n + dt²*[(0.5-β)*a_n + β*a_{n+1}]
        res_x = x_new - x_old - dt * vx_global_old - dt2 * ((0.5 - beta) * ax_global_old + beta * ax_global_new)
        res_y = y_new - y_old - dt * vy_global_old - dt2 * ((0.5 - beta) * ay_global_old + beta * ay_global_new)
        res_yaw = yaw_new - yaw_old - dt * omega_old - dt2 * ((0.5 - beta) * alpha_old + beta * alpha_new)
        
        # =====================================================================
        # DYNAMIC EQUILIBRIUM AT INTERPOLATED STATE
        # =====================================================================
        # Evaluate forces at t_{n+1-αf} for improved stability
        
        # Interpolated state for force evaluation
        x_af = (1.0 - af) * x_new + af * x_old
        y_af = (1.0 - af) * y_new + af * y_old
        yaw_af = (1.0 - af) * yaw_new + af * yaw_old
        vx_af = (1.0 - af) * vx_new + af * vx_old
        vy_af = (1.0 - af) * vy_new + af * vy_old
        omega_af = (1.0 - af) * omega_new + af * omega_old
        
        # Compute forces at interpolated state
        Fx_total, Fy_total, Mz_total = self._compute_forces_for_genalpha(
            x_af, y_af, yaw_af, vx_af, vy_af, omega_af, throttle, brake, delta, dt
        )
        
        # Dynamic equilibrium: m*a = F (evaluated at n+1-αm for acceleration)
        # Note: We use new accelerations directly for simplicity (αm ≈ 0 for moderate ρ∞)
        
        # FIX: Added kinematic terms (Coriolis/Centripetal) for rotating reference frame
        # FIX: Added kinematic terms (Coriolis/Centripetal) for rotating reference frame
        # STANDARD PHYSICS CONVENTION (Ax = Fx/m + vy*omega, Ay = Fy/m - vx*omega)
        # The "Legacy" convention (inverted) caused massive instability.
        res_ax = ax_new - (Fx_total / self.mass + vy_af * omega_af)
        res_ay = ay_new - (Fy_total / self.mass - vx_af * omega_af)
        
        res_alpha = alpha_new - Mz_total / self.Iz
        
        return np.array([res_x, res_y, res_yaw, res_vx, res_vy, res_omega, res_ax, res_ay, res_alpha])
    
    def _compute_forces_for_genalpha(self, x, y, yaw, vx, vy, omega, throttle, brake, delta, dt):
        """
        Compute total forces and moment at a given state (used by Generalized-α residual).
        
        This mirrors the force computation in compute_residual but returns forces directly
        instead of forming residual equations.
        
        Returns:
            (Fx_total, Fy_total, Mz_total): Forces/moment in body frame [N, N, Nm]
        """
        # -------------------------
        # Tire/wheel states
        # -------------------------
        Fz_local = self.compute_wheel_loads_local(vx, vy, omega)
        wheels = self.compute_wheel_local_states(vx, vy, omega, delta, Fz_local)
        
        # -------------------------
        # Longitudinal forces: drive, brake, resistance
        # -------------------------
        vehicle_components = [{
            'Cd': self.Cd,
            'A': self.A_front,
            'mass': self.mass,
            'Crr': self.Crr
        }]
        
        vx_safe = max(abs(vx), 1e-2)
        F_resist = compute_aero_and_rolling_resistance(vx_safe, vehicle_components)
        
        # Engine & braking
        rpm = self.compute_engine_rpm(vx, self.r_w)
        T_engine = throttle * self.engine_torque_curve(rpm) + throttle * self.electric_torque(vx)
        
        # TORQUE CONVERTER
        if ENABLE_TORQUE_CONVERTER and self.torque_converter is not None:
            omega_engine = rpm * 2.0 * np.pi / 60.0
            total_ratio = self.get_total_ratio()
            omega_gearbox = abs(vx) / max(self.r_w, 0.01) * total_ratio
            T_engine = self.torque_converter.compute(T_engine, omega_engine, omega_gearbox)
        
        F_drive = self.compute_drive_force(T_engine, self.r_w, vx)
        
        # PNEUMATIC BRAKE DELAY
        brake_effective = brake
        if ENABLE_BRAKE_DELAY and self.brake_system is not None:
            brake_effective = self.brake_system.update(brake, dt)
        
        F_brake_total = self.compute_brake_force(brake_effective, vx)
        brake_forces, regen_force, mechanical_force = self.distribute_braking_forces(F_brake_total, vx)
        
        # DIFFERENTIAL: distribute torque based on wheel speed difference
        if ENABLE_LSD and self.differential is not None:
            omega_rl = self.wheel_omega.get('rl', 0.0)
            omega_rr = self.wheel_omega.get('rr', 0.0)
            F_left, F_right = self.differential.distribute(F_drive, omega_rl, omega_rr)
            drive_forces_whl = {'fl': 0.0, 'fr': 0.0, 'rl': F_left, 'rr': F_right}
            Fx_drive_per_wheel = F_drive / max(len(self.driven_wheels), 1)
        else:
            Fx_drive_per_wheel = F_drive / len(self.driven_wheels) if vx > 0.01 else 0.0
            drive_forces_whl = {k: 0.0 for k in ['fl', 'fr', 'rl', 'rr']}
            for dw in self.driven_wheels:
                drive_forces_whl[dw] = Fx_drive_per_wheel
        
        # TCS modulation on drive forces
        if ENABLE_TCS and self.tcs_controller is not None and throttle > 0.01:
            wheel_kappas = getattr(self, '_wheel_kappas', {k: 0.0 for k in ['fl','fr','rl','rr']})
            drive_forces_whl, _ = self.tcs_controller.modulate(
                drive_forces_whl, wheel_kappas, vx, dt, self.driven_wheels
            )
        
        # ESC corrective braking forces
        esc_forces_whl = {k: 0.0 for k in ['fl', 'fr', 'rl', 'rr']}
        if ENABLE_ESC and self.esc_controller is not None:
            esc_forces_whl = self.esc_controller.compute(vx, omega, delta, dt)
            # Add ESC corrective forces to brake dict
            for k in esc_forces_whl:
                brake_forces[k] = brake_forces.get(k, 0.0) + esc_forces_whl.get(k, 0.0)
        
        # -------------------------
        # Per-wheel tire forces using Pacejka
        # -------------------------
        Fx_total = 0.0
        Fy_total = 0.0
        Mz_total = 0.0
        
        for key in ['fl', 'fr', 'rl', 'rr']:
            w = wheels[key]
            Fz_w = Fz_local[key]
            alpha_w = w['alpha']
            kappa_w = w.get('kappa', 0.0)
            camber_w = w['camber']
            
            # Select tire model
            if key in ['fl', 'fr']:
                tire = self.tire_front
            else:
                tire = self.tire_rear
            
            # Per-wheel friction with μ-map + hydroplaning
            mu_whl = self.mu_static
            if ENABLE_MU_MAP and self.road_friction is not None:
                cos_yaw = np.cos(self.yaw)
                sin_yaw = np.sin(self.yaw)
                t_pos = self.tire_positions
                pos_key = {'fl': 'front_left', 'fr': 'front_right', 'rl': 'rear_left', 'rr': 'rear_right'}[key]
                wp = t_pos[pos_key]
                wx = self.x + wp[0] * cos_yaw - wp[1] * sin_yaw
                wy = self.y + wp[0] * sin_yaw + wp[1] * cos_yaw
                mu_whl = self.road_friction.get_mu(wx, wy)
                
                if ENABLE_HYDROPLANING and self.hydroplaning is not None:
                    water_depth = self.road_friction.get_water_depth(wx, wy)
                    if water_depth > 0.1:
                        hydro_factor = self.hydroplaning.compute_mu_factor(abs(vx), water_depth)
                        mu_whl *= hydro_factor
            
            # DUAL-TIRE: use dual model for rear wheels
            if ENABLE_DUAL_TIRES and key in ('rl', 'rr'):
                dual_tire = self.dual_tire_rl if key == 'rl' else self.dual_tire_rr
                Fx_tire, Fy_tire = dual_tire.combined_forces(alpha_w, kappa_w, Fz_w, camber_w, mu_whl)
            else:
                # ===================================================================
                # TIRE SATURATION MODE SELECTOR
                # ===================================================================
                if TIRE_SATURATION_MODE == 'PACEJKA_DIRECT':
                    # Direct Pacejka saturation (like RoadView)
                    Fx_tire, Fy_tire = tire.combined_slip(alpha_w, kappa_w, Fz_w, camber_w, mu_whl)
                else:  # SMOOTH_TANH mode
                    # Compute longitudinal from kappa using Pacejka stiffness
                    C_kappa = tire.get_longitudinal_stiffness(Fz_w, mu_whl)
                    Fx_linear = C_kappa * kappa_w
                    Fx_max = mu_whl * max(1.0, Fz_w)
                    Fx_tire = Fx_max * np.tanh(2.0 * Fx_linear / max(Fx_max, 1.0))
                    
                    # Compute lateral with smooth saturation
                    C_alpha = tire.get_cornering_stiffness(Fz_w, mu_whl)
                    Fy_linear = C_alpha * alpha_w
                    Fy_max = mu_whl * max(1.0, Fz_w)
                    Fy_tire = Fy_max * np.tanh(2.0 * Fy_linear / max(Fy_max, 1.0))
                    
                    # Apply friction ellipse to couple Fx and Fy
                    Fx_tire, Fy_tire = friction_ellipse_limit(Fx_tire, Fy_tire, Fz_w, mu_whl)
            # FIX: Negate lateral force — Pacejka outputs Fy in the direction of slip,
            # but physics requires a restoring force (opposing slip). Without this,
            # any lateral perturbation is amplified → divergence.
            Fy_tire = -Fy_tire
            
            # Low-speed lateral force reduction (matches compute_residual)
            if vx < 1.0:
                speed_factor = max(vx, 0.0)
                Fy_tire *= speed_factor
                Fx_tire *= speed_factor
            
            # =============================================================
            # TIRE RELAXATION LENGTH - Transient Force Dynamics
            # =============================================================
            # Mirrors the relaxation logic in compute_residual (lines 3567-3596)
            # Uses history state as F_current, does NOT mutate history here
            if ENABLE_TIRE_RELAXATION:
                if key == 'fl':
                    Fx_current = self.Fx_fl_transient_hist
                    Fy_current = self.Fy_fl_transient_hist
                    relaxation = self.tire_relaxation_front
                elif key == 'fr':
                    Fx_current = self.Fx_fr_transient_hist
                    Fy_current = self.Fy_fr_transient_hist
                    relaxation = self.tire_relaxation_front
                elif key == 'rl':
                    Fx_current = self.Fx_rl_transient_hist
                    Fy_current = self.Fy_rl_transient_hist
                    relaxation = self.tire_relaxation_rear
                else:  # 'rr'
                    Fx_current = self.Fx_rr_transient_hist
                    Fy_current = self.Fy_rr_transient_hist
                    relaxation = self.tire_relaxation_rear
                
                # Compute transient forces (candidate only, no state mutation)
                Fx_tire, Fy_tire = relaxation.compute_transient_forces(
                    Fx_tire, Fy_tire, Fx_current, Fy_current, vx, dt
                )

            # Add drive force for driven wheels (TCS-modulated)
            Fx_tire += drive_forces_whl.get(key, 0.0)
            
            # Subtract brake force
            Fx_tire -= brake_forces.get(key, 0.0)
            
            # Accumulate forces
            Fx_total += Fx_tire
            Fy_total += Fy_tire
            
            # Moment contribution (tire position relative to CG)
            if key == 'fl':
                pos = self.tire_positions['front_left']
            elif key == 'fr':
                pos = self.tire_positions['front_right']
            elif key == 'rl':
                pos = self.tire_positions['rear_left']
            else:
                pos = self.tire_positions['rear_right']
            
            px, py = float(pos[0]), float(pos[1])
            Mz_total += Fy_tire * px - Fx_tire * py
        
        # -------------------------
        # Resistance and trailer drag
        # -------------------------
        Fx_total -= F_resist
        
        # -------------------------
        # IMPLICIT HITCH FORCES (Critical for Gen-Alpha)
        # -------------------------
        # This mirrors the logic in compute_residual using the interpolated state (x, y, yaw...)
        if getattr(self, 'follower_info', None):
            # 1. Calculate leader's hitch position (body -> global) at interpolated state
            # Hitch is at (-hitch_length, 0) in body frame
            hitch_x_leader = x - self.hitch_length * np.cos(yaw)
            hitch_y_leader = y - self.hitch_length * np.sin(yaw)
            
            # 2. Calculate leader's hitch velocity (body -> global)
            # Velocity of point P: v_P = v_CG + omega x r_P
            # r_P (body) = (-hitch_length, 0)
            # v_hitch_x = vx_global - (-hitch_length)*sin(yaw)*omega
            # v_hitch_y = vy_global + (-hitch_length)*cos(yaw)*omega
            # First, get global velocity of CG
            vx_global_cg = vx * np.cos(yaw) - vy * np.sin(yaw)
            vy_global_cg = vx * np.sin(yaw) + vy * np.cos(yaw)
            
            # Then adds rotation effect
            hitch_vx_leader = vx_global_cg - (-self.hitch_length) * np.sin(yaw) * omega
            hitch_vy_leader = vy_global_cg + (-self.hitch_length) * np.cos(yaw) * omega
            
            # 3. Get follower's target position/velocity (from HitchParams)
            target_pos_x, target_pos_y = self.follower_info.target_pos
            target_vel_x, target_vel_y = self.follower_info.target_vel
            
            # 4. Calculate error (delta)
            dx = hitch_x_leader - target_pos_x
            dy = hitch_y_leader - target_pos_y
            dvx = hitch_vx_leader - target_vel_x
            dvy = hitch_vy_leader - target_vel_y
            
            # 5. Global hitch force on LEADER
            # Force splits based on stiffness/damping
            Fx_hitch_global = -self.follower_info.stiffness * dx - self.follower_info.damping * dvx
            Fy_hitch_global = -self.follower_info.stiffness * dy - self.follower_info.damping * dvy
            
            # 6. Transform to body frame
            Fx_hitch_body = Fx_hitch_global * np.cos(yaw) + Fy_hitch_global * np.sin(yaw)
            Fy_hitch_body = -Fx_hitch_global * np.sin(yaw) + Fy_hitch_global * np.cos(yaw)
            
            # 7. Add to sums
            Fx_total += Fx_hitch_body
            Fy_total += Fy_hitch_body 
            
            # 8. Moment contribution
            Mz_total += (-self.hitch_length) * Fy_hitch_body

        
        # -------------------------
        # Gravity component on slopes
        # -------------------------
        g = 9.81
        Fx_total -= self.mass * g * np.sin(self.pitch)
        
        return Fx_total, Fy_total, Mz_total
    
    def compute_jacobian_genalpha(self, state_new, state_old, throttle, brake, delta, dt):
        """
        Compute 9×9 Jacobian for Generalized-α residual using central differences.
        """
        dim = 9
        J = np.zeros((dim, dim), dtype=float)
        
        for i in range(dim):
            # Adaptive perturbation
            eps = max(1e-6, abs(state_new[i]) * 1e-4)
            
            state_plus = state_new.copy()
            state_plus[i] += eps
            state_minus = state_new.copy()
            state_minus[i] -= eps
            
            f_plus = self.compute_residual_genalpha(state_plus, state_old, throttle, brake, delta, dt)
            f_minus = self.compute_residual_genalpha(state_minus, state_old, throttle, brake, delta, dt)
            
            J[:, i] = (f_plus - f_minus) / (2.0 * eps)
        
        return J



def compute_equivalent_stiffness_split(m, lf, lr, mu, B, C, Bx, Cx, h_cg, pitch_angle, tire_positions):
    g = 9.81

    # Cargas estáticas
    Fz_front_static = m * g * lr / (lf + lr)
    Fz_rear_static  = m * g * lf / (lf + lr)

    # Cargas dinámicas por pitch
    delta_Fz_front = -m * g * h_cg * math.sin(pitch_angle) * (lr / (lf + lr))
    delta_Fz_rear  = +m * g * h_cg * math.sin(pitch_angle) * (lf / (lf + lr))

    # Rigidez estática
    Cf_static = mu * Fz_front_static / 2 * B * C * 2  # 2 neumáticos
    Cr_static = mu * Fz_rear_static / 2 * B * C * 2
    Cx_static = mu * (Fz_front_static + Fz_rear_static) / 4 * Bx * Cx * 4

    # Corrección dinámica
    Cf_dynamic = mu * delta_Fz_front / 2 * B * C * 2
    Cr_dynamic = mu * delta_Fz_rear / 2 * B * C * 2
    Cx_dynamic = mu * (delta_Fz_front + delta_Fz_rear) / 4 * Bx * Cx * 4

    Cf_eq = Cf_static + Cf_dynamic
    Cr_eq = Cr_static + Cr_dynamic
    Cx_eq = Cx_static + Cx_dynamic

    return Cf_eq, Cr_eq, Cx_eq



class ArticulatedSegment():
    def __init__(self, mass, Iz, x, y, z, yaw, pitch, hitch_length, trailer_length, waypoints, articulation_yaw, articulation_yaw_rate, h_cg_trailer, track_width_trailer, has_front_axle=False):
        self.mass = mass  # kg
        self.Iz = Iz  # kg/m2
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.yaw_rate = 0.0
        self.omega=0.0
        self.vx = 0.0  # to avoid division by zero
        self.vy = 0.0
        self.pitch = pitch
        self.hitch_length = hitch_length
        self.trailer_length = trailer_length
        self.waypoints = waypoints
        self.articulation_yaw = articulation_yaw
        self.articulation_yaw_rate = articulation_yaw_rate
        self.prev_articulation_yaw = articulation_yaw  # for rate calculation
        self.lf = trailer_length/2
        self.lr = trailer_length/2
        self.track_width = track_width_trailer
        self.h_cg = h_cg_trailer
        self.mu = DEFAULT_MU  # tire-road friction coefficient (references global for sensitivity studies)
        self.toe=np.radians(0.05) # toe angle in radians
        self.static_camber = np.radians(-0.2)   # trailer static camber
        self.Cgamma = 20000.0                   # camber coefficient trailer [N/rad]
        self.has_front_axle = has_front_axle    # Store for use in update

        self.has_front_axle = has_front_axle    # Store for use in update
        self.r_w = 0.5 # Default trailer wheel radius
        self.wheel_inertia = 10.0 # Trailer wheel inertia
        self.wheel_omega = {'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0}

        self.tyre_B = 10.0
        self.tyre_C = 1.9
        self.tyre_Bx = 10.0
        self.tyre_Cx = 1.9
        
        # Define tires (Default: Rear Axle Only for Semitrailers)
        self.tire_positions = [
            {"x": -self.lr, "y":  self.track_width / 2,  "axle": "rear"},
            {"x": -self.lr, "y": -self.track_width / 2,  "axle": "rear"},
        ]
        
        # Add front axle only if explicitly requested (e.g. for full trailers/wagons)
        if self.has_front_axle:
            self.tire_positions.extend([
                {"x": self.lf,  "y":  self.track_width / 2,  "axle": "front"},
                {"x": self.lf,  "y": -self.track_width / 2,  "axle": "front"},
            ])
            
        # ====================================================================
        # HITCH JOINT PARAMETERS
        # ====================================================================
        # Rotational spring-damper at the pivot (articulation)
        # Fifth-wheel coupling has LOW rotational friction — it's designed to pivot freely.
        # High values here create lateral forces that push the tractor sideways.
        if USE_FREE_PIVOT:
            self.k_psi = 0.0
            self.c_psi = 0.0
            print("[HITCH] Using FREE PIVOT (frictionless joint)")
        else:
            self.k_psi = 5000.0     # Torsional stiffness [Nm/rad] - light centering torque
            self.c_psi = 2000.0     # Torsional damping [Nm·s/rad] - prevents rapid oscillations
        
        # Translational constraint spring-damper (keeps hitch points coincident)
        # Very stiff penalty spring → ~0.1mm position error under normal loads
        self.k_hitch = 5000000.0  # Constraint stiffness [N/m] (5 MN/m)
        self.c_hitch = 50000.0    # Constraint damping [Ns/m] (50 kNs/m)
        
        # Hard angle stops (prevent jackknifing)
        self.max_articulation = np.radians(55.0)  # ±55° real fifth-wheel limit
        self.k_stop = 500000.0    # Stop contact stiffness [Nm/rad]
        self.c_stop = 20000.0     # Stop contact damping [Nm·s/rad]
        
        # Leader's rear hitch offset (distance from leader CG to hitch point)
        # Must be set externally after construction (depends on the leader vehicle)
        self.leader_rear_offset = 0.0

        # Combined-slip Pacejka tire model for trailer
        self.tire_model = create_trailer_tire()
        
        # Wheel angular velocities (for slip ratio calculation)
        self.wheel_omega = {'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0}
        self.r_w = 0.5  # Wheel radius [m]
        self.I_wheel = 1.0 # Wheel inertia [kg m^2]

        # Roll dynamics (trailers have higher CoG than tractors due to cargo)
        self.roll = 0.0
        self.roll_rate = 0.0
        self.Ixx = 4000.0          # Roll inertia [kg m^2] (higher than tractor)
        self.K_phi = 250000.0      # Roll stiffness [Nm/rad] (softer - air suspension)
        self.C_phi = 35000.0       # Roll damping [Nms/rad]
        self.camber_gain = 0.06    # Camber change per roll angle

        # Pitch dynamics (cargo creates large pitch variations)
        self.pitch_dyn = 0.0
        self.pitch_terrain = 0.0
        self.pitch_rate = 0.0
        self.Iyy = 20000.0         # Pitch inertia [kg m^2] (LARGE due to long cargo)
        self.K_theta = 150000.0    # Pitch stiffness [Nm/rad] (softer - air suspension)
        self.C_theta = 25000.0     # Pitch damping [Nms/rad]
        self.anti_dive = 0.05      # 5% anti-dive (less than tractor)
        self.anti_squat = 0.0      # 0% anti-squat (no drive axle)
    
        # Follower drag force (from trailer behind this one)
        self.follower_drag = 0.0   # Initialize to prevent undefined behavior
        
        # Previous accelerations for explicit Euler predictor (matching tractor)
        self.ax_prev = 0.0
        self.ay_prev = 0.0
        self.alpha_prev = 0.0  # Angular acceleration
        
        # ============================================================================
        # LATERAL LOAD TRANSFER FIX: Cached lateral acceleration from force balance
        # ============================================================================
        # Cache the lateral acceleration computed from tire forces (ay = sum_Fy/m - vx*omega)
        # instead of approximating with kinematics alone (ay = vx*omega).
        # Lag by one iteration for implicit solver self-consistency.
        self._ay_for_load_transfer = 0.0  # [m/s²] body-frame lateral acceleration from forces
        # ============================================================================
        # GENERALIZED-ALPHA STATE VARIABLES
        # ============================================================================
        # The Generalized-α method requires storing accelerations as state variables
        self.ax = 0.0       # Longitudinal acceleration [m/s²]
        self.ay = 0.0       # Lateral acceleration [m/s²] (body frame)
        self.alpha = 0.0    # Yaw angular acceleration [rad/s²]
        
        # Precompute Generalized-α integration parameters from RHO_INF
        if USE_GENERALIZED_ALPHA:
            self._ga_alpha_m, self._ga_alpha_f, self._ga_beta, self._ga_gamma = compute_genalpha_params(RHO_INF)
        else:
            self._ga_alpha_m = self._ga_alpha_f = self._ga_beta = self._ga_gamma = None
        
        # ============================================================================
        # TIRE RELAXATION DYNAMICS
        # ============================================================================
        # Transient tire force development for realistic dynamic response
        if ENABLE_TIRE_RELAXATION:
            # Initialize transient tire forces to zero
            # Initialize transient tire forces to zero (HISTORY STATE)
            self.Fx_axle_front_transient_hist = 0.0
            self.Fy_axle_front_transient_hist = 0.0
            self.Fx_axle_rear_transient_hist = 0.0
            self.Fy_axle_rear_transient_hist = 0.0
            
            # Create tire relaxation dynamics object for trailer
            self.tire_relaxation = TireRelaxationDynamics(
                sigma_x=SIGMA_X_TRAILER,
                sigma_y=SIGMA_Y_TRAILER
            )
            
            print("[TIRE RELAXATION] Enabled for ArticulatedSegment (Trailer)")
            print(f"  Axles: sigma_x={SIGMA_X_TRAILER}m, sigma_y={SIGMA_Y_TRAILER}m")
        else:
            self.tire_relaxation = None

        # ============================================================================
        # ABS (ANTI-LOCK BRAKING SYSTEM)
        # ============================================================================
        if ENABLE_ABS:
            self.abs_controller = ABSController(
                slip_target=ABS_SLIP_TARGET,
                slip_threshold=ABS_SLIP_THRESHOLD,
                release_threshold=ABS_RELEASE_THRESHOLD,
                pressure_increase_rate=ABS_PRESSURE_INCREASE_RATE,
                pressure_decrease_rate=ABS_PRESSURE_DECREASE_RATE,
                min_speed=ABS_MIN_SPEED
            )
            print("[ABS] Enabled for ArticulatedSegment (Trailer)")
        else:
            self.abs_controller = None

        # ============================================================================
        # PNEUMATIC BRAKE SYSTEM (trailer-specific delay)
        # ============================================================================
        if ENABLE_BRAKE_DELAY:
            self.brake_system = PneumaticBrakeSystem(
                delay=BRAKE_DELAY_TRAILER,
                pressure_rate=BRAKE_PRESSURE_RATE,
                release_rate=BRAKE_RELEASE_RATE
            )
            print(f"[BRAKE-DELAY] Enabled for ArticulatedSegment: delay={BRAKE_DELAY_TRAILER*1000:.0f}ms")
        else:
            self.brake_system = None

        # Per-wheel kappa storage for ABS access
        self._wheel_kappas = {k: 0.0 for k in ['fl', 'fr', 'rl', 'rr']}
        self._brake_input = 0.0  # Stored by solver before compute_residual
        
        # Trailer braking parameters
        self.max_brake_decel = 0.5 * 9.81  # Max deceleration (~0.5g for trailer)
        self.brake_force_distribution_front = 0.4  # 40% front (if has front axle)
        self.brake_force_distribution_rear = 0.6   # 60% rear

    def warm_start_tire_relaxation(self):
        """
        Initialize transient force history to steady-state Pacejka forces
        at the current vehicle state. Must be called AFTER __init__ and
        initialize_wheel_states().
        """
        if not ENABLE_TIRE_RELAXATION or self.tire_relaxation is None:
            return
        
        vx = max(self.vx, 0.1)
        vy = self.vy
        omega = self.omega
        
        # Compute steady-state forces at initial condition
        alpha_f = (vy + self.lf * omega) / max(vx, 1e-3)
        alpha_r = (vy - self.lr * omega) / max(vx, 1e-3)
        kappa = 0.0
        
        if self.has_front_axle:
            Fz_front = self.mass * 9.81 * self.lr / (self.lf + self.lr)
            Fxf_ss, Fyf_ss = self.tire_model.combined_slip(
                alpha=alpha_f, kappa=kappa, Fz=Fz_front,
                gamma=self.static_camber, mu=self.mu
            )
            self.Fx_axle_front_transient_hist = Fxf_ss
            self.Fy_axle_front_transient_hist = -Fyf_ss  # Negate convention
        
        Fz_rear = self.mass * 9.81 * self.lf / (self.lf + self.lr)
        Fxr_ss, Fyr_ss = self.tire_model.combined_slip(
            alpha=alpha_r, kappa=kappa, Fz=Fz_rear,
            gamma=self.static_camber, mu=self.mu
        )
        self.Fx_axle_rear_transient_hist = Fxr_ss
        self.Fy_axle_rear_transient_hist = -Fyr_ss  # Negate convention
        
        print(f"[TIRE RELAXATION] Warm-started trailer: "
              f"F_front=({self.Fx_axle_front_transient_hist:.0f},{self.Fy_axle_front_transient_hist:.0f})N "
              f"R_rear=({self.Fx_axle_rear_transient_hist:.0f},{self.Fy_axle_rear_transient_hist:.0f})N")
        
        # Follower info for coupled solver
        self.follower_info = None

    def initialize_wheel_states(self):
        """
        Initialize trailer wheel rotational speeds to match vehicle velocity.
        """
        print(f"[INIT] Initializing Trailer wheel speeds for vx={self.vx:.2f} m/s")
        
        for tire in self.tire_positions:
            # Parse tire info
            px, py = tire['x'], tire['y']
            axle = tire['axle']
            
            # Determine key (approximate for logging)
            if axle == 'front':
                key = 'fl' if py > 0 else 'fr'
                # Check actual list if needed, but standard 4-wheel logic works
            else:
                key = 'rl' if py > 0 else 'rr'
                
            # 1. Wheel center velocity
            vwx = self.vx - self.omega * py
            vwy = self.vy + self.omega * px
            
            # 2. Wheel heading (Steer + Toe)
            # Trailers usually don't steer, just static toe
            steer = 0.0
            toe = self.toe
                
            heading = steer + toe
            
            # 3. Project velocity onto wheel plane
            v_long = vwx * np.cos(heading) + vwy * np.sin(heading)
            
            # 4. Set rotational speed
            if self.r_w > 0:
                # Store in same dict, key mapping might need care if >4 wheels
                # For now assuming standard 2 or 4 wheel setup
                self.wheel_omega[key] = v_long / self.r_w
                
            # If we have more than 4 wheels, this dict needs expanding, 
            # but current ArticulatedSegment init sets wheel_omega for fl,fr,rl,rr
            
            print(f"  Wheel {axle}_{'L' if py>0 else 'R'}: v_long={v_long:.2f} m/s -> omega={self.wheel_omega.get(key, 0):.2f} rad/s")

    def integrate_wheel_speeds(self, throttle, brake, dt):
        """
        Explicit time integration of trailer wheel speeds.
        Trailers usually have no drive, only brakes.
        """
        vx, vy, omega = self.vx, self.vy, self.omega
        
        # Calculate forces
        for tire in self.tire_positions:
            px, py = tire['x'], tire['y']
            axle = tire['axle']
            if axle == 'front': key = 'fl' if py > 0 else 'fr'
            else: key = 'rl' if py > 0 else 'rr'
            
            # Velocity
            vwx = vx - omega * py
            vwy = vy + omega * px
            
            heading = 0.0 + self.toe # No steer
            
            v_long = vwx * np.cos(heading) + vwy * np.sin(heading)
            
            # Slip
            omega_w = self.wheel_omega.get(key, 0.0)
            v_wheel = omega_w * self.r_w
            
            if abs(v_long) > 0.1:
                kappa = (v_wheel - v_long) / abs(v_long)
            else:
                kappa = 0.0
            kappa = np.clip(kappa, -1.0, 1.0)
            
             # Load (Approx static)
            if axle == 'front':
                Fz = self.mass * 9.81 * (self.lr / (self.lf + self.lr)) / 2.0
            else:
                Fz = self.mass * 9.81 * (self.lf / (self.lf + self.lr)) / 2.0
            
            # Tire Force Fx
            alpha = math.atan2(vwy, max(abs(vwx), 0.1)) - heading
            Fx, _ = self.tire_model.combined_slip(alpha, kappa, Fz, self.static_camber, self.mu)
            Fx = np.clip(Fx, -self.mu*Fz, self.mu*Fz)
            
            # Brake Torque (with pneumatic delay)
            # PNEUMATIC BRAKE DELAY: filter brake input through air brake model
            brake_eff = brake
            if ENABLE_BRAKE_DELAY and getattr(self, 'brake_system', None) is not None:
                brake_eff = self.brake_system.update(brake_eff, dt)
            
            # Brake torque sized to wheel load (pneumatic brake chambers scale with axle rating)
            brake_design_mu = 0.85
            max_brake_torque_wheel = Fz * brake_design_mu * self.r_w
            T_brake = brake_eff * max_brake_torque_wheel
            
            # ABS modulation for trailer wheel-speed integration
            if ENABLE_ABS and self.abs_controller is not None and brake > 0.01:
                # Build per-wheel brake force dict from torque
                F_brake_per_wheel = T_brake / max(self.r_w, 0.01)  # Convert torque to force
                temp_brake_dict = {key: abs(F_brake_per_wheel)}
                temp_kappas = {key: kappa}
                modulated_dict, _ = self.abs_controller.modulate(temp_brake_dict, temp_kappas, vx, dt)
                T_brake = modulated_dict.get(key, abs(F_brake_per_wheel)) * self.r_w * np.sign(omega_w)
            else:
                T_brake = T_brake * np.sign(omega_w)
            
            T_drive = 0.0 # No drive for now
            
            # Integration
            # I * dw/dt = -T_brake - Fx * r
            torque_net = T_drive - T_brake - Fx * self.r_w
            
            dw_dt = torque_net / self.wheel_inertia
            
            current_omega = self.wheel_omega.get(key, 0.0)
            self.wheel_omega[key] = current_omega + dw_dt * dt
            
            if abs(self.wheel_omega[key]) < 0.1 and abs(T_brake) > 0.0 and abs(vx) < 0.1:
                self.wheel_omega[key] = 0.0



    def advance_tire_states(self, state_new, state_old, dt):
        """
        Explicitly update tire relaxation state after solver convergence.
        """
        if not ENABLE_TIRE_RELAXATION or self.tire_relaxation is None:
            return

        # Unpack states
        rx, ry, ryaw, vx, vy, omega = state_new[0:6]
        
        # Calculate slip angles
        alpha_f = (vy + self.lf * omega) / max(vx, 1e-3)
        alpha_r = (vy - self.lr * omega) / max(vx, 1e-3)
        
        kappa_trailer = 0.0
        
        # Front axle
        if self.has_front_axle:
            Fxf_ss, Fyf_ss = self.tire_model.combined_slip(
                alpha=alpha_f, kappa=kappa_trailer, Fz=self.mass*9.81*self.lr/(self.lf+self.lr), 
                gamma=self.static_camber, mu=self.mu
            )
            # Negate Fy: match restoring force convention used in compute_residual
            Fyf_ss = -Fyf_ss
            # Update history
            self.Fx_axle_front_transient_hist, self.Fy_axle_front_transient_hist = \
                self.tire_relaxation.compute_transient_forces(
                    Fxf_ss, Fyf_ss,
                    self.Fx_axle_front_transient_hist, self.Fy_axle_front_transient_hist,
                    vx, dt
                )

        # Rear axle
        Fxr_ss, Fyr_ss = self.tire_model.combined_slip(
            alpha=alpha_r, kappa=kappa_trailer, Fz=self.mass*9.81*self.lf/(self.lf+self.lr),
            gamma=self.static_camber, mu=self.mu
        )
        # Negate Fy: match restoring force convention used in compute_residual
        Fyr_ss = -Fyr_ss
        self.Fx_axle_rear_transient_hist, self.Fy_axle_rear_transient_hist = \
            self.tire_relaxation.compute_transient_forces(
                Fxr_ss, Fyr_ss,
                self.Fx_axle_rear_transient_hist, self.Fy_axle_rear_transient_hist,
                vx, dt
            )

    def update(self, leader_x, leader_y, leader_z, leader_yaw, leader_pitch, leader_vx, leader_vy, leader_yaw_rate, leader_steer, dt, follower_info=None):
        """Update trailer state using implicit Euler with Newton-Raphson solver.
        
        Args:
            follower_info: HitchParams object with follower's target state (for implicit coupling)
                          or None if no trailer behind.
        """
        # Store follower info for use in compute_residual
        self.follower_info = follower_info
        
        # Update articulation yaw and rate
        self.articulation_yaw = normalize_angle(leader_yaw - self.yaw)
        self.articulation_yaw_rate = (self.articulation_yaw - self.prev_articulation_yaw) / dt
        self.prev_articulation_yaw = self.articulation_yaw

        # ========================================================================
        # Roll Dynamics (before implicit solver, like tractor)
        # ========================================================================
        # Centripetal acceleration causes roll moment
        ay_est = self.vx * self.omega if abs(self.vx) > 1e-3 else 0.0
        M_roll = self.mass * ay_est * self.h_cg - self.K_phi * self.roll - self.C_phi * self.roll_rate
        
        # Integrate roll
        self.roll_rate += (M_roll / self.Ixx) * dt
        self.roll += self.roll_rate * dt
        
        # Clamp roll (trailers can roll more than tractors)
        self.roll = np.clip(self.roll, -np.radians(12), np.radians(12))
        self.roll_rate = np.clip(self.roll_rate, -np.radians(60), np.radians(60))

        # ========================================================================
        # Pitch Dynamics
        # ========================================================================
        # Estimate longitudinal acceleration
        if hasattr(self, 'vx_prev'):
            ax_est = (self.vx - self.vx_prev) / dt if dt > 1e-6 else 0.0
        else:
            ax_est = 0.0
        self.vx_prev = self.vx

        # Pitch moment from longitudinal acceleration
        M_pitch = self.mass * ax_est * self.h_cg

        # Anti-dive (trailers only brake, no drive)
        if ax_est < 0:  # Braking
            M_pitch -= self.anti_dive * abs(ax_est) * self.mass * self.h_cg

        # Suspension restoring moment
        M_pitch -= self.K_theta * self.pitch_dyn + self.C_theta * self.pitch_rate

        # Integrate pitch
        self.pitch_rate += (M_pitch / self.Iyy) * dt
        self.pitch_dyn += self.pitch_rate * dt

        # Clamp pitch
        self.pitch_dyn = np.clip(self.pitch_dyn, -np.radians(6), np.radians(6))
        self.pitch_rate = np.clip(self.pitch_rate, -np.radians(30), np.radians(30))

        # Total pitch = terrain + dynamic
        self.pitch_terrain = get_terrain_pitch(self.x, self.y, self.waypoints)
        self.pitch = self.pitch_terrain + self.pitch_dyn

        # State vector: [x, y, yaw, vx, vy, omega]
        x_old = np.array([self.x, self.y, self.yaw, self.vx, self.vy, self.omega], dtype=float)
        
        # ========================================================================
        # EXPLICIT EULER PREDICTOR (matching tractor solver)
        # Uses actual accelerations from previous timestep for better initial guess
        # ========================================================================
        # NEWMARK-BASED PREDICTOR (consistent with Generalized-α)
        # ========================================================================
        vx_old = self.vx
        vy_old = self.vy
        omega_old = self.omega
        yaw_old = self.yaw
        dt2 = dt * dt
        
        # Velocity extrapolation
        vx_guess = max(0.0, vx_old + dt * self.ax)
        vy_guess = vy_old + dt * self.ay
        omega_guess = omega_old + dt * self.alpha
        
        # Yaw extrapolation (2nd order)
        yaw_guess = yaw_old + dt * omega_old + 0.5 * dt2 * self.alpha
        yaw_guess = np.arctan2(np.sin(yaw_guess), np.cos(yaw_guess))
        
        # Position extrapolation (2nd order, global frame)
        cos_yaw = np.cos(yaw_old)
        sin_yaw = np.sin(yaw_old)
        vx_global = vx_old * cos_yaw - vy_old * sin_yaw
        vy_global = vx_old * sin_yaw + vy_old * cos_yaw
        ax_global = self.ax * cos_yaw - self.ay * sin_yaw
        ay_global = self.ax * sin_yaw + self.ay * cos_yaw
        
        x_guess = x_old[0] + dt * vx_global + 0.5 * dt2 * ax_global
        y_guess = x_old[1] + dt * vy_global + 0.5 * dt2 * ay_global
        
        # Acceleration guess (will be solved for)
        ax_guess = self.ax
        ay_guess = self.ay
        alpha_guess = self.alpha
        
        # Newton-Raphson settings
        max_iterations = MAX_ITERATIONS_TRAILER

        # ========================================================================
        # GENERALIZED-ALPHA SOLVER (Pure implementation - no legacy fallback)
        # ========================================================================
        
        if not USE_GENERALIZED_ALPHA:
            raise RuntimeError("Legacy Backward Euler solver has been removed. Set USE_GENERALIZED_ALPHA = True")
        
        if USE_GENERALIZED_ALPHA:
            # ================================================================
            # GENERALIZED-ALPHA SOLVER (2nd-order accurate)
            # ================================================================
            state_old = np.array([
                self.x, self.y, self.yaw,
                vx_old, vy_old, omega_old,
                self.ax, self.ay, self.alpha
            ], dtype=float)
            
            state_new = np.array([
                x_guess, y_guess, yaw_guess,
                vx_guess, vy_guess, omega_guess,
                self.ax, self.ay, self.alpha
            ], dtype=float)
            
            # RELAXED CONVERGENCE during startup
            if hasattr(self, '_sim_time') and self._sim_time < 5.0:
                residual_threshold = RESIDUAL_THRESHOLD_STARTUP
            else:
                residual_threshold = RESIDUAL_THRESHOLD_TRAILER_STEADY
            
            state_change_threshold = 1e-7
            energy_change_threshold = 1e-6  # Energy increment convergence
            it = 0
            residual_norm = float('inf')
            solver_converged = False
            
            # Energy tracking for convergence
            E_kinetic_prev = 0.5 * self.mass * (state_old[3]**2 + state_old[4]**2) + 0.5 * self.Iz * state_old[5]**2
            energy_increment_prev = 0.0
            
            while it < max_iterations:
                f = self.compute_residual_genalpha(state_new, state_old, leader_x, leader_y, leader_yaw,
                                                    leader_vx, leader_vy, leader_yaw_rate, leader_steer, dt)
                residual_norm = np.linalg.norm(f)
                
                # Compute kinetic energy at current trial state
                vx_trial, vy_trial, omega_trial = state_new[3], state_new[4], state_new[5]
                E_kinetic_new = 0.5 * self.mass * (vx_trial**2 + vy_trial**2) + 0.5 * self.Iz * omega_trial**2
                dE = E_kinetic_new - E_kinetic_prev
                
                # Check residual convergence (primary)
                if residual_norm < residual_threshold:
                    solver_converged = True
                    break
                
                J = self.compute_jacobian_genalpha(state_new, state_old, leader_x, leader_y, leader_yaw,
                                                    leader_vx, leader_vy, leader_yaw_rate, leader_steer, dt)
                
                try:
                    dx = np.linalg.solve(J, -f)
                except np.linalg.LinAlgError:
                    dx = np.linalg.lstsq(J, -f, rcond=None)[0]
                
                # Check state change convergence (secondary)
                state_change = np.linalg.norm(dx) / (np.linalg.norm(state_new) + 1e-6)
                if state_change < state_change_threshold and residual_norm < residual_threshold * 10:
                    solver_converged = True
                    break
                
                # Check energy increment convergence (tertiary)
                if it > 0:
                    energy_increment_change = abs(dE - energy_increment_prev) / (abs(dE) + 1e-10)
                    if energy_increment_change < energy_change_threshold and residual_norm < residual_threshold * 20:
                        solver_converged = True
                        break
                
                energy_increment_prev = dE
                
                # Limit step size
                dx_norm = np.linalg.norm(dx)
                if dx_norm > 30.0:
                    dx = dx * (30.0 / dx_norm)
                
                # Simple line search
                step = 1.0
                for ls_iter in range(10):
                    state_trial = state_new + step * dx
                    f_trial = self.compute_residual_genalpha(state_trial, state_old, leader_x, leader_y, leader_yaw,
                                                              leader_vx, leader_vy, leader_yaw_rate, leader_steer, dt)
                    if np.linalg.norm(f_trial) < 0.95 * residual_norm or step < 0.01:
                        state_new = state_trial
                        break
                    step *= 0.5
                
                it += 1
            
            # Commit new state
            self.x, self.y, self.yaw = float(state_new[0]), float(state_new[1]), float(state_new[2])
            self.vx, self.vy, self.omega = float(state_new[3]), float(state_new[4]), float(state_new[5])
            self.ax, self.ay, self.alpha = float(state_new[6]), float(state_new[7]), float(state_new[8])
            
            self.ax_prev = self.ax
            self.ay_prev = self.ay
            self.alpha_prev = self.alpha
            
            # ================================================================
            # TIRE RELAXATION HISTORY UPDATE (CRITICAL for Gen-Alpha path)
            # ================================================================
            # Same fix as TractorHead: genalpha residual reads but doesn't
            # write transient_hist. Update here after convergence.
            if ENABLE_TIRE_RELAXATION:
                relax = self.tire_relaxation if hasattr(self, 'tire_relaxation') else None
                if relax is not None:
                    for key in ['rl', 'rr']:  # Trailers typically have rear axle only
                        if not hasattr(self, f'Fx_axle_{"front" if key in ("fl","fr") else "rear"}_transient_hist'):
                            continue
                        
                        # Compute steady-state forces at converged state
                        # Use simplified axle-level computation matching trailer solver
                        vwx = self.vx
                        vwy = self.vy + self.omega * (-self.lr)  # rear axle
                        alpha_w = math.atan2(vwy, max(abs(vwx), 0.1))
                        Fz_w = self.mass * 9.81 / 2  # simplified per-tire
                        
                        Fx_ss, Fy_ss = self.tire_model.combined_slip(alpha_w, 0, Fz_w, 0, self.mu)
                        Fy_ss = -Fy_ss  # Negate convention
                        
                        hist_attr_Fx = 'Fx_axle_rear_transient_hist'
                        hist_attr_Fy = 'Fy_axle_rear_transient_hist'
                        if hasattr(self, hist_attr_Fx):
                            hist_Fx = getattr(self, hist_attr_Fx)
                            hist_Fy = getattr(self, hist_attr_Fy)
                            Fx_new, Fy_new = relax.compute_transient_forces(
                                Fx_ss, Fy_ss, hist_Fx, hist_Fy, self.vx, dt)
                            setattr(self, hist_attr_Fx, Fx_new)
                            setattr(self, hist_attr_Fy, Fy_new)
                        break  # Only one rear axle update needed

        # ==================================================================
        # POST-PROCESSING
        # ==================================================================
        # Guard against negative forward speed
        self.vx = max(self.vx, 0.0)
        
        # CRITICAL FIX: Progressive low-speed stabilization to prevent spin
        # At low speeds, lateral dynamics should be heavily damped to prevent instability
        if self.vx < 0.5:  # Progressive damping below 0.5 m/s (reduced from 1.0)
            # Progressive damping: stronger at lower speeds
            # At vx=0: 100% damping (vy=0, omega=0)
            # At vx=0.5: 0% damping (full dynamics)
            damping_factor = self.vx / 0.5  # 0 to 1 range
            
            # Apply progressive damping to lateral velocity and yaw rate
            self.vy = self.vy * damping_factor
            self.omega = self.omega * damping_factor
            
            # Hard limits at EXTREMELY low speeds to prevent numerical explosion
            # Only force to zero when essentially stationary
            if self.vx < 0.001:  # Changed from 0.01 to 0.001
                self.vy = 0.0
                self.omega = 0.0
        
        # FIX 1: ARTICULATION ANGLE LIMITS (prevent jackknifing)
        # Real trailer hitches have mechanical stops around ±60-70°
        articulation_current = normalize_angle(leader_yaw - self.yaw)
        max_articulation = np.radians(70)  # ±70° typical mechanical limit
        
        if abs(articulation_current) > max_articulation:
            # Clamp yaw to enforce articulation limit
            if articulation_current > 0:
                self.yaw = normalize_angle(leader_yaw - max_articulation)
            else:
                self.yaw = normalize_angle(leader_yaw + max_articulation)
        
        # FIX 3: VELOCITY SANITY CHECKS (prevent extreme lateral velocities)
        # Lateral velocity should not exceed 2x forward speed in realistic scenarios
        if self.vx > 0.001:  # Only when moving
            max_lateral_velocity = 2.0 * self.vx
            if abs(self.vy) > max_lateral_velocity:
                self.vy = np.sign(self.vy) * max_lateral_velocity
        
        # Also cap yaw rate to reasonable limit (prevent extreme spinning)
        max_yaw_rate = 3.0  # rad/s (about 170°/s, realistic upper limit for trailers)
        if abs(self.omega) > max_yaw_rate:
            self.omega = np.sign(self.omega) * max_yaw_rate
        
        
        # Update terrain elevation (z position) - CRITICAL: Must come BEFORE return
        self.z = get_terrain_elevation(self.x, self.y, self.waypoints)
        
        # Print convergence status with detailed diagnostics
        if not solver_converged:
            articulation_deg = np.degrees(normalize_angle(leader_yaw - self.yaw))
            print(f"[TRAILER_SOLVER_FAIL] WARNING: Trailer solver did not converge! Error={residual_norm:.2e}")
            print(f"[TRAILER_SOLVER_FAIL] t={self._sim_time:.2f}s vx={self.vx:.2f} vy={self.vy:.2f} omega={self.omega:.3f}")
            print(f"[TRAILER_SOLVER_FAIL] Articulation angle={articulation_deg:.1f}° leader_vx={leader_vx:.2f} leader_steer={np.degrees(leader_steer):.1f}°")
        
        return solver_converged
    
    def compute_hitch_force(self, leader_x, leader_y, leader_yaw, leader_vx):
        """
        Compute hitch reaction force on the leader using the constraint-target model.
        
        Geometry:
          Leader CG --[leader_rear_offset]--> Pin Joint --[hitch_length + lf]--> Trailer CG
        
        Returns positive force when trailer is being pulled (leader feels drag).
        """
        # 1. Leader's rear hitch point (pin joint) in global frame
        hitch_x = leader_x - self.leader_rear_offset * np.cos(leader_yaw)
        hitch_y = leader_y - self.leader_rear_offset * np.sin(leader_yaw)
        
        # 2. Target trailer CG (hitch_length = coupling gap, lf = CG to front)
        front_offset = self.hitch_length + self.lf
        target_x = hitch_x - front_offset * np.cos(self.yaw)
        target_y = hitch_y - front_offset * np.sin(self.yaw)
        
        # 3. Position error
        err_x = self.x - target_x
        err_y = self.y - target_y
        
        # 4. Velocity error
        leader_vx_global = leader_vx * np.cos(leader_yaw)
        leader_vy_global = leader_vx * np.sin(leader_yaw)
        
        trailer_vx_global = self.vx * np.cos(self.yaw) - self.vy * np.sin(self.yaw)
        trailer_vy_global = self.vx * np.sin(self.yaw) + self.vy * np.cos(self.yaw)
        
        vel_err_x = trailer_vx_global - leader_vx_global
        vel_err_y = trailer_vy_global - leader_vy_global
        
        # 5. Constraint force in global frame (on trailer)
        Fx_hitch_global = -self.k_hitch * err_x - self.c_hitch * vel_err_x
        Fy_hitch_global = -self.k_hitch * err_y - self.c_hitch * vel_err_y
        
        # 6. Reaction on leader (Newton's 3rd law) → project onto leader's axis
        Fx_on_leader = -Fx_hitch_global
        Fy_on_leader = -Fy_hitch_global
        F_long = Fx_on_leader * np.cos(leader_yaw) + Fy_on_leader * np.sin(leader_yaw)
        
        return -F_long  # positive = drag
    
    def compute_residual(self, x_new, x_old, leader_x, leader_y, leader_yaw, leader_vx, leader_vy, leader_yaw_rate, leader_steer, dt):
        # Unpack states
        rx, ry, ryaw, vx, vy, omega = x_new
        rx_old, ry_old, ryaw_old, vx_old, vy_old, omega_old = x_old

        # Kinematic residuals (Position updates)
        pitch_candidate = get_terrain_pitch(rx, ry, self.waypoints)
        
        # Kinematic equations (global frame)
        res_x = rx - rx_old - dt * (vx * np.cos(ryaw) - vy * np.sin(ryaw))
        res_y = ry - ry_old - dt * (vx * np.sin(ryaw) + vy * np.cos(ryaw))
        res_yaw = ryaw - ryaw_old - dt * omega

        # Dynamic residuals (Velocity updates)
        # Compute vertical loads for tire model (static + longitudinal load transfer)
        L_trailer = self.lf + self.lr
        Fz_front_static = self.mass * 9.81 * self.lr / L_trailer / 2.0  # Per wheel
        Fz_rear_static = self.mass * 9.81 * self.lf / L_trailer / 2.0   # Per wheel
        
        # Longitudinal load transfer: ΔFz = m·ax·h_cg / L
        ax_trailer = getattr(self, 'ax_prev', 0.0)
        h_cg_trailer = getattr(self, 'h_cg', 1.0)
        deltaFz_pitch_trailer = (self.mass * ax_trailer * h_cg_trailer) / L_trailer / 2.0
        
        Fz_front = max(50.0, Fz_front_static - deltaFz_pitch_trailer)  # front unloads during accel
        Fz_rear  = max(50.0, Fz_rear_static + deltaFz_pitch_trailer)   # rear loads up during accel

        # Lateral load transfer: ΔFz_lateral = m·ay·h_cg / track_width
        # FIX: Use cached ay from force balance (computed in previous residual) for accuracy
        ay_cached = getattr(self, '_ay_for_load_transfer', 0.0)
        ay_for_Fz = ay_cached if abs(ay_cached) > 1e-6 else ((vx * omega) if abs(vx) > 1e-3 else 0.0)
        deltaFz_roll_trailer = (self.mass * ay_for_Fz * h_cg_trailer) / max(self.track_width, 0.1) / 2.0
        # Apply per-wheel lateral load transfer (assuming front Fz = rear Fz split symmetrically)
        # Left side unloads, right side loads (for positive ay = left turn)
        Fz_front_left  = max(50.0, Fz_front - deltaFz_roll_trailer)
        Fz_front_right = max(50.0, Fz_front + deltaFz_roll_trailer)
        Fz_rear_left   = max(50.0, Fz_rear  - deltaFz_roll_trailer)
        Fz_rear_right  = max(50.0, Fz_rear  + deltaFz_roll_trailer)

        
        # Slip angles (bicycle model approximation for trailers)
        # FIX: Removed leader_steer from alpha calculation - trailers are not steered by the tractor's wheel angle!
        # The hitch moves, but the trailer wheels don't steer.
        alpha_f = (vy + self.lf * omega) / max(vx, 1e-3)
        alpha_r = (vy - self.lr * omega) / max(vx, 1e-3)

        # =================================================================
        # COMBINED-SLIP PACEJKA TIRE MODEL FOR TRAILER
        # =================================================================
        # Compute per-wheel slip ratios from wheel angular velocities
        # (replaces hardcoded kappa_trailer = 0.0)
        omega_fl = self.wheel_omega.get('fl', 0.0)
        omega_fr = self.wheel_omega.get('fr', 0.0)
        omega_rl = self.wheel_omega.get('rl', 0.0)
        omega_rr = self.wheel_omega.get('rr', 0.0)
        
        vx_safe = max(abs(vx), 0.1)
        
        # Front axle average κ
        if self.has_front_axle:
            v_wheel_f = 0.5 * (omega_fl + omega_fr) * self.r_w
            kappa_f = np.clip((v_wheel_f - vx) / vx_safe, -1.0, 1.0)
        else:
            kappa_f = 0.0
        
        # Rear axle average κ
        v_wheel_r = 0.5 * (omega_rl + omega_rr) * self.r_w
        kappa_r = np.clip((v_wheel_r - vx) / vx_safe, -1.0, 1.0)
        
        # Store per-wheel kappas for ABS
        self._wheel_kappas['fl'] = np.clip((omega_fl * self.r_w - vx) / vx_safe, -1.0, 1.0) if self.has_front_axle else 0.0
        self._wheel_kappas['fr'] = np.clip((omega_fr * self.r_w - vx) / vx_safe, -1.0, 1.0) if self.has_front_axle else 0.0
        self._wheel_kappas['rl'] = np.clip((omega_rl * self.r_w - vx) / vx_safe, -1.0, 1.0)
        self._wheel_kappas['rr'] = np.clip((omega_rr * self.r_w - vx) / vx_safe, -1.0, 1.0)
        
        # Get tire forces from Pacejka model
        if self.has_front_axle:
            Fxf_tire_ss, Fyf_ss = self.tire_model.combined_slip(
                alpha=alpha_f,
                kappa=kappa_f,
                Fz=Fz_front * 2.0,
                gamma=self.static_camber,
                mu=self.mu
            )
            # FIX: Negate lateral force for stability
            Fyf_ss = -Fyf_ss
        else:
            Fxf_tire_ss = 0.0
            Fyf_ss = 0.0
        
        Fxr_tire_ss, Fyr_ss = self.tire_model.combined_slip(
            alpha=alpha_r,
            kappa=kappa_r,
            Fz=Fz_rear * 2.0,
            gamma=self.static_camber,
            mu=self.mu
        )
        # FIX: Negate lateral force for stability
        Fyr_ss = -Fyr_ss
        
        # Low-speed lateral force reduction
        if vx < 1.0:
            speed_factor = max(vx, 0.0)
            Fyf_ss = Fyf_ss * speed_factor
            Fyr_ss = Fyr_ss * speed_factor
            Fxf_tire_ss = Fxf_tire_ss * speed_factor
            Fxr_tire_ss = Fxr_tire_ss * speed_factor
        
        # =================================================================
        # TIRE RELAXATION
        # =================================================================
        if ENABLE_TIRE_RELAXATION:
            # Update front tires only if they exist
            if self.has_front_axle:
                # Use HISTORY state as input, return CANDIDATE state (do not mutate self)
                Fx_front_transient, Fy_front_transient = self.tire_relaxation.compute_transient_forces(
                    Fxf_tire_ss, Fyf_ss,
                    self.Fx_axle_front_transient_hist, self.Fy_axle_front_transient_hist,
                    vx, dt
                )
                Fxf_tire = Fx_front_transient
                Fyf = Fy_front_transient
            else:
                Fxf_tire = 0.0
                Fyf = 0.0

            # Update rear tires
            Fx_rear_transient, Fy_rear_transient = self.tire_relaxation.compute_transient_forces(
                Fxr_tire_ss, Fyr_ss,
                self.Fx_axle_rear_transient_hist, self.Fy_axle_rear_transient_hist,
                vx, dt
            )
            
            Fxr_tire = Fx_rear_transient
            Fyr = Fy_rear_transient
        else:
            Fxf_tire = Fxf_tire_ss
            Fyf = Fyf_ss
            Fxr_tire = Fxr_tire_ss
            Fyr = Fyr_ss

        # ====================================================================
        # ARTICULATION TORQUE (rotational spring-damper at pivot)
        # ====================================================================
        articulation_yaw = normalize_angle(leader_yaw - ryaw)
        tau_art = -self.k_psi * articulation_yaw - self.c_psi * (leader_yaw_rate - omega)
        
        # Hard angle stops: stiff contact spring beyond ±max_articulation
        if abs(articulation_yaw) > self.max_articulation:
            penetration = abs(articulation_yaw) - self.max_articulation
            art_rate = leader_yaw_rate - omega
            tau_stop = -np.sign(articulation_yaw) * (self.k_stop * penetration + self.c_stop * art_rate)
            tau_art += tau_stop

        vehicle_components = [{'Cd': 0.6, 'A': 8.0, 'mass': self.mass, 'Crr': 0.008}]
        F_load = compute_aero_and_rolling_resistance(vx, vehicle_components)

        # ====================================================================
        # HITCH COUPLING — CONSTRAINT-TARGET MODEL
        # ====================================================================
        # Geometry: Leader CG --[leader_rear_offset]--> Pin --[hitch_length+lf]--> Trailer CG
        # The pin joint is at the leader's rear hitch point.
        # The coupling gap (hitch_length) + trailer front offset (lf) define
        # the trailer arm that pivots with the trailer's yaw.
        # ====================================================================
        
        # 1. Leader's rear hitch point (pin joint, global frame)
        hitch_x = leader_x - self.leader_rear_offset * np.cos(leader_yaw)
        hitch_y = leader_y - self.leader_rear_offset * np.sin(leader_yaw)
        
        # 2. Target trailer CG position (from pin + front offset in trailer direction)
        front_offset = self.hitch_length + self.lf
        cos_yaw = np.cos(ryaw)
        sin_yaw = np.sin(ryaw)
        target_x = hitch_x - front_offset * cos_yaw
        target_y = hitch_y - front_offset * sin_yaw
        
        # 3. Position error in global frame
        err_x = rx - target_x
        err_y = ry - target_y
        
        # 4. Velocity error in global frame
        #    Trailer actual global velocity
        trailer_vx_global = vx * cos_yaw - vy * sin_yaw
        trailer_vy_global = vx * sin_yaw + vy * cos_yaw
        #    Target velocity (time-derivative of constraint):
        #    ẋ_target = ẋ_L + d_L·sin(ψ_L)·ω_L + front_offset·sin(ψ_T)·ω_T
        #    ẏ_target = ẏ_L - d_L·cos(ψ_L)·ω_L - front_offset·cos(ψ_T)·ω_T
        leader_vx_global = leader_vx * np.cos(leader_yaw) - leader_vy * np.sin(leader_yaw)
        leader_vy_global = leader_vx * np.sin(leader_yaw) + leader_vy * np.cos(leader_yaw)
        
        target_vx = leader_vx_global + self.leader_rear_offset * np.sin(leader_yaw) * leader_yaw_rate \
                                      + front_offset * sin_yaw * omega
        target_vy = leader_vy_global - self.leader_rear_offset * np.cos(leader_yaw) * leader_yaw_rate \
                                      - front_offset * cos_yaw * omega
        
        vel_err_x = trailer_vx_global - target_vx
        vel_err_y = trailer_vy_global - target_vy
        
        # 5. Constraint force on trailer in GLOBAL frame (pulls toward target)
        Fx_hitch_global = -self.k_hitch * err_x - self.c_hitch * vel_err_x
        Fy_hitch_global = -self.k_hitch * err_y - self.c_hitch * vel_err_y
        
        # 6. Rotate hitch force into trailer body frame
        Fx_hitch_body = Fx_hitch_global * cos_yaw + Fy_hitch_global * sin_yaw
        Fy_hitch_body = -Fx_hitch_global * sin_yaw + Fy_hitch_global * cos_yaw

        # ====================================================================
        # TOTAL FORCES
        # ====================================================================
        # Front axle forces (Body Frame) — tire only (hitch force separated for correct moment arm)
        Fx_front_body = Fxf_tire
        Fy_front_body = Fyf

        # Rear axle forces (Body Frame)
        Fx_rear_body = Fxr_tire
        Fy_rear_body = Fyr

        # Total forces (tires + hitch + aero/rolling resistance)
        # Apply trailer braking force (distributed per axle)
        brake_input = getattr(self, '_brake_input', 0.0)
        # PNEUMATIC BRAKE DELAY: filter brake input through air brake model
        if ENABLE_BRAKE_DELAY and getattr(self, 'brake_system', None) is not None:
            brake_input = self.brake_system.update(brake_input, dt)
        F_brake_trailer = brake_input * self.mass * self.max_brake_decel
        if F_brake_trailer > 1.0:
            F_brake_front = F_brake_trailer * self.brake_force_distribution_front if self.has_front_axle else 0.0
            F_brake_rear  = F_brake_trailer * self.brake_force_distribution_rear
            if not self.has_front_axle:
                F_brake_rear = F_brake_trailer  # All braking on rear if no front axle
            
            # Build per-wheel brake dict for ABS
            brake_forces_dict = {
                'fl': F_brake_front / 2.0 if self.has_front_axle else 0.0,
                'fr': F_brake_front / 2.0 if self.has_front_axle else 0.0,
                'rl': F_brake_rear / 2.0,
                'rr': F_brake_rear / 2.0
            }
            
            # ABS modulation
            if ENABLE_ABS and self.abs_controller is not None:
                brake_forces_dict, abs_active = self.abs_controller.modulate(
                    brake_forces_dict, self._wheel_kappas, vx, dt
                )
                if any(abs_active.values()):
                    diag = self.abs_controller.get_diagnostics()
                    pf = diag['pressure_factor']
                    print(f"[ABS TRAILER] ACTIVE pf_rl={pf['rl']:.2f} pf_rr={pf['rr']:.2f}")
            
            # Sum per-wheel brake forces back to axle level for bicycle model
            F_brake_front_eff = brake_forces_dict['fl'] + brake_forces_dict['fr']
            F_brake_rear_eff  = brake_forces_dict['rl'] + brake_forces_dict['rr']
        else:
            F_brake_front_eff = 0.0
            F_brake_rear_eff = 0.0
            # Reset ABS when not braking
            if ENABLE_ABS and self.abs_controller is not None:
                self.abs_controller.reset()
        
        sum_Fx = Fx_front_body + Fx_rear_body + Fx_hitch_body - F_load - F_brake_front_eff - F_brake_rear_eff
        sum_Fy = Fy_front_body + Fy_rear_body + Fy_hitch_body

        # ====================================================================
        # IMPLICIT FOLLOWER COUPLING (force from trailer BEHIND this one)
        # ====================================================================
        if self.follower_info:
            # 1. This unit's rear hitch point (pin joint) in global frame
            # self.lr is distance from CG to rear axle/hitch
            hitch_x_rear = rx - self.lr * np.cos(ryaw)
            hitch_y_rear = ry - self.lr * np.sin(ryaw)
            
            # 2. Follower target position target
            target_x, target_y = self.follower_info.target_pos
            
            # 3. Position error
            dx = hitch_x_rear - target_x
            dy = hitch_y_rear - target_y
            
            # 4. Velocity error
            # Current velocity of rear hitch point
            vx_hitch = vx * np.cos(ryaw) - vy * np.sin(ryaw) - self.lr * (-np.sin(ryaw) * omega)
            vy_hitch = vx * np.sin(ryaw) + vy * np.cos(ryaw) + self.lr * (np.cos(ryaw) * omega)
            
            target_vx, target_vy = self.follower_info.target_vel
            dvx = vx_hitch - target_vx
            dvy = vy_hitch - target_vy
            
            # 5. Global hitch force on THIS UNIT (pulled back by follower)
            Fx_hitch_global = -self.follower_info.stiffness * dx - self.follower_info.damping * dvx
            Fy_hitch_global = -self.follower_info.stiffness * dy - self.follower_info.damping * dvy
            
            # 6. Transform to body frame
            Fx_hitch_body = Fx_hitch_global * np.cos(ryaw) + Fy_hitch_global * np.sin(ryaw)
            Fy_hitch_body = -Fx_hitch_global * np.sin(ryaw) + Fy_hitch_global * np.cos(ryaw)
            
            # 7. Add to sums
            sum_Fx += Fx_hitch_body
            sum_Fy += Fy_hitch_body 
            
            # 8. Moment: rear hitch is at (-lr, 0)
            sum_Mz_follower = (-self.lr) * Fy_hitch_body
        else:
            sum_Mz_follower = 0.0

        # Accelerations (body-frame equations of motion)
        # dvx/dt = Fx/m + vy*omega  (Coriolis/centripetal term)
        # dvy/dt = Fy/m - vx*omega  (Coriolis/centripetal term)
        ax = sum_Fx / self.mass + vy * omega
        ay = sum_Fy / self.mass - vx * omega
        
        # ===================================================================
        # FIX: Cache the computed lateral acceleration for load transfer
        # ===================================================================
        self._ay_for_load_transfer = ay

        # Moments — separate arms for tire forces and hitch force
        Mz_front_lat = self.lf * Fy_front_body
        Mz_rear = -self.lr * Fy_rear_body
        # FIX: Hitch force acts at (hitch_length + lf) ahead of CG, not at lf
        Mz_hitch = (self.hitch_length + self.lf) * Fy_hitch_body
        
        alpha_ang_acc = (Mz_front_lat + Mz_rear + Mz_hitch + tau_art + sum_Mz_follower) / self.Iz

        # Residuals
        res_vx = vx - vx_old - dt * ax
        res_vy = vy - vy_old - dt * ay
        res_omega = omega - omega_old - dt * alpha_ang_acc

        return np.array([res_x, res_y, res_yaw, res_vx, res_vy, res_omega])

    def compute_jacobian(self, x_new, x_old, leader_x, leader_y, leader_yaw, leader_vx, leader_vy, leader_yaw_rate, leader_steer, dt):
        eps = 1e-6  # Match tractor precision
        dim = 6
        J = np.zeros((dim, dim), dtype=float)  # Explicit dtype
        f0 = self.compute_residual(x_new, x_old, leader_x, leader_y, leader_yaw,
                                    leader_vx, leader_vy, leader_yaw_rate, leader_steer, dt)
        
        # Central finite difference (2nd order accurate - matches TractorHead)
        for i in range(dim):
            x_plus = x_new.copy();  x_plus[i] += eps
            x_minus = x_new.copy(); x_minus[i] -= eps
            
            f_plus = self.compute_residual(x_plus, x_old, leader_x, leader_y, leader_yaw,
                                            leader_vx, leader_vy, leader_yaw_rate, leader_steer, dt)
            f_minus = self.compute_residual(x_minus, x_old, leader_x, leader_y, leader_yaw,
                                             leader_vx, leader_vy, leader_yaw_rate, leader_steer, dt)
            
            J[:, i] = (f_plus - f_minus) / (2.0 * eps)
        
        return J

    def compute_energy_balance(self, state_new, state_old, leader_x, leader_y, leader_yaw, leader_vx, leader_vy, leader_yaw_rate, leader_steer, dt):
        """
        Compute complete energy balance for convergence checking (ArticluatedSegment).
        """
        # Unpack states
        x_old, y_old, yaw_old = state_old[0], state_old[1], state_old[2]
        vx_old, vy_old, omega_old = state_old[3], state_old[4], state_old[5]
        
        x_new, y_new, yaw_new = state_new[0], state_new[1], state_new[2]
        vx_new, vy_new, omega_new = state_new[3], state_new[4], state_new[5]
        
        # Kinetic Energy
        E_k_old = 0.5 * self.mass * (vx_old**2 + vy_old**2) + 0.5 * self.Iz * omega_old**2
        E_k_new = 0.5 * self.mass * (vx_new**2 + vy_new**2) + 0.5 * self.Iz * omega_new**2
        dE_kinetic = E_k_new - E_k_old
        
        # Potential Energy (Springs) - Roll/Pitch
        E_p_spring_old = 0.5 * self.K_phi * self.roll**2 + 0.5 * self.K_theta * self.pitch_dyn**2
        E_p_spring_new = E_p_spring_old # Updated outside loop
        dE_spring = E_p_spring_new - E_p_spring_old
        
        # Potential Energy (Gravity)
        z_old = get_terrain_elevation(x_old, y_old, self.waypoints)
        z_new = get_terrain_elevation(x_new, y_new, self.waypoints)
        E_g_old = self.mass * 9.81 * z_old
        E_g_new = self.mass * 9.81 * z_new
        dE_gravity = E_g_new - E_g_old
        
        dE_potential = dE_spring + dE_gravity
        
        # Work Input (None for trailer unless powered axis added later)
        W_engine = 0.0
        
        # Dissipation
        rho = 1.225
        F_drag = 0.5 * rho * 0.6 * 8.0 * vx_new**2 # Approx Cd/A
        W_drag = F_drag * abs(vx_new) * dt
        
        W_roll_damping = self.C_phi * self.roll_rate**2 * dt
        W_pitch_damping = self.C_theta * self.pitch_rate**2 * dt
        
        v_lateral = abs(vy_new)
        v_slip_prod = vy_new * omega_new
        if abs(vx_new) > 0.1:
             F_lateral_approx = self.mass * abs(v_slip_prod)
        else:
             F_lateral_approx = 0.0
        W_tire_friction = F_lateral_approx * v_lateral * dt * 0.1
        
        W_dissipation = W_drag + W_roll_damping + W_pitch_damping + W_tire_friction
        
        # Energy balance
        energy_balance = dE_kinetic + dE_potential - W_engine + W_dissipation
        
        E_total = abs(E_k_new) + abs(W_dissipation) + 1e-10
        energy_balance_normalized = abs(energy_balance) / E_total
        
        return {
            'dE_kinetic': dE_kinetic,
            'dE_potential': dE_potential,
            'W_engine': W_engine,
            'W_dissipation': W_dissipation,
            'energy_balance': energy_balance,
            'energy_balance_normalized': energy_balance_normalized
        }

    
    # ========================================================================
    # GENERALIZED-ALPHA TIME INTEGRATION METHODS
    # ========================================================================
    
    def compute_residual_genalpha(self, state_new, state_old, leader_x, leader_y, leader_yaw,
                                   leader_vx, leader_vy, leader_yaw_rate, leader_steer, dt):
        """
        Generalized-α residual for ArticulatedSegment with 9-state vector.
        
        State vector: [x, y, yaw, vx, vy, omega, ax, ay, alpha]
        """
        # Unpack new state
        x_new, y_new, yaw_new = state_new[0], state_new[1], state_new[2]
        vx_new, vy_new, omega_new = state_new[3], state_new[4], state_new[5]
        ax_new, ay_new, alpha_new = state_new[6], state_new[7], state_new[8]
        
        # Unpack old state
        x_old, y_old, yaw_old = state_old[0], state_old[1], state_old[2]
        vx_old, vy_old, omega_old = state_old[3], state_old[4], state_old[5]
        ax_old, ay_old, alpha_old = state_old[6], state_old[7], state_old[8]
        
        # Get Generalized-α parameters
        am = self._ga_alpha_m
        af = self._ga_alpha_f
        beta = self._ga_beta
        gamma = self._ga_gamma
        dt2 = dt * dt
        
        # =====================================================================
        # NEWMARK KINEMATIC CONSTRAINTS
        # =====================================================================
        # Velocity update
        res_vx = vx_new - vx_old - dt * ((1.0 - gamma) * ax_old + gamma * ax_new)
        res_vy = vy_new - vy_old - dt * ((1.0 - gamma) * ay_old + gamma * ay_new)
        res_omega = omega_new - omega_old - dt * ((1.0 - gamma) * alpha_old + gamma * alpha_new)
        
        # Position update in global frame
        cos_yaw_old = np.cos(yaw_old)
        sin_yaw_old = np.sin(yaw_old)
        
        vx_global_old = vx_old * cos_yaw_old - vy_old * sin_yaw_old
        vy_global_old = vx_old * sin_yaw_old + vy_old * cos_yaw_old
        ax_global_old = ax_old * cos_yaw_old - ay_old * sin_yaw_old
        ay_global_old = ax_old * sin_yaw_old + ay_old * cos_yaw_old
        ax_global_new = ax_new * np.cos(yaw_new) - ay_new * np.sin(yaw_new)
        ay_global_new = ax_new * np.sin(yaw_new) + ay_new * np.cos(yaw_new)
        
        res_x = x_new - x_old - dt * vx_global_old - dt2 * ((0.5 - beta) * ax_global_old + beta * ax_global_new)
        res_y = y_new - y_old - dt * vy_global_old - dt2 * ((0.5 - beta) * ay_global_old + beta * ay_global_new)
        res_yaw = yaw_new - yaw_old - dt * omega_old - dt2 * ((0.5 - beta) * alpha_old + beta * alpha_new)
        
        # =====================================================================
        # DYNAMIC EQUILIBRIUM - Use existing compute_residual for force evaluation
        # =====================================================================
        # Create 6-state vectors for legacy residual computation
        x6_new = np.array([x_new, y_new, yaw_new, vx_new, vy_new, omega_new])
        x6_old = np.array([x_old, y_old, yaw_old, vx_old, vy_old, omega_old])
        
        # Get dynamic residuals from legacy method
        legacy_res = self.compute_residual(x6_new, x6_old, leader_x, leader_y, leader_yaw,
                                           leader_vx, leader_vy, leader_yaw_rate, leader_steer, dt)
        
        # Extract force residuals (last 3 elements: vx, vy, omega dynamics)
        # These are: m*a - F = 0 format.
        # NORMALIZE by dividing by Mass/Inertia to get Acceleration units [m/s^2]
        # This prevents Force magnitude (e.g. 5000N) from overwhelming the solver tolerance (1e-3).
        # res_ax = ax_new - (vx_new - vx_old)/dt + legacy_res[3]/dt
        
        res_ax = ax_new - (vx_new - vx_old) / dt + legacy_res[3] / dt
        res_ay = ay_new - (vy_new - vy_old) / dt + legacy_res[4] / dt
        res_alpha = alpha_new - (omega_new - omega_old) / dt + legacy_res[5] / dt
        
        return np.array([res_x, res_y, res_yaw, res_vx, res_vy, res_omega, res_ax, res_ay, res_alpha])
    
    def compute_jacobian_genalpha(self, state_new, state_old, leader_x, leader_y, leader_yaw,
                                   leader_vx, leader_vy, leader_yaw_rate, leader_steer, dt):
        """
        Compute 9×9 Jacobian for Generalized-α residual using central differences.
        """
        dim = 9
        J = np.zeros((dim, dim), dtype=float)
        
        for i in range(dim):
            eps = max(1e-6, abs(state_new[i]) * 1e-4)
            
            state_plus = state_new.copy()
            state_plus[i] += eps
            state_minus = state_new.copy()
            state_minus[i] -= eps
            
            f_plus = self.compute_residual_genalpha(state_plus, state_old, leader_x, leader_y, leader_yaw,
                                                     leader_vx, leader_vy, leader_yaw_rate, leader_steer, dt)
            f_minus = self.compute_residual_genalpha(state_minus, state_old, leader_x, leader_y, leader_yaw,
                                                      leader_vx, leader_vy, leader_yaw_rate, leader_steer, dt)
            
            J[:, i] = (f_plus - f_minus) / (2.0 * eps)
        
        return J
    
    def compute_equivalent_stiffness(self):
        """
        Compute per-axle vertical loads (static + pitch + roll approx) and
        return effective cornering stiffness per axle (Cf_eq, Cr_eq, Cx_eq).
        """
        g = 9.81
        pitch_angle = self.pitch

        # 1) static axle loads (pitch only)
        Fz_front_static = self.mass * g * self.lr / (self.lf + self.lr)
        Fz_rear_static  = self.mass * g * self.lf / (self.lf + self.lr)

        # 2) pitch-induced dynamic load transfer (approx)
        delta_Fz_front_pitch = -self.mass * g * self.h_cg * math.sin(pitch_angle) * (self.lr / (self.lf + self.lr))
        delta_Fz_rear_pitch  = +self.mass * g * self.h_cg * math.sin(pitch_angle) * (self.lf / (self.lf + self.lr))

        # 3) roll-induced lateral load transfer (approx)
        # use current roll and lateral acceleration estimate (small-angle)
        # we will approximate ay here with vy * yaw_rate if available; if not, 0
        ay_est = getattr(self, 'vy', 0.0) * getattr(self, 'omega', 0.0)
        delta_Fz_roll = (self.mass * ay_est * self.h_cg) / max(self.track_width, 0.1)

        # Left/right distribution (useful if splitting per wheel later)
        # For coarse per-axle effective load, assume half of roll transfer affects each axle equally:
        delta_Fz_front = delta_Fz_front_pitch - 0.5 * delta_Fz_roll
        delta_Fz_rear  = delta_Fz_rear_pitch + 0.5 * delta_Fz_roll

        # total per-axle Fz
        Fz_front = Fz_front_static + delta_Fz_front
        Fz_rear  = Fz_rear_static  + delta_Fz_rear

        # Ensure non-negative
        Fz_front = max(1.0, Fz_front)
        Fz_rear  = max(1.0, Fz_rear)

        # 4) convert to equivalent cornering stiffness (axle-level)
        # Use Cf_axle/Cr_axle as base values scaled with load (rough linear scaling)
        Cf_eq = self.Cf_axle * (Fz_front / (self.mass * g / 2.0))  # normalized by nominal half-mass
        Cr_eq = self.Cr_axle * (Fz_rear  / (self.mass * g / 2.0))

        # longitudinal stiffness (approx)
        Cx_eq = (self.tyre_Bx * self.tyre_Cx) * (Fz_front + Fz_rear) / 2.0

        return Cf_eq, Cr_eq, Cx_eq



def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle

def compute_consistent_delta_s(waypoints, dt_phys, v_min=2.0, v_max=25.0, safety_steps=5):
    """
    Compute a constant, physics-consistent delta_s for path interpolation.

    Parameters
    ----------
    waypoints : np.ndarray
        Shape (N, 2) or (N, 3). Path points (x, y, [z]).
    dt_phys : float
        Physics integration timestep [s].
    v_min, v_max : float
        Expected min/max vehicle speed [m/s].
    safety_steps : int
        Desired number of physics integration steps per Δs (5–10 typical).

    Returns
    -------
    delta_s : float
        Recommended constant spatial interpolation step [m].
    """

    # Base constant Δs from physics and speed
    delta_s = v_max * dt_phys * safety_steps  # ensures multiple physics steps per Δs

    # Compute curvature to avoid undersampling sharp turns
    if waypoints.shape[0] > 3:
        dx = np.gradient(waypoints[:, 0])
        dy = np.gradient(waypoints[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = np.abs(dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)
        curvature = np.nan_to_num(curvature)

        # Estimate minimum radius of curvature
        r_min = 1.0 / (np.max(curvature) + 1e-6)
        # Rule of thumb: Δs should be small relative to curvature radius
        delta_s_curv = 0.05 * r_min  # 5% of local radius

        # Take the more conservative (smaller) Δs
        delta_s = min(delta_s, delta_s_curv)

    # Clamp to reasonable range
    delta_s = np.clip(delta_s, v_min * dt_phys * safety_steps, 1.0)
    return float(np.round(delta_s, 3))


# ============================================================================
# MONOLITHIC SOLVER SYSTEM
# ============================================================================

class CoupledVehicleSystem:
    def __init__(self, vehicles, dt_phys):
        """
        Manages the coupled solve for a list of vehicles.
        vehicles[0] is the Tractor (Leader).
        vehicles[1..N] are Followers (Trailers).
        """
        self.vehicles = vehicles
        self.dt = dt_phys
        self.n_vehicles = len(vehicles)
        self.state_dim = 9
        self.total_dim = self.n_vehicles * self.state_dim
        
        # Solver parameters
        self.max_iter = 50
        self.tol = 1e-3
        
        # Advanced Solver Features
        self.uncertainty = UncertaintyTracker(self.total_dim)

    def pack_state(self, vehicles_list=None):
        if vehicles_list is None:
            vehicles_list = self.vehicles
        
        X = np.zeros(self.total_dim)
        for i, v in enumerate(vehicles_list):
            idx = i * self.state_dim
            # State: [x, y, yaw, vx, vy, omega, ax, ay, alpha]
            state = np.array([v.x, v.y, v.yaw, v.vx, v.vy, v.omega, v.ax, v.ay, v.alpha])
            X[idx : idx + self.state_dim] = state
        return X

    def unpack_and_update(self, X_new):
        """Update vehicle objects with new state"""
        for i, v in enumerate(self.vehicles):
            idx = i * self.state_dim
            s = X_new[idx : idx + self.state_dim]
            v.x, v.y, v.yaw = float(s[0]), float(s[1]), float(s[2])
            v.vx, v.vy, v.omega = float(s[3]), float(s[4]), float(s[5])
            v.ax, v.ay, v.alpha = float(s[6]), float(s[7]), float(s[8])
            
            # Update history (for Gen-Alpha)
            if hasattr(v, 'ax_prev'):
                v.ax_prev = v.ax
                v.ay_prev = v.ay
                v.alpha_prev = v.alpha
            
            # Update derived properties (Z, Pitch)
            if hasattr(v, 'waypoints') and v.waypoints is not None:
                v.z = get_terrain_elevation(v.x, v.y, v.waypoints)

        # Recalculate articulation angles for all follower vehicles
        # (The solver updates x, y, yaw but does NOT update the derived
        #  articulation_yaw attribute that KPI extraction relies on.)
        for i in range(1, self.n_vehicles):
            leader = self.vehicles[i - 1]
            follower = self.vehicles[i]
            new_art = normalize_angle(leader.yaw - follower.yaw)
            if hasattr(follower, 'prev_articulation_yaw'):
                follower.articulation_yaw_rate = (
                    new_art - follower.prev_articulation_yaw) / self.dt
            follower.articulation_yaw = new_art
            follower.prev_articulation_yaw = new_art

    def compute_residual_individual(self, idx, s_new, s_old, X_new, throttle, brake, steer, dt):
        """Helper to call correct residual function for vehicle type"""
        v = self.vehicles[idx]
        
        if idx == 0:
            # TRACTOR (Leader) - No explicit follower info passed
            return v.compute_residual_genalpha(s_new, s_old, throttle, brake, steer, dt)
        else:
            # TRAILER (Follower)
            # Needs Leader state for its OWN constraint (force from front)
            leader_idx = (idx - 1) * self.state_dim
            leader_state = X_new[leader_idx : leader_idx + self.state_dim]
            
            l_x, l_y, l_yaw = leader_state[0], leader_state[1], leader_state[2]
            l_vx, l_vy, l_omega = leader_state[3], leader_state[4], leader_state[5]
            
            # Pass leader kinematic state to trailer residual
            # ArticulatedSegment will compute FRONT hitch force
            # Store brake input on trailer for ABS (compute_residual doesn't take brake param)
            v._brake_input = brake
            return v.compute_residual_genalpha(s_new, s_old, l_x, l_y, l_yaw, l_vx, l_vy, l_omega, 0.0, dt)

    def compute_system_residual(self, X_new, X_old, throttle, brake, steer, dt):
        R = np.zeros(self.total_dim)
        
        # 1. Compute Individual Residuals (uncoupled / forward coupled)
        for i in range(self.n_vehicles):
            idx = i * self.state_dim
            s_new = X_new[idx : idx + self.state_dim]
            s_old = X_old[idx : idx + self.state_dim]
            
            res = self.compute_residual_individual(i, s_new, s_old, X_new, throttle, brake, steer, dt)
            R[idx : idx + self.state_dim] = res

        # 2. Add Reaction Forces (Backward Coupling)
        for i in range(self.n_vehicles - 1):
            leader = self.vehicles[i]
            follower = self.vehicles[i+1]
            
            l_idx = i * self.state_dim
            f_idx = (i + 1) * self.state_dim
            
            s_leader = X_new[l_idx : l_idx + self.state_dim]
            s_follower = X_new[f_idx : f_idx + self.state_dim]
            
            # Unpack
            lx, ly, lyaw = s_leader[0], s_leader[1], s_leader[2]
            lvx, lvy, lomega = s_leader[3], s_leader[4], s_leader[5]
            
            fx, fy, fyaw = s_follower[0], s_follower[1], s_follower[2]
            fvx, fvy, fomega = s_follower[3], s_follower[4], s_follower[5]
            
            # Parameters
            k_hitch = follower.k_hitch
            c_hitch = follower.c_hitch
            offset_leader = follower.leader_rear_offset 
            offset_follower = follower.hitch_length + follower.lf
            
            # 1. Hitch Point on Leader (Global)
            cx_leader, cy_leader = np.cos(lyaw), np.sin(lyaw)
            # Rear hitch is at (-offset, 0) in leader frame
            pA_x = lx - offset_leader * cx_leader
            pA_y = ly - offset_leader * cy_leader
            
            lvx_global = lvx * cx_leader - lvy * cy_leader
            lvy_global = lvx * cy_leader + lvy * cx_leader
            
            vA_x = lvx_global - (-offset_leader * cy_leader) * lomega
            vA_y = lvy_global + (-offset_leader * cx_leader) * lomega
            
            # 2. Target Hitch Point from Follower (Global)
            cx_follower, cy_follower = np.cos(fyaw), np.sin(fyaw)
            # Pin is AHEAD of trailer CG
            pB_x = fx + offset_follower * cx_follower
            pB_y = fy + offset_follower * cy_follower
            
            fvx_global = fvx * cx_follower - fvy * cy_follower
            fvy_global = fvx * cy_follower + fvy * cx_follower
            vB_x = fvx_global + offset_follower * -cy_follower * fomega
            vB_y = fvy_global + offset_follower *  cx_follower * fomega
            
            # Force vector PULLING A -> B (Constraint pulls leader back if trailer lags?)
            # Force on B (Trailer) = k * (pA - pB) + c * (vA - vB)
            # Confirmed in ArticulatedSegment: Force = -k*(err_x) where err = Me - Target.
            # So Force on B pulls B towards A.
            
            # Force on A (Leader) = - Force on B (Newton 3rd Law)
            # Force on A = k * (pB - pA) + c * (vB - vA)
            
            dist_x = pB_x - pA_x
            dist_y = pB_y - pA_y
            dvel_x = vB_x - vA_x
            dvel_y = vB_y - vA_y
            
            Fx_global = k_hitch * dist_x + c_hitch * dvel_x
            Fy_global = k_hitch * dist_y + c_hitch * dvel_y
            
            # Rotate into Leader Body Frame
            Fx_body = Fx_global * cx_leader + Fy_global * cy_leader
            Fy_body = -Fx_global * cy_leader + Fy_global * cx_leader
            
            # Moment on Leader (Force applied at rear hitch point)
            # r = [-offset, 0]
            # Mz = r_x * Fy - r_y * Fx = -offset * Fy
            Mz_body = -offset_leader * Fy_body
            
            # Subtract force from residual (m*a - F = 0) => if F increases, R decreases.
            # NORMALIZE by Leader Mass/Inertia (since R is now in acceleration units)
            R[l_idx + 6] -= Fx_body / leader.mass
            R[l_idx + 7] -= Fy_body / leader.mass
            R[l_idx + 8] -= Mz_body / leader.Iz

        return R

    def compute_system_jacobian(self, X_new, X_old, throttle, brake, steer, dt):
        """Finite Difference Jacobian for the whole system"""
        dim = self.total_dim
        J = np.zeros((dim, dim))
        
        # Base residual
        R0 = self.compute_system_residual(X_new, X_old, throttle, brake, steer, dt)
        
        for i in range(dim):
            X_plus = X_new.copy()
            eps = 1e-5 # Fixed small epsilon for better stability with stiff constraints
            X_plus[i] += eps
            
            R_plus = self.compute_system_residual(X_plus, X_old, throttle, brake, steer, dt)
            
            J[:, i] = (R_plus - R0) / eps
            
        return J

    def solve_step(self, throttle, brake, steer, dt):
        X_old = self.pack_state()
        
        # ====================================================================
        # KOOPMAN PREDICTOR (System-wide)
        # ====================================================================
        # [DISABLED TEMPORARILY] - Mixing Koopman (Tractor) and Euler (Trailers)
        # causes large initial discontinuities in stiff hitch constraints.
        # Needs a full system Koopman operator or coupled prediction.
        
        X_pred_koopman = None
        # tractor = self.vehicles[0]
        
        # if hasattr(tractor, 'koopman') and tractor.koopman.is_ready():
        #     # Get tractor state
        #     tractor_state_old = X_old[0:9]
        #     tractor_pred = tractor.koopman.predict(tractor_state_old)
        #     
        #     if tractor_pred is not None:
        #         X_pred_koopman = X_old.copy() # Start with old state
        #         X_pred_koopman[0:9] = tractor_pred # Overwrite tractor part
        #         # (Trailers will be updated by Euler below)
        
        # Predictor: explicit euler extrapolation (Base)
        X_new = X_old.copy()
        
        n_sub = self.n_vehicles
        for i in range(n_sub):
            idx = i * 9
            
            # If we have a Koopman prediction for this vehicle (currently only tractor i=0), use it
            # if i == 0 and X_pred_koopman is not None:
            #    X_new[idx:idx+9] = X_pred_koopman[idx:idx+9]
            #    continue
                
            # Otherwise use Euler/Kinematic extrapolation
            # Unpack
            x, y, yaw = X_old[idx], X_old[idx+1], X_old[idx+2]
            vx, vy, omega = X_old[idx+3], X_old[idx+4], X_old[idx+5]
            ax, ay, alpha = X_old[idx+6], X_old[idx+7], X_old[idx+8]
            
            # Global velocities 
            vx_g = vx * np.cos(yaw) - vy * np.sin(yaw)
            vy_g = vx * np.sin(yaw) + vy * np.cos(yaw)
            
            X_new[idx+0] += vx_g * dt
            X_new[idx+1] += vy_g * dt
            X_new[idx+2] += omega * dt
            X_new[idx+3] += ax * dt
            X_new[idx+4] += ay * dt
            X_new[idx+5] += alpha * dt
        
        converged = False
        res_norm = 0.0
        
        residual_threshold = 1e-3
        state_change_threshold = 1e-6
        energy_balance_threshold = 1e-4

        for it in range(self.max_iter):
            R = self.compute_system_residual(X_new, X_old, throttle, brake, steer, dt)
            res_norm = np.linalg.norm(R)
            # DEBUG: Detailed Logging
            # if it < 2 or it > self.max_iter - 2:
            #     print(f"DEBUG: Solver iter {it}, Res: {res_norm:.4e}, Energy: {avg_energy_error if 'avg_energy_error' in locals() else 0.0:.4e}")
            
            # ================================================================
            # ENERGY BALANCE CHECK (System-wide)
            # ================================================================
            total_energy_error_norm = 0.0
            
            # Sum up energy balance errors from all vehicles
            # We need to temporarily unpack X_new to individual vehicles to call compute_energy_balance
            # Note: This is a bit expensive but robust
            
            for i, v in enumerate(self.vehicles):
                idx = i * 9
                s_new = X_new[idx : idx + 9]
                s_old = X_old[idx : idx + 9]
                
                # Get energy data (passing dummy leader info for trailers as energy is internal)
                # Note: compute_energy_balance signature differs by vehicle type
                if i == 0:
                    e_data = v.compute_energy_balance(s_new, s_old, throttle, brake, steer, dt)
                else:
                    # For trailers, pass zeros for leader info as it's not used in energy calculation proper
                    # (Hitch work is implicit in position change or ignored in this simplified check)
                    e_data = v.compute_energy_balance(s_new, s_old, 0, 0, 0, 0, 0, 0, 0, dt)
                    
                total_energy_error_norm += e_data['energy_balance_normalized']
            
            avg_energy_error = total_energy_error_norm / self.n_vehicles

            # Primary Convergence Check (Residual)
            if res_norm < self.tol:
                converged = True
                break
            
            # Jacobian update every iteration
            J = self.compute_system_jacobian(X_new, X_old, throttle, brake, steer, dt)
            
            # ================================================================
            # ENTROPY & UNCERTAINTY UPDATE
            # ================================================================
            try:
                sign, logdet = np.linalg.slogdet(J)
                entropy = logdet
            except:
                entropy = 100.0
                
            self.uncertainty.update_from_solver(J, res_norm)
            uncertainty_trace = self.uncertainty.get_trace()
            
            # Solve Newton step
            try:
                delta = np.linalg.solve(J, -R)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(J, -R, rcond=None)[0]
            
            # Secondary Convergence Check (Energy + Entropy + Uncertainty)
            if it > 0:
                combined_ok = (
                    avg_energy_error < energy_balance_threshold and
                    entropy < 50.0 and # Loose bound
                    uncertainty_trace < 1e-5
                )
                if combined_ok and res_norm < self.tol * 10:
                    converged = True
                    break

            # Simple line search
            step = 1.0
            for ls in range(5):
                X_trial = X_new + step * delta
                R_trial = self.compute_system_residual(X_trial, X_old, throttle, brake, steer, dt)
                if np.linalg.norm(R_trial) < res_norm:
                    X_new = X_trial
                    break
                step *= 0.5
            else:
                X_new += step * delta # Force update if LS fails
            
        if converged:
             self.unpack_and_update(X_new)
             
             # ADVANCE TIRE RELAXATION STATES (History Update)
             # Now that we have a converged solution, update the history terms for next step
             for i, vehicle in enumerate(self.vehicles):
                 idx = i * 9
                 s_new = X_new[idx : idx + 9]
                 s_old = X_old[idx : idx + 9]
                 if hasattr(vehicle, 'advance_tire_states'):
                     vehicle.advance_tire_states(s_new, s_old, dt)
             
             # Add sample to Tractor's Koopman (for online learning)
             if hasattr(self.vehicles[0], 'koopman'):
                 self.vehicles[0].koopman.add_sample(X_old[0:9], X_new[0:9])
        else:
             if self.dt > 1e-4:
                 print(f"WARN: Monolithic solver not converged. Res: {res_norm:.2e} EnergyErr: {avg_energy_error:.2e}")
             self.unpack_and_update(X_new)
             
        return converged


class Controller2D(object):
    """
    2D Controller for waypoint following.
    - Longitudinal: PID speed control
    - Lateral: Stanley steering control
    """
    def __init__(self, waypoints, dt):
        self._current_x = 0
        self._current_y = 0
        self._current_yaw = 0
        self._current_speed = 0
        self._desired_speed = 0
        self._current_frame = 0
        self._current_timestamp = 0
        self.throttle = 0.0
        self.brake = 0.0
        self.steer = 0.0
        self._waypoints = np.array(waypoints) if not isinstance(waypoints, np.ndarray) else waypoints
        self._conv_rad_to_steer = 180.0 / 70.0 / np.pi
        self._pi = np.pi
        self._2pi = 2.0 * np.pi
        self.e_buffer = deque(maxlen=20)
        self._e = 0
        self.dt = dt

        # PID parameters for speed controller
        self.K_P = 1.0
        self.K_D = 0.001
        self.K_I = 0.3

        # Initialize MPC steering controller if enabled
        if USE_MPC_STEERING:
            self.mpc = MPCSteeringController(
                N=MPC_PREDICTION_HORIZON,
                M=MPC_CONTROL_HORIZON,
                dt=self.dt,
                wheelbase=3.7,  # TractorHead wheelbase
                max_steer=np.radians(30.0),  # TractorHead max_steer
                max_steer_rate=np.radians(MPC_MAX_STEER_RATE),
                weight_cte=MPC_WEIGHT_CTE,
                weight_heading=MPC_WEIGHT_HEADING,
                weight_effort=MPC_WEIGHT_EFFORT,
                weight_rate=MPC_WEIGHT_RATE,
                # Dynamic bicycle model params (shared with EKF)
                mass=EKF_MASS,
                Iz=EKF_IZ,
                lf=EKF_LF,
                lr=EKF_LR,
                Cf=EKF_CF,
                Cr=EKF_CR
            )
            print(f"[CONTROLLER] MPC steering enabled (dynamic bicycle): N={MPC_PREDICTION_HORIZON}, M={MPC_CONTROL_HORIZON}")
        else:
            self.mpc = None
            print("[CONTROLLER] Stanley steering controller enabled")
        
        # Initialize EKF state estimator if enabled
        if ENABLE_EKF and USE_MPC_STEERING:
            self.ekf = VehicleEKF(
                dt=self.dt,
                L=3.7,
                lf=EKF_LF,
                lr=EKF_LR,
                mass=EKF_MASS,
                Iz=EKF_IZ,
                Cf=EKF_CF,
                Cr=EKF_CR
            )
            self._ekf_initialized = False
            self._last_delta_rad = 0.0  # Track last steering for EKF predict
            # Filtered state storage
            self._filtered_vy = 0.0
            self._filtered_omega = 0.0
            print(f"[CONTROLLER] EKF state estimator enabled (6D state)")
        else:
            self.ekf = None
            self._ekf_initialized = False
            self._filtered_vy = 0.0
            self._filtered_omega = 0.0
            if not ENABLE_EKF:
                print("[CONTROLLER] EKF disabled — MPC uses perfect state")

    def update_values(self, x, y, yaw, speed, vy=0.0, omega=0.0, timestamp=0.0):
        """
        Update current vehicle state and run EKF estimation cycle.
        
        When EKF is enabled:
          1. EKF predict step (using last MPC steering command)
          2. Simulate noisy sensor measurements from true state
          3. EKF update step (correct with noisy measurements)
          4. Store filtered state for MPC to use
        
        Args:
            x, y: True position [m]
            yaw: True heading [rad]
            speed: True longitudinal velocity vx [m/s]
            vy: True lateral velocity [m/s] (needed for EKF noise simulation)
            omega: True yaw rate [rad/s] (needed for EKF noise simulation)
            timestamp: Current simulation time [s]
        """
        self._current_timestamp = timestamp
        
        if self.ekf is not None:
            # --- EKF ESTIMATION CYCLE ---
            
            # Initialize EKF on first call with true state
            if not self._ekf_initialized:
                self.ekf.initialize([x, y, yaw, speed, vy, omega])
                self._ekf_initialized = True
                print(f"[EKF] Initialized at ({x:.1f}, {y:.1f}), v={speed:.1f} m/s")
            
            # Step 1: PREDICT using last MPC steering command (MPC→EKF coupling)
            self.ekf.predict(self._last_delta_rad)
            
            # Step 2: Simulate noisy sensor measurements
            true_state = (x, y, yaw, speed, vy, omega)
            z_noisy = self.ekf.add_noise(true_state)
            
            # Step 3: UPDATE with noisy measurements (Sensor→EKF coupling)
            self.ekf.update(z_noisy)
            
            # Step 4: Extract filtered state for MPC
            x_f, y_f, yaw_f, vx_f, vy_f, omega_f = self.ekf.get_state()
            
            # Store filtered state for use by update_controls
            self._current_x = x_f
            self._current_y = y_f
            self._current_yaw = yaw_f
            self._current_speed = vx_f
            self._filtered_vy = vy_f
            self._filtered_omega = omega_f
            
            # Log EKF diagnostics periodically
            if timestamp <= 10.0 or np.random.rand() < 0.01:
                ekf_diag = self.ekf.get_diagnostics()
                # Estimation errors (we know truth in simulation)
                err_x = x_f - x
                err_y = y_f - y
                err_yaw = np.arctan2(np.sin(yaw_f - yaw), np.cos(yaw_f - yaw))
                err_vy = vy_f - vy
                print(f"[EKF] t={timestamp:.2f}s est_err: pos=({err_x:.3f},{err_y:.3f})m "
                      f"yaw={np.degrees(err_yaw):.2f}° vy_err={err_vy:.3f}m/s "
                      f"std_pos=({ekf_diag['std_x']:.3f},{ekf_diag['std_y']:.3f}) "
                      f"NIS={ekf_diag['NIS']:.2f}")
        else:
            # No EKF: use true state directly (original behavior)
            self._current_x = x
            self._current_y = y
            self._current_yaw = yaw
            self._current_speed = speed
            self._filtered_vy = vy
            self._filtered_omega = omega

    def update_desired_speed(self):
        """Find nearest waypoint and set desired speed"""
        min_idx = 0
        min_dist = float("inf")
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                self._waypoints[i][0] - self._current_x,
                self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        
        # Use constant target speed (can be enhanced to use waypoint speeds)
        self._desired_speed = DEFAULT_TARGET_SPEED

    def update_waypoints(self, new_waypoints):
        """Update waypoint reference"""
        self._waypoints = np.array(new_waypoints) if not isinstance(new_waypoints, np.ndarray) else new_waypoints

    def update_controls(self):
        """Compute throttle, brake, and steering commands"""
        x = self._current_x
        y = self._current_y
        yaw = self._current_yaw
        v = self._current_speed
        self.update_desired_speed()
        v_desired = self._desired_speed
        waypoints = self._waypoints
        
        # Ensure waypoints is numpy array (safety check)
        if not isinstance(waypoints, np.ndarray):
            waypoints = np.array(waypoints)

        # ==================================
        # LONGITUDINAL CONTROLLER (PID)
        # ==================================
        self._e = v_desired - v
        self.e_buffer.append(self._e)

        # Derivative and integral with proper time scaling
        if len(self.e_buffer) >= 2:
            _de = (self.e_buffer[-1] - self.e_buffer[-2]) / self.dt
            _ie = sum(self.e_buffer) * self.dt
        else:
            _de = 0.0
            _ie = 0.0

        # PID terms
        p_term = self.K_P * self._e
        d_term = self.K_D * _de
        i_term = self.K_I * _ie

        # Combined control signal
        control_signal = p_term + d_term + i_term
        
        # Anti-windup: prevent integral accumulation when saturated
        if control_signal > 1.0 or control_signal < -1.0:
            # Control is saturated, don't accumulate more integral error
            if len(self.e_buffer) > 1:
                self.e_buffer.pop()  # Remove last error to prevent windup
        
        # Smooth transition between throttle and brake
        if control_signal > 0:
            # Accelerating: use throttle
            self.throttle = float(np.clip(control_signal, 0.0, 1.0))
            self.brake = 0.0
        else:
            # Decelerating: use brake
            self.throttle = 0.0
            self.brake = float(np.clip(-control_signal, 0.0, 1.0))


        # ==================================
        # LATERAL CONTROLLER (MPC or Stanley)
        # ==================================
        if USE_MPC_STEERING and self.mpc is not None:
            # MPC STEERING CONTROLLER
            try:
                # Compute optimal steering using MPC (with EKF-filtered vy, omega)
                delta_opt, pred_traj = self.mpc.compute_control(
                    x=x, y=y, yaw=yaw, v=v, waypoints=waypoints,
                    vy=self._filtered_vy, omega=self._filtered_omega
                )
                
                # Get diagnostics
                diag = self.mpc.get_diagnostics()
                
                # Calculate cross-track error and heading error for detailed diagnostics
                current_xy = np.array([x, y])
                distances = np.linalg.norm(current_xy - waypoints[:, :2], axis=1)
                nearest_idx = np.argmin(distances)
                crosstrack_error = distances[nearest_idx]
                
                # Calculate heading error
                lookahead_idx = min(nearest_idx + 3, len(waypoints) - 1)
                yaw_path = np.arctan2(
                    waypoints[lookahead_idx][1] - waypoints[nearest_idx][1],
                    waypoints[lookahead_idx][0] - waypoints[nearest_idx][0]
                )
                heading_error = yaw_path - yaw
                # Normalize to [-pi, pi]
                heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
                
                # ENHANCED DIAGNOSTICS: Log every control cycle for first 10 seconds
                # Suppress when fixed steering is active (MPC output is not used)
                current_time = self._current_timestamp
                if current_time <= 10.0 and not USE_FIXED_STEERING:
                    print(f"[MPC_DETAIL] t={current_time:.2f}s pos=({x:.1f},{y:.1f}) v={v:.1f}m/s " +
                          f"cte={crosstrack_error:.2f}m heading_err={np.degrees(heading_error):.1f}° " +
                          f"delta={np.degrees(delta_opt):.1f}° cost={diag['recent_cost']:.1f} " +
                          f"solve_time={diag['solve_time_ms']:.1f}ms iters={diag['iterations']}")
                    
                    # Log first 3 predicted trajectory points
                    if pred_traj is not None and len(pred_traj) > 0:
                        print(f"[MPC_PRED] pred[0]=({pred_traj[0][0]:.1f},{pred_traj[0][1]:.1f}) " +
                              f"pred[1]=({pred_traj[1][0] if len(pred_traj)>1 else 0:.1f},{pred_traj[1][1] if len(pred_traj)>1 else 0:.1f}) " +
                              f"pred[2]=({pred_traj[2][0] if len(pred_traj)>2 else 0:.1f},{pred_traj[2][1] if len(pred_traj)>2 else 0:.1f})")
                
                # Periodic logging after 10 seconds (1% sample to avoid spam)
                elif np.random.rand() < 0.01 and not USE_FIXED_STEERING:
                    print(f"[MPC] t={current_time:.1f}s solve_time={diag['solve_time_ms']:.2f}ms iterations={diag['iterations']} cost={diag['recent_cost']:.3f}")
                
                # SANITY CHECK: Alert if steering output is extreme
                if abs(delta_opt) > np.radians(25.0) and not USE_FIXED_STEERING:
                    print(f"[MPC_WARNING] Extreme steering command: {np.degrees(delta_opt):.1f}° at t={current_time:.2f}s")
                
                steer_output = delta_opt  # Already in radians
                
            except Exception as e:
                # Fallback to simple heading control if MPC fails
                print(f"[MPC ERROR] {e}, using fallback steering")
                # Simple proportional heading control
                distances = np.linalg.norm(np.array([x, y]) - waypoints[:, :2], axis=1)
                nearest_idx = np.argmin(distances)
                lookahead_idx = min(nearest_idx + 5, len(waypoints) - 1)
                yaw_path = np.arctan2(
                    waypoints[lookahead_idx][1] - waypoints[nearest_idx][1],
                    waypoints[lookahead_idx][0] - waypoints[nearest_idx][0]
                )
                yaw_diff = yaw_path - yaw
                yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
                steer_output = np.clip(yaw_diff, -1.22, 1.22)
        
        else:
            # STANLEY STEERING CONTROLLER (Legacy)
            k_e = 0.3  # Crosstrack gain
            k_v = 20   # Speed normalization

            # 1. Heading error
            yaw_path = np.arctan2(
                waypoints[-1][1] - waypoints[0][1],
                waypoints[-1][0] - waypoints[0][0]
            )
            yaw_diff = yaw_path - yaw
            # Normalize to [-pi, pi]
            if yaw_diff > np.pi:
                yaw_diff -= 2 * np.pi
            if yaw_diff < -np.pi:
                yaw_diff += 2 * np.pi

            # 2. Crosstrack error (FIXED: use actual distance, not squared)
            current_xy = np.array([x, y])
            distances = np.linalg.norm(current_xy - waypoints[:, :2], axis=1)
            min_dist_idx = np.argmin(distances)
            crosstrack_dist = distances[min_dist_idx]

            # Determine sign of crosstrack error
            yaw_cross_track = np.arctan2(
                y - waypoints[min_dist_idx][1],
                x - waypoints[min_dist_idx][0]
            )
            yaw_path2ct = yaw_path - yaw_cross_track
            if yaw_path2ct > np.pi:
                yaw_path2ct -= 2 * np.pi
            if yaw_path2ct < -np.pi:
                yaw_path2ct += 2 * np.pi
            
            # Signed crosstrack error
            if yaw_path2ct > 0:
                crosstrack_error = crosstrack_dist
            else:
                crosstrack_error = -crosstrack_dist

            # Stanley crosstrack term
            yaw_diff_crosstrack = np.arctan(k_e * crosstrack_error / (k_v + v))

            # print(crosstrack_error, yaw_diff, yaw_diff_crosstrack)
            # DEBUG: Print steering components if steering is suspiciously low but error exists
            if abs(crosstrack_error) > 1.0 or abs(yaw_diff) > 0.1:
                 print(f"[STANLEY_DEBUG] cte={crosstrack_error:.2f} yaw_err={yaw_diff:.2f} steer_cte={yaw_diff_crosstrack:.2f}")

            # 3. Combined steering command
            steer_expect = yaw_diff + yaw_diff_crosstrack
            if steer_expect > np.pi:
                steer_expect -= 2 * np.pi
            if steer_expect < -np.pi:
                steer_expect += 2 * np.pi
            steer_output = np.clip(steer_expect, -1.22, 1.22)

        # 4. Convert to normalized output [-1, 1]
        input_steer = self._conv_rad_to_steer * steer_output
        self.steer = np.clip(input_steer, -1.0, 1.0)
        
        # 5. Store steering in radians for EKF predict step (MPC → EKF feedback)
        if hasattr(self, '_last_delta_rad'):
            self._last_delta_rad = steer_output


def export_data_to_csv(t_hist, x_history, y_history, z_history, 
                       speed_history, vy_history, omega_history, 
                       yaw_history, pitch_history, roll_history,
                       ax_history, ay_history,
                       x_trailer1_history, y_trailer1_history, z_trailer1_history,
                       vx_trailer1_history, vy_trailer1_history, omega_trailer1_history,
                       articulation1_history, pitch_trailer1_history, roll_trailer1_history,
                       x_dolly_history, y_dolly_history, z_dolly_history,
                       vx_dolly_history, vy_dolly_history, omega_dolly_history,
                       articulation2_history, pitch_dolly_history, roll_dolly_history,
                       x_trailer2_history, y_trailer2_history, z_trailer2_history,
                       vx_trailer2_history, vy_trailer2_history, omega_trailer2_history,
                       articulation3_history, pitch_trailer2_history, roll_trailer2_history,
                       throttle_history, brake_history, steer_history, shift_history,
                       speed_ref, speed_error,
                       filename):
    """
    Export all simulation data to a CSV file.
    
    Args:
        Multiple history arrays for all vehicle states
        filename: Output CSV filename (with path)
    """
    # Create a DataFrame with all the data
    data = {
        'time_s': t_hist,
        # Tractor position
        'tractor_x_m': x_history,
        'tractor_y_m': y_history,
        'tractor_z_m': z_history,
        # Tractor velocities
        'tractor_vx_ms': speed_history,
        'tractor_vy_ms': vy_history,
        'tractor_omega_rads': omega_history,
        # Tractor orientation
        'tractor_yaw_rad': yaw_history,
        'tractor_pitch_rad': pitch_history,
        'tractor_roll_rad': roll_history,
        # Tractor accelerations
        'tractor_ax_ms2': ax_history,
        'tractor_ay_ms2': ay_history,
        # Trailer 1 position
        'trailer1_x_m': x_trailer1_history,
        'trailer1_y_m': y_trailer1_history,
        'trailer1_z_m': z_trailer1_history,
        # Trailer 1 velocities
        'trailer1_vx_ms': vx_trailer1_history,
        'trailer1_vy_ms': vy_trailer1_history,
        'trailer1_omega_rads': omega_trailer1_history,
        # Trailer 1 orientation
        'trailer1_articulation_rad': articulation1_history,
        'trailer1_pitch_rad': pitch_trailer1_history,
        'trailer1_roll_rad': roll_trailer1_history,
        # Dolly position
        'dolly_x_m': x_dolly_history,
        'dolly_y_m': y_dolly_history,
        'dolly_z_m': z_dolly_history,
        # Dolly velocities
        'dolly_vx_ms': vx_dolly_history,
        'dolly_vy_ms': vy_dolly_history,
        'dolly_omega_rads': omega_dolly_history,
        # Dolly orientation
        'dolly_articulation_rad': articulation2_history,
        'dolly_pitch_rad': pitch_dolly_history,
        'dolly_roll_rad': roll_dolly_history,
        # Trailer 2 position
        'trailer2_x_m': x_trailer2_history,
        'trailer2_y_m': y_trailer2_history,
        'trailer2_z_m': z_trailer2_history,
        # Trailer 2 velocities
        'trailer2_vx_ms': vx_trailer2_history,
        'trailer2_vy_ms': vy_trailer2_history,
        'trailer2_omega_rads': omega_trailer2_history,
        # Trailer 2 orientation
        'trailer2_articulation_rad': articulation3_history,
        'trailer2_pitch_rad': pitch_trailer2_history,
        'trailer2_roll_rad': roll_trailer2_history,
        # Control inputs
        'throttle': throttle_history,
        'brake': brake_history,
        'steer_rad': steer_history,
        'gear': shift_history,
        # Reference signals
        'speed_ref_ms': speed_ref,
        'speed_error_ms': speed_error,
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Data exported to: {filename} ({len(df)} rows, {len(df.columns)} columns)")


def main():
    global dt_phys  # Allow modification of global dt_phys variable
    
    #############################################
    # Setup rolling debug logging (separate file per time window)
    #############################################
    log_file = None  # Will be created dynamically per time window
    log_window_start = 0.0
    log_window_end = 10.0
    
    # Create a wrapper for print that logs to both console and file
    original_print = print
    start_time = None
    
    def logging_print(*args, **kwargs):
        nonlocal start_time, log_file
        if start_time is None:
            start_time = 0  # Will be set when simulation starts
        
        # Format message
        message = ' '.join(str(arg) for arg in args)
        
        # Print to console with encoding fallback for Windows
        try:
            original_print(*args, **kwargs)
        except UnicodeEncodeError:
            # Fallback: encode to ascii with replacement, then decode
            safe_args = [str(a).encode('ascii', 'replace').decode('ascii') for a in args]
            original_print(*safe_args, **kwargs)
        
        # Log ALL output to file (no filtering)
        if log_file:
            try:
                log_file.write(message + '\n')
                log_file.flush()
            except UnicodeEncodeError:
                safe_message = message.encode('ascii', 'replace').decode('ascii')
                log_file.write(safe_message + '\n')
                log_file.flush()
    
    # Replace print globally
    import builtins
    builtins.print = logging_print
    
    #############################################
    # Load Waypoints
    #############################################
    waypoints_file = WAYPOINTS_FILENAME
    waypoints_np = load_path(waypoints_file)
    
    # Save the actual path used (including extension) for plotting consistency
    try:
        pd.DataFrame(waypoints_np, columns=['x', 'y', 'z']).to_csv('sim_track.csv', index=False)
        print("Exported actual simulation path to sim_track.csv")
    except Exception as e:
        print(f"Warning: Could not save sim_track.csv: {e}")
    # Linear interpolation computations, we can also use spine interpolation
    wp_distance = []  # distance array
    for i in range(1, waypoints_np.shape[0]):
        wp_distance.append(
            np.sqrt((waypoints_np[i, 0] - waypoints_np[i - 1, 0]) ** 2 +
                    (waypoints_np[i, 1] - waypoints_np[i - 1, 1]) ** 2))
    # last distance is 0 because it is the distance from the last waypoint to the last waypoint
    wp_distance.append(0)

    # Linearly interpolate between waypoints and store in a list
    wp_interp = []  # interpolated values
    # (rows = waypoints, columns = [x, y, v])
    wp_interp_hash = []
    # hash table which indexes waypoints_np to the index of the waypoint in wp_interp
    interp_counter = 0  # counter for current interpolated point index
    reached_the_end = False

    delta_s = compute_consistent_delta_s(waypoints_np, dt_phys, v_min=2, v_max=25, safety_steps=5)

    # Waypoint progress tracking (robust and directly correlates with path completion)

    for i in range(waypoints_np.shape[0] - 1):
        # Add original waypoint to interpolated waypoints list (and append
        # it to the hash table)
        wp_interp.append(list(waypoints_np[i]))
        wp_interp_hash.append(interp_counter)
        interp_counter += 1

        # Interpolate to the next waypoint. First compute the number of
        # points to interpolate based on the desired resolution and
        # incrementally add interpolated points until the next waypoint
        # is about to be reached.
        num_pts_to_interp = int(np.floor(wp_distance[i] / float(delta_s)) - 1)
        wp_vector = waypoints_np[i + 1] - waypoints_np[i]
        wp_uvector = wp_vector / np.linalg.norm(wp_vector)
        for j in range(num_pts_to_interp):
            next_wp_vector = delta_s * float(j + 1) * wp_uvector
            wp_interp.append(list(waypoints_np[i] + next_wp_vector))
            interp_counter += 1
    # add last waypoint at the end
    wp_interp.append(list(waypoints_np[-1]))
    wp_interp_hash.append(interp_counter)
    interp_counter += 1

    controller = Controller2D(waypoints_np, dt=dt_controller)

    # Extract initial position and orientation
    # Use same yaw calculation as Stanley controller (global path direction)
    # to avoid initial heading error that causes detour
    wp0 = waypoints_np[0]
    wp1 = waypoints_np[1]
    vec0 = wp1 - wp0
    # FIX: Use LOCAL path direction (first two waypoints) instead of GLOBAL (first to last)
    # Global direction can differ significantly from local if path has curves,
    # causing initial heading error and large lateral forces
    yaw0 = np.arctan2(vec0[1], vec0[0])  # Local direction at start
    pitch0 = np.arctan2(vec0[2], np.hypot(vec0[0], vec0[1]))

    tractor = TractorHead(
        x=wp0[0],
        y=wp0[1],
        z=get_terrain_elevation(wp0[0], wp0[1], waypoints_np),
        yaw=yaw0,
        pitch=pitch0,
        vx=INITIAL_SPEED,  # Global config: synchronized start speed for all vehicles
        # No arranca si la velocidad inicial es inferior al ralentí del motor
        vy=0.0,
        omega=0.0,
        waypoints=waypoints_np
    )
    
    # Initialize tractor wheel speeds to match initial velocity
    tractor.initialize_wheel_states()

    start_x, start_y, start_yaw = tractor.x, tractor.y, tractor.yaw
    # FIX: REMOVED problematic initial update call!
    # This was calling solver with vx=20 m/s but zero throttle/brake/steer,
    # creating large deceleration forces and potential numerical instability
    # The first real update will happen in the simulation loop with proper controller inputs
    # tractor.update(throttle=0, brake=0, delta=0, dt=dt_phys)  # REMOVED!
    
    # Tractor position and orientation history
    x_history = [start_x]
    y_history = [start_y]
    start_z = get_terrain_elevation(start_x, start_y, waypoints_np)
    z_history = [start_z]
    yaw_history = [start_yaw]
    pitch_history = [tractor.pitch]
    roll_history = [tractor.roll]
    
    # Tractor velocities and accelerations
    speed_history = [tractor.vx]
    vy_history = [tractor.vy]
    omega_history = [tractor.omega]
    ax_history = [0.0]
    ay_history = [0.0]
    
    # Tractor control and state
    shift_history = [1]
    brake_history = [0.0]
    steer_history = [0.0]
    

    # Index of waypoint that is currently closest to the car, assumed to be the first index
    closest_index = 0
    max_closest_index = 0
    steps = 0
    # reference track and speed for plotting usage
    x_ref = list(waypoints_np[:, 0])
    y_ref = list(waypoints_np[:, 1])
    speed_ref = [0]
    # for debug
    speed_error = [0]
    throttle_history = [1]

    # Trailer 1 behind tractor
    # Position: tractor hitch point is at (x - hitch_length*cos(yaw), y - hitch_length*sin(yaw))
    # Trailer1 front should connect there, and its CG is trailer_length/2 further back
    trailer1_length = 8.0
    trailer1_hitch = 1.5
    trailer1_x = tractor.x - (tractor.hitch_length + trailer1_hitch + trailer1_length/2) * np.cos(yaw0)
    trailer1_y = tractor.y - (tractor.hitch_length + trailer1_hitch + trailer1_length/2) * np.sin(yaw0)

    trailer1 = ArticulatedSegment(
        mass=6000.0,
        Iz=5000.0,
        x=trailer1_x,
        y=trailer1_y,
        z=get_terrain_elevation(trailer1_x, trailer1_y, waypoints_np),
        yaw=yaw0,  # Same yaw as tractor for perfect alignment
        pitch=get_terrain_pitch(trailer1_x, trailer1_y, waypoints_np),
        hitch_length=trailer1_hitch,
        trailer_length=trailer1_length,
        waypoints=waypoints_np,
        articulation_yaw=0.0,  # Perfectly aligned
        articulation_yaw_rate=0.0,
        h_cg_trailer=1.4,
        track_width_trailer=2.0
    )
    # Set initial velocities to match tractor (global config)
    trailer1.vx = INITIAL_SPEED  # Synchronized with tractor
    trailer1.vy = tractor.vy
    trailer1.omega = tractor.omega
    # Leader's CG-to-hitch distance (tractor rear hitch point)
    trailer1.leader_rear_offset = tractor.hitch_length  # 0.7m
    # STABILITY FIX: Reduced stiffness after fixing unit scaling bug (was 2e6)
    # Stability: Restored to 2e6 after adding kinematic terms to solver
    trailer1.k_hitch = 2000000.0  # Moderate stiffness (2 MN/m)
    trailer1.c_hitch = 500000.0   # Critical damping for 6000kg
    trailer1.initialize_wheel_states()

    # Dolly behind trailer1
    dolly_length = 1.9
    dolly_hitch = 1.2
    dolly_x = trailer1_x - (trailer1_length/2 + dolly_hitch + dolly_length/2) * np.cos(yaw0)
    dolly_y = trailer1_y - (trailer1_length/2 + dolly_hitch + dolly_length/2) * np.sin(yaw0)

    dolly = ArticulatedSegment(
        mass=1200.0,
        Iz=1500.0,
        x=dolly_x,
        y=dolly_y,
        z=get_terrain_elevation(dolly_x, dolly_y, waypoints_np),
        yaw=yaw0,
        pitch=get_terrain_pitch(dolly_x, dolly_y, waypoints_np),
        hitch_length=dolly_hitch,
        trailer_length=dolly_length,
        waypoints=waypoints_np,
        articulation_yaw=0.0,
        articulation_yaw_rate=0.0,
        h_cg_trailer=1.1,
        track_width_trailer=2.0
    )
    # Set initial velocities to match tractor (global config)
    dolly.vx = INITIAL_SPEED
    dolly.vy = tractor.vy
    dolly.omega = tractor.omega
    # Leader's CG-to-hitch distance (trailer1 rear = lr = trailer1_length/2)
    dolly.leader_rear_offset = trailer1.lr  # 4.0m
    # Stability: Restored to 2e6 matching Trailer1
    dolly.k_hitch = 2000000.0
    dolly.c_hitch = 500000.0   # Critical damping
    dolly.initialize_wheel_states()

    # Trailer 2 behind dolly
    trailer2_length = 12.0
    trailer2_hitch = 2.0
    trailer2_x = dolly_x - (dolly_length/2 + trailer2_hitch + trailer2_length/2) * np.cos(yaw0)
    trailer2_y = dolly_y - (dolly_length/2 + trailer2_hitch + trailer2_length/2) * np.sin(yaw0)

    trailer2 = ArticulatedSegment(
        mass=5000.0,
        Iz=5000.0,
        x=trailer2_x,
        y=trailer2_y,
        z=get_terrain_elevation(trailer2_x, trailer2_y, waypoints_np),
        yaw=yaw0,
        pitch=get_terrain_pitch(trailer2_x, trailer2_y, waypoints_np),
        hitch_length=trailer2_hitch,
        trailer_length=trailer2_length,
        waypoints=waypoints_np,
        articulation_yaw=0.0,
        articulation_yaw_rate=0.0,
        h_cg_trailer=1.45,
        track_width_trailer=2.0
    )
    # Set initial velocities to match tractor (global config)
    trailer2.vx = INITIAL_SPEED
    trailer2.vy = tractor.vy
    trailer2.omega = tractor.omega
    # Leader's CG-to-hitch distance (dolly rear = lr = dolly_length/2)
    trailer2.leader_rear_offset = dolly.lr  # 0.95m
    # Stability: Restored to 2e6
    trailer2.k_hitch = 2000000.0
    trailer2.c_hitch = 500000.0
    trailer2.initialize_wheel_states()

    # ========================================================================
    # WARM-START TIRE RELAXATION (Prevent zero-force cold start)
    # ========================================================================
    # Without warm-start, transient force history = 0, leaving tires with
    # ~5% grip for the first ~0.1s of simulation. This causes uncontrolled
    # drift, especially in trailers.
    if ENABLE_TIRE_RELAXATION:
        tractor.warm_start_tire_relaxation()
        trailer1.warm_start_tire_relaxation()
        dolly.warm_start_tire_relaxation()
        trailer2.warm_start_tire_relaxation()

    # ========================================================================
    # AUTOMATIC TIMESTEP CALCULATION (Eigenvalue-based)
    # ========================================================================
    if AUTO_CALCULATE_DT:
        # Calculate optimal dt based on system dynamics
        dt_optimal, diagnostics = compute_optimal_dt(
            tractor=tractor,
            trailers=[trailer1, dolly, trailer2],
            rho_inf=RHO_INF,
            samples_per_period=10,  # Accuracy: 10 samples per oscillation
            safety_factor=0.5,      # Conservative: 2× safety margin
            enable_tire_relaxation=ENABLE_TIRE_RELAXATION,
            dt_controller=dt_controller  # Ensure dt_phys is compatible with controller
        )
        
        # Print detailed analysis
        print_dt_analysis(dt_optimal, diagnostics)
        
        # Override dt_phys with calculated value
        dt_phys = dt_optimal
        original_print(f"[AUTO] Using AUTO-CALCULATED dt_phys = {dt_phys*1000:.4f} ms ({1/dt_phys:.0f} Hz)")
    else:
        original_print(f"[MANUAL] Using MANUAL dt_phys = {dt_phys*1000:.4f} ms ({1/dt_phys:.0f} Hz)")

    # ========================================================================
    # Initialize trailer velocities to match tractor (critical for smooth start)
    # ========================================================================
    # Without this, trailers start at vx=0 while tractor has vx=0.1,
    # causing large initial hitch forces and jerky motion
    trailer1.vx = tractor.vx  # Match tractor longitudinal velocity
    trailer1.vy = 0.0
    
    dolly.vx = tractor.vx
    dolly.vy = 0.0
    
    trailer2.vx = tractor.vx
    trailer2.vy = 0.0

    # ========================================================================
    # Initialize Monolithic Solver System
    # ========================================================================
    system = CoupledVehicleSystem([tractor, trailer1, dolly, trailer2], dt_phys)
    original_print(f"[SYSTEM] Monolithic solver initialized with {system.total_dim} states")


    # Trailer 1 history
    x_trailer1_history, y_trailer1_history, z_trailer1_history = [trailer1.x], [trailer1.y], [trailer1.z]
    vx_trailer1_history, vy_trailer1_history = [trailer1.vx], [trailer1.vy]
    omega_trailer1_history = [trailer1.omega]
    articulation1_history = [trailer1.articulation_yaw]
    pitch_trailer1_history = [trailer1.pitch]
    roll_trailer1_history = [trailer1.roll]
    
    # Dolly history
    x_dolly_history, y_dolly_history, z_dolly_history = [dolly.x], [dolly.y], [dolly.z]
    vx_dolly_history, vy_dolly_history = [dolly.vx], [dolly.vy]
    omega_dolly_history = [dolly.omega]
    articulation2_history = [dolly.articulation_yaw]
    pitch_dolly_history = [dolly.pitch]
    roll_dolly_history = [dolly.roll]
    
    # Trailer 2 history
    x_trailer2_history, y_trailer2_history, z_trailer2_history = [trailer2.x], [trailer2.y], [trailer2.z]
    vx_trailer2_history, vy_trailer2_history = [trailer2.vx], [trailer2.vy]
    omega_trailer2_history = [trailer2.omega]
    articulation3_history = [trailer2.articulation_yaw]
    pitch_trailer2_history = [trailer2.pitch]
    roll_trailer2_history = [trailer2.roll]

    # --- two-rate simulation setup
    t = 0.0
    t_hist=[t]
    t_next_ctrl = 0.0
    # Progress tracking uses waypoint index only (more robust than arc-length)
    reached_the_end = False
    
    # Create initial log file for first time window (0-10s)
    log_filename = f"results/debug_{int(log_window_start):04d}-{int(log_window_end):04d}s.log"
    log_file = open(log_filename, 'w')
    original_print(f"[LOG] Starting simulation with log file: {log_filename}")
    original_print("")

    # --- Periodic data export setup (every 10 seconds)
    next_export_time = 10.0
    export_counter = 0

    while t < MAX_SIM_TIME:
            # ----------------------
            # CONTROLLER UPDATE (at lower frequency)
            # ----------------------
            if t >= t_next_ctrl:
                # Update controller with current tractor state
                # Pass vy and omega for EKF noise simulation (when enabled)
                controller.update_values(
                    tractor.x, tractor.y, tractor.yaw, tractor.vx,
                    vy=tractor.vy, omega=tractor.omega, timestamp=t
                )
                
                # Compute new throttle, brake, steering commands
                controller.update_controls()
                
                # Log controller state for debug
                if len(controller.e_buffer) >= 2:
                    de = (controller.e_buffer[-1] - controller.e_buffer[-2]) / controller.dt
                    ie = sum(controller.e_buffer) * controller.dt
                else:
                    de = 0.0
                    ie = 0.0
                
                print(f"[CTRL] v={tractor.vx:.2f} v_des={controller._desired_speed:.2f} " +
                      f"err={controller._e:.3f} throttle={controller.throttle:.3f} " +
                      f"brake={controller.brake:.3f} steer={controller.steer:.3f} " +
                      f"p={controller.K_P * controller._e:.3f} " +
                      f"d={controller.K_D * de:.3f} " +
                      f"i={controller.K_I * ie:.3f}")
                
                t_next_ctrl += dt_controller
            
            # ----------------------
            # PHYSICS STEP (high frequency)
            # ----------------------
            # Advance the tractor and trailers by DT_PHYS
            # Use current controller outputs (stale between controller updates)
            
            # FIXED: Limit throttle at low speeds, but not too aggressively
            # We need SOME torque at startup to overcome rolling resistance
            throttle_limited = controller.throttle
            
            # FIXED CONTROL OVERRIDES (User Requested)
            if USE_FIXED_SPEED:
                # Force constant speed (Infinite power / Cruise Control)
                tractor.vx = FIXED_SPEED_VALUE
                tractor.ax = 0.0
                throttle_limited = 0.0 # Zero out for logging/physics, state is forced
                controller.brake = 0.0 # CRITICAL: Ensure no braking eats up tire grip
                
                # If fixed speed is used, we bypass governor and soft start
            else:
                # NORMAL OPERATION: Use controller input with governor and soft start
                
                # Speed governor with ACTIVE braking at maximum speed
                max_speed = DEFAULT_TARGET_SPEED+5  # m/s (72 km/h - realistic for heavy truck)
                
                if tractor.vx >= max_speed:
                    # CRITICAL: Actively brake when at max speed
                    throttle_limited = 0.0
                    # Override controller brake with governor brake if needed
                    governor_brake = min(1.0, (tractor.vx - max_speed) * 0.5)  # Proportional brake
                    controller.brake = max(controller.brake, governor_brake)
                else:
                    # Below max speed: trust the controller completely
                    throttle_limited = controller.throttle
                
                # SOFT START: Gradually ramp throttle over first 3 seconds to prevent solver divergence
                if t < 3.0 and tractor.vx < 2.0:
                    soft_start_scale = t / 3.0  # Linear ramp: 0% at t=0 → 100% at t=3s
                    throttle_limited = throttle_limited * soft_start_scale
                    if steps % 50 == 0:  # Debug output
                        print(f"[SOFT_START] t={t:.2f}s scale={soft_start_scale:.2f} throttle={throttle_limited:.3f}")
            
            # Convert controller steering from normalized [-1, 1] to radians
            if USE_FIXED_STEERING:
                # Direct fixed steering angle
                delta_rad = np.radians(FIXED_STEERING_VALUE)
            else:
                # Controller: 1.0 = 70°, TractorHead expects radians (max ±30°)
                delta_rad = controller.steer * np.radians(70.0)  # [-1.22, 1.22] rad
            
            # ========================================================================
            # MONOLITHIC SOLVER UPDATE
            # ========================================================================
            # Solves the entire coupled system (Tractor + Trailers) simultaneously
            
            # FIXED SLOPE OVERRIDE (User Requested)
            if USE_FIXED_SLOPE:
                # Override pitch (slope) for all vehicles before solver uses it for gravity
                # Positive pitch = Uphill (Gravity pulls back in x)
                pitch_rad = np.radians(FIXED_SLOPE_DEG)
                tractor.pitch = pitch_rad
                trailer1.pitch = pitch_rad
                dolly.pitch = pitch_rad
                trailer2.pitch = pitch_rad

            # Update simulation time for all vehicles (needed for logging/plots)
            tractor._sim_time = t
            trailer1._sim_time = t
            dolly._sim_time = t
            trailer2._sim_time = t
            
            converged = system.solve_step(throttle=throttle_limited, brake=controller.brake, steer=delta_rad, dt=dt_phys)
            
            # CRITICAL: Check if solver converged - if not, ABORT simulation
            if not converged:
                print(f"\n{'='*80}")
                print(f"SIMULATION ABORTED: Solver failed to converge")
                print(f"Time: {t:.2f}s, Steps: {steps}, Speed: {tractor.vx:.2f} m/s")
                print(f"Saving log and aborting...")
                print(f"{'='*80}\n")
                # Generate plots with partial data before aborting
                original_print("Generating plots with partial simulation data...")
                # Flush and close log file
                if log_file:
                    log_file.flush()
                    log_file.close()
                # Raise exception to stop simulation
                raise RuntimeError(
                    f"Solver convergence failure at t={t:.2f}s. "
                    f"Check debug_output_full_simulation.log for details. "
                    f"Last state: vx={tractor.vx:.2f} m/s, throttle={throttle_limited:.3f}, brake={controller.brake:.3f}"
                )

            # ========================================================================
            # VEHICLE STATE SANITY CHECKS (Detect unphysical behavior early)
            # ========================================================================
            
            # Check 1: Lateral velocity sanity (heavy trucks should NOT have extreme lateral slip)
            if abs(tractor.vy) > 5.0:  # 5 m/s is extremely high for a truck
                print(f"[PHYSICS_WARNING] t={t:.2f}s EXTREME lateral velocity: vy={tractor.vy:.2f} m/s")
                print(f"  Position: ({tractor.x:.1f}, {tractor.y:.1f}), Yaw: {np.degrees(tractor.yaw):.1f}°")
                print(f"  Longitudinal: vx={tractor.vx:.2f} m/s, Yaw rate: {tractor.omega:.3f} rad/s")
                print(f"  Steering: {np.degrees(delta_rad):.1f}°, Throttle: {throttle_limited:.3f}, Brake: {controller.brake:.3f}")
                
            # Check 2: Total velocity magnitude sanity
            total_velocity = np.sqrt(tractor.vx**2 + tractor.vy**2)
            if total_velocity > 30.0:  # 108 km/h is unrealistic for this scenario
                print(f"[PHYSICS_WARNING] t={t:.2f}s EXTREME total velocity: {total_velocity:.2f} m/s (vx={tractor.vx:.2f}, vy={tractor.vy:.2f})")
            
            # Check 3: Articulation angle sanity
            if abs(trailer1.articulation_yaw) > np.radians(45.0):
                print(f"[ARTICULATION_WARNING] t={t:.2f}s EXTREME trailer1 articulation: {np.degrees(trailer1.articulation_yaw):.1f}°")
                print(f"  Tractor yaw: {np.degrees(tractor.yaw):.1f}°, Trailer1 position: ({trailer1.x:.1f}, {trailer1.y:.1f})")
                
            if abs(dolly.articulation_yaw) > np.radians(45.0):
                print(f"[ARTICULATION_WARNING] t={t:.2f}s EXTREME dolly articulation: {np.degrees(dolly.articulation_yaw):.1f}°")
                
            if abs(trailer2.articulation_yaw) > np.radians(45.0):
                print(f"[ARTICULATION_WARNING] t={t:.2f}s EXTREME trailer2 articulation: {np.degrees(trailer2.articulation_yaw):.1f}°")
            
            # Check 4: Hitch constraint verification (trailers should stay connected!)
            # Use the constraint-target model's expected distance (leader_rear_offset + hitch_length + lf)
            expected_dist = trailer1.leader_rear_offset + trailer1.hitch_length + trailer1.lf
            hitch_error_trailer1 = abs(np.sqrt((trailer1.x - tractor.x)**2 + (trailer1.y - tractor.y)**2) - expected_dist)
            
            if hitch_error_trailer1 > 0.5 and t > 1.0:  # 0.5m tolerance
                print(f"[HITCH_WARNING] t={t:.2f}s Trailer1 hitch error: {hitch_error_trailer1:.2f}m")
                print(f"  Tractor: ({tractor.x:.1f}, {tractor.y:.1f}), Trailer1: ({trailer1.x:.1f}, {trailer1.y:.1f})")
            
            # Detailed diagnostics for first 10 seconds
            if t <= 10.0 and steps % 50 == 0:
                print(f"[STATE_DETAIL] t={t:.2f}s tractor: pos=({tractor.x:.1f},{tractor.y:.1f}) " +
                      f"vel=(vx={tractor.vx:.2f},vy={tractor.vy:.2f}) yaw={np.degrees(tractor.yaw):.1f}°")
                print(f"  trailer1: pos=({trailer1.x:.1f},{trailer1.y:.1f}) " +
                      f"vel=(vx={trailer1.vx:.2f},vy={trailer1.vy:.2f}) artic={np.degrees(trailer1.articulation_yaw):.1f}°")
                print(f"  dolly: pos=({dolly.x:.1f},{dolly.y:.1f}) artic={np.degrees(dolly.articulation_yaw):.1f}°")
                print(f"  trailer2: pos=({trailer2.x:.1f},{trailer2.y:.1f}) artic={np.degrees(trailer2.articulation_yaw):.1f}°")
            
            # Update simulation time
            t += dt_phys

            # Update closest waypoint index
            dists = np.linalg.norm(waypoints_np[:, :2] - np.array([tractor.x, tractor.y]), axis=1)
            closest_index = np.argmin(dists)
            
            # SAFETY CHECK: Detect position teleporting (solver divergence)
            # If closest_index jumped by more than 50 waypoints in one step, something is wrong
            if steps > 100:  # After initial phase
                if abs(closest_index - max_closest_index) > 50:
                    # Vehicle likely teleported - reset to maintain monotonic progress
                    closest_index = max_closest_index
                    print(f"WARNING: Position jump detected at t={t:.2f}s, clamping waypoint index")
            
            max_closest_index = max(max_closest_index, closest_index)

            # Waypoint-based progress tracking
            # More robust than arc-length as it directly tracks path progression
            percent_complete = 100.0 * (max_closest_index / max(1, (waypoints_np.shape[0] - 1)))
            percent_complete = min(100.0, max(0.0, percent_complete))

            # optional: print progress occasionally
            if steps % 100 == 0:
                # Diagnostic output: include solver convergence info
                waypoint_jump = closest_index - (0 if steps == 0 else (closest_index_prev if 'closest_index_prev' in locals() else closest_index))
                print(f"t={t:.2f}s  speed={tractor.vx:.2f} m/s  progress={percent_complete:.1f}%  waypoint={max_closest_index}/{waypoints_np.shape[0]-1}")
                closest_index_prev = closest_index
            steps += 1

            # Update position, timestamp
            current_x, current_y, current_yaw = tractor.x, tractor.y, tractor.yaw
            x_history.append(current_x)
            y_history.append(current_y)
            current_z = tractor.z  # update from vehicle state after terrain query
            z_history.append(current_z)
            z_ref = list(waypoints_np[:, 2])
            yaw_history.append(current_yaw)
            pitch_history.append(tractor.pitch)
            roll_history.append(tractor.roll)
            
            # Tractor velocities and accelerations
            speed_history.append(tractor.vx)
            current_speed=tractor.vx
            vy_history.append(tractor.vy)
            omega_history.append(tractor.omega)
            ax_history.append(tractor.ax_prev)  # Last computed acceleration
            ay_history.append(tractor.ay_prev)  # Last computed acceleration
            
            # Tractor control and state
            shift_history.append(tractor.current_gear+1)
            brake_history.append(controller.brake)
            steer_history.append(controller.steer)
            t_hist.append(t)

            # Trailer 1 state
            x_trailer1_history.append(trailer1.x)
            y_trailer1_history.append(trailer1.y)
            z_trailer1_history.append(trailer1.z)
            vx_trailer1_history.append(trailer1.vx)
            vy_trailer1_history.append(trailer1.vy)
            omega_trailer1_history.append(trailer1.omega)
            articulation1_history.append(trailer1.articulation_yaw)
            pitch_trailer1_history.append(trailer1.pitch)
            roll_trailer1_history.append(trailer1.roll)

            # Dolly state
            x_dolly_history.append(dolly.x)
            y_dolly_history.append(dolly.y)
            z_dolly_history.append(dolly.z)
            vx_dolly_history.append(dolly.vx)
            vy_dolly_history.append(dolly.vy)
            omega_dolly_history.append(dolly.omega)
            articulation2_history.append(dolly.articulation_yaw)
            pitch_dolly_history.append(dolly.pitch)
            roll_dolly_history.append(dolly.roll)

            # Trailer 2 state
            x_trailer2_history.append(trailer2.x)
            y_trailer2_history.append(trailer2.y)
            z_trailer2_history.append(trailer2.z)
            vx_trailer2_history.append(trailer2.vx)
            vy_trailer2_history.append(trailer2.vy)
            omega_trailer2_history.append(trailer2.omega)
            articulation3_history.append(trailer2.articulation_yaw)
            pitch_trailer2_history.append(trailer2.pitch)
            roll_trailer2_history.append(trailer2.roll)

            # ----------------------
            # CONTROLLER UPDATE (lower frequency)
            # ----------------------
            if t >= t_next_ctrl:
                # Option A: query the controller by current (x,y) as before,
                # but the controller now runs at DT_CTRL and internal PID uses self.dt_ctrl.
                current_x, current_y, current_yaw = tractor.x, tractor.y, tractor.yaw
                current_speed = tractor.vx

                # Find closest_index as you did (you may keep your existing code),
                # or optionally use s_progress to fetch a spatial reference.
                # I'll reuse your "closest_index" logic for compatibility:
                closest_distance = np.linalg.norm(np.array([
                    waypoints_np[closest_index, 0] - current_x,
                    waypoints_np[closest_index, 1] - current_y]))
                new_distance = closest_distance
                new_index = closest_index
                while new_distance <= closest_distance:
                    closest_distance = new_distance
                    closest_index = new_index
                    new_index += 1
                    if new_index >= waypoints_np.shape[0]:
                        break
                    new_distance = np.linalg.norm(np.array([
                        waypoints_np[new_index, 0] - current_x,
                        waypoints_np[new_index, 1] - current_y]))

                new_distance = closest_distance
                new_index = closest_index
                while new_distance <= closest_distance:
                    closest_distance = new_distance
                    closest_index = new_index
                    new_index -= 1
                    if new_index < 0:
                        break
                    new_distance = np.linalg.norm(np.array([
                        waypoints_np[new_index, 0] - current_x,
                        waypoints_np[new_index, 1] - current_y]))

                waypoint_subset_first_index = closest_index - 1
                if waypoint_subset_first_index < 0:
                    waypoint_subset_first_index = 0

                waypoint_subset_last_index = closest_index
                total_distance_ahead = 0
                while total_distance_ahead < INTERP_LOOKAHEAD_DISTANCE:
                    total_distance_ahead += wp_distance[waypoint_subset_last_index]
                    waypoint_subset_last_index += 1
                    if waypoint_subset_last_index >= waypoints_np.shape[0]:
                        waypoint_subset_last_index = waypoints_np.shape[0] - 1
                        break

                new_waypoints = wp_interp[wp_interp_hash[waypoint_subset_first_index]:
                                         wp_interp_hash[waypoint_subset_last_index] + 1]

                # update controller with the waypoints and current states
                controller.update_waypoints(new_waypoints)
                controller.update_values(current_x, current_y, current_yaw, current_speed, timestamp=t)
                controller.update_controls()
                # schedule next controller execution
                t_next_ctrl += dt_controller
            # Keep the append on the wile loop for plotting later (same dimensions as t_hist)
            speed_ref.append(controller._desired_speed)
            throttle_history.append(controller.throttle)
            speed_error.append(controller._e)
            
            # ----------------------
            # ROLLING MEMORY MANAGEMENT (every 10 seconds)
            # ----------------------
            if t >= next_export_time:
                # Close previous log file and open new one for this window
                if log_file:
                    log_file.close()
                
                # Create new log file for this time window
                log_filename = f"results/debug_{int(log_window_start):04d}-{int(log_window_end):04d}s.log"
                log_file = open(log_filename, 'w')
                original_print(f"[LOG] New log file: {log_filename}")
                
                # Export CSV for this time window
                export_filename = f"results/simulation_data_{int(log_window_start):04d}-{int(log_window_end):04d}s.csv"
                export_data_to_csv(
                    t_hist, x_history, y_history, z_history,
                    speed_history, vy_history, omega_history,
                    yaw_history, pitch_history, roll_history,
                    ax_history, ay_history,
                    x_trailer1_history, y_trailer1_history, z_trailer1_history,
                    vx_trailer1_history, vy_trailer1_history, omega_trailer1_history,
                    articulation1_history, pitch_trailer1_history, roll_trailer1_history,
                    x_dolly_history, y_dolly_history, z_dolly_history,
                    vx_dolly_history, vy_dolly_history, omega_dolly_history,
                    articulation2_history, pitch_dolly_history, roll_dolly_history,
                    x_trailer2_history, y_trailer2_history, z_trailer2_history,
                    vx_trailer2_history, vy_trailer2_history, omega_trailer2_history,
                    articulation3_history, pitch_trailer2_history, roll_trailer2_history,
                    throttle_history, brake_history, steer_history, shift_history,
                    speed_ref, speed_error,
                    export_filename
                )
                
                original_print(f"[EXPORT] Exported {len(t_hist)} timesteps to {export_filename}")
                original_print(f"[CLEANUP] Clearing memory: keeping last 5 timesteps, freeing {len(t_hist)-5} entries")
                
                # ========================================================================
                # MEMORY WIPE: Keep only last 5 timesteps
                # ========================================================================
                # This ensures constant memory usage regardless of simulation length
                KEEP_LAST_N = 5
                
                # Time and reference
                t_hist = t_hist[-KEEP_LAST_N:]
                
                # Tractor state
                x_history = x_history[-KEEP_LAST_N:]
                y_history = y_history[-KEEP_LAST_N:]
                z_history = z_history[-KEEP_LAST_N:]
                yaw_history = yaw_history[-KEEP_LAST_N:]
                pitch_history = pitch_history[-KEEP_LAST_N:]
                roll_history = roll_history[-KEEP_LAST_N:]
                speed_history = speed_history[-KEEP_LAST_N:]
                vy_history = vy_history[-KEEP_LAST_N:]
                omega_history = omega_history[-KEEP_LAST_N:]
                ax_history = ax_history[-KEEP_LAST_N:]
                ay_history = ay_history[-KEEP_LAST_N:]
                
                # Tractor control
                shift_history = shift_history[-KEEP_LAST_N:]
                brake_history = brake_history[-KEEP_LAST_N:]
                steer_history = steer_history[-KEEP_LAST_N:]
                throttle_history = throttle_history[-KEEP_LAST_N:]
                
                # Trailer 1 state
                x_trailer1_history = x_trailer1_history[-KEEP_LAST_N:]
                y_trailer1_history = y_trailer1_history[-KEEP_LAST_N:]
                z_trailer1_history = z_trailer1_history[-KEEP_LAST_N:]
                vx_trailer1_history = vx_trailer1_history[-KEEP_LAST_N:]
                vy_trailer1_history = vy_trailer1_history[-KEEP_LAST_N:]
                omega_trailer1_history = omega_trailer1_history[-KEEP_LAST_N:]
                articulation1_history = articulation1_history[-KEEP_LAST_N:]
                pitch_trailer1_history = pitch_trailer1_history[-KEEP_LAST_N:]
                roll_trailer1_history = roll_trailer1_history[-KEEP_LAST_N:]
                
                # Dolly state
                x_dolly_history = x_dolly_history[-KEEP_LAST_N:]
                y_dolly_history = y_dolly_history[-KEEP_LAST_N:]
                z_dolly_history = z_dolly_history[-KEEP_LAST_N:]
                vx_dolly_history = vx_dolly_history[-KEEP_LAST_N:]
                vy_dolly_history = vy_dolly_history[-KEEP_LAST_N:]
                omega_dolly_history = omega_dolly_history[-KEEP_LAST_N:]
                articulation2_history = articulation2_history[-KEEP_LAST_N:]
                pitch_dolly_history = pitch_dolly_history[-KEEP_LAST_N:]
                roll_dolly_history = roll_dolly_history[-KEEP_LAST_N:]
                
                # Trailer 2 state
                x_trailer2_history = x_trailer2_history[-KEEP_LAST_N:]
                y_trailer2_history = y_trailer2_history[-KEEP_LAST_N:]
                z_trailer2_history = z_trailer2_history[-KEEP_LAST_N:]
                vx_trailer2_history = vx_trailer2_history[-KEEP_LAST_N:]
                vy_trailer2_history = vy_trailer2_history[-KEEP_LAST_N:]
                omega_trailer2_history = omega_trailer2_history[-KEEP_LAST_N:]
                articulation3_history = articulation3_history[-KEEP_LAST_N:]
                pitch_trailer2_history = pitch_trailer2_history[-KEEP_LAST_N:]
                roll_trailer2_history = roll_trailer2_history[-KEEP_LAST_N:]
                
                # Reference signals
                speed_ref = speed_ref[-KEEP_LAST_N:]
                speed_error = speed_error[-KEEP_LAST_N:]
                
                # Update counters for next window
                export_counter += 1
                log_window_start = log_window_end
                log_window_end += 10.0
                next_export_time += 10.0
                
                original_print(f"[CLEANUP] Memory cleared. Next export at t={next_export_time}s")
                original_print(f"   Current memory: {KEEP_LAST_N} timesteps (~{KEEP_LAST_N * 40 * 8 / 1024:.1f} KB)")
                original_print("")


            # ----------------------
            # Termination check (use latest tractor.x,y)
            # ----------------------
            # If using FIXED controls (Open Loop), we ignore path termination 
            # and run until MAX_SIM_TIME
            if not (USE_FIXED_STEERING or USE_FIXED_SPEED):
                dist_to_last_waypoint = np.linalg.norm(np.array([
                    waypoints_np[-1][0] - tractor.x,
                    waypoints_np[-1][1] - tractor.y]))
                if dist_to_last_waypoint < DIST_THRESHOLD_TO_LAST_WAYPOINT or tractor.x > 320:
                    reached_the_end = True
                    print("Reached the end of path. Exporting final data ...")
                
                    # Export final simulation data
                    export_data_to_csv(
                        t_hist, x_history, y_history, z_history,
                        speed_history, vy_history, omega_history,
                        yaw_history, pitch_history, roll_history,
                        ax_history, ay_history,
                        x_trailer1_history, y_trailer1_history, z_trailer1_history,
                        vx_trailer1_history, vy_trailer1_history, omega_trailer1_history,
                        articulation1_history, pitch_trailer1_history, roll_trailer1_history,
                        x_dolly_history, y_dolly_history, z_dolly_history,
                        vx_dolly_history, vy_dolly_history, omega_dolly_history,
                        articulation2_history, pitch_dolly_history, roll_dolly_history,
                        x_trailer2_history, y_trailer2_history, z_trailer2_history,
                        vx_trailer2_history, vy_trailer2_history, omega_trailer2_history,
                        articulation3_history, pitch_trailer2_history, roll_trailer2_history,
                        throttle_history, brake_history, steer_history, shift_history,
                        speed_ref, speed_error,
                        "results/simulation_data_final.csv"
                    )
                    
                    # final plot and break

                    # shift_history=shift_history+1 # to make gear start from 1 instead of 0

                    break
    
    # If we exited the loop due to MAX_SIM_TIME (not reaching the end), generate plots
    if not reached_the_end:
        print(f"\nSimulation reached MAX_SIM_TIME={MAX_SIM_TIME}s limit without completing path.")
        print(f"Final progress: {percent_complete:.1f}% (waypoint {max_closest_index}/{waypoints_np.shape[0]-1})")
        print("Exporting final data and partial simulation data...")
        
        # Export final simulation data
        export_data_to_csv(
            t_hist, x_history, y_history, z_history,
            speed_history, vy_history, omega_history,
            yaw_history, pitch_history, roll_history,
            ax_history, ay_history,
            x_trailer1_history, y_trailer1_history, z_trailer1_history,
            vx_trailer1_history, vy_trailer1_history, omega_trailer1_history,
            articulation1_history, pitch_trailer1_history, roll_trailer1_history,
            x_dolly_history, y_dolly_history, z_dolly_history,
            vx_dolly_history, vy_dolly_history, omega_dolly_history,
            articulation2_history, pitch_dolly_history, roll_dolly_history,
            x_trailer2_history, y_trailer2_history, z_trailer2_history,
            vx_trailer2_history, vy_trailer2_history, omega_trailer2_history,
            articulation3_history, pitch_trailer2_history, roll_trailer2_history,
            throttle_history, brake_history, steer_history, shift_history,
            speed_ref, speed_error,
            "results/simulation_data_final.csv"
        )
    # Close log file if still open
    if log_file:
        log_file.close()


if __name__ == '__main__':
    main()  