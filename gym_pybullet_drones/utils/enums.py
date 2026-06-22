from enum import Enum

class DroneModel(Enum):
    """Drone models enumeration class."""

    CF2X = "cf2x"   # Bitcraze Craziflie 2.0 in the X configuration
    CF2P = "cf2p"   # Bitcraze Craziflie 2.0 in the + configuration
    RACE = "racer"  # Racer drone in the X configuration


################################################################################

class Physics(Enum):
    """Physics implementations enumeration class."""

    PYB = "pyb"                         # Base PyBullet physics update
    DYN = "dyn"                         # Explicit dynamics model
    PYB_GND = "pyb_gnd"                 # PyBullet physics update with ground effect
    PYB_DRAG = "pyb_drag"               # PyBullet physics update with drag
    PYB_DW = "pyb_dw"                   # PyBullet physics update with downwash
    PYB_GND_DRAG_DW = "pyb_gnd_drag_dw" # PyBullet physics update with ground effect, drag, and downwash

################################################################################

class ImageType(Enum):
    """Camera capture image type enumeration class."""

    RGB = 0     # Red, green, blue (and alpha)
    DEP = 1     # Depth
    SEG = 2     # Segmentation by object id
    BW = 3      # Black and white

################################################################################

class ActionType(Enum):
    """Action type enumeration class."""
    RPM = "rpm"                 # RPMS
    PID = "pid"                 # PID control
    VEL = "vel"                 # Velocity input (using PID control)
    ONE_D_RPM = "one_d_rpm"     # 1D (identical input to all motors) with RPMs
    ONE_D_PID = "one_d_pid"     # 1D (identical input to all motors) with PID control

################################################################################

class ObservationType(Enum):
    """Observation type enumeration class."""
    KIN = "kin"     # Kinematic information (pose, linear and angular velocities)
    RGB = "rgb"     # RGB camera capture in each drone's POV
    DEP = "dep"     # Depth camera capture in each drone's POV
    ALL = "all"     # Kinematic + RGBD

################################################################################

class BehaviorType(Enum):
    """High-level (macro) behavior enumeration class.

    Each value abstracts one of the four core tactical behaviors that an RL
    policy can command. The low-level trajectory generation is hand-coded in
    `gym_pybullet_drones.behaviors`.
    """
    IDLE    = "idle"     # Hold the current position (default / between macros)
    TRANSIT = "transit"  # (1) Point-to-point move along the line of sight
    RECON   = "recon"    # (2) Cover a disk around a center (lawnmower / spiral)
    LOITER  = "loiter"   # (3) Circular orbit that tracks a (moving) target
    STRIKE  = "strike"   # (4) Terminal dash that drives error to zero
