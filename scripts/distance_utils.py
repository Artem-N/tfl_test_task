"""
distance_utils.py

Purpose (per task section 6):
    Approximate real-world distance (meters) of a detected/GT vehicle from the
    camera, using bbox pixel size as a proxy, and bucket it into distance bands.

Assumptions (task section 6):
    - Average vehicle length ~= 4.5 m (REAL_VEHICLE_LENGTH_M)
    - Camera horizontal FOV ~= 90 degrees (HORIZONTAL_FOV_DEG) -- measured/known
      value for the drone camera used here (supersedes the task's generic 84 deg
      placeholder assumption; see README "Distance Estimation" section).

Formulas:
    f_px = frame_width_px / (2 * tan(horizontal_FOV_rad / 2))
    distance_m = real_vehicle_length_m * f_px / bbox_size_px
    bbox_size_px = max(box_width_px, box_height_px)

`frame_width_px` must be the pixel width (x-dimension) of the actual frame as
stored/loaded, regardless of the video's landscape/portrait orientation - the
eval videos are not all the same orientation (see evaluate.py).

Bands:
    0-200 m   -> "near"
    200-400 m -> "far"
    >400 m    -> ignored for these metrics
"""

import math

REAL_VEHICLE_LENGTH_M = 4.5
HORIZONTAL_FOV_DEG = 90.0

NEAR_BAND = "near"       # 0-200 m
FAR_BAND = "far"         # 200-400 m
IGNORE_BAND = "ignore"   # >400 m

NEAR_MAX_M = 200.0
FAR_MAX_M = 400.0


def compute_focal_length_px(frame_width_px: int, horizontal_fov_deg: float = HORIZONTAL_FOV_DEG) -> float:
    """Compute the effective focal length in pixels from frame width and horizontal FOV."""
    return frame_width_px / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))


def estimate_distance_m(
    box_width_px: float,
    box_height_px: float,
    focal_length_px: float,
    real_vehicle_length_m: float = REAL_VEHICLE_LENGTH_M,
) -> float:
    """Estimate distance in meters from a bbox size and precomputed focal length.

    bbox_size_px = max(box_width_px, box_height_px)
    distance_m = real_vehicle_length_m * focal_length_px / bbox_size_px
    """
    bbox_size_px = max(box_width_px, box_height_px)
    if bbox_size_px <= 0:
        return math.inf
    return real_vehicle_length_m * focal_length_px / bbox_size_px


def distance_to_band(distance_m: float) -> str:
    """Map a distance in meters to a band: 'near' (0-200), 'far' (200-400), or 'ignore' (>400)."""
    if distance_m <= NEAR_MAX_M:
        return NEAR_BAND
    if distance_m <= FAR_MAX_M:
        return FAR_BAND
    return IGNORE_BAND
