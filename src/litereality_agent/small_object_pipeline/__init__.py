"""RGB-D grounded small-object reconstruction for LiteReality captures.

Phase 1/2 deliberately contains no detector or SAM3D dependency.  It establishes the only
coordinate contract later stages may use: canonical object geometry is placed in the scanner's
ARKit/RoomPlan world from measured RGB-D points.
"""

from .coordinate_frames import CameraIntrinsics, apply_transform, backproject_pixels
from .scanner_frame import ScannerFrame, load_scanner_frame

__all__ = [
    "CameraIntrinsics",
    "ScannerFrame",
    "apply_transform",
    "backproject_pixels",
    "load_scanner_frame",
]
