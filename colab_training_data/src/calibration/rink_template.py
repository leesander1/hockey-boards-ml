"""
NHL Rink Template - Canonical geometry in world coordinates (feet).

Coordinate system:
  x = along the rink length (0 = left end boards, 200 = right end boards)
  y = across the rink width  (0 = near/camera-side boards, 85 = far boards)

The broadcast camera sits on the y=0 side, elevated, looking across the rink.
In the video frame:
  - x-axis of rink  → left/right in the image
  - y-axis of rink  → depth (near=bottom of frame, far=top of frame)
  - Rink lines (blue/red) appear as roughly vertical stripes in the image
"""

# ── Rink dimensions ──────────────────────────────────────────────────────────
RINK_LENGTH = 200.0   # feet, x-axis
RINK_WIDTH  =  85.0   # feet, y-axis

# ── Key line x-positions (feet along the rink length) ─────────────────────
CENTER_LINE_X       = 100.0
BLUE_LINE_LEFT_X    =  75.0   # left of center
BLUE_LINE_RIGHT_X   = 125.0   # right of center
GOAL_LINE_LEFT_X    =  11.0
GOAL_LINE_RIGHT_X   = 189.0

# ── Board / ad-zone geometry ──────────────────────────────────────────────
# Dasher boards sit on the ice surface.  The "ad strip" is the vertical face
# of the boards, which projects to a thin polygon just OUTSIDE the ice perimeter
# in the floor plane (because the boards have ~1 ft of physical thickness plus
# we want a few feet of visible face).  We model the near/far board strips as
# rectangles that extend 5 ft outside y=0 and y=85 respectively.
BOARD_STRIP_FT = 5.0   # feet outside the ice edge to cover the visible board face

# Correspondence points used for homography calibration.
# Each entry is (world_x, world_y) in feet.
# world_y=0  → where lines meet the near (camera-side) boards
# world_y=85 → where lines meet the far boards
CALIBRATION_WORLD_POINTS = [
    (BLUE_LINE_LEFT_X,   0.0),          # left blue line,  near boards
    (BLUE_LINE_LEFT_X,   RINK_WIDTH),   # left blue line,  far  boards
    (CENTER_LINE_X,      0.0),          # red  center line, near boards
    (CENTER_LINE_X,      RINK_WIDTH),   # red  center line, far  boards
    (BLUE_LINE_RIGHT_X,  0.0),          # right blue line, near boards
    (BLUE_LINE_RIGHT_X,  RINK_WIDTH),   # right blue line, far  boards
]


def get_near_board_polygon_world():
    """
    Returns the near-side board polygon (camera side, y ≈ 0) in world coords.
    Represented as a list of (x, y) tuples covering the full rink length.
    """
    return [
        (0.0,            -BOARD_STRIP_FT),
        (RINK_LENGTH,    -BOARD_STRIP_FT),
        (RINK_LENGTH,     0.0),
        (0.0,             0.0),
    ]


def get_far_board_polygon_world():
    """
    Returns the far-side board polygon in world coords.
    """
    return [
        (0.0,         RINK_WIDTH),
        (RINK_LENGTH, RINK_WIDTH),
        (RINK_LENGTH, RINK_WIDTH + BOARD_STRIP_FT),
        (0.0,         RINK_WIDTH + BOARD_STRIP_FT),
    ]


def get_end_board_polygons_world():
    """
    Returns the two end-board polygons (left and right ends).
    """
    left = [
        (-BOARD_STRIP_FT, 0.0),
        (0.0,             0.0),
        (0.0,             RINK_WIDTH),
        (-BOARD_STRIP_FT, RINK_WIDTH),
    ]
    right = [
        (RINK_LENGTH,              0.0),
        (RINK_LENGTH + BOARD_STRIP_FT, 0.0),
        (RINK_LENGTH + BOARD_STRIP_FT, RINK_WIDTH),
        (RINK_LENGTH,              RINK_WIDTH),
    ]
    return left, right
