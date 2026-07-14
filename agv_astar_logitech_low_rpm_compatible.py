import math
import time
import json
import threading

import cv2
import serial
from pupil_apriltags import Detector


#Config Files
# AprilTag heading is sent to ESP32 as the ground-truth current heading.
# The permanent target remains the map heading (0, 90, 180, or -90).
TAG_HEADING_OFFSET_GAIN = 1.0
MAX_TAG_HEADING_OFFSET_DEG = 1.0

# global variables for threading
latest_frame = None
latest_detections = []
latest_pose = None
latest_key = -1
latest_frame_id = 0

latest_lock = threading.Lock()
camera_running = True

MAP_FILE = "./maps/testbed.json"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

USB_CAMERA_DEVICE = "/dev/video0"
CAMERA_FPS = 30
CAMERA_FOURCC = "MJPG"

FX = 615.0
FY = 615.0
CX = FRAME_WIDTH / 2.0
CY = FRAME_HEIGHT / 2.0
CAMERA_PARAMS = (FX, FY, CX, CY)

TAG_SIZE_M = 0.010
APRILTAG_FAMILY = "tag36h11"

SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 115200
 # Docking variables
DOCK_NODE = 0
FIRST_NODE = 1

# Low-speed test profile using the same 7-field serial protocol as the 40 RPM ESP32 firmware.
DRIVE_VELOCITY_MPS = 0.050
ARRIVAL_VELOCITY_MPS = 0.025

# Every adjacent map node is one 0.50 m segment.
SEGMENT_DISTANCE_M = 0.50

# A segment ending at a turn or final goal must already be at the slow
# arrival speed before the landmark centre. Pass-through segments keep
# cruise speed through the endpoint.
TURN_GOAL_END_VELOCITY_MPS = ARRIVAL_VELOCITY_MPS
PASSTHROUGH_END_VELOCITY_MPS = DRIVE_VELOCITY_MPS

HELPER_SPACING_M = 0.015

TAG_HEADING_GAIN = 0.40
MAX_TAG_HEADING_CORRECTION_DEG = 2.0

MAX_ACCEPTED_LATERAL_M = 0.100

# Now this is used relative to active path heading, not absolute tag heading.
MAX_ACCEPTED_HEADING_DEG = 25.0

TURN_HEADING_THRESHOLD_DEG = 1.0

MODE_DOCK_WAIT = "DOCK_WAIT"
MODE_DOCK_TO_TAG1 = "DOCK_TO_TAG1"
MODE_WAIT_TASK = "WAIT_TASK"
MODE_RUN_PATH = "RUN_PATH"

DOCK_HEADING_DEG = 0.0

# Map loading
def load_map(filename):
    with open(filename, "r") as f:
        return json.load(f)


MAP_DATA = load_map(MAP_FILE)


# Helper function

def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_angle(angle_deg):
    while angle_deg > 180:
        angle_deg -= 360
    while angle_deg <= -180:
        angle_deg += 360
    return angle_deg


def tag_area(detection):
    corners = getattr(detection, "corners", None)

    if corners is None or len(corners) != 4:
        return 0.0

    x0, y0 = corners[0]
    x1, y1 = corners[1]
    x2, y2 = corners[2]
    x3, y3 = corners[3]

    return 0.5 * abs(
        x0 * y1 + x1 * y2 + x2 * y3 + x3 * y0
        - y0 * x1 - y1 * x2 - y2 * x3 - y3 * x0
    )


# Tag priority and helper offset

def helper_lateral_offset_for_heading(position, heading_deg):
    """
    Convert the selected helper-tag position into the lateral offset from
    that helper tag to the landmark's main centre tag.

    The raw camera lateral measurement keeps its camera-frame sign.
    Only the map-position helper offset changes with travel direction.
    """
    h = normalize_angle(heading_deg)

    east_group = {"east", "north_east", "south_east"}
    west_group = {"west", "north_west", "south_west"}
    south_group = {"south", "south_west", "south_east"}
    north_group = {"north", "north_west", "north_east"}

    # Map heading 0 degrees.
    if abs(normalize_angle(h - 0.0)) < 5.0:
        if position in east_group:
            return +HELPER_SPACING_M
        if position in west_group:
            return -HELPER_SPACING_M
        return 0.0

    # Map heading 180 degrees.
    if abs(normalize_angle(h - 180.0)) < 5.0:
        if position in east_group:
            return -HELPER_SPACING_M
        if position in west_group:
            return +HELPER_SPACING_M
        return 0.0

    # Map heading +90 degrees.
    # User-defined rule:
    # south-west, south, south-east -> +15 mm
    # north-west, north, north-east -> -15 mm
    if abs(normalize_angle(h - 90.0)) < 5.0:
        if position in south_group:
            return +HELPER_SPACING_M
        if position in north_group:
            return -HELPER_SPACING_M
        return 0.0

    # Map heading -90 degrees: opposite of +90 degrees.
    if abs(normalize_angle(h - (-90.0))) < 5.0:
        if position in south_group:
            return -HELPER_SPACING_M
        if position in north_group:
            return +HELPER_SPACING_M
        return 0.0

    return 0.0


def lateral_pose_for_heading(pose, active_heading):
    if pose is None:
        return 0.0

    raw_x = pose["raw_lateral"]

    helper_offset = helper_lateral_offset_for_heading(
        pose["position"],
        active_heading,
    )

    return raw_x + helper_offset


def tag_heading_error_for_log(map_heading, tag_heading):
    """
    Diagnostic only. The target is not changed by this error.
    ESP32 aligns its IMU estimate to tag_heading and then controls toward
    map_heading using normalized angular error.
    """
    if tag_heading is None:
        return None

    error = normalize_angle(map_heading - tag_heading)

    print(
        f"TAG_HEADING_GROUND_TRUTH "
        f"map={map_heading:.2f} "
        f"tag={tag_heading:.2f} "
        f"map_error={error:.2f}"
    )

    return error


TAG_PRIORITY_BY_POSITION = {
    "center": 1,

    "north": 2,
    "east": 2,
    "south": 2,
    "west": 2,

    "north_west": 3,
    "north_east": 3,
    "south_west": 3,
    "south_east": 3,
}


def tag_priority(position):
    return TAG_PRIORITY_BY_POSITION.get(position, 99)


#apriltag pose helpers

def compute_heading(detection):
    if detection.pose_R is None:
        return None

    r = detection.pose_R

    return normalize_angle(
        math.degrees(math.atan2(r[1, 0], r[0, 0]))
    )


def compute_lateral(detection):
    if detection.pose_t is None:
        return None

    return float(detection.pose_t[0][0])


def compute_forward(detection):
    if detection.pose_t is None:
        return None

    return float(detection.pose_t[1][0])

# Map loading

def landmark_by_id(landmark_id):
    for landmark in MAP_DATA["landmarks"]:
        if int(landmark["id"]) == int(landmark_id):
            return landmark

    return None


def find_tag(tag_id):
    for landmark in MAP_DATA["landmarks"]:
        for position, mapped_tag_id in landmark["tags"].items():
            if int(mapped_tag_id) == int(tag_id):
                return {
                    "landmark": landmark,
                    "id": int(landmark["id"]),
                    "position": position,
                }

    return None


def neighbors(landmark_id):
    current = landmark_by_id(landmark_id)

    if current is None:
        return []

    out = []

    for landmark in MAP_DATA["landmarks"]:
        if int(landmark["id"]) == int(landmark_id):
            continue

        dr = abs(landmark["row"] - current["row"])
        dc = abs(landmark["column"] - current["column"])

        if dr + dc == 1:
            out.append(int(landmark["id"]))

    return out


def heuristic(a, b):
    node_a = landmark_by_id(a)
    node_b = landmark_by_id(b)

    if node_a is None or node_b is None:
        return 999999

    return abs(node_a["row"] - node_b["row"]) + abs(
        node_a["column"] - node_b["column"]
    )


def find_path(start_id, goal_id):
    if landmark_by_id(start_id) is None or landmark_by_id(goal_id) is None:
        return []

    open_list = [(0, start_id)]
    came_from = {}
    g_score = {start_id: 0}

    while open_list:
        open_list.sort(key=lambda item: item[0])
        _, current = open_list.pop(0)

        if current == goal_id:
            path = [current]

            while current in came_from:
                current = came_from[current]
                path.append(current)

            path.reverse()
            return path

        for nxt in neighbors(current):
            tentative_g = g_score[current] + 1

            if nxt not in g_score or tentative_g < g_score[nxt]:
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                f_score = tentative_g + heuristic(nxt, goal_id)
                open_list.append((f_score, nxt))

    return []


def map_heading(current_id, next_id):
    """
    row +1    -> 0 deg
    column +1 -> 90 deg
    row -1    -> 180 deg
    column -1 -> -90 deg
    """

    current = landmark_by_id(current_id)
    target = landmark_by_id(next_id)

    if current is None or target is None:
        return None

    dr = target["row"] - current["row"]
    dc = target["column"] - current["column"]

    if dr == 1 and dc == 0:
        return 0.0

    if dr == 0 and dc == 1:
        return 90.0

    if dr == -1 and dc == 0:
        return 180.0

    if dr == 0 and dc == -1:
        return -90.0

    return None


# Arrival Gate

def waypoint_positions_for_heading(heading_deg):
    heading_deg = normalize_angle(heading_deg)

    # Moving north/south: center horizontal row is valid.
    if abs(normalize_angle(heading_deg - 0.0)) < 1.0:
        return {"west", "center", "east"}

    if abs(normalize_angle(heading_deg - 180.0)) < 1.0:
        return {"west", "center", "east"}

    # Moving east/west: center vertical column is valid.
    if abs(normalize_angle(heading_deg - 90.0)) < 1.0:
        return {"north", "center", "south"}

    if abs(normalize_angle(heading_deg - -90.0)) < 1.0:
        return {"north", "center", "south"}

    return {"center"}

# Check tag1 centre reached

def tag1_centre_reached(pose):
    if pose is None:
        return False
    
    if pose ["landmark_id"] != FIRST_NODE:
        return False
    
    return pose["position"] in {"west", "center", "east"}

def waypoint_reached_for_segment(pose, active_to, active_heading):
    if pose is None:
        return False

    if pose["landmark_id"] != active_to:
        return False

    allowed_positions = waypoint_positions_for_heading(active_heading)

    return pose["position"] in allowed_positions


# CAMERA AND DETECTOR


def start_camera():
    """Open a Logitech USB/UVC camera through OpenCV and V4L2."""
    camera = cv2.VideoCapture(USB_CAMERA_DEVICE, cv2.CAP_V4L2)

    if not camera.isOpened():
        raise RuntimeError(
            f"Cannot open USB camera {USB_CAMERA_DEVICE}. "
            "Check the /dev/video* device number and camera permissions."
        )

    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*CAMERA_FOURCC),
    )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    for _ in range(10):
        ok, _ = camera.read()
        if not ok:
            time.sleep(0.05)

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = camera.get(cv2.CAP_PROP_FPS)

    print(
        f"USB_CAMERA_OPEN device={USB_CAMERA_DEVICE} "
        f"size={actual_width}x{actual_height} "
        f"fps={actual_fps:.1f}"
    )

    return camera

def create_detector():
    return Detector(
        families=APRILTAG_FAMILY,
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=True,
    )


def detect_tags(detector, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=CAMERA_PARAMS,
        tag_size=TAG_SIZE_M,
    )

    for det in detections:
        det.heading = compute_heading(det)
        det.lateral = compute_lateral(det)
        det.forward = compute_forward(det)
        det.area = tag_area(det)

    return detections


# ============================================================================
# PRIORITY-BASED POSE ESTIMATION
# ============================================================================

def estimate_pose_from_tags(detections):
    candidates = []
    unknown_tags = []

    for det in detections:
        result = find_tag(det.tag_id)

        if result is None:
            unknown_tags.append(int(det.tag_id))
            continue

        if det.heading is None or det.lateral is None:
            continue

        position = result["position"]
        # Store raw camera lateral only. The heading-aware helper offset is
        # applied later, when the active segment heading is known.
        offset = 0.0
        corrected_lateral = det.lateral

        if abs(corrected_lateral) > MAX_ACCEPTED_LATERAL_M:
            continue

        # IMPORTANT:
        # Do not reject by absolute heading here.
        # In multi-direction navigation, valid tag heading can be 0, 90, 180, or -90.

        area = max(float(getattr(det, "area", 0.0)), 1.0)

        candidates.append(
            {
                "tag": int(det.tag_id),
                "landmark": result["landmark"],
                "landmark_id": result["id"],
                "position": position,
                "priority": tag_priority(position),
                "heading": det.heading,
                "raw_lateral": det.lateral,
                "offset": offset,
                "corrected_lateral": corrected_lateral,
                "forward": det.forward,
                "area": area,
            }
        )

    if unknown_tags:
        print(f"Unknown tags detected: {unknown_tags}")

    if not candidates:
        return None

    selected = min(
        candidates,
        key=lambda item: (
            item["priority"],
            -item["area"],
        ),
    )

    visible_same_landmark = [
        item["tag"]
        for item in candidates
        if item["landmark_id"] == selected["landmark_id"]
    ]

    return {
        "valid": True,

        "landmark": selected["landmark"],
        "landmark_id": selected["landmark_id"],

        "tag": selected["tag"],
        "position": selected["position"],
        "priority": selected["priority"],

        "heading": selected["heading"],
        "lateral": selected["corrected_lateral"],
        "raw_lateral": selected["raw_lateral"],
        "center_lateral_offset": selected["offset"],
        "forward": selected["forward"],

        "visible_tags": visible_same_landmark,
        "used_count": 1,
        "quality": selected["area"],
    }


# ============================================================================
# NAVIGATION
# ============================================================================

def compute_navigation_for_segment(
    pose,
    active_to,
    active_heading,
    goal_node,
    velocity_mps,
):
    if pose is None:
        return None

    reached_waypoint = waypoint_reached_for_segment(
        pose,
        active_to,
        active_heading,
    )

    x_corrected = lateral_pose_for_heading(
        pose,
        active_heading,
    )

    tag_heading_error_for_log(
        active_heading,
        pose["heading"],
    )

    if reached_waypoint and active_to == goal_node:
        return {
            "current": pose["landmark_id"],
            "next": None,
            "map_heading": active_heading,
            "tag_heading": pose["heading"],
            "desired_heading": active_heading,
            "lateral_error": 0.0,
            "velocity": 0.0,
            "reached_waypoint": True,
            "final_arrival": True,
            "arrival_mode": False,
        }

    return {
        "current": pose["landmark_id"],
        "next": active_to,
        "map_heading": active_heading,
        "tag_heading": pose["heading"],
        "desired_heading": active_heading,
        "lateral_error": x_corrected,
        "velocity": (
            ARRIVAL_VELOCITY_MPS
            if pose["landmark_id"] == active_to and not reached_waypoint
            else velocity_mps
        ),
        "reached_waypoint": reached_waypoint,
        "final_arrival": False,
        "arrival_mode": (
            pose["landmark_id"] == active_to and not reached_waypoint
        ),
    }


# ============================================================================
# SERIAL COMMUNICATION
# ============================================================================

def open_serial(port=SERIAL_PORT, baud=SERIAL_BAUD):
    ser = serial.Serial(port, baud, timeout=0.2)

    time.sleep(2.0)

    ser.reset_input_buffer()
    ser.reset_output_buffer()

    return ser


def read_line(ser):
    line = ser.readline().decode(errors="ignore").strip()

    if line == "":
        return None

    return line


def read_available_lines(ser):
    lines = []

    while ser.in_waiting > 0:
        line = read_line(ser)

        if line is None:
            break

        lines.append(line)

    return lines


def wait_for_ack(ser, max_wait_s=1.0):
    deadline = time.monotonic() + max_wait_s

    while time.monotonic() < deadline:
        line = read_line(ser)

        if line is None:
            continue

        if line == "ACK" or line.startswith("ACK"):
            return True

        if line.startswith("ERR"):
            print("ESP32 error:", line)
            return False

        if line.startswith("FAULT"):
            print("ESP32 fault:", line)
            return False

        print("ESP32:", line)

    return False


def send_command_wait_ack(ser, command, max_wait_s=1.0):
    ser.reset_input_buffer()
    ser.write((command + "\n").encode())
    ser.flush()

    return wait_for_ack(ser, max_wait_s=max_wait_s)


def send_velocity(
    ser,
    velocity_mps,
    map_heading_deg,
    tag_heading_deg,
    lateral_error_m,
    segment_id,
    segment_distance_m,
    end_velocity_mps,
):
    """
    Start or update one distance-based segment motion profile.

    Repeated commands with the same segment_id update heading/lateral
    correction without restarting the travelled-distance profile.
    """
    command = (
        f"VEL {velocity_mps:.3f}"
        f" {map_heading_deg:.2f}"
        f" {tag_heading_deg:.2f}"
        f" {lateral_error_m:.4f}"
        f" {int(segment_id)}"
        f" {segment_distance_m:.3f}"
        f" {end_velocity_mps:.3f}"
    )

    print("TX:", command)

    return send_command_wait_ack(ser, command, max_wait_s=1.0)


# Send approach with map target, AprilTag ground-truth heading, lateral x,
# and forward y.
def send_approach(
    ser,
    velocity_mps,
    map_heading_deg,
    tag_heading_deg,
    x_lateral_m,
    y_lateral_m,
):
    command = (
        f"APP {velocity_mps:.3f}"
        f" {map_heading_deg:.2f}"
        f" {tag_heading_deg:.2f}"
        f" {x_lateral_m:.4f}"
        f" {y_lateral_m:.4f}"
    )

    print("TX:", command)

    return send_command_wait_ack(ser, command, max_wait_s=1.0)


def send_turn_wait_done(ser, target_heading_deg, max_wait_s=15.0):
    command = f"TURN {target_heading_deg:.2f}"

    print("TX:", command)

    ser.reset_input_buffer()
    ser.write((command + "\n").encode())
    ser.flush()

    deadline = time.monotonic() + max_wait_s
    got_ack = False

    while time.monotonic() < deadline:
        line = read_line(ser)

        if line is None:
            continue

        print("ESP32:", line)

        if line == "ACK" or line.startswith("ACK"):
            got_ack = True
            continue

        if line.startswith("TURN_DONE"):
            return True

        if line.startswith("ERR") or line.startswith("FAULT"):
            return False

    if not got_ack:
        print("No ACK for TURN.")

    print("TURN timeout.")

    return False


def calibrate_imu(ser):
    return send_command_wait_ack(ser, "CAL", max_wait_s=15.0)


def zero_heading(ser):
    return send_command_wait_ack(ser, "ZERO", max_wait_s=1.0)


def enable_motors(ser):
    return send_command_wait_ack(ser, "EN", max_wait_s=2.0)


def stop_robot(ser):
    return send_command_wait_ack(ser, "STOP", max_wait_s=1.0)


def disable_motors(ser):
    return send_command_wait_ack(ser, "DIS", max_wait_s=1.0)


# ============================================================================
# VIEWER
# ============================================================================

def draw_detections(frame, detections, pose=None, nav=None):
    height, width = frame.shape[:2]

    image_center_x = width // 2
    image_center_y = height // 2

    cv2.line(
        frame,
        (0, image_center_y),
        (width, image_center_y),
        (128, 128, 128),
        1,
    )

    cv2.line(
        frame,
        (image_center_x, 0),
        (image_center_x, height),
        (128, 128, 128),
        1,
    )

    cv2.circle(
        frame,
        (image_center_x, image_center_y),
        4,
        (128, 128, 128),
        -1,
    )

    selected_tag = None

    if pose is not None:
        selected_tag = pose["tag"]

    for det in detections:
        corners = det.corners.astype(int)
        tag_center = tuple(det.center.astype(int))

        if selected_tag is not None and int(det.tag_id) == int(selected_tag):
            color = (0, 255, 255)
            thickness = 3
        else:
            color = (0, 255, 0)
            thickness = 2

        for i in range(4):
            p1 = tuple(corners[i])
            p2 = tuple(corners[(i + 1) % 4])
            cv2.line(frame, p1, p2, color, thickness)

        cv2.circle(frame, tag_center, 5, (0, 0, 255), -1)

        cv2.line(
            frame,
            (image_center_x, image_center_y),
            tag_center,
            color,
            1,
        )

        x = int(corners[0][0])
        y = int(corners[0][1])

        cv2.putText(
            frame,
            f"ID:{det.tag_id}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        if det.heading is not None:
            cv2.putText(
                frame,
                f"H:{det.heading:.1f}",
                (x, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        if det.lateral is not None:
            cv2.putText(
                frame,
                f"L:{det.lateral:.3f}",
                (x, y + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

    y = 25

    if pose is not None:
        text = (
            f"POSE lm={pose['landmark_id']} "
            f"tag={pose['tag']} "
            f"pos={pose['position']} "
            f"pri={pose['priority']} "
            f"lat={pose['lateral']:.4f} "
            f"fwd={pose['forward']:.4f} "
            f"h={pose['heading']:.2f} "
            f"visible={pose['visible_tags']}"
        )

        cv2.putText(
            frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        y += 25

    if nav is not None:
        text = (
            f"NAV cur={nav['current']} "
            f"next={nav['next']} "
            f"des={nav['desired_heading']:.2f} "
            f"lat={nav['lateral_error']:.4f} "
            f"vel={nav['velocity']:.3f} "
            f"arr={nav.get('arrival_mode')} "
            f"reached={nav['reached_waypoint']} "
            f"final={nav['final_arrival']}"
        )

        cv2.putText(
            frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

    return frame


# ============================================================================
# TURN VALIDATION
# ============================================================================

def wait_for_landmark_pose(camera, detector, landmark_id, max_wait_s=5.0):
    deadline = time.monotonic() + max_wait_s

    while time.monotonic() < deadline:
        with latest_lock:
            pose = latest_pose

        if pose is not None and pose["landmark_id"] == landmark_id:
            return pose

        time.sleep(0.01)

    return None

# camera threading

def camera_worker(camera, detector):
    global latest_frame
    global latest_detections
    global latest_pose
    global latest_key
    global latest_frame_id
    global camera_running

    consecutive_read_failures = 0

    while camera_running:
        ok, frame = camera.read()

        if not ok or frame is None:
            consecutive_read_failures += 1
            if consecutive_read_failures == 1 or consecutive_read_failures % 30 == 0:
                print(f"USB_CAMERA_READ_FAILED count={consecutive_read_failures}")
            time.sleep(0.02)
            continue

        consecutive_read_failures = 0
        detections = detect_tags(detector, frame)
        pose = estimate_pose_from_tags(detections)

        display_frame = draw_detections(
            frame.copy(),
            detections,
            pose,
            None,
        )

        cv2.imshow("AGV Single File - Logitech USB", display_frame)
        key = cv2.waitKey(1) & 0xFF

        with latest_lock:
            latest_frame = frame
            latest_detections = detections
            latest_pose = pose
            latest_key = key
            latest_frame_id += 1

        time.sleep(0.002)

def helper_forward_offset_for_heading(position, heading_deg):
    heading_deg = normalize_angle(heading_deg)

    if abs(normalize_angle(heading_deg - 0.0)) < 1.0:
        if position in ("south", "south_west", "south_east"):
            return HELPER_SPACING_M
        if position in ("north", "north_west", "north_east"):
            return -HELPER_SPACING_M

    if abs(normalize_angle(heading_deg - 180.0)) < 1.0:
        if position in ("north", "north_west", "north_east"):
            return HELPER_SPACING_M
        if position in ("south", "south_west", "south_east"):
            return -HELPER_SPACING_M

    if abs(normalize_angle(heading_deg - 90.0)) < 1.0:
        if position in ("west", "north_west", "south_west"):
            return HELPER_SPACING_M
        if position in ("east", "north_east", "south_east"):
            return -HELPER_SPACING_M

    if abs(normalize_angle(heading_deg - -90.0)) < 1.0:
        if position in ("east", "north_east", "south_east"):
            return HELPER_SPACING_M
        if position in ("west", "north_west", "south_west"):
            return -HELPER_SPACING_M

    return 0.0
# path planning
def segment_type_for_path(path, path_index, goal_node):
    active_from = path[path_index]
    active_to = path[path_index + 1]
    active_heading = map_heading(active_from, active_to)

    if active_to == goal_node:
        return "goal"

    if path_index + 2 >= len(path):
        return "goal"
    
    next_to = path[path_index + 2]
    next_heading = map_heading(active_to, next_to)

    heading_change = normalize_angle(next_heading - active_heading)

    if abs(heading_change) > TURN_HEADING_THRESHOLD_DEG:
        return "turn"

    return "passthrough"

def end_velocity_for_segment_type(segment_type):
    if segment_type in ("turn", "goal"):
        return TURN_GOAL_END_VELOCITY_MPS

    return PASSTHROUGH_END_VELOCITY_MPS


def corrected_forward_for_heading(pose, active_heading):
    if pose is None:
        return None

    y = pose["forward"]

    if y is None:
        return None

    return y + helper_forward_offset_for_heading(
        pose["position"],
        active_heading,
    )

def is_center_zone_for_heading(pose, active_heading):
    if pose is None:
        return False

    allowed_positions = waypoint_positions_for_heading(active_heading)

    return pose["position"] in allowed_positions

# Heading handling rule:
# - active_heading remains the permanent map target.
# - pose["heading"] is sent separately as AprilTag ground truth.
# - ESP32 aligns the IMU estimate to the tag and computes:
#       normalizeAngle(map_heading - corrected_imu_heading)
# No map_heading + error target is created here.


# Main

def main():
    global camera_running
    global latest_key
    camera = start_camera()
    detector = create_detector()
    ser = open_serial()
    
    camera_thread = threading.Thread(
        target=camera_worker,
        args=(camera, detector),
        daemon=True,
    )

    camera_thread.start()

    started = False

    path = []
    path_index = 0

    mode = MODE_DOCK_WAIT

    dock_reference_pose = None
    dock_reference_saved = False
    last_tag1_forward = None

    last_arrival_forward = None
    handled_passthrough_nodes = set()
    leaving_ignore_landmark = None

    start_node = None
    goal_node = None

    active_from = None
    active_to = None
    active_heading = None
    current_robot_heading = DOCK_HEADING_DEG

    # Increment only when a new physical map segment begins.
    # Repeated helper/centre corrections keep the same ID, so the ESP32
    # does not restart its acceleration/deceleration profile.
    motion_segment_id = 0
    active_segment_end_velocity = DRIVE_VELOCITY_MPS

    last_sent_segment = None
    last_sent_final_arrival = None
    last_sent_pose_landmark = None
    last_sent_pose_tag = None
    last_sent_arrival_mode = None

    last_processed_frame_id = -1

    print("==========================================")
    print("AGV A* Logitech USB - Low Speed Test")
    print("Press 's' to calibrate/start.")
    print("Press 'q' to quit.")
    print("==========================================")
    
    try:
        while True:
            with latest_lock:
                pose = latest_pose
                key = latest_key
                frame_id = latest_frame_id
                latest_key = -1
            
            if frame_id == 0:
                time.sleep(0.01)
                continue

            if frame_id == last_processed_frame_id:
                time.sleep(0.002)
                continue

            last_processed_frame_id = frame_id

            for line in read_available_lines(ser):
                print("ESP32:", line)

            # Dock tag0 to tag 1 logic rule

            if mode == MODE_DOCK_TO_TAG1:
                active_heading = DOCK_HEADING_DEG

                # ------------------------------------------------------------
                # Rule 2:
                # Target tag 1 uses continuous latest frame.
                # Send APP with x_lateral and y_lateral.
                # ------------------------------------------------------------

                if pose is None:
                    continue

                if pose["landmark_id"] == FIRST_NODE:
                    x_error = lateral_pose_for_heading(pose, active_heading)

                    raw_y_error = pose["forward"]
                    y_error = raw_y_error

                    if y_error is not None:
                        y_error = y_error + helper_forward_offset_for_heading(
                            pose["position"],
                            active_heading,
                        )

                    raw_y_text = "None" if raw_y_error is None else f"{raw_y_error:.4f}"
                    corr_y_text = "None" if y_error is None else f"{y_error:.4f}"

                    print(
                        f"TAG1_APPROACH "
                        f"tag={pose['tag']} "
                        f"pos={pose['position']} "
                        f"x={x_error:.4f} "
                        f"raw_y={raw_y_text} "
                        f"corr_y={corr_y_text}"
                    )

                    reached_y_centre = False

                    if y_error is not None:
                        if last_tag1_forward is not None:
                            if last_tag1_forward > 0.0 and y_error <= 0.0:
                                reached_y_centre = True
                        else:
                            if y_error <= 0.0:
                                reached_y_centre = True

                        last_tag1_forward = y_error

                    if reached_y_centre:
                        send_velocity(
                            ser,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            motion_segment_id,
                            SEGMENT_DISTANCE_M,
                            0.0,
                        )

                        print("Tag1 corrected y-center crossed.")
                        print("Tag1 reached.")
                        print("Enter start and goal node.")

                        handled_passthrough_nodes = set()
                        last_arrival_forward = None
                        leaving_ignore_landmark = None
                        current_robot_heading = DOCK_HEADING_DEG

                        mode = MODE_WAIT_TASK
                        started = False

                        print("Ready for A* input.")
                        continue

                    if y_error is None:
                        print("TAG1_APPROACH y=None, sending nothing")
                        continue

                    tag_heading_error_for_log(
                        active_heading,
                        pose["heading"],
                    )

                    send_approach(
                        ser,
                        ARRIVAL_VELOCITY_MPS,
                        active_heading,
                        pose["heading"],
                        x_error,
                        y_error,
                    )

                    continue

                # ------------------------------------------------------------
                # Rule 1:
                # Later tag 0 frames are ignored.
                # Send nothing, but still update video.
                # ------------------------------------------------------------
                if pose["landmark_id"] == DOCK_NODE:
                    print(
                        f"LEAVING_DOCK_IGNORE_TAG0 "
                        f"tag={pose['tag']} "
                        f"pos={pose['position']} "
                        f"sending nothing"
                    )

                    continue

                # ------------------------------------------------------------
                # Any other unexpected landmark during dock movement.
                # According to your rule: send nothing.
                # ------------------------------------------------------------
                print(
                    f"UNEXPECTED_LANDMARK_DURING_DOCK "
                    f"lm={pose['landmark_id']} "
                    f"tag={pose['tag']} "
                    f"sending nothing"
                )

                continue
            
            if mode == MODE_WAIT_TASK:

                try:
                    start_node = int(input("Enter start node: "))
                    goal_node = int(input("Enter goal node: "))
                except ValueError:
                    print("Invalid input.Enter numbers only.")
                    continue
                print(f"Start Node: {start_node}")
                print(f"Goal Node: {goal_node}")
                with latest_lock:
                    pose = latest_pose

                if pose is None:
                    print("No valid current pose. Cannot start.")
                    continue

                if pose["landmark_id"] != start_node:
                    print(
                        f"Start node mismatch. "
                        f"Robot is seeing tag {pose['landmark_id']}, "
                        f"but you entered start {start_node}."
                    )
                    continue

                if landmark_by_id(start_node) is None:
                    print("Invalid start node.")
                    continue

                if landmark_by_id(goal_node) is None:
                    print("Invalid goal node.")
                    continue

                path = find_path(start_node, goal_node)

                if len(path) < 2:
                    print(f"No path from {start_node} to {goal_node}")
                    continue

                print(f"A* path: {' -> '.join(map(str, path))}")

                path_index = 0
                active_from = path[path_index]
                active_to = path[path_index + 1]
                active_heading = map_heading(active_from, active_to)

                print(
                    f"READY FOR PATH: {active_from}->{active_to} "
                    f"heading={active_heading:.1f}"
                )

                with latest_lock:
                    pose = latest_pose

                if pose is None or pose["landmark_id"] != start_node:
                    print("No valid start pose at entered start node. Cannot leave.")
                    continue

                first_heading_change = normalize_angle(active_heading - current_robot_heading)

                if abs(first_heading_change) > TURN_HEADING_THRESHOLD_DEG:
                    print(
                        f"START_TURN_NEEDED "
                        f"current_heading={current_robot_heading:.1f} "
                        f"required_heading={active_heading:.1f} "
                        f"turn={first_heading_change:.1f}"
                    )

                    ok = send_turn_wait_done(
                        ser,
                        active_heading,
                        max_wait_s=15.0,
                    )

                    if not ok:
                        print("Start turn failed. Aborting path.")
                        stop_robot(ser)
                        mode = MODE_WAIT_TASK
                        started = False
                        continue

                    current_robot_heading = active_heading

                    print("Start turn complete. Waiting for current start tag pose.")

                    pose_after_turn = wait_for_landmark_pose(
                        camera,
                        detector,
                        start_node,
                        max_wait_s=5.0,
                    )

                    if pose_after_turn is None:
                        print("No valid start tag pose after turn. Cannot leave.")
                        mode = MODE_WAIT_TASK
                        started = False
                        continue

                    pose = pose_after_turn
                else:
                    current_robot_heading = active_heading

                print(
                    f"PATH_START_DEPARTURE "
                    f"from={active_from} "
                    f"to={active_to} "
                    f"current_heading={current_robot_heading:.1f} "
                    f"tag_heading={pose['heading']:.2f} "
                    f"x={pose['lateral']:.4f}"
                )
                tag_heading_error_for_log(
                    active_heading,
                    pose["heading"],
                )
                x_cmd = lateral_pose_for_heading(
                    pose,
                    active_heading,
                )
                current_type = segment_type_for_path(
                    path,
                    path_index,
                    goal_node,
                )
                active_segment_end_velocity = (
                    end_velocity_for_segment_type(current_type)
                )
                motion_segment_id += 1

                print(
                    f"SEGMENT_PROFILE_START "
                    f"id={motion_segment_id} "
                    f"type={current_type} "
                    f"distance={SEGMENT_DISTANCE_M:.3f} "
                    f"cruise={DRIVE_VELOCITY_MPS:.3f} "
                    f"end={active_segment_end_velocity:.3f}"
                )

                send_velocity(
                    ser,
                    DRIVE_VELOCITY_MPS,
                    active_heading,
                    pose["heading"],
                    x_cmd,
                    motion_segment_id,
                    SEGMENT_DISTANCE_M,
                    active_segment_end_velocity,
                )

                leaving_ignore_landmark = start_node
                last_arrival_forward = None
                handled_passthrough_nodes = set()

                mode = MODE_RUN_PATH
                started = False
                continue
                
            if mode == MODE_RUN_PATH:
                if len(path) < 2:
                    print("No active path.")
                    mode = MODE_WAIT_TASK
                    continue

                if path_index >= len(path) - 1:
                    print("Path completed.")
                    mode = MODE_WAIT_TASK
                    continue

                active_from = path[path_index]
                active_to = path[path_index + 1]
                active_heading = map_heading(active_from, active_to)

                current_type = segment_type_for_path(
                    path,
                    path_index,
                    goal_node,
                )
                if pose is None:
                    continue

                if leaving_ignore_landmark is not None:
                    if pose["landmark_id"] == leaving_ignore_landmark:
                        print(
                            f"LEAVING_IGNORE "
                            f"lm={pose['landmark_id']} "
                            f"tag={pose['tag']} "
                            f"pos={pose['position']} "
                            f"sending nothing"
                        )
                        continue
                    else:
                        leaving_ignore_landmark = None

                # ------------------------------------------------------------
                # Ignore unrelated landmarks.
                # ------------------------------------------------------------
                if pose["landmark_id"] != active_to:
                    print(
                        f"PATH_IGNORE_OTHER "
                        f"expected={active_to} "
                        f"seen={pose['landmark_id']} "
                        f"tag={pose['tag']} "
                        f"sending nothing"
                    )
                    continue

                # ------------------------------------------------------------
                # PASS-THROUGH TAG RULE
                # No stop. No APP.
                # Use only center-zone frame once.
                # ------------------------------------------------------------
                if current_type == "passthrough":
                    if active_to in handled_passthrough_nodes:
                        print(
                            f"PASSTHROUGH_ALREADY_HANDLED "
                            f"lm={active_to} "
                            f"tag={pose['tag']} "
                            f"pos={pose['position']} "
                            f"sending nothing"
                        )
                        continue

                    tag_heading_error_for_log(
                        active_heading,
                        pose["heading"],
                    )

                    x_cmd = lateral_pose_for_heading(
                        pose,
                        active_heading,
                    )

                    # Entry/exit helpers are used for lateral pull-back.
                    # They do not advance the path index.
                    if not is_center_zone_for_heading(pose, active_heading):
                        print(
                            f"PASSTHROUGH_HELPER_CORRECT "
                            f"lm={active_to} "
                            f"tag={pose['tag']} "
                            f"pos={pose['position']} "
                            f"map_heading={active_heading:.2f} "
                            f"tag_heading={pose['heading']:.2f} "
                            f"x_raw={pose['raw_lateral']:.4f} "
                            f"x_corrected={x_cmd:.4f}"
                        )

                        send_velocity(
                            ser,
                            DRIVE_VELOCITY_MPS,
                            active_heading,
                            pose["heading"],
                            x_cmd,
                            motion_segment_id,
                            SEGMENT_DISTANCE_M,
                            active_segment_end_velocity,
                        )
                        continue

                    print(
                        f"PASSTHROUGH_CENTER "
                        f"lm={active_to} "
                        f"tag={pose['tag']} "
                        f"pos={pose['position']} "
                        f"map_heading={active_heading:.2f} "
                        f"tag_heading={pose['heading']:.2f} "
                        f"x_raw={pose['raw_lateral']:.4f} "
                        f"x_corrected={x_cmd:.4f}"
                    )

                    handled_passthrough_nodes.add(active_to)
                    path_index += 1

                    if path_index >= len(path) - 1:
                        print("Path ended after pass-through.")
                        mode = MODE_WAIT_TASK
                        continue

                    # The next segment has the same heading for a true
                    # pass-through node. Use that next map heading explicitly.
                    active_from = path[path_index]
                    active_to = path[path_index + 1]
                    active_heading = map_heading(active_from, active_to)
                    current_robot_heading = active_heading

                    tag_heading_error_for_log(
                        active_heading,
                        pose["heading"],
                    )

                    x_cmd = lateral_pose_for_heading(
                        pose,
                        active_heading,
                    )

                    next_segment_type = segment_type_for_path(
                        path,
                        path_index,
                        goal_node,
                    )
                    active_segment_end_velocity = (
                        end_velocity_for_segment_type(next_segment_type)
                    )
                    motion_segment_id += 1

                    print(
                        f"SEGMENT_PROFILE_START "
                        f"id={motion_segment_id} "
                        f"type={next_segment_type} "
                        f"distance={SEGMENT_DISTANCE_M:.3f} "
                        f"cruise={DRIVE_VELOCITY_MPS:.3f} "
                        f"end={active_segment_end_velocity:.3f}"
                    )

                    send_velocity(
                        ser,
                        DRIVE_VELOCITY_MPS,
                        active_heading,
                        pose["heading"],
                        x_cmd,
                        motion_segment_id,
                        SEGMENT_DISTANCE_M,
                        active_segment_end_velocity,
                    )

                    leaving_ignore_landmark = active_from
                    last_arrival_forward = None
                    continue

                # ------------------------------------------------------------
                # GOAL TAG OR TURNING TAG ARRIVAL RULE
                # Continuous APP until corrected y crosses center.
                # ------------------------------------------------------------
                if current_type in ("goal", "turn"):
                    x_error = lateral_pose_for_heading(pose, active_heading)
                    raw_y_error = pose["forward"]
                    y_error = corrected_forward_for_heading(pose, active_heading)

                    raw_y_text = "None" if raw_y_error is None else f"{raw_y_error:.4f}"
                    corr_y_text = "None" if y_error is None else f"{y_error:.4f}"

                    print(
                        f"{current_type.upper()}_APP "
                        f"lm={active_to} "
                        f"tag={pose['tag']} "
                        f"pos={pose['position']} "
                        f"x={x_error:.4f} "
                        f"raw_y={raw_y_text} "
                        f"corr_y={corr_y_text} "
                        f"heading={pose['heading']:.2f}"
                    )

                    reached_y_centre = False
                    center_zone = is_center_zone_for_heading(pose, active_heading)
                    
                    if y_error is None:
                        print(f"{current_type.upper()}_APP y=None, sending nothing")
                        continue

                    if center_zone:
                        if last_arrival_forward is not None:
                            if last_arrival_forward > 0.0 and y_error <= 0.0:
                                reached_y_centre = True

                        else:
                            if y_error <= 0.0:
                                reached_y_centre = True

                        last_arrival_forward = y_error
                    
                    else:
                        print(
                            f"{current_type.upper()}_HELPER_NO_STOP "
                            f"lm={active_to} "
                            f"tag={pose['tag']} "
                            f"pos={pose['position']} "
                            f"x={x_error:.4f} "
                            f"corr_y={y_error:.4f}"
                        )

                    tag_heading_error_for_log(
                        active_heading,
                        pose["heading"],
                    )

                    if not reached_y_centre:
                        send_approach(
                            ser,
                            ARRIVAL_VELOCITY_MPS,
                            active_heading,
                            pose["heading"],
                            x_error,
                            y_error,
                        )
                        continue

                    print(
                        f"{current_type.upper()}_CENTER_REACHED "
                        f"lm={active_to}"
                    )

                    send_velocity(
                        ser,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        motion_segment_id,
                        SEGMENT_DISTANCE_M,
                        0.0,
                    )

                    # --------------------------------------------------------
                    # FINAL GOAL
                    # --------------------------------------------------------
                    if current_type == "goal":
                        print(f"FINAL GOAL REACHED: {goal_node}")
                        current_robot_heading = active_heading
                        mode = MODE_WAIT_TASK
                        started = False
                        last_arrival_forward = None
                        leaving_ignore_landmark = None
                        continue

                    # --------------------------------------------------------
                    # TURNING TAG
                    # Stop, turn, then leave using first valid post-turn frame.
                    # --------------------------------------------------------
                    old_heading = active_heading

                    next_from = path[path_index + 1]
                    next_to = path[path_index + 2]
                    next_heading = map_heading(next_from, next_to)

                    heading_change = normalize_angle(next_heading - old_heading)

                    print(
                        f"TURN_TAG_REACHED "
                        f"lm={active_to} "
                        f"old_heading={old_heading:.1f} "
                        f"next_heading={next_heading:.1f} "
                        f"turn={heading_change:.1f}"
                    )

                    ok = send_turn_wait_done(
                        ser,
                        next_heading,
                        max_wait_s=15.0,
                    )

                    if not ok:
                        print("Turn failed. Aborting navigation.")
                        stop_robot(ser)
                        mode = MODE_WAIT_TASK
                        started = False
                        last_arrival_forward = None
                        continue
                    current_robot_heading = next_heading

                    print("Turn complete. Waiting for first valid post-turn frame.")

                    pose_after_turn = wait_for_landmark_pose(
                        camera,
                        detector,
                        next_from,
                        max_wait_s=5.0,
                    )

                    if pose_after_turn is None:
                        print("No valid post-turn tag. Not sending departure command.")
                        mode = MODE_WAIT_TASK
                        started = False
                        last_arrival_forward = None
                        continue

                    print(
                        f"POST_TURN_DEPARTURE_FRAME "
                        f"lm={pose_after_turn['landmark_id']} "
                        f"tag={pose_after_turn['tag']} "
                        f"pos={pose_after_turn['position']} "
                        f"heading={pose_after_turn['heading']:.2f} "
                        f"x={pose_after_turn['lateral']:.4f}"
                    )

                    path_index += 1
                    
                    tag_heading_error_for_log(
                        next_heading,
                        pose_after_turn["heading"],
                    )

                    post_turn_x = lateral_pose_for_heading(
                        pose_after_turn,
                        next_heading,
                    )

                    next_segment_type = segment_type_for_path(
                        path,
                        path_index,
                        goal_node,
                    )
                    active_segment_end_velocity = (
                        end_velocity_for_segment_type(next_segment_type)
                    )
                    motion_segment_id += 1

                    print(
                        f"SEGMENT_PROFILE_START "
                        f"id={motion_segment_id} "
                        f"type={next_segment_type} "
                        f"distance={SEGMENT_DISTANCE_M:.3f} "
                        f"cruise={DRIVE_VELOCITY_MPS:.3f} "
                        f"end={active_segment_end_velocity:.3f}"
                    )

                    send_velocity(
                        ser,
                        DRIVE_VELOCITY_MPS,
                        next_heading,
                        pose_after_turn["heading"],
                        post_turn_x,
                        motion_segment_id,
                        SEGMENT_DISTANCE_M,
                        active_segment_end_velocity,
                    )

                    leaving_ignore_landmark = next_from
                    last_arrival_forward = None
                    continue

            if key == ord("q"):
                break

            if key == ord("s") and mode == MODE_DOCK_WAIT:
                if pose is None:
                    print("No valid localization. Cannot start.")
                    continue

                if pose["landmark_id"] != DOCK_NODE:
                    print(
                        f"Robot must start at dock tag 0. "
                        f"Detected={pose['landmark_id']}"
                    )
                    continue

                # Save only the first valid dock frame.
                # Later tag 0 helper frames will not update correction.
                if not dock_reference_saved:
                    dock_reference_pose = pose
                    dock_reference_saved = True

                    print(
                        f"DOCK_REFERENCE_SAVED "
                        f"tag={pose['tag']} "
                        f"pos={pose['position']} "
                        f"x={pose['lateral']:.4f} "
                        f"y={pose['forward']:.4f}"
                    )

                print("Dock tag 0 detected.")
                print("Calibrating IMU...")

                if not calibrate_imu(ser):
                    print("IMU calibration failed.")
                    continue

                print("Zeroing heading...")

                if not zero_heading(ser):
                    print("Failed to zero heading.")
                    continue
                print("Enabling motors...")

                if not enable_motors(ser):
                    print("Failed to enable motors.")
                    continue

                print("Moving from dock tag 0 to tag 1.")
                
                dock_x = lateral_pose_for_heading(
                    dock_reference_pose,
                    DOCK_HEADING_DEG,
                )

                tag_heading_error_for_log(
                    DOCK_HEADING_DEG,
                    dock_reference_pose["heading"],
                )

                print(
                    f"DOCK_START_COMMAND "
                    f"map_heading={DOCK_HEADING_DEG:.2f} "
                    f"tag_heading={dock_reference_pose['heading']:.2f} "
                    f"x_raw={dock_reference_pose['raw_lateral']:.4f} "
                    f"x_corrected={dock_x:.4f}"
                )

                motion_segment_id += 1
                active_segment_end_velocity = ARRIVAL_VELOCITY_MPS

                print(
                    f"SEGMENT_PROFILE_START "
                    f"id={motion_segment_id} "
                    f"type=dock_goal "
                    f"distance={SEGMENT_DISTANCE_M:.3f} "
                    f"cruise={DRIVE_VELOCITY_MPS:.3f} "
                    f"end={active_segment_end_velocity:.3f}"
                )

                send_velocity(
                    ser,
                    DRIVE_VELOCITY_MPS,
                    DOCK_HEADING_DEG,
                    dock_reference_pose["heading"],
                    dock_x,
                    motion_segment_id,
                    SEGMENT_DISTANCE_M,
                    active_segment_end_velocity,
                )
                last_tag1_forward = None

                mode = MODE_DOCK_TO_TAG1
                started = False

                continue

    finally:
        camera_running = False
        try:
            camera_thread.join(timeout=1.0)
        except Exception as exc:
            print("Camera thread warning:", exc)
        try:
            stop_robot(ser)
            disable_motors(ser)
            ser.close()
        except Exception as exc:
            print("Serial cleanup warning:", exc)

        try:
            camera.release()
        except Exception as exc:
            print("Camera cleanup warning:", exc)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()