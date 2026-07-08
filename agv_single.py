import math
import time
import json
import threading

import cv2
import serial
from picamera2 import Picamera2
from pupil_apriltags import Detector


#Config Files

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

DRIVE_VELOCITY_MPS = 0.050
ARRIVAL_VELOCITY_MPS = 0.025

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

def helper_lateral_offset(position):
    if position in ("east", "north_east", "south_east"):
        return HELPER_SPACING_M

    if position in ("west", "north_west", "south_west"):
        return -HELPER_SPACING_M

    return 0.0

def compute_desired_heading_from_tag(pose, active_heading):
    tag_heading_error = normalize_angle(pose["heading"] - active_heading)

    if abs(tag_heading_error) > MAX_ACCEPTED_HEADING_DEG:
        tag_heading_error =0.0

    tag_heading_correction = -TAG_HEADING_GAIN * tag_heading_error
    tag_heading_correction = clamp(tag_heading_correction, -MAX_TAG_HEADING_CORRECTION_DEG, MAX_TAG_HEADING_CORRECTION_DEG)

    return normalize_angle(active_heading + tag_heading_correction)
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


# ============================================================================
# APRILTAG POSE HELPERS
# ============================================================================

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


# ============================================================================
# MAP AND A* PATH
# ============================================================================

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


# ============================================================================
# WAYPOINT / ARRIVAL GATE
# ============================================================================

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


# ============================================================================
# CAMERA AND DETECTOR
# ============================================================================

def start_camera():
    camera = Picamera2()

    camera.set_controls(
        {
            "AwbMode": False,
            "ExposureTime": 5000,
            "AnalogueGain": 1.0,
            "AwbEnable": False,
            "ColourGains": (1.7, 1.7),
        }
    )

    config = camera.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
    )

    camera.configure(config)
    camera.start()

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
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

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
        offset = helper_lateral_offset(position)
        corrected_lateral = det.lateral + offset

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

    # ------------------------------------------------------------------------
    # ARRIVAL MODE
    # When entering target landmark but not yet at center row/column:
    # do not use lateral correction, do not use tag heading correction.
    # Just move slowly straight until center row/column is reached.
    # ------------------------------------------------------------------------
    if pose["landmark_id"] == active_to and not reached_waypoint:
        return {
            "current": pose["landmark_id"],
            "next": active_to,
            "desired_heading": active_heading,
            "lateral_error": 0.0,
            "velocity": ARRIVAL_VELOCITY_MPS,
            "reached_waypoint": False,
            "final_arrival": False,
            "arrival_mode": True,
        }

    # ------------------------------------------------------------------------
    # FINAL GOAL STOP
    # ------------------------------------------------------------------------
    if reached_waypoint and active_to == goal_node:
        return {
            "current": pose["landmark_id"],
            "next": None,
            "desired_heading": 0.0,
            "lateral_error": 0.0,
            "velocity": 0.0,
            "reached_waypoint": True,
            "final_arrival": True,
            "arrival_mode": False,
        }

    # ------------------------------------------------------------------------
    # NORMAL TRAVEL MODE
    # Heading error is relative to the active segment heading.
    # This fixes the 90-degree helper tag rejection problem.
    # ------------------------------------------------------------------------
    tag_heading_error = normalize_angle(pose["heading"] - active_heading)

    if abs(tag_heading_error) > MAX_ACCEPTED_HEADING_DEG:
        tag_heading_error = 0.0

    tag_heading_correction = -TAG_HEADING_GAIN * tag_heading_error

    tag_heading_correction = clamp(
        tag_heading_correction,
        -MAX_TAG_HEADING_CORRECTION_DEG,
        MAX_TAG_HEADING_CORRECTION_DEG,
    )

    desired_heading = normalize_angle(active_heading + tag_heading_correction)

    return {
        "current": pose["landmark_id"],
        "next": active_to,
        "desired_heading": desired_heading,
        "lateral_error": pose["lateral"],
        "velocity": velocity_mps,
        "reached_waypoint": reached_waypoint,
        "final_arrival": False,
        "arrival_mode": False,
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


def send_velocity(ser, velocity_mps, desired_heading_deg, lateral_error_m):
    command = f"VEL {velocity_mps:.3f} {desired_heading_deg:.2f} {lateral_error_m:.4f}"

    print("TX:", command)

    return send_command_wait_ack(ser, command, max_wait_s=1.0)

# send approach for lateral x and y correction when reaching tag1, or goal node

def send_approach(ser, velocity_mps, desired_heading_deg, x_lateral_m, y_lateral_m):
    command = (f"APP {velocity_mps:.3f}"
              f" {desired_heading_deg:.2f}"
              f" {x_lateral_m:.4f}"
              f" {y_lateral_m:.4f}")
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

    while camera_running:
        frame = camera.capture_array()
        detections = detect_tags(detector, frame)
        pose = estimate_pose_from_tags(detections)

        display_frame = draw_detections(
            frame.copy(),
            detections,
            pose,
            None,
        )

        cv2.imshow(
            "AGV Single File",
            cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR),
        )

        key = cv2.waitKey(1) & 0xFF

        with latest_lock:
            latest_frame = frame
            latest_detections = detections
            latest_pose = pose
            latest_key = key
            latest_frame_id += 1

        time.sleep(0.002)
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

    start_node = None
    goal_node = None

    active_from = None
    active_to = None
    active_heading = None

    last_sent_segment = None
    last_sent_final_arrival = None
    last_sent_pose_landmark = None
    last_sent_pose_tag = None
    last_sent_arrival_mode = None

    last_processed_frame_id = -1

    print("==========================================")
    print("AGV A* Single File Controller")
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



            nav = None

            if started:
                nav = compute_navigation_for_segment(
                    pose,
                    active_to,
                    active_heading,
                    goal_node,
                    DRIVE_VELOCITY_MPS,
                )

            for line in read_available_lines(ser):
                print("ESP32:", line)

            # Dock tag0 to tag 1 logic rule

            if mode == MODE_DOCK_TO_TAG1:
                active_heading = DOCK_HEADING_DEG

                # ------------------------------------------------------------
                # No tag visible between tag 0 and tag 1 is normal.
                # According to your rule: send nothing.
                # But still update camera window before continue.
                # ------------------------------------------------------------
                if pose is None:
                    print("No_TAG_GAP 0->1, sending nothing.")
                    continue

                # ------------------------------------------------------------
                # Rule 2:
                # Target tag 1 uses continuous latest frame.
                # Send APP with x_lateral and y_lateral.
                # ------------------------------------------------------------
                if pose["landmark_id"] == FIRST_NODE:
                    x_error = pose["lateral"]

                    raw_y_error = pose["forward"]
                    y_error = raw_y_error

                    if y_error is not None:
                        y_error = y_error + helper_forward_offset_for_heading(
                            pose["position"],
                            active_heading,
                        )

                    print(
                        f"TAG1_APPROACH "
                        f"tag={pose['tag']} "
                        f"pos={pose['position']} "
                        f"x={x_error:.4f} "
                        f"raw_y={raw_y_error:.4f} "
                        f"corr_y={y_error:.4f}"
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
                        send_velocity(ser, 0.0, 0.0, 0.0)

                        print("Tag1 corrected y-center crossed.")
                        print("Tag1 reached.")
                        print("Enter start and goal node.")

                        mode = MODE_WAIT_TASK
                        started = False
                        continue

                    send_approach(
                        ser,
                        ARRIVAL_VELOCITY_MPS,
                        active_heading,
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
                if start_node != FIRST_NODE:
                    print("Robot is physically at tag 1, so start node should be 1.")
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

                # Do not run the full path yet.
                # We stop here for this milestone.
                mode = MODE_RUN_PATH
                continue
                

            if started and nav is not None:
                current_segment = (active_from, active_to)

                if nav["final_arrival"]:
                    print(f"FINAL GOAL REACHED: {goal_node}")

                    send_velocity(
                        ser,
                        0.0,
                        0.0,
                        0.0,
                    )

                    started = False
                    nav = None

                else:
                    if nav["reached_waypoint"] and active_to != goal_node:
                        print(f"PASSED WAYPOINT {active_to}")

                        old_heading = active_heading

                        path_index += 1

                        active_from = path[path_index]
                        active_to = path[path_index + 1]
                        active_heading = map_heading(active_from, active_to)

                        heading_change = normalize_angle(active_heading - old_heading)

                        print(
                            f"NEXT SEGMENT {active_from}->{active_to} "
                            f"heading={active_heading:.1f} "
                            f"turn={heading_change:.1f}"
                        )

                        if abs(heading_change) > TURN_HEADING_THRESHOLD_DEG:
                            print("TURN NEEDED. Stopping before pivot turn.")

                            send_velocity(
                                ser,
                                0.0,
                                0.0,
                                0.0,
                            )

                            ok = send_turn_wait_done(
                                ser,
                                active_heading,
                                max_wait_s=15.0,
                            )

                            if not ok:
                                print("Turn failed. Aborting navigation.")
                                stop_robot(ser)
                                started = False
                                nav = None
                                continue

                            print("Turn complete. Checking for valid helper/center tag.")

                            pose_after_turn = wait_for_landmark_pose(
                                camera,
                                detector,
                                active_from,
                                max_wait_s=5.0,
                            )

                            if pose_after_turn is None:
                                print(
                                    "No valid tag after turn. "
                                    "Continuing with last known pose."
                                )
                            else:
                                pose = pose_after_turn

                                print(
                                    f"Valid tag after turn: "
                                    f"lm={pose['landmark_id']} "
                                    f"tag={pose['tag']} "
                                    f"pos={pose['position']} "
                                    f"lat={pose['lateral']:.4f} "
                                    f"h={pose['heading']:.2f}"
                                )

                        nav = compute_navigation_for_segment(
                            pose,
                            active_to,
                            active_heading,
                            goal_node,
                            DRIVE_VELOCITY_MPS,
                        )

                        last_sent_segment = None
                        last_sent_final_arrival = None
                        last_sent_pose_landmark = None
                        last_sent_pose_tag = None
                        last_sent_arrival_mode = None

                        current_segment = (active_from, active_to)

                    if nav is not None:
                        should_send = (
                            current_segment != last_sent_segment
                            or nav["final_arrival"] != last_sent_final_arrival
                            or pose["landmark_id"] != last_sent_pose_landmark
                            or pose["tag"] != last_sent_pose_tag
                            or nav.get("arrival_mode") != last_sent_arrival_mode
                            or nav["reached_waypoint"]
                        )

                        if should_send:
                            print(
                                f"SEND VEL {nav['velocity']:.3f} "
                                f"{nav['desired_heading']:.2f} "
                                f"{nav['lateral_error']:.4f} "
                                f"tag={pose['tag']} "
                                f"pos={pose['position']} "
                                f"priority={pose['priority']} "
                                f"raw_lat={pose['raw_lateral']:.4f} "
                                f"offset={pose['center_lateral_offset']:.4f} "
                                f"corr_lat={pose['lateral']:.4f} "
                                f"tag_h={pose['heading']:.2f} "
                                f"segment={active_from}->{active_to} "
                                f"arrival_mode={nav.get('arrival_mode')} "
                                f"reached={nav['reached_waypoint']} "
                                f"final={nav['final_arrival']}"
                            )

                            send_velocity(
                                ser,
                                nav["velocity"],
                                nav["desired_heading"],
                                nav["lateral_error"],
                            )

                            last_sent_segment = current_segment
                            last_sent_final_arrival = nav["final_arrival"]
                            last_sent_pose_landmark = pose["landmark_id"]
                            last_sent_pose_tag = pose["tag"]
                            last_sent_arrival_mode = nav.get("arrival_mode")

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
                
                dock_desired_heading = dock_reference_pose["heading"]
                print(
                    f"DOCK_START_COMMAND "
                    f"desired_heading={dock_desired_heading:.2f} "
                    f"x_lateral={dock_reference_pose['lateral']:.4f} "
                    f"tag_heading={dock_reference_pose['heading']:.2f}"
                )
                send_velocity(
                    ser,
                    DRIVE_VELOCITY_MPS,
                    dock_desired_heading,
                    dock_reference_pose["lateral"],
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
            camera.stop()
        except Exception as exc:
            print("Camera cleanup warning:", exc)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()