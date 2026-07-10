import math
import time
import json
import threading

import cv2
import serial
from picamera2 import Picamera2
from pupil_apriltags import Detector


# ============================================================
# CONFIG
# ============================================================

TAG_HEADING_OFFSET_GAIN = 1.0
MAX_TAG_HEADING_OFFSET_DEG = 1.0

DOCK_NODE = 0
FIRST_NODE = 1

DOCK_NODE_HEADING_DEG = 0.0

DRIVE_VELOCITY_MPS = 0.050
ARRIVAL_VELOCITY_MPS = 0.025

HELPER_SPACING_M = 0.015

MAX_ACCEPTED_LATERAL_M = 0.100
TURN_HEADING_THRESHOLD_DEG = 1.0

MODE_DOCK_WAIT = "DOCK_WAIT"
MODE_DOCK_TO_TAG1 = "DOCK_TO_TAG1"
MODE_WAIT_TASK = "WAIT_TASK"
MODE_RUN_STRAIGHT = "RUN_STRAIGHT"

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


# ============================================================
# GLOBAL CAMERA STATE
# ============================================================

latest_frame = None
latest_detections = []
latest_pose = None
latest_key = -1
latest_frame_id = 0

latest_lock = threading.Lock()
camera_running = True


# ============================================================
# BASIC HELPERS
# ============================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_angle(angle_deg):
    while angle_deg > 180:
        angle_deg -= 360
    while angle_deg <= -180:
        angle_deg += 360
    return angle_deg


# ============================================================
# MAP HELPERS
# ============================================================

def load_map(filename):
    with open(filename, "r") as f:
        return json.load(f)


MAP_DATA = load_map(MAP_FILE)


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


def build_straight_0_180_path(start_node, goal_node):
    """
    Only same-column straight test.

    row increasing  -> heading 0 deg
    row decreasing  -> heading 180 deg

    Examples:
      1  -> 5 -> 9 -> 13   heading 0
      13 -> 9 -> 5 -> 1    heading 180
    """
    start = landmark_by_id(start_node)
    goal = landmark_by_id(goal_node)

    if start is None or goal is None:
        return [], None

    if start["column"] != goal["column"]:
        return [], None

    if goal["row"] > start["row"]:
        heading = 0.0
        step = 1
    elif goal["row"] < start["row"]:
        heading = 180.0
        step = -1
    else:
        return [], None

    path = []
    row = start["row"]
    col = start["column"]

    while True:
        found = None

        for landmark in MAP_DATA["landmarks"]:
            if landmark["row"] == row and landmark["column"] == col:
                found = int(landmark["id"])
                break

        if found is None:
            return [], None

        path.append(found)

        if row == goal["row"]:
            break

        row += step

    return path, heading


# ============================================================
# APRILTAG POSE
# ============================================================

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

def helper_lateral_offset_for_heading(position, heading_deg):
    h = normalize_angle(heading_deg)

    # 0 degree movement
    if abs(normalize_angle(h - 0.0)) < 5.0:
        if position in ("east", "north_east", "south_east"):
            return HELPER_SPACING_M
        if position in ("west", "north_west", "south_west"):
            return -HELPER_SPACING_M
        return 0.0

    # 180 degree movement
    if abs(abs(h) - 180.0) < 5.0:
        if position in ("east", "north_east", "south_east"):
            return -HELPER_SPACING_M
        if position in ("west", "north_west", "south_west"):
            return HELPER_SPACING_M
        return 0.0

    return 0.0

def lateral_pose_for_heading(pose, active_heading):
    if pose is None:
        return 0.0

    raw_x = pose["raw_lateral"]

    offset = helper_lateral_offset_for_heading(
        pose["position"],
        active_heading,
    )

    return raw_x + offset

def helper_lateral_offset(position):
    if position in ("east", "north_east", "south_east"):
        return HELPER_SPACING_M

    if position in ("west", "north_west", "south_west"):
        return -HELPER_SPACING_M

    return 0.0


def compute_heading(detection):
    if detection.pose_R is None:
        return None

    r = detection.pose_R
    return normalize_angle(math.degrees(math.atan2(r[1, 0], r[0, 0])))


def compute_lateral(detection):
    if detection.pose_t is None:
        return None

    return float(detection.pose_t[0][0])


def compute_forward(detection):
    if detection.pose_t is None:
        return None

    return float(detection.pose_t[1][0])


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


# ============================================================
# HEADING / X / Y CORRECTION
# ============================================================

def helper_forward_offset_for_heading(position, heading_deg):
    heading_deg = normalize_angle(heading_deg)

    # Heading 0: moving row increasing.
    if abs(normalize_angle(heading_deg - 0.0)) < 1.0:
        if position in ("south", "south_west", "south_east"):
            return HELPER_SPACING_M
        if position in ("north", "north_west", "north_east"):
            return -HELPER_SPACING_M

    # Heading 180: moving row decreasing.
    if abs(abs(heading_deg) - 180.0) < 1.0:
        if position in ("north", "north_west", "north_east"):
            return HELPER_SPACING_M
        if position in ("south", "south_west", "south_east"):
            return -HELPER_SPACING_M

    return 0.0


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


def waypoint_positions_for_heading(heading_deg):
    heading_deg = normalize_angle(heading_deg)

    if abs(normalize_angle(heading_deg - 0.0)) < 1.0:
        return {"west", "center", "east"}

    if abs(abs(heading_deg) - 180.0) < 1.0:
        return {"west", "center", "east"}

    return {"center"}


def is_center_zone_for_heading(pose, active_heading):
    if pose is None:
        return False

    allowed_positions = waypoint_positions_for_heading(active_heading)
    return pose["position"] in allowed_positions


def corrected_map_heading_from_tag(map_heading, tag_heading):
    """
    Rule:
      tag_error = map_heading - tag_heading
      corrected_heading = map_heading + clamp(tag_error)
    """
    if tag_heading is None:
        return normalize_angle(map_heading)

    raw_error = normalize_angle(map_heading - tag_heading)

    tag_error = clamp(
        raw_error,
        -MAX_TAG_HEADING_OFFSET_DEG,
        MAX_TAG_HEADING_OFFSET_DEG,
    )

    tag_offset = TAG_HEADING_OFFSET_GAIN * tag_error
    corrected_heading = normalize_angle(map_heading + tag_offset)

    print(
        f"TAG_HEADING_VERIFY "
        f"map={map_heading:.2f} "
        f"tag={tag_heading:.2f} "
        f"raw_error={raw_error:.2f} "
        f"tag_error={tag_error:.2f} "
        f"tag_offset={tag_offset:.2f} "
        f"corrected={corrected_heading:.2f}"
    )

    return corrected_heading


def lateral_command_for_heading(x_corrected, active_heading):
    h = normalize_angle(active_heading)

    # 180 degree steering command sign is opposite
    if abs(abs(h) - 180.0) < 5.0:
        return -x_corrected

    return x_corrected


# ============================================================
# CAMERA
# ============================================================

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


def draw_detections(frame, detections, pose=None):
    height, width = frame.shape[:2]

    image_center_x = width // 2
    image_center_y = height // 2

    cv2.line(frame, (0, image_center_y), (width, image_center_y), (128, 128, 128), 1)
    cv2.line(frame, (image_center_x, 0), (image_center_x, height), (128, 128, 128), 1)
    cv2.circle(frame, (image_center_x, image_center_y), 4, (128, 128, 128), -1)

    selected_tag = pose["tag"] if pose is not None else None

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
        cv2.line(frame, (image_center_x, image_center_y), tag_center, color, 1)

        x = int(corners[0][0])
        y = int(corners[0][1])

        cv2.putText(frame, f"ID:{det.tag_id}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if det.heading is not None:
            cv2.putText(frame, f"H:{det.heading:.1f}", (x, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if det.lateral is not None:
            cv2.putText(frame, f"L:{det.lateral:.3f}", (x, y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    if pose is not None:
        text = (
            f"POSE lm={pose['landmark_id']} tag={pose['tag']} "
            f"pos={pose['position']} lat={pose['lateral']:.4f} "
            f"fwd={pose['forward']:.4f} h={pose['heading']:.2f}"
        )

        cv2.putText(frame, text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    return frame


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

        display_frame = draw_detections(frame.copy(), detections, pose)

        cv2.imshow(
            "AGV Straight 0/180 Test",
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


def get_latest_pose():
    with latest_lock:
        return latest_pose


def wait_for_landmark_pose(landmark_id, max_wait_s=5.0):
    deadline = time.monotonic() + max_wait_s

    while time.monotonic() < deadline:
        pose = get_latest_pose()

        if pose is not None and pose["landmark_id"] == landmark_id:
            return pose

        time.sleep(0.01)

    return None


# ============================================================
# SERIAL
# ============================================================

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


def send_approach(ser, velocity_mps, desired_heading_deg, x_lateral_m, y_lateral_m):
    command = (
        f"APP {velocity_mps:.3f}"
        f" {desired_heading_deg:.2f}"
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


# ============================================================
# MAIN
# ============================================================

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

    mode = MODE_DOCK_WAIT

    dock_reference_pose = None
    dock_reference_saved = False

    current_robot_heading = DOCK_NODE_HEADING_DEG

    path = []
    path_index = 0
    test_heading = None
    start_node = None
    goal_node = None

    leaving_ignore_landmark = None
    last_tag1_forward = None
    last_goal_forward = None

    last_processed_frame_id = -1

    print("==========================================")
    print("AGV Straight 0/180 Test Controller")
    print("Press 's' at dock tag 0 to calibrate/start.")
    print("Only same-column straight paths are allowed.")
    print("Examples: 1->13 heading 0, 13->1 heading 180")
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

            if key == ord("q"):
                break

            # ------------------------------------------------------------
            # START FROM DOCK
            # ------------------------------------------------------------
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

                if not dock_reference_saved:
                    dock_reference_pose = pose
                    dock_reference_saved = True

                    print(
                        f"DOCK_REFERENCE_SAVED "
                        f"tag={pose['tag']} "
                        f"pos={pose['position']} "
                        f"x={pose['lateral']:.4f} "
                        f"y={pose['forward']:.4f} "
                        f"heading={pose['heading']:.2f}"
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

                dock_corrected_heading = corrected_map_heading_from_tag(
                    DOCK_NODE_HEADING_DEG,
                    dock_reference_pose["heading"],
                )

                print(
                    f"DOCK_START_COMMAND "
                    f"map_heading={DOCK_NODE_HEADING_DEG:.2f} "
                    f"desired_heading={dock_corrected_heading:.2f} "
                    f"x_lateral={dock_reference_pose['lateral']:.4f} "
                    f"tag_heading={dock_reference_pose['heading']:.2f}"
                )

                send_velocity(
                    ser,
                    DRIVE_VELOCITY_MPS,
                    dock_corrected_heading,
                    dock_reference_pose["lateral"],
                )

                last_tag1_forward = None
                mode = MODE_DOCK_TO_TAG1
                continue

            # ------------------------------------------------------------
            # DOCK -> TAG 1
            # ------------------------------------------------------------
            if mode == MODE_DOCK_TO_TAG1:
                active_heading = DOCK_NODE_HEADING_DEG

                if pose is None:
                    continue

                if pose["landmark_id"] == DOCK_NODE:
                    print(
                        f"LEAVING_DOCK_IGNORE_TAG0 "
                        f"tag={pose['tag']} "
                        f"pos={pose['position']} "
                        f"sending nothing"
                    )
                    continue

                if pose["landmark_id"] != FIRST_NODE:
                    print(
                        f"UNEXPECTED_LANDMARK_DURING_DOCK "
                        f"lm={pose['landmark_id']} "
                        f"tag={pose['tag']} "
                        f"sending nothing"
                    )
                    continue

                x_error = pose["lateral"]
                y_error = corrected_forward_for_heading(pose, active_heading)

                raw_y_text = "None" if pose["forward"] is None else f"{pose['forward']:.4f}"
                corr_y_text = "None" if y_error is None else f"{y_error:.4f}"

                print(
                    f"TAG1_APPROACH "
                    f"tag={pose['tag']} "
                    f"pos={pose['position']} "
                    f"x={x_error:.4f} "
                    f"raw_y={raw_y_text} "
                    f"corr_y={corr_y_text} "
                    f"heading={pose['heading']:.2f}"
                )

                if y_error is None:
                    print("TAG1_APPROACH y=None, sending nothing")
                    continue

                reached_y_centre = False
                center_zone = is_center_zone_for_heading(pose, active_heading)

                if center_zone:
                    if last_tag1_forward is not None:
                        if last_tag1_forward > 0.0 and y_error <= 0.0:
                            reached_y_centre = True
                    else:
                        if y_error <= 0.0:
                            reached_y_centre = True

                    last_tag1_forward = y_error
                else:
                    print(
                        f"TAG1_HELPER_NO_STOP "
                        f"tag={pose['tag']} "
                        f"pos={pose['position']} "
                        f"corr_y={y_error:.4f}"
                    )

                if reached_y_centre:
                    send_velocity(ser, 0.0, 0.0, 0.0)

                    print("Tag1 corrected y-center crossed.")
                    print("Tag1 reached.")
                    print("Ready for straight 0/180 input.")

                    current_robot_heading = DOCK_NODE_HEADING_DEG
                    leaving_ignore_landmark = None
                    last_goal_forward = None
                    mode = MODE_WAIT_TASK
                    continue

                corrected_heading = corrected_map_heading_from_tag(
                    active_heading,
                    pose["heading"],
                )

                send_approach(
                    ser,
                    ARRIVAL_VELOCITY_MPS,
                    corrected_heading,
                    x_error,
                    y_error,
                )
                continue

            # ------------------------------------------------------------
            # ASK START / GOAL
            # ------------------------------------------------------------
            if mode == MODE_WAIT_TASK:
                try:
                    start_node = int(input("Enter start node: "))
                    goal_node = int(input("Enter goal node: "))
                except ValueError:
                    print("Invalid input. Enter numbers only.")
                    continue

                print(f"Start Node: {start_node}")
                print(f"Goal Node: {goal_node}")

                pose = get_latest_pose()

                if pose is None:
                    print("No valid current pose. Cannot start.")
                    continue

                if pose["landmark_id"] != start_node:
                    print(
                        f"Start node mismatch. "
                        f"Robot is seeing {pose['landmark_id']}, "
                        f"but you entered {start_node}."
                    )
                    continue

                path, test_heading = build_straight_0_180_path(
                    start_node,
                    goal_node,
                )

                if len(path) < 2:
                    print("Only same-column straight 0/180 paths are allowed.")
                    continue

                print(
                    f"STRAIGHT_PATH: {' -> '.join(map(str, path))} "
                    f"heading={test_heading:.1f}"
                )

                turn_amount = normalize_angle(test_heading - current_robot_heading)

                if abs(turn_amount) > TURN_HEADING_THRESHOLD_DEG:
                    print(
                        f"START_TURN_NEEDED "
                        f"current_heading={current_robot_heading:.1f} "
                        f"required_heading={test_heading:.1f} "
                        f"turn={turn_amount:.1f}"
                    )

                    ok = send_turn_wait_done(
                        ser,
                        test_heading,
                        max_wait_s=15.0,
                    )

                    if not ok:
                        print("Start turn failed. Aborting.")
                        stop_robot(ser)
                        mode = MODE_WAIT_TASK
                        continue

                    current_robot_heading = test_heading

                    print("Start turn complete. Waiting for start tag pose.")
                    pose_after_turn = wait_for_landmark_pose(
                        start_node,
                        max_wait_s=5.0,
                    )

                    if pose_after_turn is None:
                        print("No valid start pose after turn. Cannot leave.")
                        mode = MODE_WAIT_TASK
                        continue

                    pose = pose_after_turn
                else:
                    current_robot_heading = test_heading

                path_index = 0
                leaving_ignore_landmark = start_node
                last_goal_forward = None

                corrected_heading = corrected_map_heading_from_tag(
                    test_heading,
                    pose["heading"],
                )

                x_corrected = lateral_pose_for_heading(pose, test_heading)
                x_cmd = lateral_command_for_heading(x_corrected, test_heading)

                print(
                    f"STRAIGHT_START_DEPARTURE "
                    f"from={path[0]} "
                    f"to={path[1]} "
                    f"heading={test_heading:.1f} "
                    f"tag_heading={pose['heading']:.2f} "
                    f"x_raw={pose['raw_lateral']:.4f} "
                    f"x_corrected={x_corrected:.4f} "
                    f"x_cmd={x_cmd:.4f}"
                )

                send_velocity(
                    ser,
                    DRIVE_VELOCITY_MPS,
                    corrected_heading,
                    x_cmd,
                )

                mode = MODE_RUN_STRAIGHT
                continue

            # ------------------------------------------------------------
            # RUN STRAIGHT PATH
            # ------------------------------------------------------------
            if mode == MODE_RUN_STRAIGHT:
                if len(path) < 2:
                    print("No active straight path.")
                    mode = MODE_WAIT_TASK
                    continue

                if path_index >= len(path) - 1:
                    print("Path completed.")
                    mode = MODE_WAIT_TASK
                    continue

                active_from = path[path_index]
                active_to = path[path_index + 1]
                active_heading = test_heading

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

                if pose["landmark_id"] != active_to:
                    print(
                        f"STRAIGHT_IGNORE_OTHER "
                        f"expected={active_to} "
                        f"seen={pose['landmark_id']} "
                        f"tag={pose['tag']} "
                        f"sending nothing"
                    )
                    continue

                is_goal = active_to == goal_node

                # --------------------------------------------------------
                # PASS THROUGH
                # --------------------------------------------------------
                if not is_goal:
                    if not is_center_zone_for_heading(pose, active_heading):
                        print(
                            f"PASSTHROUGH_ENTRY_EXIT_IGNORE "
                            f"lm={active_to} "
                            f"tag={pose['tag']} "
                            f"pos={pose['position']} "
                            f"sending nothing"
                        )
                        continue

                    corrected_heading = corrected_map_heading_from_tag(
                        active_heading,
                        pose["heading"],
                    )

                    x_corrected = lateral_pose_for_heading(pose, active_heading)
                    x_cmd = lateral_command_for_heading(x_corrected, active_heading)

                    print(
                        f"PASSTHROUGH_CENTER "
                        f"lm={active_to} "
                        f"tag={pose['tag']} "
                        f"pos={pose['position']} "
                        f"heading={pose['heading']:.2f} "
                        f"x_raw={pose['lateral']:.4f} "
                        f"x_cmd={x_cmd:.4f}"
                    )

                    path_index += 1

                    send_velocity(
                        ser,
                        DRIVE_VELOCITY_MPS,
                        corrected_heading,
                        x_cmd,
                    )

                    leaving_ignore_landmark = active_to
                    last_goal_forward = None
                    continue

                # --------------------------------------------------------
                # FINAL GOAL
                # --------------------------------------------------------
                x_corrected = lateral_pose_for_heading(pose, active_heading)
                x_cmd = lateral_command_for_heading(x_corrected, active_heading)
                y_error = corrected_forward_for_heading(pose, active_heading)

                raw_y_text = "None" if pose["forward"] is None else f"{pose['forward']:.4f}"
                corr_y_text = "None" if y_error is None else f"{y_error:.4f}"

                print(
                    f"GOAL_APP "
                    f"lm={active_to} "
                    f"tag={pose['tag']} "
                    f"pos={pose['position']} "
                    f"x_raw={pose['lateral']:.4f} "
                    f"x_cmd={x_cmd:.4f} "
                    f"raw_y={raw_y_text} "
                    f"corr_y={corr_y_text} "
                    f"heading={pose['heading']:.2f}"
                )

                if y_error is None:
                    print("GOAL_APP y=None, sending nothing")
                    continue

                reached_y_centre = False
                center_zone = is_center_zone_for_heading(pose, active_heading)

                if center_zone:
                    if last_goal_forward is not None:
                        if last_goal_forward > 0.0 and y_error <= 0.0:
                            reached_y_centre = True
                    else:
                        if y_error <= 0.0:
                            reached_y_centre = True

                    last_goal_forward = y_error
                else:
                    print(
                        f"GOAL_HELPER_NO_STOP "
                        f"lm={active_to} "
                        f"tag={pose['tag']} "
                        f"pos={pose['position']} "
                        f"corr_y={y_error:.4f}"
                    )

                if reached_y_centre:
                    print(f"GOAL_CENTER_REACHED lm={active_to}")
                    send_velocity(ser, 0.0, 0.0, 0.0)

                    print(f"FINAL GOAL REACHED: {goal_node}")

                    current_robot_heading = active_heading
                    leaving_ignore_landmark = None
                    last_goal_forward = None
                    mode = MODE_WAIT_TASK
                    continue

                corrected_heading = corrected_map_heading_from_tag(
                    active_heading,
                    pose["heading"],
                )

                send_approach(
                    ser,
                    ARRIVAL_VELOCITY_MPS,
                    corrected_heading,
                    x_cmd,
                    y_error,
                )
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
