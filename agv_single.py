import math
import time
import json

import cv2
import serial
from picamera2 import Picamera2
from pupil_apriltags import Detector


# ============================================================================
# CONFIGURATION
# ============================================================================

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

DOCK_NODE = 0
FIRST_NODE = 1

DRIVE_VELOCITY_MPS = 0.050
ARRIVAL_VELOCITY_MPS = 0.025

HELPER_SPACING_M = 0.015

MAX_ACCEPTED_LATERAL_M = 0.100

MODE_DOCK_WAIT = "DOCK_WAIT"
MODE_DOCK_TO_TAG1 = "DOCK_TO_TAG1"
MODE_WAIT_TASK = "WAIT_TASK"
MODE_DONE = "DONE"


# ============================================================================
# MAP LOADING
# ============================================================================

def load_map(filename):
    with open(filename, "r") as f:
        return json.load(f)


MAP_DATA = load_map(MAP_FILE)


# ============================================================================
# BASIC HELPERS
# ============================================================================

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


# ============================================================================
# TAG PRIORITY AND HELPER OFFSET
# ============================================================================

def helper_lateral_offset(position):
    if position in ("east", "north_east", "south_east"):
        return HELPER_SPACING_M

    if position in ("west", "north_west", "south_west"):
        return -HELPER_SPACING_M

    return 0.0


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
# DOCK TO TAG 1 HELPERS
# ============================================================================

def tag1_center_reached(pose):
    if pose is None:
        return False

    if pose["landmark_id"] != FIRST_NODE:
        return False

    # Dock 0 -> tag 1 heading is 0 deg.
    # For 0 deg movement, center row is west / center / east.
    return pose["position"] in {"west", "center", "east"}


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


def send_approach(ser, velocity_mps, desired_heading_deg, x_lateral_m, y_lateral_m):
    command = (
        f"APP {velocity_mps:.3f} "
        f"{desired_heading_deg:.2f} "
        f"{x_lateral_m:.4f} "
        f"{y_lateral_m:.4f}"
    )

    print("TX:", command)

    return send_command_wait_ack(ser, command, max_wait_s=1.0)


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

def draw_detections(frame, detections, pose=None, mode=None):
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

    cv2.putText(
        frame,
        f"MODE {mode}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
    )

    y += 25

    if pose is not None:
        text = (
            f"POSE lm={pose['landmark_id']} "
            f"tag={pose['tag']} "
            f"pos={pose['position']} "
            f"pri={pose['priority']} "
            f"x={pose['lateral']:.4f} "
            f"y={pose['forward']:.4f} "
            f"h={pose['heading']:.2f} "
            f"visible={pose['visible_tags']}"
        )

        cv2.putText(
            frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
        )

    return frame


# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    camera = start_camera()
    detector = create_detector()
    ser = open_serial()

    mode = MODE_DOCK_WAIT

    print("==========================================")
    print("AGV Dock-to-Tag1 Controller")
    print("Press 's' to start from dock tag 0.")
    print("Press 'q' to quit.")
    print("==========================================")

    try:
        while True:
            # ---------------------------------------------------------------
            # Every loop captures the latest camera frame.
            # Correction is always based on the latest selected pose.
            # ---------------------------------------------------------------
            frame = camera.capture_array()
            detections = detect_tags(detector, frame)
            pose = estimate_pose_from_tags(detections)

            for line in read_available_lines(ser):
                print("ESP32:", line)

            # ---------------------------------------------------------------
            # MODE: Move from dock tag 0 to tag 1
            # ---------------------------------------------------------------
            if mode == MODE_DOCK_TO_TAG1:
                active_heading = 0.0

                if pose is None:
                    print("No pose during dock-to-tag1. Stopping.")
                    send_velocity(ser, 0.0, 0.0, 0.0)
                    continue

                if tag1_center_reached(pose):
                    send_velocity(ser, 0.0, 0.0, 0.0)

                    print("TAG 1 REACHED")
                    print("Enter start and goal node.")

                    mode = MODE_WAIT_TASK
                    continue

                if pose["landmark_id"] == FIRST_NODE:
                    # Latest frame sees tag 1 helper, but not center row yet.
                    # Send APP with x and y correction.
                    print(
                        f"TAG1_APPROACH "
                        f"tag={pose['tag']} "
                        f"pos={pose['position']} "
                        f"x={pose['lateral']:.4f} "
                        f"y={pose['forward']:.4f}"
                    )

                    send_approach(
                        ser,
                        ARRIVAL_VELOCITY_MPS,
                        active_heading,
                        pose["lateral"],
                        pose["forward"],
                    )

                else:
                    # Tag 1 not visible yet. Usually the camera still sees dock tag 0.
                    print("DOCK_CRUISE 0 -> 1")

                    send_velocity(
                        ser,
                        DRIVE_VELOCITY_MPS,
                        active_heading,
                        0.0,
                    )

                continue

            # ---------------------------------------------------------------
            # MODE: After tag 1 is reached, get start and goal from terminal
            # ---------------------------------------------------------------
            if mode == MODE_WAIT_TASK:
                send_velocity(ser, 0.0, 0.0, 0.0)

                try:
                    start_node = int(input("Enter start node: "))
                    goal_node = int(input("Enter goal node: "))
                except ValueError:
                    print("Invalid input. Enter numbers only.")
                    continue

                print(f"Start node: {start_node}")
                print(f"Goal node: {goal_node}")

                if start_node != FIRST_NODE:
                    print("Robot is physically at tag 1, so start node should be 1.")
                    continue

                path = find_path(start_node, goal_node)

                if len(path) < 2:
                    print(f"No path from {start_node} to {goal_node}")
                    continue

                print(f"A* path: {' -> '.join(map(str, path))}")
                print("Dock-to-tag1 milestone complete. Path driving is not connected yet.")

                mode = MODE_DONE
                continue

            # ---------------------------------------------------------------
            # Draw/debug window
            # ---------------------------------------------------------------
            draw_detections(frame, detections, pose, mode)

            cv2.imshow(
                "AGV Dock-to-Tag1",
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            # ---------------------------------------------------------------
            # Start from dock tag 0
            # ---------------------------------------------------------------
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

                mode = MODE_DOCK_TO_TAG1
                continue

            if mode == MODE_DONE:
                # Keep camera/debug open. Press q to quit.
                pass

    finally:
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
