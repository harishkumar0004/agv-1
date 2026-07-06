import math
import time
from collections import deque

import cv2
import serial
from picamera2 import Picamera2
from pupil_apriltags import Detector

# Configuration

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

FX = 615.0 #focal length in x direction, in pixels
FY = 615.0 #focal length in y direction, in pixels
CX = FRAME_WIDTH / 2.0 #optical center in x direction
CY = FRAME_HEIGHT / 2.0 #optical center in y direction
CAMERA_PARAMS = (FX, FY, CX, CY) #camera parameters

TAG_SIZE_M = 0.010
APRILTAG_FAMILY = "tag36h11"

SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 115200

START_LANDMARK = 0
TARGET_LANDMARK = 1
DRIVE_VELOCITY_MPS = 0.050

#Helper tags are 15 mm from centre tag
HELPER_SPACING_M = 0.015

TAG_HEADING_GAIN = 0.40
MAX_TAG_HEADING_CORRECTION_DEG = 2.0

# Measurement safety limits

MAX_ACCEPTED_LATERAL_M = 0.100
MAX_ACCEPTED_HEADING_DEG = 25.0

# Tested Map

MAP_DATA = {
    "grid": {
        "rows": 5,
        "columns": 4,
        "landmark_spacing_m": 0.500,
        "helper_spacing_m": HELPER_SPACING_M,
        "auto_neighbors": True,
    },
    "landmarks": [
        {
            "id": 0,
            "name": "Dock",
            "type": "dock",
            "row": -1,
            "column": 0,
            "tags": {
                "north_west": 268,
                "north": 261,
                "north_east": 262,
                "west": 267,
                "center": 0,
                "east": 263,
                "south_west": 266,
                "south": 265,
                "south_east": 264,
            },
        },
        {
            "id": 1,
            "name": "Landmark 1",
            "type": "normal",
            "row": 0,
            "column": 0,
            "tags": {
                "north_west": 108,
                "north": 101,
                "north_east": 102,
                "west": 107,
                "center": 1,
                "east": 103,
                "south_west": 106,
                "south": 105,
                "south_east": 104,
            },
        },
    ],
}

def clamp(value, low, high):
    return max(low, min(high, value))

def normalize_angle(angle_deg):
    while angle_deg > 180:
        angle_deg -= 360
    while angle_deg <= -180:
        angle_deg += 360
    return angle_deg

def average_angles_deg(weighted_angles):
    if not weighted_angles:
        return None
    
    x_sum = 0.0
    y_sum = 0.0

    for angle_deg, weight in weighted_angles:
        angle_rad = math.radians(angle_deg)
        x_sum += math.cos(angle_rad) * weight
        y_sum += math.sin(angle_rad) * weight

    if abs(x_sum) < 1e-9 and abs(y_sum) < 1e-9:
        return None
    
    return normalize_angle(math.degrees(math.atan2(y_sum, x_sum)))

def tag_area(detection):
    corners = getattr(detection, "corners", None)
    if corners is None or len(corners) != 4:
        return 0.0
    
    x0, y0 = corners[0]
    x1, y1 = corners[1]
    x2, y2 = corners[2] 
    x3, y3 = corners[3] 

    return 0.5 * abs(x0*y1 + x1*y2 + x2*y3 + x3*y0 - y0*x1 - y1*x2 - y2*x3 - y3*x0)

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

# Map and navigation helpers

def landmark_by_id(landmark_id):
    for landmark in MAP_DATA["landmarks"]:
        if landmark["id"] == landmark_id:
            return landmark
    return None

def find_tag(tag_id):
    for landmark in MAP_DATA["landmarks"]:
        for position, mapped_tag_id  in landmark["tags"].items():
            if int(mapped_tag_id) == int(tag_id):
                return {
                    "landmark": landmark,
                    "id": landmark["id"],
                    "position": position,
                }
    return None

def neighbors(landmark_id):
    current = landmark_by_id(landmark_id)
    if current is None:
        return []

    out = []
    for landmark in MAP_DATA["landmarks"]:
        if landmark["id"] == landmark_id:
            continue
        dr = abs(landmark["row"] - current["row"])
        dc = abs(landmark["column"] - current["column"])
        if dr + dc == 1:
            out.append(landmark["id"])
    return out

# BFS Search for pathfinding between landmarks
def find_path(start_id, goal_id):
    if landmark_by_id(start_id) is None or landmark_by_id(goal_id) is None:
        return []
    
    queue = deque([[start_id]])
    visited = {start_id}

    while queue:
        path = queue.popleft()
        current = path[-1]

        if current == goal_id:
            return path

        for nxt in neighbors(current):
            if nxt not in visited:
                visited.add(nxt)
                queue.append(path + [nxt])
    return []

def map_heading(current_id, next_id):

    """
    row +1 -> 0 deg
    column +1 -> 90 deg
    row -1 -> 180 deg
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

# camera
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

# This is priority-based pose estimate function
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

        if abs(det.heading) > MAX_ACCEPTED_HEADING_DEG:
            continue

        area = max(float(getattr(det, "area", 0.0)), 1.0)

        candidates.append({
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
        })

    if unknown_tags:
        print(f"Unknown tags detected: {unknown_tags}")

    if not candidates:
        return None

    selected = min(candidates,key=lambda item: (item["priority"],-item["area"],),)

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
    
def compute_navigation(pose, target_id, velocity_mps):
    if pose is None:
        return None
    
    current = pose["landmark_id"]
    if current == target_id:
        return {
            "current" : current,
            "next" : None,
            "desired_heading" : 0.0,
            "lateral_error" : 0.0,
            "velocity" : 0.0,
            "path": [current],
        }
    path = find_path(current, target_id)
    if len(path) < 2:
        return None
    nxt = path[1]
    base_heading = map_heading(current, nxt)
    if base_heading is None:
        return None
    
    tag_heading_correction = -TAG_HEADING_GAIN * pose["heading"]
    tag_heading_correction = clamp(tag_heading_correction, -MAX_TAG_HEADING_CORRECTION_DEG, MAX_TAG_HEADING_CORRECTION_DEG)
    desired_heading = normalize_angle(base_heading + tag_heading_correction)
    return {
        "current": current,
        "next": nxt,
        "desired_heading": desired_heading,
        "lateral_error": pose["lateral"],
        "velocity": velocity_mps,
        "path": path,
    }

# ESP32 Serial Communication

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

# Viewer

def draw_detections(frame, detections, pose=None, nav=None):
    height, width = frame.shape[:2]
    image_center_x = width // 2
    image_center_y = height // 2

    # Draw camera center crosshair
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
            color = (0, 255, 255)   # selected tag
            thickness = 3
        else:
            color = (0, 255, 0)     # normal detected tag
            thickness = 2

        # Draw tag border
        for i in range(4):
            p1 = tuple(corners[i])
            p2 = tuple(corners[(i + 1) % 4])
            cv2.line(frame, p1, p2, color, thickness)

        # Draw tag center
        cv2.circle(frame, tag_center, 5, (0, 0, 255), -1)

        # Draw line from camera/image center to tag center
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
            f"vel={nav['velocity']:.3f}"
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

# Main loop

def main():
    camera = start_camera()
    detector = create_detector()
    ser = open_serial()

    started = False
    last_sent_landmark = None

    print("==========================================")
    print("AGV Minimal Single File Controller")
    print(f"Start landmark: {START_LANDMARK}")
    print(f"Target landmark: {TARGET_LANDMARK}")
    print("Press 's' to calibrate/start. Press 'q' to quit.")
    print("==========================================")

    try:
        while True:
            frame = camera.capture_array()
            detections = detect_tags(detector, frame)
            pose = estimate_pose_from_tags(detections)
            nav = compute_navigation(pose, TARGET_LANDMARK, DRIVE_VELOCITY_MPS)

            for line in read_available_lines(ser):
                if line.startswith("STATUS") or line.startswith("FAULT") or line.startswith("ERR"):
                    print("ESP32:", line)
                else:
                    print("ESP32:", line)

            if started and nav is not None:
                should_send = nav["current"] != last_sent_landmark

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
                        f"visible={pose['visible_tags']} "
                        f"used={pose['used_count']} "
                        f"current={nav['current']} "
                        f"next={nav['next']}"
                    )

                    send_velocity(
                        ser,
                        nav["velocity"],
                        nav["desired_heading"],
                        nav["lateral_error"],
                    )

                    last_sent_landmark = nav["current"]

            draw_detections(frame, detections, pose, nav)
            cv2.imshow("AGV Single File", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s") and not started:
                if pose is None:
                    print("No valid localization. Cannot start.")
                    continue

                if pose["landmark_id"] != START_LANDMARK:
                    print(f"Robot must start at landmark {START_LANDMARK}. Current={pose['landmark_id']}")
                    continue

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

                started = True
                last_sent_landmark = None
                print("Autonomous navigation started.")

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
