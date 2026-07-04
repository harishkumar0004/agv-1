import cv2

from camera import Camera
from detector import AprilTagDetector
from viewer import Viewer

import geometry

from landmark_map import LandmarkMap
from localization import Localization
from navigation import Navigation
from serial_manager import SerialManager


def main():
    camera = Camera()
    detector = AprilTagDetector()
    viewer = Viewer()

    world = LandmarkMap("./maps/testbed.json")

    localization = Localization(world)

    navigation = Navigation(world)
    navigation.set_target(1)
    navigation.set_velocity(0.05)

    serial = SerialManager("/dev/ttyUSB0")

    camera.start()

    # if not serial.ping():
    #     print("Unable to communicate with low-level controller.")
    #     return

    # newly updated
    last_sent_tag = None
    last_sent_landmark = None

    started = False

    print("==========================================")
    print("Robot Ready")
    print("Place the robot at Landmark 0.")
    print("Align it with the corridor.")
    print("Press 's' to calibrate and start.")
    print("Press 'q' to quit.")
    print("==========================================")

    try:
        while True:

            frame = camera.get_frame()

            detections = detector.detect(frame)

            geometry.update(detections)

            localization.update(detections)

            navigation.update(localization)

            if started and navigation.valid():

                velocity = navigation.velocity

                if navigation.current == navigation.target:
                    velocity = 0.0

                should_send = False

                if navigation.current != last_sent_landmark:
                    should_send = True

                if should_send:
                    print(
                        f"SEND VEL {velocity:.3f} "
                        f"{navigation.desired_heading:.2f} "
                        f"{navigation.lateral_error:.4f} "
                        f"tag={localization.tag} "
                        f"pos={localization.position} "
                        f"raw_lat={localization.raw_lateral:.4f} "
                        f"offset={localization.center_lateral_offset:.4f} "
                        f"corr_lat={localization.lateral:.4f} "
                        f"visible={localization.visible_tags} "
                        f"current={navigation.current} "
                        f"next={navigation.next}"
                    )

                    if serial.send_velocity(
                        velocity,
                        navigation.desired_heading,
                        navigation.lateral_error,
                    ):
                        last_sent_landmark = navigation.current
                    else:
                        print("VEL command was not acknowledged by ESP32.")

            viewer.draw(frame, detections)
            viewer.show(frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s") and not started:

                if not localization.valid():
                    print("No valid localization. Cannot start.")
                    continue

                if localization.landmark["id"] != 0:
                    print("Robot must be positioned at Landmark 0.")
                    continue

                print("Calibrating IMU...")

                if not serial.calibrate():
                    print("IMU calibration failed.")
                    continue

                print("Zeroing heading...")

                if not serial.zero_heading():
                    print("Failed to zero heading.")
                    continue

                print("Enabling motors...")

                if not serial.enable():
                    print("Failed to enable motors.")
                    continue

                started = True

                print("Autonomous navigation started.")

    finally:

        serial.stop()
        serial.disable()
        serial.close()

        camera.release()
        viewer.close()


if __name__ == "__main__":
    main()