from time import time

import serial


class SerialManager:
    def __init__(self, port, baudrate=115200):
        self.serial = serial.Serial(
            port,
            baudrate,
            timeout=0.2,
        )

        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()

    def close(self):
        self.serial.close()

    def send_velocity(
        self,
        velocity_mps,
        desired_heading_deg,
        lateral_error_m,
    ):
        command = (
            f"VEL "
            f"{velocity_mps:.3f} "
            f"{desired_heading_deg:.2f} "
            f"{lateral_error_m:.4f}\n"
        )

        self.serial.write(command.encode())

        return self.wait_for_ack(max_wait_s = 1.0)

    def enable(self):
        self.serial.write(b"EN\n")
        return self.wait_for_ack(max_wait_s = 1.0)

    def disable(self):
        self.serial.write(b"DIS\n")
        return self.wait_for_ack(max_wait_s = 1.0)

    def stop(self):
        self.serial.write(b"STOP\n")
        return self.wait_for_ack(max_wait_s = 1.0)

    def ping(self):
        self.serial.write(b"PING\n")
        return self.wait_for_ack(max_wait_s = 1.0)

    def zero_heading(self):
        self.serial.write(b"ZERO\n")
        return self.wait_for_ack(max_wait_s = 1.0)

    def calibrate(self):
        self.serial.write(b"CAL\n")
        return self.wait_for_ack(max_wait_s = 5.0)

    def request_status(self):
        self.serial.write(b"STATUS\n")
        return self.read_line()

    def wait_for_ack(self, max_wait_s = 1.0):
        deadline = time.monotonic() + max_wait_s

        while time.monotonic() < deadline:
            line = self.read_line()

            if line is None or line == "":
                continue

            if line == "ACK":
                return True

            if line.startswith("ACK"):
                return True

            if line.startswith("ERR"):
                print("ESP32 error:", line)
                return False

            if line.startswith("FAULT"):
                print("ESP32 fault:", line)
                return False

            # Ignore STATUS / AGV Ready / other debug lines while waiting for ACK.
            print("ESP32:", line)

        return False

    # def read_line(self):
    #     if self.serial.in_waiting == 0:
    #         return None

    #     return self.serial.readline().decode().strip()

    def read_line(self):
        line = self.serial.readline().decode(errors="ignore").strip()

        if line == "":
            return None

        return line