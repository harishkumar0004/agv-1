import time

import serial


class SerialManager:
    def __init__(self, port, baudrate=115200):
        self.serial = serial.Serial(
            port,
            baudrate,
            timeout=0.2,
        )

        time.sleep(2.0)

        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()

    def close(self):
        self.serial.close()

    def clear_input(self):
        self.serial.reset_input_buffer()

    def send_command_wait_ack(self, command, max_wait_s=1.0):
        self.serial.reset_input_buffer()

        self.serial.write((command + "\n").encode())
        self.serial.flush()

        return self.wait_for_ack(max_wait_s=max_wait_s)

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
            f"{lateral_error_m:.4f}"
        )

        print("TX:", command)

        return self.send_command_wait_ack(
            command,
            max_wait_s=1.0,
        )

    def enable(self):
        return self.send_command_wait_ack(
            "EN",
            max_wait_s=2.0,
        )

    def disable(self):
        return self.send_command_wait_ack(
            "DIS",
            max_wait_s=1.0,
        )

    def stop(self):
        return self.send_command_wait_ack(
            "STOP",
            max_wait_s=1.0,
        )

    def ping(self):
        return self.send_command_wait_ack(
            "PING",
            max_wait_s=1.0,
        )

    def zero_heading(self):
        return self.send_command_wait_ack(
            "ZERO",
            max_wait_s=1.0,
        )

    def calibrate(self):
        return self.send_command_wait_ack(
            "CAL",
            max_wait_s=15.0,
        )

    def request_status(self):
        self.serial.write(b"STATUS\n")
        self.serial.flush()

        deadline = time.monotonic() + 1.0

        while time.monotonic() < deadline:
            line = self.read_line()

            if line is None:
                continue

            if line.startswith("STATUS"):
                return line

            print("ESP32:", line)

        return None

    def wait_for_ack(self, max_wait_s=1.0):
        deadline = time.monotonic() + max_wait_s

        while time.monotonic() < deadline:
            line = self.read_line()

            if line is None:
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

            if line.startswith("STATUS"):
                print("ESP32:", line)
                continue

            if line.startswith("AGV Ready"):
                print("ESP32:", line)
                continue

            print("ESP32:", line)

        return False
        
    def read_available_lines(self):
        lines = []

        while self.serial.in_waiting > 0:
            line = self.read_line()

            if line is None:
                break

            lines.append(line)

        return lines

    def read_line(self):
        line = self.serial.readline().decode(
            errors="ignore"
        ).strip()

        if line == "":
            return None

        return line
