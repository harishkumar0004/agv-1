#!/usr/bin/env python3
"""
AGV-OS v2.1.1
==============
Industrial Autonomous Guided Vehicle Operating System

DEVELOPMENT MODE:
    Run manually: python3 agv_os.py
    Fullscreen window opens, covers desktop
    Exit: ESC key or window close button
    Returns to Raspberry Pi desktop unchanged

DEPLOYMENT MODE (optional, configured separately):
    Auto-starts after Linux boot via systemd
    Desktop hidden from end user

Hardware: RPi4 + ESP32 + T60 Driver + 57AM23ED + MPU6050 + Waveshare 7"
Industrial Standard: ISO 3691-4, VDA 5050 compliant logging
"""

import sys
import os
import time
import json
import math
import signal
import threading
import logging
from datetime import datetime
from pathlib import Path
from enum import Enum, auto
from typing import Optional, Dict, List, Tuple

# Third-party imports (must be installed)
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("WARNING: OpenCV not installed. Display disabled.")

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

try:
    from pupil_apriltags import Detector
    APRILTAG_AVAILABLE = True
except ImportError:
    APRILTAG_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """AGV-OS configuration."""
    # Display (Waveshare 7" DPI LCD typical resolution)
    DISPLAY_WIDTH = 1024
    DISPLAY_HEIGHT = 600

    # Serial - will auto-detect if default not found
    SERIAL_PORT = "/dev/ttyUSB0"
    SERIAL_BAUD = 115200
    SERIAL_TIMEOUT = 2.0

    # Camera (matches your existing code)
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480

    # AprilTag (matches your existing code)
    TAG_SIZE_M = 0.010
    APRILTAG_FAMILY = "tag36h11"
    CAMERA_PARAMS = (615.0, 615.0, CAMERA_WIDTH / 2.0, CAMERA_HEIGHT / 2.0)

    # Map
    MAP_FILE = "./maps/testbed.json"

    # Timing
    SPLASH_DURATION_S = 2.5
    BOOT_STAGE_TIMEOUT_S = 15.0
    ESP32_BOOT_WAIT_S = 3.0

    # Paths
    LOG_DIR = Path("./logs")
    ASSETS_DIR = Path("./assets")


# ============================================================================
# INDUSTRIAL LOGGER
# ============================================================================

class AGVLogger:
    """Industrial-grade logger with file and console output."""

    ERROR_CODES = {
        'INIT': 'AGV-001', 'HW': 'AGV-002', 'COMM': 'AGV-003',
        'SAFETY': 'AGV-004', 'SENSOR': 'AGV-005', 'NAV': 'AGV-006',
        'READY': 'AGV-007', 'SHUTDOWN': 'AGV-008', 'EMERGENCY': 'AGV-900',
        'RUNTIME': 'AGV-010', 'DASHBOARD': 'AGV-020'
    }

    def __init__(self):
        self.logger = logging.getLogger('AGV_OS')
        self.logger.setLevel(logging.DEBUG)

        # Console
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(
            '[%(asctime)s] [%(levelname)-8s] %(message)s',
            datefmt='%H:%M:%S'
        ))
        self.logger.addHandler(console)

        # File
        try:
            Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(
                Config.LOG_DIR / f"agv_os_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                '[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s'
            ))
            self.logger.addHandler(fh)
        except Exception:
            pass

        # Display buffer for on-screen logs
        self.display_buffer: List[Tuple[str, str, str]] = []
        self._buffer_lock = threading.Lock()
        self.max_buffer_lines = 100

    def log(self, level, subsystem, message, code=None):
        code_str = f"[{code}] " if code else ""
        self.logger.log(level, f"[{subsystem:>6}] {code_str}{message}")

        with self._buffer_lock:
            self.display_buffer.append((
                logging.getLevelName(level), subsystem, message
            ))
            if len(self.display_buffer) > self.max_buffer_lines:
                self.display_buffer.pop(0)

    def info(self, subsys, msg, code=None):
        self.log(logging.INFO, subsys, msg, code)

    def warning(self, subsys, msg, code=None):
        self.log(logging.WARNING, subsys, msg, code)

    def error(self, subsys, msg, code=None):
        self.log(logging.ERROR, subsys, msg, code)

    def critical(self, subsys, msg, code='EMERGENCY'):
        self.log(logging.CRITICAL, subsys, msg, code)

    def get_display_lines(self, count=20):
        with self._buffer_lock:
            return self.display_buffer[-count:]


# ============================================================================
# DISPLAY ENGINE (OpenCV Fullscreen Window)
# ============================================================================

class DisplayEngine:
    """
    Industrial HMI display engine.
    Runs as a normal OpenCV window that can go fullscreen.
    Window title: "AGV-OS" - appears in taskbar like any application.
    """

    # Industrial color palette (BGR format for OpenCV)
    C_BG = (16, 16, 22)           # Deep navy background
    C_PANEL = (26, 26, 34)        # Panel background
    C_PANEL_LIGHT = (36, 36, 46)  # Highlighted panel
    C_ACCENT = (0, 180, 255)      # Cyan accent
    C_SUCCESS = (0, 220, 120)     # Green
    C_WARNING = (0, 180, 255)     # Amber
    C_ERROR = (60, 60, 255)       # Red
    C_TEXT = (230, 230, 240)      # Primary text
    C_TEXT_DIM = (140, 140, 150)  # Secondary text
    C_BORDER = (55, 55, 70)       # Borders
    C_HEADER_BG = (22, 22, 30)    # Header bar
    C_PROGRESS_BG = (40, 40, 50)  # Progress bar background
    C_EMERGENCY = (0, 0, 200)     # Emergency red
    C_WHITE = (255, 255, 255)

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_MONO = cv2.FONT_HERSHEY_PLAIN

    def __init__(self, logger: AGVLogger):
        self.log = logger
        self.width = Config.DISPLAY_WIDTH
        self.height = Config.DISPLAY_HEIGHT
        self.frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        self._running = True
        self._view = "splash"  # splash | boot | dashboard | emergency | shutdown | error
        self._lock = threading.Lock()

        # Boot state
        self.boot_stage = 0
        self.boot_stage_name = "Initializing"
        self.boot_progress = 0.0
        self.hardware_status = {}
        self.system_info = {}

        # Dashboard state
        self.dashboard_data = {}

        # Emergency state
        self.emergency_reason = ""
        self.emergency_time = 0.0

        # Error state
        self.error_message = ""
        self.error_details = []

        # Create window - normal first, then fullscreen
        if CV2_AVAILABLE:
            cv2.namedWindow("AGV-OS", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("AGV-OS", self.width, self.height)
            cv2.setWindowProperty("AGV-OS", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            self.log.info('DISP', "Display engine initialized: 1024x600 fullscreen", 'DASHBOARD')

        self._render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self._render_thread.start()

    def stop(self):
        self._running = False
        if self._render_thread:
            self._render_thread.join(timeout=1.0)
        if CV2_AVAILABLE:
            try:
                cv2.destroyWindow("AGV-OS")
            except:
                pass

    def set_view(self, view: str):
        with self._lock:
            self._view = view

    def set_boot_progress(self, stage: int, name: str, progress: float):
        with self._lock:
            self.boot_stage = stage
            self.boot_stage_name = name
            self.boot_progress = progress

    def set_hardware_status(self, status: Dict):
        with self._lock:
            self.hardware_status = status.copy()

    def set_system_info(self, info: Dict):
        with self._lock:
            self.system_info = info.copy()

    def set_dashboard_data(self, data: Dict):
        with self._lock:
            self.dashboard_data = data.copy()

    def set_error(self, message: str, details: List[str]):
        with self._lock:
            self.error_message = message
            self.error_details = details
            self._view = "error"

    def trigger_emergency(self, reason: str):
        with self._lock:
            self._view = "emergency"
            self.emergency_reason = reason
            self.emergency_time = time.time()

    def trigger_shutdown(self):
        with self._lock:
            self._view = "shutdown"

    def _render_loop(self):
        while self._running:
            with self._lock:
                view = self._view

            self.frame[:] = self.C_BG

            if view == "splash":
                self._draw_splash()
            elif view == "boot":
                self._draw_boot()
            elif view == "dashboard":
                self._draw_dashboard()
            elif view == "emergency":
                self._draw_emergency()
            elif view == "shutdown":
                self._draw_shutdown()
            elif view == "error":
                self._draw_error()

            if CV2_AVAILABLE:
                cv2.imshow("AGV-OS", self.frame)
                key = cv2.waitKey(33) & 0xFF
                if key == 27:  # ESC key
                    self._running = False

    # ========================================================================
    # DRAWING PRIMITIVES
    # ========================================================================

    def _panel(self, x, y, w, h, title=""):
        """Draw a panel with optional title."""
        cv2.rectangle(self.frame, (x, y), (x+w, y+h), self.C_PANEL, -1)
        cv2.rectangle(self.frame, (x, y), (x+w, y+h), self.C_BORDER, 1)
        if title:
            cv2.rectangle(self.frame, (x, y), (x+w, y+26), self.C_BORDER, -1)
            cv2.putText(self.frame, title, (x+10, y+18), self.FONT, 0.5, self.C_ACCENT, 1)
        return y + 30

    def _centered_text(self, text, y, size=0.7, color=None, thickness=1):
        color = color or self.C_TEXT
        (tw, th), _ = cv2.getTextSize(text, self.FONT, size, thickness)
        x = (self.width - tw) // 2
        cv2.putText(self.frame, text, (x, y), self.FONT, size, color, thickness)

    def _right_text(self, text, y, x_right, size=0.5, color=None, thickness=1):
        color = color or self.C_TEXT
        (tw, th), _ = cv2.getTextSize(text, self.FONT, size, thickness)
        cv2.putText(self.frame, text, (x_right - tw, y), self.FONT, size, color, thickness)

    def _progress_bar(self, x, y, w, h, progress, label=""):
        """Draw a progress bar with percentage."""
        cv2.rectangle(self.frame, (x, y), (x+w, y+h), self.C_PROGRESS_BG, -1)
        cv2.rectangle(self.frame, (x, y), (x+w, y+h), self.C_BORDER, 1)

        fill_w = int(w * min(1.0, max(0.0, progress)))
        if fill_w > 0:
            for i in range(fill_w):
                ratio = i / w
                b = int(self.C_ACCENT[0])
                g = int(self.C_ACCENT[1] * (0.7 + 0.3 * (1 - ratio)))
                r = int(self.C_ACCENT[2] * (0.8 + 0.2 * (1 - ratio)))
                cv2.line(self.frame, (x+i, y+1), (x+i, y+h-1), (b, g, r), 1)

        pct = int(progress * 100)
        cv2.putText(self.frame, f"{pct}%", (x + w - 45, y + h - 4),
                    self.FONT, 0.45, self.C_TEXT, 1)
        if label:
            cv2.putText(self.frame, label, (x + 5, y + h - 4),
                        self.FONT, 0.4, self.C_TEXT_DIM, 1)

    def _status_indicator(self, x, y, label, status, detail=""):
        """Draw a status indicator line."""
        if status is True:
            color = self.C_SUCCESS
            symbol = "OK"
        elif status is False:
            color = self.C_ERROR
            symbol = "FAIL"
        else:
            color = self.C_WARNING
            symbol = "WAIT"

        cv2.putText(self.frame, f"[{symbol}]", (x, y), self.FONT, 0.5, color, 1)
        cv2.putText(self.frame, label, (x + 55, y), self.FONT, 0.5, self.C_TEXT, 1)
        if detail:
            cv2.putText(self.frame, detail, (x + 200, y), self.FONT, 0.4, self.C_TEXT_DIM, 1)

    # ========================================================================
    # VIEWS
    # ========================================================================

    def _draw_splash(self):
        """Professional AGV splash screen."""
        cx = self.width // 2
        cy = self.height // 2 - 30

        # Hexagon logo
        hex_r = 70
        hex_pts = []
        for i in range(6):
            a = math.radians(i * 60 - 30)
            hex_pts.append([
                int(cx + hex_r * math.cos(a)),
                int(cy - 40 + hex_r * math.sin(a))
            ])
        hex_pts = np.array(hex_pts, np.int32)
        cv2.fillPoly(self.frame, [hex_pts], self.C_ACCENT)
        cv2.polylines(self.frame, [hex_pts], True, (0, 200, 255), 2)

        # AGV text inside hexagon
        cv2.putText(self.frame, "AGV", (cx - 42, cy - 32), self.FONT,
                    1.2, self.C_BG, 3)

        # Company name
        self._centered_text("AUTONOMOUS GUIDED VEHICLE", cy + 55, 0.65, self.C_TEXT, 1)

        # Hardware spec
        self._centered_text("RPi4 + ESP32  |  T60 Driver  |  57AM23ED  |  MPU6050",
                            cy + 85, 0.45, self.C_TEXT_DIM, 1)

        # Version and standard
        self._centered_text("AGV-OS v2.1.1  |  ISO 3691-4  |  VDA 5050",
                            cy + 110, 0.4, self.C_TEXT_DIM, 1)

        # Animated loading dots
        t = time.time()
        dots = int((t * 2.5) % 4)
        self._centered_text("SYSTEM INITIALIZING" + "." * dots,
                              cy + 155, 0.55, self.C_ACCENT, 1)

    def _draw_boot(self):
        """Boot progress screen with live diagnostics."""
        # Header bar
        cv2.rectangle(self.frame, (0, 0), (self.width, 44), self.C_HEADER_BG, -1)
        cv2.putText(self.frame, "AGV-OS  BOOT SEQUENCE", (15, 30),
                    self.FONT, 0.75, self.C_ACCENT, 2)

        # Stage indicator
        stage_text = f"STAGE {self.boot_stage}/7: {self.boot_stage_name}"
        cv2.putText(self.frame, stage_text, (15, 68), self.FONT, 0.5, self.C_TEXT, 1)

        # Progress bar
        self._progress_bar(15, 78, self.width - 30, 18, self.boot_progress)

        # Three-column layout
        col1_w = 300
        col2_w = self.width - col1_w - 300
        col3_w = 300

        # Column 1: Hardware Status
        content_y = self._panel(15, 110, col1_w - 15, self.height - 140, "HARDWARE STATUS")
        y = content_y
        for name, status in self.hardware_status.items():
            self._status_indicator(25, y, name.replace('_', ' ').title(), status)
            y += 24

        # Column 2: System Logs
        content_y = self._panel(col1_w, 110, col2_w, self.height - 140, "DIAGNOSTIC LOGS")
        y = content_y
        for level, subsys, msg in self.log.get_display_lines(14):
            if level in ("ERROR", "CRITICAL"):
                color = self.C_ERROR
            elif level == "WARNING":
                color = self.C_WARNING
            else:
                color = self.C_TEXT
            line = f"[{subsys:>4}] {msg[:48]}"
            cv2.putText(self.frame, line, (col1_w + 10, y), self.FONT_MONO, 0.9, color, 1)
            y += 16

        # Column 3: System Info
        content_y = self._panel(col1_w + col2_w, 110, col3_w - 15, self.height - 140, "SYSTEM INFO")
        info_items = [
            ("Platform", "Raspberry Pi 4"),
            ("Display", "Waveshare 7\" 1024x600"),
            ("Motor", "57AM23ED (2.3 N.m)"),
            ("Driver", "T60 (1/16 microstep)"),
            ("IMU", "MPU6050 (I2C 0x68)"),
            ("Camera", "PiCamera2"),
            ("Tags", "AprilTag 36h11"),
            ("Serial", f"{Config.SERIAL_BAUD} baud"),
        ]
        y = content_y
        for label, value in info_items:
            cv2.putText(self.frame, f"{label}:", (col1_w + col2_w + 10, y),
                        self.FONT, 0.42, self.C_TEXT_DIM, 1)
            cv2.putText(self.frame, value, (col1_w + col2_w + 110, y),
                        self.FONT, 0.42, self.C_TEXT, 1)
            y += 22

        # Footer
        cv2.rectangle(self.frame, (0, self.height - 24), (self.width, self.height),
                      self.C_HEADER_BG, -1)
        boot_time = self.system_info.get('boot_time', 0.0)
        cv2.putText(self.frame, f"Boot Time: {boot_time:.1f}s  |  ESC to abort",
                    (15, self.height - 7), self.FONT, 0.4, self.C_TEXT_DIM, 1)

    def _draw_dashboard(self):
        """Operational dashboard - main operator interface."""
        # Header
        cv2.rectangle(self.frame, (0, 0), (self.width, 40), self.C_HEADER_BG, -1)
        cv2.putText(self.frame, "AGV-OS  OPERATOR DASHBOARD", (15, 28),
                    self.FONT, 0.75, self.C_ACCENT, 2)

        mode = self.dashboard_data.get('mode', 'STANDBY')
        mode_color = self.C_SUCCESS if mode == 'RUNNING' else self.C_WARNING
        self._right_text(f"MODE: {mode}", 28, self.width - 15, 0.55, mode_color, 1)

        # Camera feed (center, large)
        cam_x, cam_y = 330, 48
        cam_w, cam_h = 500, 380
        self._panel(cam_x, cam_y, cam_w, cam_h, "VISION SYSTEM")

        # Placeholder for actual camera feed
        cv2.putText(self.frame, "[ Camera Feed ]", (cam_x + cam_w//2 - 90, cam_y + cam_h//2),
                    self.FONT, 0.7, self.C_TEXT_DIM, 1)
        cv2.putText(self.frame, "AprilTag Detection Active",
                    (cam_x + cam_w//2 - 110, cam_y + cam_h//2 + 25),
                    self.FONT, 0.45, self.C_TEXT_DIM, 1)

        # Left column: Motion
        y = self._panel(8, 48, 315, 195, "MOTION CONTROL")
        motor = self.dashboard_data.get('motors', {})
        items = [
            ("Left Velocity", f"{motor.get('left_vel', 0):.3f} m/s"),
            ("Right Velocity", f"{motor.get('right_vel', 0):.3f} m/s"),
            ("Left Steps", f"{motor.get('left_steps', 0)}"),
            ("Right Steps", f"{motor.get('right_steps', 0)}"),
            ("Linear Speed", f"{motor.get('linear', 0):.3f} m/s"),
            ("Angular Rate", f"{motor.get('angular', 0):.4f} rad/s"),
        ]
        for label, value in items:
            cv2.putText(self.frame, label, (18, y), self.FONT, 0.42, self.C_TEXT_DIM, 1)
            cv2.putText(self.frame, value, (140, y), self.FONT, 0.42, self.C_TEXT, 1)
            y += 22

        # Left bottom: IMU
        y = self._panel(8, 250, 315, 195, "IMU / HEADING")
        imu = self.dashboard_data.get('imu', {})
        items = [
            ("Heading", f"{imu.get('heading', 0):.2f} deg"),
            ("Gyro Z", f"{imu.get('gyro', 0):.2f} deg/s"),
            ("Calibrated", "YES" if imu.get('calibrated') else "NO"),
            ("Tag Aligned", "YES" if imu.get('aligned') else "NO"),
            ("Bias", f"{imu.get('bias', 0):.4f} deg/s"),
        ]
        for label, value in items:
            cv2.putText(self.frame, label, (18, y), self.FONT, 0.42, self.C_TEXT_DIM, 1)
            cv2.putText(self.frame, value, (140, y), self.FONT, 0.42, self.C_TEXT, 1)
            y += 22

        # Right column: Navigation
        y = self._panel(self.width - 323, 48, 315, 195, "NAVIGATION")
        nav = self.dashboard_data.get('navigation', {})
        items = [
            ("Current Node", f"{nav.get('current', '--')}"),
            ("Next Node", f"{nav.get('next', '--')}"),
            ("Target Heading", f"{nav.get('target_heading', 0):.1f} deg"),
            ("Lateral Error", f"{nav.get('lateral_error', 0):.4f} m"),
            ("Path Progress", f"{nav.get('path_index', 0)}/{nav.get('path_len', 0)}"),
        ]
        for label, value in items:
            cv2.putText(self.frame, label, (self.width - 313, y), self.FONT, 0.42, self.C_TEXT_DIM, 1)
            cv2.putText(self.frame, value, (self.width - 190, y), self.FONT, 0.42, self.C_TEXT, 1)
            y += 22

        # Right bottom: AprilTag
        y = self._panel(self.width - 323, 250, 315, 195, "APRILTAG LOCALIZATION")
        tag = self.dashboard_data.get('apriltag', {})
        items = [
            ("Tag ID", f"{tag.get('tag_id', '--')}"),
            ("Landmark", f"{tag.get('landmark', '--')}"),
            ("Position", f"{tag.get('position', '--')}"),
            ("Distance Y", f"{tag.get('forward', 0):.4f} m"),
            ("Lateral X", f"{tag.get('lateral', 0):.4f} m"),
            ("Detected", "YES" if tag.get('detected') else "NO"),
        ]
        for label, value in items:
            cv2.putText(self.frame, label, (self.width - 313, y), self.FONT, 0.42, self.C_TEXT_DIM, 1)
            cv2.putText(self.frame, value, (self.width - 190, y), self.FONT, 0.42, self.C_TEXT, 1)
            y += 22

        # Command bar
        cv2.rectangle(self.frame, (0, self.height - 44), (self.width, self.height),
                      self.C_HEADER_BG, -1)
        cv2.putText(self.frame, "[S] Dock  |  [G] Goal  |  [SPACE] E-Stop  |  [ESC] Exit",
                    (50, self.height - 14), self.FONT, 0.5, self.C_TEXT, 1)

    def _draw_emergency(self):
        """Emergency stop screen - full screen warning."""
        # Flashing background
        t = time.time() - self.emergency_time
        flash = int((t * 3) % 2)
        bg_color = (0, 0, 60) if flash else (0, 0, 30)
        self.frame[:] = bg_color

        cx = self.width // 2
        cy = self.height // 2

        # Border
        border_color = self.C_ERROR if flash else (0, 0, 80)
        cv2.rectangle(self.frame, (30, 30), (self.width - 30, self.height - 30), border_color, 6)

        # EMERGENCY text
        self._centered_text("EMERGENCY STOP", cy - 80, 1.6, self.C_ERROR, 3)
        self._centered_text("ACTIVE", cy - 30, 1.0, self.C_ERROR, 2)

        # Reason
        self._centered_text(f"Cause: {self.emergency_reason}", cy + 20, 0.6, self.C_TEXT, 1)

        # Status boxes
        box_y = cy + 60
        cv2.rectangle(self.frame, (cx - 180, box_y), (cx + 180, box_y + 40), self.C_PANEL, -1)
        cv2.putText(self.frame, "MOTORS: DISABLED", (cx - 140, box_y + 28),
                    self.FONT, 0.6, self.C_SUCCESS, 2)

        cv2.rectangle(self.frame, (cx - 180, box_y + 50), (cx + 180, box_y + 90), self.C_PANEL, -1)
        cv2.putText(self.frame, "SERIAL: CLOSED", (cx - 120, box_y + 78),
                    self.FONT, 0.6, self.C_SUCCESS, 2)

        # Recovery
        self._centered_text("RECOVERY PROCEDURE", cy + 170, 0.55, self.C_WARNING, 1)
        self._centered_text("1. Power cycle ESP32", cy + 195, 0.45, self.C_TEXT_DIM, 1)
        self._centered_text("2. Press ESP32 reset button", cy + 215, 0.45, self.C_TEXT_DIM, 1)
        self._centered_text("3. Re-run AGV-OS boot sequence", cy + 235, 0.45, self.C_TEXT_DIM, 1)

    def _draw_shutdown(self):
        """Shutdown screen - clean power-off."""
        self.frame[:] = self.C_BG
        cx = self.width // 2
        cy = self.height // 2

        self._centered_text("SYSTEM SHUTDOWN", cy - 60, 1.3, self.C_ACCENT, 2)

        t = time.time()
        dots = int((t * 2) % 4)
        self._centered_text("Powering off" + "." * dots, cy - 10, 0.6, self.C_TEXT, 1)

        # Checklist
        checklist = [
            ("Motors disabled", True),
            ("Serial port closed", True),
            ("Camera stopped", True),
            ("Logs saved", True),
        ]
        y = cy + 40
        for item, ok in checklist:
            color = self.C_SUCCESS if ok else self.C_ERROR
            cv2.putText(self.frame, f"[OK] {item}", (cx - 130, y), self.FONT, 0.5, color, 1)
            y += 25

        self._centered_text("It is now safe to remove power.", cy + 160, 0.55, self.C_WARNING, 1)

    def _draw_error(self):
        """Boot error screen - shows when boot fails."""
        self.frame[:] = self.C_BG
        cx = self.width // 2
        cy = self.height // 2 - 50

        # Error header
        cv2.rectangle(self.frame, (0, 0), (self.width, 50), (0, 0, 60), -1)
        cv2.putText(self.frame, "AGV-OS  BOOT ERROR", (15, 35),
                    self.FONT, 0.85, self.C_ERROR, 2)

        # Error icon (X)
        cv2.line(self.frame, (cx - 40, cy - 40), (cx + 40, cy + 40), self.C_ERROR, 6)
        cv2.line(self.frame, (cx + 40, cy - 40), (cx - 40, cy + 40), self.C_ERROR, 6)

        # Error message
        self._centered_text("BOOT SEQUENCE FAILED", cy + 20, 0.9, self.C_ERROR, 2)
        self._centered_text(self.error_message, cy + 55, 0.55, self.C_TEXT, 1)

        # Error details
        y = cy + 90
        for detail in self.error_details[:6]:
            cv2.putText(self.frame, f"  > {detail}", (cx - 280, y),
                        self.FONT, 0.45, self.C_TEXT_DIM, 1)
            y += 22

        # Recovery instructions
        cv2.rectangle(self.frame, (cx - 250, cy + 220), (cx + 250, cy + 320), self.C_PANEL, -1)
        cv2.rectangle(self.frame, (cx - 250, cy + 220), (cx + 250, cy + 320), self.C_BORDER, 1)

        self._centered_text("RECOVERY STEPS", cy + 245, 0.55, self.C_WARNING, 1)
        self._centered_text("1. Check ESP32 USB connection", cy + 270, 0.45, self.C_TEXT_DIM, 1)
        self._centered_text("2. Verify ESP32 is powered and programmed", cy + 290, 0.45, self.C_TEXT_DIM, 1)
        self._centered_text("3. Press ESC to exit, then re-run AGV-OS", cy + 310, 0.45, self.C_TEXT_DIM, 1)

        # Footer
        cv2.rectangle(self.frame, (0, self.height - 30), (self.width, self.height),
                      self.C_HEADER_BG, -1)
        cv2.putText(self.frame, "ESC to exit to desktop",
                    (15, self.height - 8), self.FONT, 0.4, self.C_TEXT_DIM, 1)


# ============================================================================
# HARDWARE MANAGER
# ============================================================================

class HardwareManager:
    """Manages all AGV hardware interactions."""

    def __init__(self, logger: AGVLogger):
        self.log = logger
        self.esp32: Optional[serial.Serial] = None
        self.camera = None
        self.detector = None
        self.map_data = None

        self.tests = {
            'power': False,
            'serial_port': False,
            'esp32_comm': False,
            'imu_calibrated': False,
            'camera': False,
            'apriltag': False,
            'map_loaded': False,
            'safety_ok': False,
        }

    def detect_serial(self):
        """Find ESP32 serial port with improved detection."""
        self.log.info('HW', "Scanning serial ports...", 'HW')
        ports = list(serial.tools.list_ports.comports())

        if not ports:
            self.log.error('HW', "No serial ports found", 'HW')
            return False

        # Log all found ports with full details
        for p in ports:
            self.log.info('HW', f"  {p.device} | {p.description} | {p.hwid}", 'HW')

        # Priority 1: Check expected port
        if Path(Config.SERIAL_PORT).exists():
            self.tests['serial_port'] = True
            self.log.info('HW', f"Found expected port: {Config.SERIAL_PORT}", 'HW')
            return True

        # Priority 2: Check common ESP32 port patterns
        common_ports = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0', 
                        '/dev/ttyACM1', '/dev/ttyAMA0']
        for port_path in common_ports:
            if Path(port_path).exists():
                Config.SERIAL_PORT = port_path
                self.tests['serial_port'] = True
                self.log.info('HW', f"Found ESP32 on: {port_path}", 'HW')
                return True

        # Priority 3: Auto-detect by description
        esp32_keywords = ['usb', 'uart', 'cp210', 'ch340', 'ftdi', 'esp32', 
                         'silicon labs', 'bridge']
        for p in ports:
            desc = (p.description + ' ' + p.hwid).lower()
            if any(kw in desc for kw in esp32_keywords):
                Config.SERIAL_PORT = p.device
                self.tests['serial_port'] = True
                self.log.info('HW', f"Auto-detected ESP32: {p.device} ({p.description})", 'HW')
                return True

        self.log.error('HW', "ESP32 not found. Check USB connection.", 'HW')
        self.log.error('HW', f"Available ports: {[p.device for p in ports]}", 'HW')
        return False

    def open_serial(self):
        """Open serial connection to ESP32."""
        try:
            self.esp32 = serial.Serial(
                Config.SERIAL_PORT, Config.SERIAL_BAUD, timeout=Config.SERIAL_TIMEOUT
            )
            time.sleep(Config.ESP32_BOOT_WAIT_S)
            self.esp32.reset_input_buffer()
            self.esp32.reset_output_buffer()
            self.log.info('COMM', f"Serial opened: {Config.SERIAL_PORT}@{Config.SERIAL_BAUD}", 'COMM')
            return True
        except Exception as e:
            self.log.error('COMM', f"Serial open failed: {e}", 'COMM')
            return False

    def ping_esp32(self):
        """Test ESP32 communication."""
        for attempt in range(3):
            try:
                self.esp32.write(b"PING\n")
                self.esp32.flush()
                time.sleep(0.5)
                response = self.esp32.readline().decode(errors='ignore').strip()
                if response == "ACK":
                    self.log.info('COMM', f"PING OK (attempt {attempt+1}/3)", 'COMM')
                    self.tests['esp32_comm'] = True
                    return True
                else:
                    self.log.warning('COMM', f"Unexpected response: '{response}'", 'COMM')
            except Exception as e:
                self.log.warning('COMM', f"PING {attempt+1} failed: {e}", 'COMM')
            time.sleep(1.0)

        self.log.error('COMM', "ESP32 not responding to PING", 'COMM')
        self.log.error('COMM', "Check: Is ESP32 programmed with low_level_controller.ino?", 'COMM')
        return False

    def check_safety(self):
        """Verify safe state."""
        try:
            self.esp32.write(b"STATUS\n")
            self.esp32.flush()
            time.sleep(0.3)
            response = self.esp32.readline().decode(errors='ignore').strip()

            if "EN=0" in response or "EN=false" in response.lower():
                self.log.info('SAFETY', "Motors DISABLED (safe state)", 'SAFETY')
                self.tests['safety_ok'] = True
                return True
            else:
                self.log.warning('SAFETY', "Motors enabled! Sending STOP+DIS...", 'SAFETY')
                self.esp32.write(b"STOP\nDIS\n")
                self.esp32.flush()
                self.tests['safety_ok'] = True
                return True
        except Exception as e:
            self.log.warning('SAFETY', f"Safety check error: {e}", 'SAFETY')
            return False

    def calibrate_imu(self):
        """Calibrate IMU via ESP32."""
        self.log.info('SENSOR', "Starting IMU calibration...", 'SENSOR')
        self.log.info('SENSOR', ">>> KEEP AGV STATIONARY <<<")

        try:
            self.esp32.write(b"CAL\n")
            self.esp32.flush()

            deadline = time.monotonic() + Config.BOOT_STAGE_TIMEOUT_S
            while time.monotonic() < deadline:
                line = self.esp32.readline().decode(errors='ignore').strip()
                if line:
                    self.log.info('SENSOR', f"  ESP32: {line}", 'SENSOR')
                    if line == "ACK":
                        self.tests['imu_calibrated'] = True
                        self.log.info('SENSOR', "IMU calibration complete", 'SENSOR')
                        return True
                    elif line.startswith("ERR"):
                        self.log.error('SENSOR', f"IMU CAL failed: {line}", 'SENSOR')
                        return False
                time.sleep(0.1)

            self.log.error('SENSOR', "IMU calibration timeout", 'SENSOR')
            return False
        except Exception as e:
            self.log.error('SENSOR', f"IMU calibration exception: {e}", 'SENSOR')
            return False

    def zero_heading(self):
        """Zero heading reference."""
        try:
            self.esp32.write(b"ZERO\n")
            self.esp32.flush()
            time.sleep(0.5)
            if self.esp32.readline().decode(errors='ignore').strip() == "ACK":
                self.log.info('SENSOR', "Heading zeroed", 'SENSOR')
                return True
        except Exception as e:
            self.log.warning('SENSOR', f"Zero heading failed: {e}", 'SENSOR')
        return False

    def init_camera(self):
        """Initialize PiCamera2."""
        try:
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": (Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT), "format": "RGB888"}
            )
            self.camera.configure(config)
            self.camera.start()
            time.sleep(Config.ESP32_BOOT_WAIT_S)

            test = self.camera.capture_array()
            if test is not None and test.size > 0:
                self.tests['camera'] = True
                self.log.info('HW', f"Camera: {test.shape[1]}x{test.shape[0]} RGB888", 'HW')
                return True
        except Exception as e:
            self.log.error('HW', f"Camera init failed: {e}", 'HW')
        return False

    def init_detector(self):
        """Initialize AprilTag detector."""
        try:
            self.detector = Detector(
                families=Config.APRILTAG_FAMILY, nthreads=4,
                quad_decimate=1.0, quad_sigma=0.0, refine_edges=True,
            )
            self.tests['apriltag'] = True
            self.log.info('HW', f"AprilTag detector: {Config.APRILTAG_FAMILY}", 'HW')
            return True
        except Exception as e:
            self.log.error('HW', f"Detector init failed: {e}", 'HW')
        return False

    def load_map(self):
        """Load navigation map."""
        try:
            map_path = Path(Config.MAP_FILE)
            if map_path.exists():
                with open(map_path, 'r') as f:
                    self.map_data = json.load(f)
                count = len(self.map_data.get('landmarks', []))
                self.tests['map_loaded'] = True
                self.log.info('HW', f"Map loaded: {count} landmarks", 'HW')
                return True
        except Exception as e:
            self.log.warning('HW', f"Map load failed: {e}", 'HW')
        return False

    def test_tag_detection(self):
        """Quick AprilTag detection test."""
        if not self.tests['camera'] or not self.tests['apriltag']:
            return False

        self.log.info('NAV', "Testing AprilTag detection...", 'NAV')
        for attempt in range(30):
            try:
                frame = self.camera.capture_array()
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                detections = self.detector.detect(
                    gray, estimate_tag_pose=True,
                    camera_params=Config.CAMERA_PARAMS,
                    tag_size=Config.TAG_SIZE_M,
                )
                if detections:
                    det = detections[0]
                    self.log.info('NAV', f"Tag {det.tag_id} detected (attempt {attempt+1})", 'NAV')
                    return True
            except Exception:
                pass
            time.sleep(0.1)

        self.log.warning('NAV', "No tags visible - check environment", 'NAV')
        return False

    def send_stop(self):
        """Send STOP to ESP32."""
        if self.esp32 and self.esp32.is_open:
            try:
                self.esp32.write(b"STOP\n")
                self.esp32.flush()
            except:
                pass

    def send_disable(self):
        """Send DIS (disable motors) to ESP32."""
        if self.esp32 and self.esp32.is_open:
            try:
                self.esp32.write(b"DIS\n")
                self.esp32.flush()
            except:
                pass

    def close(self):
        """Close all hardware."""
        self.send_stop()
        self.send_disable()
        time.sleep(0.2)

        if self.esp32 and self.esp32.is_open:
            try:
                self.esp32.close()
            except:
                pass

        if self.camera:
            try:
                self.camera.stop()
            except:
                pass


# ============================================================================
# EMERGENCY STOP HANDLER
# ============================================================================

class EmergencyHandler:
    """
    Handles emergency stop conditions.

    WHAT HAPPENS:
    1. Display shows full-screen flashing EMERGENCY STOP
    2. STOP + DIS sent to ESP32 (motors halt and disable)
    3. Serial port closed
    4. System enters safe state
    5. Manual recovery required (power cycle ESP32)
    """

    def __init__(self, logger: AGVLogger, display: DisplayEngine, hardware: HardwareManager):
        self.log = logger
        self.display = display
        self.hardware = hardware
        self._active = False

    def trigger(self, reason: str = "OPERATOR_REQUESTED"):
        if self._active:
            return
        self._active = True

        self.log.critical('EMERGENCY', f"EMERGENCY STOP: {reason}", 'EMERGENCY')

        # 1. Display emergency screen
        self.display.trigger_emergency(reason)

        # 2. Stop motors
        self.hardware.send_stop()
        time.sleep(0.1)

        # 3. Disable motors
        self.hardware.send_disable()
        time.sleep(0.1)

        # 4. Close serial
        self.hardware.close()

        self.log.critical('EMERGENCY', "System in SAFE STATE", 'EMERGENCY')
        self.log.critical('EMERGENCY', "Recovery: Power cycle ESP32, re-run AGV-OS", 'EMERGENCY')

    def is_active(self):
        return self._active


# ============================================================================
# SHUTDOWN HANDLER
# ============================================================================

class ShutdownHandler:
    """
    Handles clean shutdown.

    WHAT HAPPENS:
    1. Display shows shutdown screen with checklist
    2. STOP + DIS sent to ESP32
    3. Camera stopped
    4. Serial closed
    5. Logs saved
    6. Display stopped
    7. Returns to Raspberry Pi desktop
    """

    def __init__(self, logger: AGVLogger, display: DisplayEngine, hardware: HardwareManager):
        self.log = logger
        self.display = display
        self.hardware = hardware
        self._requested = False

    def request(self):
        if self._requested:
            return
        self._requested = True

        self.log.info('SHUTDOWN', "Clean shutdown initiated", 'SHUTDOWN')

        # Show shutdown screen
        self.display.trigger_shutdown()
        time.sleep(0.5)

        # Stop hardware
        self.hardware.send_stop()
        self.hardware.send_disable()
        time.sleep(0.3)
        self.hardware.close()

        self.log.info('SHUTDOWN', "Hardware stopped", 'SHUTDOWN')

        # Stop display
        time.sleep(1.5)
        self.display.stop()

        self.log.info('SHUTDOWN', "AGV-OS shutdown complete", 'SHUTDOWN')
        sys.exit(0)


# ============================================================================
# BOOT SEQUENCE CONTROLLER
# ============================================================================

class BootSequence:
    """7-stage power-on self-test."""

    STAGES = [
        (1, "POWER-ON & SYSTEM INITIALIZATION"),
        (2, "HARDWARE DETECTION"),
        (3, "COMMUNICATION HANDSHAKE"),
        (4, "SAFETY SYSTEM VERIFICATION"),
        (5, "SENSOR CALIBRATION"),
        (6, "NAVIGATION INITIALIZATION"),
        (7, "FINAL READINESS CHECK"),
    ]

    def __init__(self, logger: AGVLogger, display: DisplayEngine, hardware: HardwareManager):
        self.log = logger
        self.display = display
        self.hardware = hardware
        self.start_time = time.monotonic()
        self._failed = False
        self._error_message = ""
        self._error_details = []

    def _update_display(self, stage_num, stage_name, progress):
        self.display.set_boot_progress(stage_num, stage_name, progress)
        self.display.set_hardware_status(self.hardware.tests)
        self.display.set_system_info({'boot_time': time.monotonic() - self.start_time})

    def _fail(self, message, details=None):
        self._failed = True
        self._error_message = message
        self._error_details = details or []
        self.log.error('SYS', message, 'EMERGENCY')
        for d in self._error_details:
            self.log.error('SYS', f"  > {d}", 'EMERGENCY')

    def run(self):
        """Execute all 7 boot stages. Returns True if ready."""
        self.display.set_view("splash")
        time.sleep(Config.SPLASH_DURATION_S)
        self.display.set_view("boot")

        # Stage 1: Power-on
        self._update_display(1, "POWER-ON & SYSTEM INITIALIZATION", 0.1)
        self.log.info('INIT', "AGV-OS v2.1.1 starting", 'INIT')
        self.log.info('INIT', "Hardware: RPi4 + ESP32 + T60 + 57AM23ED + MPU6050", 'INIT')
        self.hardware.tests['power'] = True
        self._update_display(1, "POWER-ON & SYSTEM INITIALIZATION", 1.0)
        time.sleep(0.3)

        # Stage 2: Hardware detection
        self._update_display(2, "HARDWARE DETECTION", 0.1)
        if not self.hardware.detect_serial():
            self._fail("ESP32 not found", [
                "Check USB cable is connected to ESP32",
                "Verify ESP32 has power (LED should be on)",
                "Try different USB port on Raspberry Pi",
                "Check 'ls /dev/ttyUSB*' in terminal"
            ])
            return False

        self._update_display(2, "HARDWARE DETECTION", 0.4)
        self.hardware.init_camera()
        self._update_display(2, "HARDWARE DETECTION", 0.7)
        self.hardware.init_detector()
        self.hardware.load_map()
        self._update_display(2, "HARDWARE DETECTION", 1.0)
        time.sleep(0.3)

        # Stage 3: Communication
        self._update_display(3, "COMMUNICATION HANDSHAKE", 0.1)
        if not self.hardware.open_serial():
            self._fail("Cannot open serial port", [
                f"Port: {Config.SERIAL_PORT}",
                "Check ESP32 USB cable has data lines (not power-only)",
                "Verify ESP32 is programmed with low_level_controller.ino",
                "Try: ls -la /dev/ttyUSB* /dev/ttyACM*"
            ])
            return False

        self._update_display(3, "COMMUNICATION HANDSHAKE", 0.5)
        if not self.hardware.ping_esp32():
            self._fail("ESP32 not responding", [
                "ESP32 may not be programmed with correct firmware",
                "Check baud rate is 115200 in low_level_controller.ino",
                "Try pressing ESP32 reset button",
                "Re-flash low_level_controller.ino to ESP32"
            ])
            return False

        self._update_display(3, "COMMUNICATION HANDSHAKE", 1.0)
        time.sleep(0.3)

        # Stage 4: Safety
        self._update_display(4, "SAFETY SYSTEM VERIFICATION", 0.3)
        self.hardware.check_safety()
        self._update_display(4, "SAFETY SYSTEM VERIFICATION", 1.0)
        time.sleep(0.3)

        # Stage 5: Calibration
        self._update_display(5, "SENSOR CALIBRATION", 0.1)
        if not self.hardware.calibrate_imu():
            self._fail("IMU calibration failed", [
                "Keep AGV completely stationary during calibration",
                "Check MPU6050 I2C wiring: GPIO 21=SDA, GPIO 22=SCL",
                "Verify MPU6050 has 3.3V power",
                "Check I2C is enabled: sudo raspi-config -> Interface Options"
            ])
            return False

        self._update_display(5, "SENSOR CALIBRATION", 0.6)
        self.hardware.zero_heading()
        self._update_display(5, "SENSOR CALIBRATION", 1.0)
        time.sleep(0.3)

        # Stage 6: Navigation
        self._update_display(6, "NAVIGATION INITIALIZATION", 0.3)
        self.hardware.test_tag_detection()
        self._update_display(6, "NAVIGATION INITIALIZATION", 1.0)
        time.sleep(0.3)

        # Stage 7: Ready check
        self._update_display(7, "FINAL READINESS CHECK", 0.3)
        boot_time = time.monotonic() - self.start_time

        critical = ['serial_port', 'esp32_comm', 'imu_calibrated', 'camera', 'safety_ok']
        all_pass = all(self.hardware.tests[k] for k in critical)

        self.log.info('READY', "=" * 40, 'READY')
        self.log.info('READY', "SYSTEM TEST SUMMARY", 'READY')
        for name, status in self.hardware.tests.items():
            status_str = "PASS" if status else "FAIL"
            self.log.info('READY', f"  [{status_str:>4}] {name.replace('_', ' ').title()}", 'READY')
        self.log.info('READY', f"Boot Time: {boot_time:.1f}s", 'READY')

        if all_pass:
            self.log.info('READY', "AGV-OS READY FOR OPERATION", 'COMPLETE')
            self._update_display(7, "SYSTEM READY", 1.0)
            time.sleep(1.0)
            self.display.set_view("dashboard")
            return True
        else:
            self._fail("Critical tests failed", ["Review log for details"])
            return False

    def get_error(self):
        return self._error_message, self._error_details


# ============================================================================
# MAIN AGV-OS APPLICATION
# ============================================================================

class AGVOS:
    """
    AGV-OS Main Application.

    DEVELOPMENT: Run manually from terminal. Exits to desktop.
    DEPLOYMENT:  Configure systemd auto-start (separate step).
    """

    def __init__(self):
        self.log = AGVLogger()
        self.display = DisplayEngine(self.log)
        self.hardware = HardwareManager(self.log)
        self.emergency = EmergencyHandler(self.log, self.display, self.hardware)
        self.shutdown = ShutdownHandler(self.log, self.display, self.hardware)

        self._running = True
        self._boot_complete = False

        # Signal handlers for clean exit
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        self.log.info('SHUTDOWN', f"Received {sig_name}", 'SHUTDOWN')
        self.shutdown.request()

    def run(self):
        """Main AGV-OS entry point."""
        print("\n")
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║                    AGV-OS  v2.1.1                                  ║")
        print("║     Industrial Autonomous Guided Vehicle Operating System            ║")
        print("║     RPi4 + ESP32 + T60 + 57AM23ED + MPU6050 + Waveshare 7\"         ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        print("\n  DEVELOPMENT MODE: Run manually from terminal")
        print("  Press ESC or close window to exit to desktop\n")

        # Run boot sequence
        boot = BootSequence(self.log, self.display, self.hardware)
        self._boot_complete = boot.run()

        if not self._boot_complete:
            error_msg, error_details = boot.get_error()
            self.log.error('SYS', f"Boot failed: {error_msg}", 'EMERGENCY')

            # Show error screen instead of just halting
            self.display.set_error(error_msg, error_details)

            # Wait for user to press ESC
            self.log.info('SYS', "Press ESC to exit to desktop", 'SHUTDOWN')
            while self._running:
                if CV2_AVAILABLE:
                    key = cv2.waitKey(100) & 0xFF
                    if key == 27:  # ESC
                        break
                time.sleep(0.1)

            self.display.stop()
            return 1

        # Boot passed - run operational loop
        self.log.info('SYS', "Entering operational mode", 'RUNTIME')
        self._operational_loop()

        return 0

    def _operational_loop(self):
        """Main operational loop - runs until shutdown."""
        while self._running:
            # Update dashboard with dummy data (replace with real telemetry)
            self.display.set_dashboard_data({
                'mode': 'STANDBY',
                'motors': {'left_vel': 0.0, 'right_vel': 0.0, 'left_steps': 0,
                          'right_steps': 0, 'linear': 0.0, 'angular': 0.0},
                'imu': {'heading': 0.0, 'gyro': 0.0, 'calibrated': True,
                       'aligned': False, 'bias': 0.0},
                'navigation': {'current': '--', 'next': '--', 'target_heading': 0.0,
                              'lateral_error': 0.0, 'path_index': 0, 'path_len': 0},
                'apriltag': {'tag_id': '--', 'landmark': '--', 'position': '--',
                            'forward': 0.0, 'lateral': 0.0, 'detected': False},
            })

            # Check for key input
            if CV2_AVAILABLE:
                key = cv2.waitKey(100) & 0xFF
                if key == 27:  # ESC
                    self.log.info('SYS', "ESC pressed - shutting down", 'SHUTDOWN')
                    self.shutdown.request()
                elif key == ord(' '):  # SPACE = Emergency Stop
                    self.emergency.trigger("OPERATOR_ESTOP")

            time.sleep(0.05)


def main():
    """AGV-OS entry point."""
    agv = AGVOS()
    return agv.run()


if __name__ == "__main__":
    sys.exit(main())
