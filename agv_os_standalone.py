#!/usr/bin/env python3
"""AGV-OS development HMI (standalone, no Raspberry Pi OS changes).

Run manually:
    python3 agv_os_standalone.py --port /dev/ttyUSB0
    python3 agv_os_standalone.py --simulate

ESC or the window close button exits the application and restores the desktop.
This program never changes boot settings, services, display configuration, or
any Raspberry Pi OS file.  It uses Tkinter (included with Raspberry Pi OS) and
optionally pyserial for ESP32 communication.

ESP32 protocol expected by default (one command per line):
    PING -> ACK
    STATUS -> a line containing EN=0 (motor outputs disabled)
    CAL -> ACK after stationary IMU calibration
    ZERO -> ACK
    STOP, DIS -> ACK optional (commands are sent independently)

IMPORTANT: This is an HMI and supervisory start-up check, not a replacement
for a certified safety system.  A real AGV needs independent E-stop hardware,
safety-rated power interruption, safety scanners, risk assessment, and
validation to applicable regulations before operating near people.
"""

from __future__ import annotations

import argparse
import logging
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None


APP_NAME = "AGV-OS"
DISPLAY_W, DISPLAY_H = 1024, 600


@dataclass(frozen=True)
class Settings:
    port: Optional[str]
    baud: int
    timeout: float
    simulate: bool
    company: str
    log_dir: Path


class AppLogger:
    def __init__(self, log_dir: Path) -> None:
        self.events: queue.Queue[tuple[str, str, str]] = queue.Queue()
        self.logger = logging.getLogger("agv_os")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        self.logger.addHandler(console)
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / f"agv_os_{datetime.now():%Y%m%d_%H%M%S}.log")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except OSError as exc:
            self.logger.warning("Cannot create log file: %s", exc)

    def write(self, level: str, subsystem: str, message: str) -> None:
        getattr(self.logger, level.lower())("[%s] %s", subsystem, message)
        self.events.put((level, subsystem, message))

    def info(self, subsystem: str, message: str) -> None:
        self.write("INFO", subsystem, message)

    def warning(self, subsystem: str, message: str) -> None:
        self.write("WARNING", subsystem, message)

    def error(self, subsystem: str, message: str) -> None:
        self.write("ERROR", subsystem, message)


class ESP32Controller:
    """Small, serialized interface to the motion-controller protocol."""
    def __init__(self, settings: Settings, log: AppLogger) -> None:
        self.settings, self.log = settings, log
        self.connection = None
        self.lock = threading.Lock()

    def discover(self) -> Optional[str]:
        if self.settings.simulate:
            return "SIMULATED"
        if serial is None:
            self.log.error("COMM", "pyserial is not installed; cannot communicate with ESP32")
            return None
        if self.settings.port:
            return self.settings.port if Path(self.settings.port).exists() else None
        candidates = []
        for port in serial.tools.list_ports.comports():
            identity = f"{port.description} {port.hwid}".lower()
            if any(x in identity for x in ("cp210", "ch340", "ftdi", "esp32", "usb serial")):
                candidates.append(port.device)
        return candidates[0] if candidates else None

    def open(self, port: str) -> bool:
        if self.settings.simulate:
            self.log.warning("COMM", "Simulation mode: no physical serial device opened")
            return True
        try:
            self.connection = serial.Serial(port, self.settings.baud, timeout=self.settings.timeout)
            # Many ESP32 boards reset on DTR; allow firmware to become ready.
            time.sleep(1.5)
            self.connection.reset_input_buffer()
            self.connection.reset_output_buffer()
            self.log.info("COMM", f"ESP32 port opened: {port} @ {self.settings.baud}")
            return True
        except Exception as exc:
            self.log.error("COMM", f"Cannot open {port}: {exc}")
            return False

    def command(self, command: str, expect_ack: bool = True) -> Optional[str]:
        if self.settings.simulate:
            if command == "STATUS":
                return "EN=0 SIMULATED"
            return "ACK"
        if self.connection is None or not self.connection.is_open:
            return None
        try:
            with self.lock:
                self.connection.write((command + "\n").encode("ascii"))
                self.connection.flush()
                if not expect_ack:
                    return "SENT"
                line = self.connection.readline().decode("utf-8", errors="replace").strip()
            return line or None
        except Exception as exc:
            self.log.error("COMM", f"{command} transport failure: {exc}")
            return None

    def stop_and_disable(self) -> bool:
        """Best effort, but explicitly report whether controller acknowledged safe state."""
        stop = self.command("STOP")
        disable = self.command("DIS")
        status = self.command("STATUS")
        safe = bool(status and "EN=0" in status.upper())
        if not safe:
            self.log.error("SAFETY", f"Unable to prove disabled state (STOP={stop}, DIS={disable}, STATUS={status})")
        return safe

    def close(self) -> None:
        if self.connection is not None:
            try:
                self.connection.close()
            except Exception:
                pass
            self.connection = None


class HMI:
    BG = "#101016"; PANEL = "#1a1a22"; BORDER = "#393947"
    TEXT = "#e8e8f0"; DIM = "#9a9aa8"; CYAN = "#00b4ff"
    GREEN = "#00dc78"; AMBER = "#ffb400"; RED = "#ff3c3c"

    def __init__(self, app: "AGVApplication") -> None:
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk, self.app = tk, ttk, app
        self.root = tk.Tk()
        self.root.title(APP_NAME)
        self.root.configure(bg=self.BG)
        self.root.attributes("-fullscreen", True)
        self.root.protocol("WM_DELETE_WINDOW", app.request_exit)
        self.root.bind("<Escape>", lambda _e: app.request_exit())
        self.root.bind("<space>", lambda _e: app.request_estop())
        self.root.bind("<F11>", lambda _e: self.toggle_fullscreen())
        self.status_vars: dict[str, tk.StringVar] = {}
        self.progress = tk.DoubleVar(value=0)
        self.stage = tk.StringVar(value="Preparing application")
        self.clock = tk.StringVar()
        self._build()

    def toggle_fullscreen(self) -> None:
        self.root.attributes("-fullscreen", not bool(self.root.attributes("-fullscreen")))

    def label(self, parent, text="", size=12, color=None, **kw):
        background = kw.pop("bg", self.PANEL)
        return self.tk.Label(
            parent,
            text=text,
            font=("DejaVu Sans", size),
            fg=color or self.TEXT,
            bg=background,
            **kw
        )

    def panel(self, parent, title: str):
        box = self.tk.Frame(parent, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1)
        self.label(box, title, 11, self.CYAN, anchor="w").pack(fill="x", padx=12, pady=(8, 5))
        return box

    def _build(self) -> None:
        root = self.root
        head = self.tk.Frame(root, bg="#16161e", height=52); head.pack(fill="x"); head.pack_propagate(False)
        self.tk.Label(head, text=f"{APP_NAME}  |  INDUSTRIAL VEHICLE CONTROL", font=("DejaVu Sans", 17, "bold"),
                      fg=self.CYAN, bg="#16161e").pack(side="left", padx=20, pady=12)
        self.tk.Label(head, textvariable=self.clock, font=("DejaVu Sans", 11), fg=self.DIM, bg="#16161e").pack(side="right", padx=20)
        self.content = self.tk.Frame(root, bg=self.BG); self.content.pack(fill="both", expand=True, padx=16, pady=14)
        self.show_splash()
        self.root.after(200, self._tick)

    def clear(self) -> None:
        for child in self.content.winfo_children(): child.destroy()

    def show_splash(self) -> None:
        self.clear()
        space = self.tk.Frame(self.content, bg=self.BG); space.pack(fill="both", expand=True)
        self.tk.Label(space, text="⬡", font=("DejaVu Sans", 112), fg=self.CYAN, bg=self.BG).pack(pady=(80, 0))
        self.tk.Label(space, text=self.app.settings.company.upper(), font=("DejaVu Sans", 27, "bold"), fg=self.TEXT, bg=self.BG).pack()
        self.tk.Label(space, text="AUTONOMOUS GUIDED VEHICLE OPERATING SYSTEM", font=("DejaVu Sans", 14), fg=self.DIM, bg=self.BG).pack(pady=12)
        self.tk.Label(space, text="Raspberry Pi 4  •  ESP32 Motion Controller  •  1024 × 600 HMI", font=("DejaVu Sans", 11), fg=self.DIM, bg=self.BG).pack()
        self.label(space, "SYSTEM INITIALIZING", 14, self.CYAN, bg=self.BG).pack(pady=48)

    def show_boot(self) -> None:
        self.clear()
        self.label(self.content, textvariable=self.stage, size=15, color=self.TEXT, bg=self.BG, anchor="w").pack(fill="x", pady=(4, 8))
        bar = self.ttk.Progressbar(self.content, maximum=100, variable=self.progress, mode="determinate")
        bar.pack(fill="x", pady=(0, 12))
        grid = self.tk.Frame(self.content, bg=self.BG); grid.pack(fill="both", expand=True)
        grid.grid_columnconfigure(0, weight=1); grid.grid_columnconfigure(1, weight=2); grid.grid_columnconfigure(2, weight=1)
        hardware = self.panel(grid, "HARDWARE STATUS"); hardware.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        logs = self.panel(grid, "DIAGNOSTIC LOG"); logs.grid(row=0, column=1, sticky="nsew", padx=8)
        info = self.panel(grid, "SYSTEM INFORMATION"); info.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        for key in ("Power", "Serial port", "ESP32 handshake", "Safety state", "IMU calibration", "Camera (optional)", "AprilTag (optional)"):
            var = self.tk.StringVar(value="WAIT   " + key); self.status_vars[key] = var
            self.label(hardware, textvariable=var, size=11, color=self.DIM, anchor="w").pack(fill="x", padx=12, pady=7)
        self.log_box = self.tk.Text(logs, bg=self.PANEL, fg=self.TEXT, bd=0, font=("DejaVu Sans Mono", 9), wrap="word", state="disabled")
        self.log_box.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        for line in ("Platform: Raspberry Pi OS (unchanged)", "Display: Waveshare HDMI IPS 7 inch", "Resolution: 1024 x 600", "Motion: ESP32 via USB serial", "Safety: external hardware required", "Exit: ESC / window close"):
            self.label(info, line, 10, self.DIM, anchor="w").pack(fill="x", padx=12, pady=7)

    def show_dashboard(self) -> None:
        self.clear()
        top = self.tk.Frame(self.content, bg=self.BG); top.pack(fill="x")
        self.label(top, "SYSTEM READY", 18, self.GREEN, bg=self.BG).pack(side="left")
        self.label(top, "MODE: STANDBY", 14, self.AMBER, bg=self.BG).pack(side="right")
        row = self.tk.Frame(self.content, bg=self.BG); row.pack(fill="both", expand=True, pady=12)
        motion = self.panel(row, "MOTION CONTROL"); motion.pack(side="left", fill="both", expand=True, padx=(0, 8))
        vision = self.panel(row, "VISION / LOCALIZATION"); vision.pack(side="left", fill="both", expand=True, padx=8)
        nav = self.panel(row, "NAVIGATION"); nav.pack(side="left", fill="both", expand=True, padx=(8, 0))
        for value in ("Motor state: DISABLED", "Linear velocity: 0.000 m/s", "Angular velocity: 0.000 rad/s", "ESP32: ONLINE"):
            self.label(motion, value, 12, self.TEXT, anchor="w").pack(fill="x", padx=14, pady=12)
        self.label(vision, "Camera / AprilTag integration point", 13, self.DIM).pack(expand=True)
        for value in ("Current node: --", "Target node: --", "Path status: IDLE", "Safety state: VERIFIED"):
            self.label(nav, value, 12, self.TEXT, anchor="w").pack(fill="x", padx=14, pady=12)
        footer = self.tk.Frame(self.content, bg="#16161e"); footer.pack(fill="x")
        self.tk.Button(footer, text="EMERGENCY STOP", command=self.app.request_estop, bg="#b00020", fg="white", font=("DejaVu Sans", 12, "bold"), bd=0, padx=20, pady=8).pack(side="left", padx=8, pady=6)
        self.label(footer, "SPACE: E-stop     ESC: exit to Raspberry Pi desktop", 11, self.DIM, bg="#16161e").pack(side="right", padx=14)

    def show_fault(self, reason: str) -> None:
        self.clear(); self.content.configure(bg="#330b0b")
        self.tk.Label(self.content, text="FAULT — MOTION INHIBITED", font=("DejaVu Sans", 28, "bold"), fg=self.RED, bg="#330b0b").pack(pady=(100, 25))
        self.tk.Label(self.content, text=reason, font=("DejaVu Sans", 15), fg=self.TEXT, bg="#330b0b", wraplength=850).pack(padx=30)
        self.tk.Label(self.content, text="Verify external safety hardware and controller wiring. ESC returns to the desktop.", font=("DejaVu Sans", 12), fg=self.DIM, bg="#330b0b").pack(pady=35)

    def status(self, name: str, passed: bool, detail: str = "") -> None:
        var = self.status_vars.get(name)
        if var: var.set(("PASS" if passed else "FAIL") + "   " + name + (f" — {detail}" if detail else ""))

    def append_log(self, level: str, subsystem: str, message: str) -> None:
        if not hasattr(self, "log_box"): return
        color = self.RED if level == "ERROR" else self.AMBER if level == "WARNING" else self.TEXT
        self.log_box.configure(state="normal"); self.log_box.insert("end", f"[{subsystem}] {message}\n", level)
        self.log_box.tag_configure(level, foreground=color); self.log_box.see("end"); self.log_box.configure(state="disabled")

    def _tick(self) -> None:
        self.clock.set(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        while True:
            try: level, system, message = self.app.log.events.get_nowait()
            except queue.Empty: break
            self.append_log(level, system, message)
        self.root.after(200, self._tick)


class AGVApplication:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings; self.log = AppLogger(settings.log_dir)
        self.controller = ESP32Controller(settings, self.log); self.hmi = HMI(self)
        self.exiting = False; self.estop_active = False

    def boot(self) -> None:
        self.hmi.show_splash(); self.hmi.root.after(1800, self._start_boot_thread)

    def _start_boot_thread(self) -> None:
        self.hmi.show_boot(); threading.Thread(target=self._boot_worker, daemon=True).start()

    def _stage(self, number: int, title: str, action: Callable[[], tuple[bool, str]], mandatory: bool = True) -> bool:
        self.hmi.root.after(0, lambda: (self.hmi.stage.set(f"STAGE {number}/6 — {title}"), self.hmi.progress.set((number - 1) * 100 / 6)))
        self.log.info("BOOT", title)
        ok, detail = action()
        self.log.info("BOOT", detail) if ok else self.log.error("BOOT", detail)
        if not ok and mandatory: return False
        return True

    def _boot_worker(self) -> None:
        tests: list[tuple[str, bool, str]] = []
        tests.append(("Power", *self._power_check()))
        if not self._stage(1, "SYSTEM INITIALIZATION", lambda: (tests[-1][1], tests[-1][2])): return self._fault(tests[-1][2])
        port_box: dict[str, Optional[str]] = {"port": None}
        def find():
            port_box["port"] = self.controller.discover(); return (port_box["port"] is not None, f"ESP32 port: {port_box['port'] or 'not found'}")
        if not self._stage(2, "HARDWARE DETECTION", find): return self._fault("ESP32 serial port not found. Connect the controller or use --simulate for HMI development.")
        tests.append(("Serial port", self.controller.open(port_box["port"] or ""), "Serial port opened"))
        if not self._stage(3, "CONTROLLER CONNECTION", lambda: (tests[-1][1], tests[-1][2])): return self._fault("Cannot open ESP32 serial port.")
        response = self.controller.command("PING")
        tests.append(("ESP32 handshake", response == "ACK", f"PING response: {response or 'no response'}"))
        if not self._stage(3, "COMMUNICATION HANDSHAKE", lambda: (tests[-1][1], tests[-1][2])): return self._fault("ESP32 did not acknowledge PING. Motion remains inhibited.")
        status = self.controller.command("STATUS")
        tests.append(("Safety state", bool(status and "EN=0" in status.upper()), f"Controller status: {status or 'no response'}"))
        if not self._stage(4, "SAFETY VERIFICATION", lambda: (tests[-1][1], tests[-1][2])): return self._fault("Controller output is not proven disabled. Correct this before operation.")
        cal = self.controller.command("CAL")
        tests.append(("IMU calibration", cal == "ACK", f"Calibration response: {cal or 'no response'}"))
        if not self._stage(5, "IMU CALIBRATION — KEEP VEHICLE STILL", lambda: (tests[-1][1], tests[-1][2])): return self._fault("IMU calibration failed. Keep vehicle stationary and inspect MPU6050/ESP32 firmware.")
        zero = self.controller.command("ZERO")
        if zero != "ACK": self.log.warning("SENSOR", f"Heading zero response: {zero or 'no response'}")
        tests.extend([("Camera (optional)", True, "Not initialized by this standalone HMI"), ("AprilTag (optional)", True, "Integration ready")])
        self._stage(6, "FINAL READINESS CHECK", lambda: (True, "All mandatory startup checks passed; motion remains disabled until your supervised control layer enables it."))
        for name, passed, detail in tests:
            self.hmi.root.after(0, lambda n=name, p=passed, d=detail: self.hmi.status(n, p, d))
        self.hmi.root.after(0, lambda: (self.hmi.progress.set(100), self.hmi.show_dashboard()))

    def _power_check(self) -> tuple[bool, str]:
        return (True, "Application started; Linux desktop and OS configuration are unchanged")

    def _fault(self, reason: str) -> None:
        self.controller.stop_and_disable(); self.log.error("SAFETY", reason)
        self.hmi.root.after(0, lambda: self.hmi.show_fault(reason))

    def request_estop(self) -> None:
        if self.estop_active: return
        self.estop_active = True; self.log.error("SAFETY", "Operator emergency-stop request")
        threading.Thread(target=lambda: self._fault("Operator E-stop requested. STOP and DIS were sent to the ESP32; validate physical safety before restarting."), daemon=True).start()

    def request_exit(self) -> None:
        if self.exiting: return
        self.exiting = True; self.log.info("SHUTDOWN", "Closing AGV-OS; returning to Raspberry Pi desktop")
        threading.Thread(target=self._safe_exit, daemon=True).start()

    def _safe_exit(self) -> None:
        self.controller.stop_and_disable(); self.controller.close()
        self.hmi.root.after(0, self.hmi.root.destroy)

    def run(self) -> int:
        self.boot(); self.hmi.root.mainloop(); return 0


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description="Standalone AGV-OS development HMI")
    parser.add_argument("--port", help="ESP32 serial port, e.g. /dev/ttyUSB0; auto-detect if omitted")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--simulate", action="store_true", help="Run the HMI without any vehicle hardware")
    parser.add_argument("--company", default="YOUR COMPANY")
    parser.add_argument("--log-dir", type=Path, default=Path.cwd() / "agv_os_logs")
    args = parser.parse_args()
    return Settings(args.port, args.baud, args.timeout, args.simulate, args.company, args.log_dir)


def main() -> int:
    app = AGVApplication(parse_args())
    signal.signal(signal.SIGINT, lambda *_: app.request_exit())
    signal.signal(signal.SIGTERM, lambda *_: app.request_exit())
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
