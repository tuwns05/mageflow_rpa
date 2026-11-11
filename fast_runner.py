from __future__ import annotations

import ctypes
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import mss
import numpy as np
import pyautogui
import pygetwindow as gw

from workflow_schema import BashStepPayload, GUIStepPayload, WorkflowDocument, WorkflowStep

if sys.platform == "win32":
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.02


@dataclass
class UIShiftException(Exception):
    """Ngoại lệ ném ra khi Fast Runner không tìm thấy thành phần giao diện đủ độ tin cậy."""

    step: WorkflowStep
    screenshot_bgr: np.ndarray
    best_confidence: float
    expected_screen_coords: Tuple[int, int]
    window_origin: Tuple[int, int]
    reason: str

    def __str__(self) -> str:
        return (
            f"UIShiftException tại Step #{self.step.step_id} ({self.step.description}): "
            f"{self.reason} | best_conf={self.best_confidence:.3f}, "
            f"expected={self.expected_screen_coords}"
        )


class FastRunner:
    """
    Bộ thực thi kịch bản tốc độ cao (Deterministic Runner) không gọi AI
    trừ khi phát hiện giao diện thay đổi / vượt ngưỡng sai số (UIShiftException).
    """

    def __init__(
        self,
        base_dir: str | Path,
        log_callback: Optional[Callable[[str, str], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.log_callback = log_callback or (lambda msg, level="INFO": print(f"[{level}] {msg}"))
        self.stop_event = stop_event or threading.Event()

    def log(self, message: str, level: str = "INFO") -> None:
        self.log_callback(message, level)

    def capture_screen_bgr(self) -> np.ndarray:
        """Chụp toàn màn hình chính siêu tốc bằng mss và trả về ảnh BGR cho OpenCV."""
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            sct_img = sct.grab(monitor)
            img_bgra = np.array(sct_img, dtype=np.uint8)
            return cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)

    @staticmethod
    def crop_template_around(
        screenshot_bgr: np.ndarray,
        center_x: int,
        center_y: int,
        width: int = 80,
        height: int = 40,
    ) -> np.ndarray:
        """Cắt vùng ảnh mẫu (mặc định 80x40px) quanh tọa độ tâm (center_x, center_y)."""
        h, w = screenshot_bgr.shape[:2]
        half_w, half_h = width // 2, height // 2
        x1 = max(0, center_x - half_w)
        y1 = max(0, center_y - half_h)
        x2 = min(w, center_x + half_w)
        y2 = min(h, center_y + half_h)
        return screenshot_bgr[y1:y2, x1:x2].copy()

    def get_window_origin(self, window_title: str) -> Tuple[int, int]:
        """
        Tìm cửa sổ theo tiêu đề (khớp một phần, không phân biệt hoa thường)
        và trả về tọa độ góc trái trên (win_x, win_y). Nếu không chỉ định hoặc
        không tìm thấy cửa sổ, trả về (0, 0) kèm cảnh báo.
        """
        if not window_title.strip():
            return (0, 0)

        target_lower = window_title.strip().lower()
        candidates = []
        for win in gw.getAllWindows():
            title = (win.title or "").strip()
            if title and target_lower in title.lower():
                candidates.append(win)

        if not candidates:
            self.log(
                f"Không tìm thấy cửa sổ '{window_title}', sử dụng gốc màn hình (0, 0).",
                "WARN",
            )
            return (0, 0)

        win = candidates[0]
        try:
            if win.isMinimized:
                win.restore()
                time.sleep(0.15)
        except Exception:
            pass

        return (int(win.left), int(win.top))

    def _resolve_template_path(self, rel_or_abs: str) -> Path:
        p = Path(rel_or_abs)
        if not p.is_absolute():
            p = self.base_dir / p
        return p

    def match_template(
        self,
        screenshot_bgr: np.ndarray,
        template_bgr: np.ndarray,
        expected_x: int,
        expected_y: int,
        search_margin: int = 120,
    ) -> Tuple[float, Tuple[int, int]]:
        """
        Dùng cv2.matchTemplate tìm template_bgr ưu tiên quanh vùng (expected_x, expected_y)
        với bán kính search_margin, sau đó fallback tìm trên toàn màn hình nếu độ khớp thấp.
        Trả về (best_confidence, (matched_center_x, matched_center_y)).
        """
        screen_h, screen_w = screenshot_bgr.shape[:2]
        tpl_h, tpl_w = template_bgr.shape[:2]

        if tpl_h > screen_h or tpl_w > screen_w:
            return 0.0, (expected_x, expected_y)

        # 1. Tìm trong vùng lân cận (Local ROI) để tăng tốc độ và tránh nhầm nút giống nhau
        pad_x = max(search_margin, tpl_w)
        pad_y = max(search_margin, tpl_h)
        roi_x1 = max(0, expected_x - pad_x)
        roi_y1 = max(0, expected_y - pad_y)
        roi_x2 = min(screen_w, expected_x + pad_x)
        roi_y2 = min(screen_h, expected_y + pad_y)

        roi = screenshot_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.shape[0] >= tpl_h and roi.shape[1] >= tpl_w:
            res_local = cv2.matchTemplate(roi, template_bgr, cv2.TM_CCOEFF_NORMED)
            _, max_val_local, _, max_loc_local = cv2.minMaxLoc(res_local)
            center_local = (
                roi_x1 + max_loc_local[0] + tpl_w // 2,
                roi_y1 + max_loc_local[1] + tpl_h // 2,
            )
            if max_val_local >= 0.85:
                return float(max_val_local), center_local
        else:
            max_val_local = 0.0
            center_local = (expected_x, expected_y)

        # 2. Fallback tìm toàn màn hình nếu cửa sổ/nút bấm dịch chuyển xa hơn search_margin
        res_global = cv2.matchTemplate(screenshot_bgr, template_bgr, cv2.TM_CCOEFF_NORMED)
        _, max_val_global, _, max_loc_global = cv2.minMaxLoc(res_global)
        center_global = (
            max_loc_global[0] + tpl_w // 2,
            max_loc_global[1] + tpl_h // 2,
        )

        if max_val_global >= max_val_local:
            return float(max_val_global), center_global
        return float(max_val_local), center_local

    def type_unicode_text(self, text: str) -> None:
        """Gõ chuỗi ký tự Unicode (hỗ trợ đầy đủ tiếng Việt có dấu) qua Windows SendInput."""
        if sys.platform != "win32":
            pyautogui.write(text, interval=0.01)
            return

        from ctypes import wintypes

        user32 = ctypes.windll.user32

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_size_t),
            ]

        class DATA(ctypes.Union):
            _fields_ = [
                ("ki", KEYBDINPUT),
                ("padding", ctypes.c_byte * (32 if ctypes.sizeof(ctypes.c_void_p) == 8 else 24)),
            ]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("data", DATA)]

        def send_char(ch: str) -> None:
            raw = ch.encode("utf-16-le")
            for i in range(0, len(raw), 2):
                if self.stop_event.is_set():
                    raise InterruptedError("Đã nhận tín hiệu dừng khẩn cấp.")
                code = int.from_bytes(raw[i : i + 2], "little")
                for flags in (4, 6):  # KEYEVENTF_UNICODE, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP
                    event = INPUT(type=1, data=DATA(ki=KEYBDINPUT(0, code, flags, 0, 0)))
                    user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT))
                time.sleep(0.015)

        normalized = str(text).replace("\r\n", "\n")
        lines = normalized.split("\n")
        for idx, line in enumerate(lines):
            for ch in line:
                if ch == "\t":
                    pyautogui.press("tab")
                else:
                    send_char(ch)
            if idx < len(lines) - 1:
                pyautogui.press("enter")

    def press_hotkey_windows(self, keys: list[str]) -> None:
        """Nhấn tổ hợp phím qua Win32 keybd_event (tương thích hoàn toàn với phím Win, Alt+F2->Win+R, F1..F24)."""
        norm_keys = [str(k).strip().lower() for k in keys if str(k).strip()]
        if not norm_keys:
            return
        if set(norm_keys) == {"alt", "f2"}:
            norm_keys = ["win", "r"]

        if sys.platform != "win32":
            pyautogui.hotkey(*norm_keys)
            return

        aliases = {
            "ctrl": 17,
            "control": 17,
            "alt": 18,
            "shift": 16,
            "super": 91,
            "win": 91,
            "windows": 91,
            "enter": 13,
            "return": 13,
            "tab": 9,
            "esc": 27,
            "escape": 27,
            "space": 32,
            "backspace": 8,
            "delete": 46,
            "left": 37,
            "up": 38,
            "right": 39,
            "down": 40,
            "home": 36,
            "end": 35,
            "pageup": 33,
            "pagedown": 34,
        }
        user32 = ctypes.windll.user32
        codes = []
        for k in norm_keys:
            if k in aliases:
                codes.append(aliases[k])
            elif len(k) == 1 and k.isascii() and k.isalnum():
                codes.append(ord(k.upper()))
            elif re.fullmatch(r"f([1-9]|1[0-9]|2[0-4])", k):
                codes.append(111 + int(k[1:]))
            else:
                pyautogui.hotkey(*norm_keys)
                return

        try:
            for code in codes:
                user32.keybd_event(code, 0, 0, 0)
        finally:
            for code in reversed(codes):
                user32.keybd_event(code, 0, 2, 0)

        if any(k in ("enter", "return", "win", "windows") for k in norm_keys):
            time.sleep(1.2)
        else:
            time.sleep(0.3)

    def execute_gui_action(
        self,
        gui: GUIStepPayload,
        target_x: int,
        target_y: int,
    ) -> None:
        """Thực thi thao tác chuột/bàn phím tại (target_x, target_y)."""
        action = gui.action_type
        if action == "click":
            pyautogui.click(target_x, target_y)
            time.sleep(0.3)
        elif action == "double_click":
            pyautogui.doubleClick(target_x, target_y)
            time.sleep(0.8)
        elif action == "right_click":
            pyautogui.rightClick(target_x, target_y)
            time.sleep(0.3)
        elif action == "type":
            if target_x > 0 or target_y > 0:
                pyautogui.click(target_x, target_y)
                time.sleep(0.08)
            if gui.text_input:
                self.type_unicode_text(gui.text_input)
            time.sleep(0.3)
        elif action == "hotkey":
            if gui.keys:
                self.press_hotkey_windows(gui.keys)
        elif action == "scroll":
            if target_x > 0 or target_y > 0:
                pyautogui.moveTo(target_x, target_y)
            scroll_amount = int(gui.text_input or "-300")
            pyautogui.scroll(scroll_amount)
            time.sleep(0.3)
        else:
            raise ValueError(f"Unsupported GUI action_type: {action}")

    def run_gui_step(self, step: WorkflowStep) -> Tuple[int, int, float]:
        """
        Thực thi 1 bước GUI bằng OpenCV template matching trong khoảng timeout_seconds.
        Nếu thất bại hoặc độ khớp < confidence_threshold, ném UIShiftException.
        """
        assert step.gui is not None
        gui = step.gui

        win_x, win_y = self.get_window_origin(gui.window_title)
        if gui.window_title.strip() and (win_x != 0 or win_y != 0):
            expected_x = win_x + gui.relative_coords[0]
            expected_y = win_y + gui.relative_coords[1]
        else:
            expected_x, expected_y = gui.screen_coords

        # Nếu là hotkey / type / scroll thuần túy không gắn ảnh mẫu
        if not gui.template_image_path and gui.action_type in ("hotkey", "type", "scroll"):
            self.execute_gui_action(gui, expected_x, expected_y)
            return expected_x, expected_y, 1.0

        tpl_path = self._resolve_template_path(gui.template_image_path)
        if not tpl_path.exists():
            screenshot_bgr = self.capture_screen_bgr()
            raise UIShiftException(
                step=step,
                screenshot_bgr=screenshot_bgr,
                best_confidence=0.0,
                expected_screen_coords=(expected_x, expected_y),
                window_origin=(win_x, win_y),
                reason=f"Không tìm thấy file ảnh mẫu: {tpl_path}",
            )

        template_bgr = cv2.imread(str(tpl_path), cv2.IMREAD_COLOR)
        if template_bgr is None or template_bgr.size == 0:
            screenshot_bgr = self.capture_screen_bgr()
            raise UIShiftException(
                step=step,
                screenshot_bgr=screenshot_bgr,
                best_confidence=0.0,
                expected_screen_coords=(expected_x, expected_y),
                window_origin=(win_x, win_y),
                reason=f"File ảnh mẫu bị hỏng hoặc rỗng: {tpl_path}",
            )

        deadline = time.perf_counter() + max(0.2, step.timeout_seconds)
        best_conf = 0.0
        best_coords = (expected_x, expected_y)
        last_screenshot = None

        while time.perf_counter() <= deadline:
            if self.stop_event.is_set():
                raise InterruptedError("Đã nhận tín hiệu dừng khẩn cấp từ người dùng.")

            last_screenshot = self.capture_screen_bgr()
            conf, matched_coords = self.match_template(
                screenshot_bgr=last_screenshot,
                template_bgr=template_bgr,
                expected_x=expected_x,
                expected_y=expected_y,
                search_margin=gui.search_margin,
            )
            if conf > best_conf:
                best_conf = conf
                best_coords = matched_coords

            if conf >= gui.confidence_threshold:
                self.execute_gui_action(gui, matched_coords[0], matched_coords[1])
                return matched_coords[0], matched_coords[1], conf

            time.sleep(0.15)

        if last_screenshot is None:
            last_screenshot = self.capture_screen_bgr()

        raise UIShiftException(
            step=step,
            screenshot_bgr=last_screenshot,
            best_confidence=best_conf,
            expected_screen_coords=(expected_x, expected_y),
            window_origin=(win_x, win_y),
            reason=(
                f"Độ khớp ảnh mẫu ({best_conf:.3f}) thấp hơn ngưỡng "
                f"({gui.confidence_threshold:.2f}) sau {step.timeout_seconds:.1f}s chờ."
            ),
        )

    def run_bash_step(self, step: WorkflowStep) -> str:
        """Thực thi bước lệnh ngầm (Bash/Command hoặc Python script) và kiểm tra điều kiện."""
        assert step.bash is not None
        bash: BashStepPayload = step.bash

        if bash.python_script.strip():
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
                tmp.write(bash.python_script)
                tmp_path = Path(tmp.name)
            try:
                proc = subprocess.run(
                    [sys.executable, str(tmp_path)],
                    capture_output=True,
                    text=True,
                    timeout=bash.timeout_seconds,
                    cwd=bash.working_dir or str(self.base_dir),
                )
            finally:
                tmp_path.unlink(missing_ok=True)
        else:
            if sys.platform == "win32":
                proc = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", bash.command],
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=bash.timeout_seconds,
                    cwd=bash.working_dir or str(self.base_dir),
                )
                if re.search(r"\bStart-Process\b", bash.command, re.IGNORECASE):
                    time.sleep(1.5)
            else:
                proc = subprocess.run(
                    bash.command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=bash.timeout_seconds,
                    cwd=bash.working_dir or str(self.base_dir),
                )

        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        exit_code = proc.returncode

        # Kiểm tra điều kiện thành công
        ok = True
        if bash.verify_condition.strip() == "exit_code == 0":
            ok = exit_code == 0
        else:
            try:
                ok = bool(
                    eval(
                        bash.verify_condition,
                        {"__builtins__": {}},
                        {"exit_code": exit_code, "stdout": stdout, "stderr": stderr},
                    )
                )
            except Exception as exc:
                ok = exit_code == 0
                self.log(f"Cảnh báo biểu thức verify_condition lỗi ({exc}), dùng exit_code == 0.", "WARN")

        if ok and bash.expected_output_contains:
            ok = bash.expected_output_contains in stdout

        if not ok:
            screenshot_bgr = self.capture_screen_bgr()
            raise UIShiftException(
                step=step,
                screenshot_bgr=screenshot_bgr,
                best_confidence=0.0,
                expected_screen_coords=(0, 0),
                window_origin=(0, 0),
                reason=f"Lệnh Bash/Python thất bại (exit_code={exit_code}, stderr={stderr[:200]})",
            )

        return stdout

    def run_workflow_iterations(
        self,
        workflow: WorkflowDocument,
        workflow_path: Path,
        iterations: int,
        healer_callback: Optional[Callable[[UIShiftException, WorkflowDocument, Path], bool]] = None,
        step_status_callback: Optional[Callable[[int, str, str], None]] = None,
    ) -> None:
        """
        Chạy kịch bản `workflow` cho `iterations` vòng lặp.
        - Nếu mọi bước khớp >= confidence_threshold: chạy tốc độ cao không gọi AI.
        - Nếu gặp UIShiftException: gọi `healer_callback` (MageHealer) để tự chữa lành,
          cập nhật workflow.json và tiếp tục ngay tại bước đó.
        """
        for loop_idx in range(1, iterations + 1):
            if self.stop_event.is_set():
                self.log("Đã dừng vòng lặp theo yêu cầu người dùng.", "WARN")
                break

            self.log(f"=== Bắt đầu vòng lặp {loop_idx}/{iterations} ===", "FAST")
            for step in workflow.steps:
                if self.stop_event.is_set():
                    break

                if step_status_callback:
                    step_status_callback(step.step_id, "RUNNING (Fast)", "Đang dò mẫu OpenCV...")

                try:
                    t0 = time.perf_counter()
                    if step.channel == "gui":
                        x, y, conf = self.run_gui_step(step)
                        elapsed_ms = (time.perf_counter() - t0) * 1000
                        msg = (
                            f"[FastRunner] Step #{step.step_id} '{step.description}' -> "
                            f"{step.gui.action_type} tại ({x}, {y}) | conf={conf:.3f} ({elapsed_ms:.0f}ms)"
                        )
                        self.log(msg, "FAST")
                        if step_status_callback:
                            step_status_callback(
                                step.step_id,
                                "OK (Fast)",
                                f"({x}, {y}) conf={conf:.2f} [{elapsed_ms:.0f}ms]",
                            )
                    else:
                        out = self.run_bash_step(step)
                        elapsed_ms = (time.perf_counter() - t0) * 1000
                        self.log(
                            f"[FastRunner] Step #{step.step_id} [BASH] '{step.description}' hoàn tất ({elapsed_ms:.0f}ms)",
                            "FAST",
                        )
                        if step_status_callback:
                            step_status_callback(
                                step.step_id,
                                "OK (Bash)",
                                out[:60] if out else f"exit=0 [{elapsed_ms:.0f}ms]",
                            )

                except UIShiftException as ui_exc:
                    self.log(f"[CẢNH BÁO] {ui_exc}", "WARN")
                    if step_status_callback:
                        step_status_callback(
                            step.step_id,
                            "HEALING (Mage AI)",
                            f"Độ khớp thấp ({ui_exc.best_confidence:.2f}) -> Đang gọi Mage-VL-Cowork...",
                        )

                    if healer_callback is None:
                        raise

                    healed = healer_callback(ui_exc, workflow, workflow_path)
                    if not healed:
                        if step_status_callback:
                            step_status_callback(
                                step.step_id,
                                "FAILED",
                                "Mage-VL-Cowork không thể tự chữa lành bước này.",
                            )
                        raise RuntimeError(
                            f"Dừng quy trình tại Step #{step.step_id}: Mage-VL-Cowork không thể phục hồi."
                        ) from ui_exc

                    if step_status_callback:
                        step_status_callback(
                            step.step_id,
                            "HEALED (Mage AI)",
                            f"Đã cập nhật tọa độ & template mới (healed #{step.healed_count})",
                        )

                if step.wait_after_ms > 0:
                    time.sleep(step.wait_after_ms / 1000.0)

            self.log(f"=== Hoàn thành vòng lặp {loop_idx}/{iterations} ===", "FAST")
