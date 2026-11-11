from __future__ import annotations

import base64
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
from urllib import error, request

import cv2
import numpy as np
import pygetwindow as gw

from fast_runner import FastRunner, UIShiftException
from workflow_schema import BashStepPayload, GUIStepPayload, WorkflowDocument, WorkflowStep


SELF_HEALING_SYSTEM_PROMPT = """You are Mage-VL-Cowork acting as a Self-Healing Recovery Engine for a Desktop RPA system.

CONTEXT:
The deterministic automation runner paused at Step #{step_id} because the target UI element could not be found at its expected coordinates ({expected_x}, {expected_y}) within window "{window_title}".
- Target Step Description: "{description}"
- Intended Action Type: "{action_type}"
- Best OpenCV Template Match Confidence: {best_confidence:.3f} (Required >= {confidence_threshold:.2f})
- Screen Resolution: {screen_width}x{screen_height}

YOUR OBJECTIVES:
1. Inspect the attached current screenshot carefully.
2. Check if an unexpected modal, notification, alert, or pop-up dialog is obstructing the target UI element.
   - If YES: Set "status" to "POPUP_BLOCKED" and return the exact coordinates (x, y) or hotkey needed to dismiss/close the pop-up first.
3. If NO pop-up is blocking, locate the new position of the target UI element described in "{description}".
   - If FOUND: Set "status" to "RELOCATED" and provide the new pixel coordinates (new_x, new_y) at the center of the target UI element so the RPA engine can execute the action and re-crop a fresh 80x40 template image.
4. If the target element is completely absent and cannot be recovered on this screen, set "status" to "UNRECOVERABLE" with a clear diagnosis.

OUTPUT FORMAT:
Respond ONLY with a valid JSON object matching this exact schema (no markdown explanation outside JSON):
{{
  "status": "RELOCATED" | "POPUP_BLOCKED" | "UNRECOVERABLE",
  "reasoning": "Brief explanation of what shifted or what pop-up is blocking",
  "popup_dismiss": {{
    "action_type": "click" | "hotkey",
    "x": 0,
    "y": 0,
    "keys": ["esc"]
  }},
  "healed_action": {{
    "channel": "gui" | "bash",
    "action_type": "click" | "double_click" | "right_click" | "type" | "hotkey",
    "new_x": 0,
    "new_y": 0,
    "coord_space": "pixel" | "normalized_1000",
    "text_input": null,
    "keys": []
  }}
}}"""


RECORD_WORKFLOW_SYSTEM_PROMPT = """You are Mage-VL-Cowork acting as an Autonomous Computer-Use Workflow Generator for MageFlow RPA.

USER GOAL:
"{goal_prompt}"

CURRENT SCREEN RESOLUTION: {screen_width}x{screen_height}
COMPLETED STEPS SO FAR:
{history_summary}

YOUR TASK:
Analyze the current screenshot and decide the SINGLE NEXT STEP required to achieve the user's goal.
- Prefer "gui" channel when interacting with visual application buttons, input fields, or menus.
- Prefer "bash" channel when background file manipulation, CLI commands, or Python data processing is faster and more reliable.
- When the entire user goal is completely achieved on screen, set "is_finished" to true.

OUTPUT FORMAT:
Respond ONLY with a valid JSON object matching this exact schema:
{{
  "is_finished": false,
  "step_description": "Clear natural-language description of this step (e.g., 'Bấm nút Xuất Excel')",
  "channel": "gui" | "bash",
  "window_title": "Title of the target application window (if gui)",
  "gui": {{
    "action_type": "click" | "double_click" | "right_click" | "type" | "hotkey" | "scroll",
    "x": 0,
    "y": 0,
    "coord_space": "pixel" | "normalized_1000",
    "text_input": null,
    "keys": []
  }},
  "bash": {{
    "command": "",
    "python_script": "",
    "verify_condition": "exit_code == 0"
  }}
}}"""


class MageHealer:
    """
    Bộ não Cứu viện & Tự chữa lành (Module 3):
    Giao tiếp với mô hình Mage-VL-Cowork (4B) qua chuẩn OpenAI-compatible Vision API
    (SGLang / vLLM server cục bộ) cho cả 2 chế độ:
      1. Record/Generate Mode (Học và sinh workflow.json + cắt ảnh mẫu 80x40px).
      2. Self-Healing Mode (Xử lý UIShiftException, tắt pop-up, cập nhật tọa độ & template mới).
    """

    def __init__(
        self,
        base_dir: str | Path,
        runner: FastRunner,
        api_base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "EMPTY",
        model_name: str = "Mage-VL-Cowork-4B",
        log_callback: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.templates_dir = self.base_dir / "assets" / "templates"
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.runner = runner
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.log_callback = log_callback or (lambda msg, level="MAGE": print(f"[{level}] {msg}"))

    def log(self, message: str, level: str = "MAGE") -> None:
        self.log_callback(message, level)

    @staticmethod
    def encode_bgr_to_base64_png(image_bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode(".png", image_bgr)
        if not ok:
            raise RuntimeError("Không thể mã hóa ảnh chụp màn hình sang định dạng PNG.")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    @staticmethod
    def extract_json_object(raw_text: str) -> Dict:
        """Trích xuất JSON an toàn từ phản hồi của mô hình VLM (kể cả khi bọc trong ```json ... ```)."""
        cleaned = raw_text.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
        if fenced:
            cleaned = fenced.group(1)
        else:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                cleaned = cleaned[start : end + 1]
        return json.loads(cleaned)

    @staticmethod
    def normalize_coords_to_pixel(
        x: float,
        y: float,
        screen_w: int,
        screen_h: int,
        coord_space: str = "pixel",
    ) -> Tuple[int, int]:
        """Chuyển đổi tọa độ về pixel thực tế nếu VLM trả về hệ quy chiếu chuẩn hóa 0..1000."""
        if coord_space == "normalized_1000":
            px = int(round((x / 1000.0) * screen_w))
            py = int(round((y / 1000.0) * screen_h))
        else:
            px, py = int(round(x)), int(round(y))
        px = max(0, min(screen_w - 1, px))
        py = max(0, min(screen_h - 1, py))
        return px, py

    def call_mage_vl(
        self,
        system_prompt: str,
        user_prompt: str,
        screenshot_bgr: np.ndarray,
        temperature: float = 0.1,
    ) -> Dict:
        """Gửi request Multimodal (Image + Prompt) tới SGLang/vLLM OpenAI-compatible server."""
        b64_img = self.encode_bgr_to_base64_png(screenshot_bgr)
        endpoint = f"{self.api_base_url}/chat/completions"

        payload = {
            "model": self.model_name,
            "temperature": temperature,
            "max_tokens": 768,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
        }

        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=45) as resp:
                resp_body = resp.read().decode("utf-8")
                data = json.loads(resp_body)
                content = data["choices"][0]["message"]["content"]
                return self.extract_json_object(content)
        except error.URLError as exc:
            raise ConnectionError(
                f"Không thể kết nối tới Mage-VL-Cowork server tại {endpoint}: {exc}"
            ) from exc

    def save_step_template(
        self,
        screenshot_bgr: np.ndarray,
        step_id: int,
        center_x: int,
        center_y: int,
        width: int = 80,
        height: int = 40,
    ) -> str:
        """
        Cắt vùng ảnh 80x40px quanh (center_x, center_y) và lưu vào `assets/templates/`.
        Trả về đường dẫn tương đối để lưu vào `workflow.json`.
        """
        crop = self.runner.crop_template_around(
            screenshot_bgr=screenshot_bgr,
            center_x=center_x,
            center_y=center_y,
            width=width,
            height=height,
        )
        filename = f"step_{step_id:03d}_tpl.png"
        full_path = self.templates_dir / filename
        cv2.imwrite(str(full_path), crop)
        return f"assets/templates/{filename}"

    @staticmethod
    def detect_active_window_info(target_title_hint: str = "") -> Tuple[str, Tuple[int, int]]:
        """Lấy tiêu đề và tọa độ gốc (win_x, win_y) của cửa sổ đích hoặc cửa sổ đang active."""
        if target_title_hint.strip():
            hint_lower = target_title_hint.strip().lower()
            for w in gw.getAllWindows():
                if w.title and hint_lower in w.title.lower():
                    return w.title.strip(), (int(w.left), int(w.top))

        active = gw.getActiveWindow()
        if active and active.title:
            return active.title.strip(), (int(active.left), int(active.top))
        return "", (0, 0)

    # =========================================================================
    # NHIỆM VỤ 1: CHẾ ĐỘ HỌC / TẠO QUY TRÌNH TỰ ĐỘNG (RECORD / GENERATE MODE)
    # =========================================================================
    def generate_workflow_from_prompt(
        self,
        goal_prompt: str,
        workflow_path: Path,
        max_steps: int = 15,
        step_added_callback: Optional[Callable[[WorkflowStep], None]] = None,
    ) -> WorkflowDocument:
        """
        Nhận yêu cầu ban đầu từ người dùng -> Chụp màn hình -> Gọi Mage-VL-Cowork
        thực hiện từng bước (gui hoặc bash).
        Ở mỗi bước GUI tại (x, y), cắt vùng ảnh 80x40px TRƯỚC khi click lưu thành .png
        và ghi toàn bộ quy trình vào `workflow.json`.
        """
        now_iso = datetime.now().isoformat(timespec="seconds")
        doc = WorkflowDocument(
            name=f"AutoFlow - {goal_prompt[:40]}",
            goal_prompt=goal_prompt,
            created_at=now_iso,
            updated_at=now_iso,
            steps=[],
        )

        self.log(f"[RecordMode] Bắt đầu tạo quy trình với Mage-VL-Cowork: '{goal_prompt}'", "MAGE")
        history_lines = []

        for step_idx in range(1, max_steps + 1):
            if self.runner.stop_event.is_set():
                self.log("[RecordMode] Đã dừng quá trình học theo yêu cầu người dùng.", "WARN")
                break

            screenshot_bgr = self.runner.capture_screen_bgr()
            screen_h, screen_w = screenshot_bgr.shape[:2]

            sys_prompt = RECORD_WORKFLOW_SYSTEM_PROMPT.format(
                goal_prompt=goal_prompt,
                screen_width=screen_w,
                screen_height=screen_h,
                history_summary="\n".join(history_lines) if history_lines else "None (First step)",
            )
            user_prompt = (
                f"Determine step #{step_idx} for goal: '{goal_prompt}'. "
                "Return JSON only."
            )

            self.log(f"[RecordMode] Đang hỏi Mage-VL-Cowork cho bước #{step_idx}...", "MAGE")
            decision = self.call_mage_vl(
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                screenshot_bgr=screenshot_bgr,
            )

            if decision.get("is_finished"):
                self.log("[RecordMode] Mage-VL-Cowork xác nhận đã hoàn thành mục tiêu!", "MAGE")
                break

            channel = decision.get("channel", "gui")
            description = decision.get("step_description", f"Bước tự động #{step_idx}")

            if channel == "gui":
                gui_data = decision.get("gui", {})
                action_type = gui_data.get("action_type", "click")
                coord_space = gui_data.get("coord_space", "pixel")
                px, py = self.normalize_coords_to_pixel(
                    float(gui_data.get("x", 0)),
                    float(gui_data.get("y", 0)),
                    screen_w,
                    screen_h,
                    coord_space=coord_space,
                )

                win_title_hint = decision.get("window_title", "")
                win_title, (win_x, win_y) = self.detect_active_window_info(win_title_hint)
                rel_x, rel_y = px - win_x, py - win_y

                # Cắt ảnh mẫu 80x40px TRƯỚC khi thao tác làm thay đổi giao diện
                tpl_rel_path = ""
                if action_type != "hotkey":
                    tpl_rel_path = self.save_step_template(
                        screenshot_bgr=screenshot_bgr,
                        step_id=step_idx,
                        center_x=px,
                        center_y=py,
                        width=80,
                        height=40,
                    )

                gui_payload = GUIStepPayload(
                    action_type=action_type,
                    window_title=win_title,
                    relative_coords=(rel_x, rel_y),
                    screen_coords=(px, py),
                    template_image_path=tpl_rel_path,
                    confidence_threshold=0.85,
                    text_input=gui_data.get("text_input"),
                    keys=list(gui_data.get("keys") or []),
                )
                step = WorkflowStep(
                    step_id=step_idx,
                    description=description,
                    channel="gui",
                    gui=gui_payload,
                )

                # Thực thi bước vừa học trên màn hình thật
                self.runner.execute_gui_action(gui_payload, px, py)
                self.log(
                    f"[RecordMode] Đã ghi & thực thi Step #{step_idx} [GUI]: {description} tại ({px}, {py}) -> {tpl_rel_path}",
                    "MAGE",
                )
            else:
                bash_data = decision.get("bash", {})
                bash_payload = BashStepPayload(
                    command=bash_data.get("command", ""),
                    python_script=bash_data.get("python_script", ""),
                    verify_condition=bash_data.get("verify_condition", "exit_code == 0"),
                )
                step = WorkflowStep(
                    step_id=step_idx,
                    description=description,
                    channel="bash",
                    bash=bash_payload,
                )
                self.runner.run_bash_step(step)
                self.log(f"[RecordMode] Đã ghi & thực thi Step #{step_idx} [BASH]: {description}", "MAGE")

            doc.steps.append(step)
            doc.updated_at = datetime.now().isoformat(timespec="seconds")
            doc.save(workflow_path)
            history_lines.append(f"Step #{step_idx} ({channel}): {description}")

            if step_added_callback:
                step_added_callback(step)

            time.sleep(0.5)

        doc.save(workflow_path)
        self.log(f"[RecordMode] Đã lưu kịch bản gồm {len(doc.steps)} bước vào {workflow_path}", "MAGE")
        return doc

    # =========================================================================
    # NHIỆM VỤ 2: CHẾ ĐỘ TỰ CHỮA LÀNH (SELF-HEALING MODE)
    # =========================================================================
    def heal_ui_shift(
        self,
        ui_exc: UIShiftException,
        workflow: WorkflowDocument,
        workflow_path: Path,
        max_popup_retries: int = 2,
    ) -> bool:
        """
        Xử lý ngoại lệ `UIShiftException` từ `FastRunner`:
        1. Gửi ảnh chụp màn hình hiện tại + thông tin bước bị kẹt + SELF_HEALING_SYSTEM_PROMPT cho Mage-VL-Cowork.
        2. Nếu phát hiện Pop-up che khuất (`POPUP_BLOCKED`): tự động tắt Pop-up trước, chụp lại màn hình.
        3. Khi tìm được tọa độ mới (`RELOCATED`) tại `(new_x, new_y)`:
           - Cắt lại ảnh template 80x40px mới quanh `(new_x, new_y)` và ghi đè `template_image_path`.
           - Thực thi thao tác chuột/bàn phím tại `(new_x, new_y)`.
           - Cập nhật `relative_coords` và `screen_coords` mới vào `workflow.json` để các vòng lặp sau
             chạy bằng FastRunner tốc độ cao mà không cần gọi lại AI.
        """
        step = ui_exc.step
        screenshot_bgr = ui_exc.screenshot_bgr

        for attempt in range(1, max_popup_retries + 2):
            screen_h, screen_w = screenshot_bgr.shape[:2]
            expected_x, expected_y = ui_exc.expected_screen_coords
            win_title = step.gui.window_title if step.gui else ""
            action_type = step.gui.action_type if step.gui else "bash"
            conf_thresh = step.gui.confidence_threshold if step.gui else 0.85

            sys_prompt = SELF_HEALING_SYSTEM_PROMPT.format(
                step_id=step.step_id,
                expected_x=expected_x,
                expected_y=expected_y,
                window_title=win_title or "Desktop",
                description=step.description,
                action_type=action_type,
                best_confidence=ui_exc.best_confidence,
                confidence_threshold=conf_thresh,
                screen_width=screen_w,
                screen_height=screen_h,
            )

            user_prompt = (
                f"Step #{step.step_id} ('{step.description}') failed OpenCV matching "
                f"(best_confidence={ui_exc.best_confidence:.3f}). "
                "Inspect the screenshot, handle any blocking popup or locate the new coordinates, and return JSON."
            )

            self.log(
                f"[Self-Healing] Đang gửi ảnh màn hình Step #{step.step_id} tới Mage-VL-Cowork (lần thử {attempt})...",
                "MAGE",
            )
            result = self.call_mage_vl(
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                screenshot_bgr=screenshot_bgr,
            )

            status = result.get("status", "UNRECOVERABLE")
            reasoning = result.get("reasoning", "")
            self.log(f"[Self-Healing] Phản hồi Mage-VL-Cowork: status={status} | {reasoning}", "MAGE")

            # Trường hợp 1: Có Pop-up che khuất -> Tắt Pop-up trước rồi chụp lại màn hình
            if status == "POPUP_BLOCKED":
                popup_info = result.get("popup_dismiss", {})
                p_action = popup_info.get("action_type", "click")
                if p_action == "hotkey":
                    keys = popup_info.get("keys") or ["esc"]
                    self.log(f"[Self-Healing] Đóng Pop-up bằng phím tắt: {keys}", "MAGE")
                    dummy_gui = GUIStepPayload(action_type="hotkey", keys=keys)
                    self.runner.execute_gui_action(dummy_gui, 0, 0)
                else:
                    coord_space = result.get("healed_action", {}).get("coord_space", "pixel")
                    px, py = self.normalize_coords_to_pixel(
                        float(popup_info.get("x", 0)),
                        float(popup_info.get("y", 0)),
                        screen_w,
                        screen_h,
                        coord_space=coord_space,
                    )
                    self.log(f"[Self-Healing] Đóng Pop-up bằng click tại ({px}, {py})", "MAGE")
                    dummy_gui = GUIStepPayload(action_type="click")
                    self.runner.execute_gui_action(dummy_gui, px, py)

                time.sleep(0.45)
                screenshot_bgr = self.runner.capture_screen_bgr()
                continue

            # Trường hợp 2: Đã định vị được tọa độ mới của nút bấm (RELOCATED)
            if status == "RELOCATED":
                healed = result.get("healed_action", {})
                coord_space = healed.get("coord_space", "pixel")
                new_x, new_y = self.normalize_coords_to_pixel(
                    float(healed.get("new_x", expected_x)),
                    float(healed.get("new_y", expected_y)),
                    screen_w,
                    screen_h,
                    coord_space=coord_space,
                )

                if step.channel == "gui" and step.gui is not None:
                    win_x, win_y = self.runner.get_window_origin(step.gui.window_title)
                    new_rel_x = new_x - win_x
                    new_rel_y = new_y - win_y

                    # 1. Cắt lại ảnh template 80x40px mới TRƯỚC khi click làm đổi trạng thái nút
                    new_tpl_path = self.save_step_template(
                        screenshot_bgr=screenshot_bgr,
                        step_id=step.step_id,
                        center_x=new_x,
                        center_y=new_y,
                        width=80,
                        height=40,
                    )

                    # 2. Thực thi thao tác tại tọa độ mới
                    self.runner.execute_gui_action(step.gui, new_x, new_y)

                    # 3. Cập nhật lại relative_coords, screen_coords và template_image_path vào workflow.json
                    step.gui.relative_coords = (new_rel_x, new_rel_y)
                    step.gui.screen_coords = (new_x, new_y)
                    step.gui.template_image_path = new_tpl_path
                    step.healed_count += 1
                    step.last_healed_at = datetime.now().isoformat(timespec="seconds")

                    workflow.updated_at = step.last_healed_at
                    workflow.save(workflow_path)

                    self.log(
                        f"[Self-Healing] ĐÃ CHỮA LÀNH Step #{step.step_id}: "
                        f"tọa độ mới screen=({new_x}, {new_y}), rel=({new_rel_x}, {new_rel_y}), "
                        f"đã ghi đè template '{new_tpl_path}' và lưu workflow.json!",
                        "MAGE",
                    )
                    return True

            # Trường hợp 3: Không thể phục hồi
            self.log(
                f"[Self-Healing] Không thể phục hồi Step #{step.step_id}: {reasoning}",
                "ERROR",
            )
            return False

        return False
