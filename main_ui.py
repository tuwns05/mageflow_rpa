from __future__ import annotations

import html
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from pynput import keyboard as pynput_keyboard
from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from fast_runner import FastRunner, UIShiftException
from mage_healer import MageHealer
from workflow_schema import GUIStepPayload, WorkflowDocument, WorkflowStep


class UIBridge(QObject):
    """Cầu nối tín hiệu an toàn giữa các luồng (Worker Threads / Hotkey Thread -> Qt UI Thread)."""

    log_signal = Signal(str, str)
    step_status_signal = Signal(int, str, str)
    workflow_loaded_signal = Signal(object)
    step_added_signal = Signal(object)
    worker_finished_signal = Signal(str)


class MageFlowMainWindow(QMainWindow):
    def __init__(self, base_dir: Path) -> None:
        super().__init__()
        self.base_dir = base_dir
        self.workflow_path = self.base_dir / "workflows" / "workflow.json"
        self.workflow_path.parent.mkdir(parents=True, exist_ok=True)
        (self.base_dir / "assets" / "templates").mkdir(parents=True, exist_ok=True)

        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self.current_workflow: Optional[WorkflowDocument] = None

        self.bridge = UIBridge()
        self.bridge.log_signal.connect(self._append_log)
        self.bridge.step_status_signal.connect(self._update_step_row_status)
        self.bridge.workflow_loaded_signal.connect(self._populate_table)
        self.bridge.step_added_signal.connect(self._append_step_row)
        self.bridge.worker_finished_signal.connect(self._on_worker_finished)

        self.runner = FastRunner(
            base_dir=self.base_dir,
            log_callback=self.emit_log,
            stop_event=self.stop_event,
        )
        self.healer = MageHealer(
            base_dir=self.base_dir,
            runner=self.runner,
            api_base_url="http://127.0.0.1:8000/v1",
            model_name="Mage-VL-Cowork-4B",
            log_callback=self.emit_log,
        )

        self._init_ui()
        self._start_global_hotkey_listener()
        self._load_or_init_sample_workflow()

    def _init_ui(self) -> None:
        self.setWindowTitle("MageFlow RPA — Self-Healing Hybrid Automation (FastRunner + Mage-VL-Cowork 4B)")
        self.resize(1180, 760)
        self.setStyleSheet(
            """
            QMainWindow { background-color: #0f172a; color: #f8fafc; }
            QLabel { color: #e2e8f0; font-size: 13px; }
            QLineEdit, QSpinBox {
                background-color: #1e293b; color: #f8fafc;
                border: 1px solid #334155; border-radius: 6px; padding: 6px 10px;
                font-size: 13px;
            }
            QPushButton {
                border-radius: 6px; padding: 8px 14px; font-weight: bold; font-size: 13px;
            }
            QTableWidget {
                background-color: #1e293b; color: #f8fafc;
                gridline-color: #334155; border: 1px solid #334155; border-radius: 6px;
            }
            QHeaderView::section {
                background-color: #0f172a; color: #94a3b8;
                padding: 6px; border: 1px solid #334155; font-weight: bold;
            }
            QTextEdit {
                background-color: #090d16; color: #e2e8f0;
                border: 1px solid #334155; border-radius: 6px; font-family: Consolas, monospace;
            }
            """
        )

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(10)

        # Hàng 1: Cấu hình Mage-VL-Cowork Endpoint + File kịch bản
        endpoint_row = QHBoxLayout()
        endpoint_row.addWidget(QLabel("Mage-VL API:"))
        self.api_input = QLineEdit("http://127.0.0.1:8000/v1")
        endpoint_row.addWidget(self.api_input, stretch=2)

        endpoint_row.addWidget(QLabel("Model:"))
        self.model_input = QLineEdit("Mage-VL-Cowork-4B")
        endpoint_row.addWidget(self.model_input, stretch=1)

        endpoint_row.addWidget(QLabel("API Key:"))
        self.api_key_input = QLineEdit("EMPTY")
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        endpoint_row.addWidget(self.api_key_input, stretch=1)

        self.btn_open_json = QPushButton("Mở workflow.json")
        self.btn_open_json.setStyleSheet("background-color: #334155; color: #f8fafc;")
        self.btn_open_json.clicked.connect(self._choose_workflow_file)
        endpoint_row.addWidget(self.btn_open_json)
        main_layout.addLayout(endpoint_row)

        # Hàng 2: Ô nhập Prompt công việc & Số vòng lặp
        prompt_row = QHBoxLayout()
        prompt_row.addWidget(QLabel("Yêu cầu công việc (Prompt):"))
        self.prompt_input = QLineEdit(
            "Mở bảng báo cáo doanh thu, bấm nút Xuất Excel và lưu file vào thư mục Báo cáo"
        )
        prompt_row.addWidget(self.prompt_input, stretch=4)

        prompt_row.addWidget(QLabel("Số lần lặp:"))
        self.loop_spin = QSpinBox()
        self.loop_spin.setRange(1, 10000)
        self.loop_spin.setValue(5)
        prompt_row.addWidget(self.loop_spin)
        main_layout.addLayout(prompt_row)

        # Hàng 3: Các nút điều khiển chính
        btn_row = QHBoxLayout()
        self.btn_generate = QPushButton("✨ Tạo quy trình bằng AI (Record Mode)")
        self.btn_generate.setStyleSheet("background-color: #7c3aed; color: white;")
        self.btn_generate.clicked.connect(self.start_ai_generation)
        btn_row.addWidget(self.btn_generate)

        self.btn_fast_run = QPushButton("⚡ Chạy vòng lặp tốc độ cao (Fast Runner + Self-Healing)")
        self.btn_fast_run.setStyleSheet("background-color: #059669; color: white;")
        self.btn_fast_run.clicked.connect(self.start_fast_runner)
        btn_row.addWidget(self.btn_fast_run)

        self.btn_stop = QPushButton("🛑 Tạm dừng / Dừng khẩn cấp (F12)")
        self.btn_stop.setStyleSheet("background-color: #dc2626; color: white;")
        self.btn_stop.clicked.connect(self.trigger_emergency_stop)
        btn_row.addWidget(self.btn_stop)
        main_layout.addLayout(btn_row)

        # Khung giữa: Bảng các bước kịch bản & Khung Log thời gian thực
        splitter = QSplitter(Qt.Orientation.Vertical)

        self.step_table = QTableWidget(0, 7)
        self.step_table.setHorizontalHeaderLabels(
            [
                "ID",
                "Mô tả bước (Description)",
                "Kênh",
                "Thao tác / Lệnh",
                "Tọa độ (Rel / Screen)",
                "Template / Điều kiện",
                "Trạng thái thực thi",
            ]
        )
        header = self.step_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        splitter.addWidget(self.step_table)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Consolas", 10))
        splitter.addWidget(self.log_view)
        splitter.setSizes([360, 260])

        main_layout.addWidget(splitter, stretch=1)

    def _start_global_hotkey_listener(self) -> None:
        """Lắng nghe phím F12 toàn cục (Global Hotkey) để dừng khẩn cấp ngay cả khi cửa sổ khác đang active."""

        def on_press(key):
            if key == pynput_keyboard.Key.f12:
                self.trigger_emergency_stop()

        self._hotkey_listener = pynput_keyboard.Listener(on_press=on_press)
        self._hotkey_listener.daemon = True
        self._hotkey_listener.start()

    def emit_log(self, message: str, level: str = "INFO") -> None:
        self.bridge.log_signal.emit(message, level)

    @Slot(str, str)
    def _append_log(self, message: str, level: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        color_map = {
            "FAST": "#34d399",   # Xanh lá: Fast Runner tốc độ cao
            "MAGE": "#c084fc",   # Tím: Mage-VL-Cowork cứu viện / tạo kịch bản
            "WARN": "#fbbf24",   # Vàng: Cảnh báo lệch giao diện
            "ERROR": "#f87171",  # Đỏ: Lỗi hoặc dừng khẩn cấp
            "INFO": "#93c5fd",   # Xanh dương: Thông tin hệ thống
        }
        color = color_map.get(level, "#e2e8f0")
        safe_msg = html.escape(message)
        line_html = f'<span style="color:#64748b;">[{ts}]</span> <b style="color:{color};">[{level}]</b> <span style="color:{color};">{safe_msg}</span>'
        self.log_view.append(line_html)

    def _load_or_init_sample_workflow(self) -> None:
        if self.workflow_path.exists():
            try:
                doc = WorkflowDocument.load(self.workflow_path)
                self.current_workflow = doc
                self._populate_table(doc)
                self.emit_log(f"Đã tải kịch bản '{doc.name}' ({len(doc.steps)} bước) từ {self.workflow_path}", "INFO")
                return
            except Exception as exc:
                self.emit_log(f"Không thể đọc {self.workflow_path}: {exc}", "WARN")

        # Khởi tạo kịch bản mẫu minh họa cấu trúc
        now_iso = datetime.now().isoformat(timespec="seconds")
        sample = WorkflowDocument(
            name="Báo cáo Doanh thu Tự động",
            goal_prompt=self.prompt_input.text(),
            created_at=now_iso,
            updated_at=now_iso,
            steps=[
                WorkflowStep(
                    step_id=1,
                    description="Bấm nút Xuất Excel trên cửa sổ phần mềm kế toán",
                    channel="gui",
                    gui=GUIStepPayload(
                        action_type="click",
                        window_title="Excel",
                        relative_coords=(180, 95),
                        screen_coords=(380, 195),
                        template_image_path="assets/templates/step_001_tpl.png",
                        confidence_threshold=0.85,
                    ),
                )
            ],
        )
        sample.save(self.workflow_path)
        self.current_workflow = sample
        self._populate_table(sample)
        self.emit_log(f"Đã khởi tạo file kịch bản mẫu tại {self.workflow_path}", "INFO")

    def _choose_workflow_file(self) -> None:
        file_str, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn file kịch bản JSON",
            str(self.workflow_path.parent),
            "JSON Files (*.json)",
        )
        if file_str:
            self.workflow_path = Path(file_str)
            self.current_workflow = WorkflowDocument.load(self.workflow_path)
            self._populate_table(self.current_workflow)
            self.emit_log(f"Đã chuyển sang kịch bản: {self.workflow_path}", "INFO")

    @Slot(object)
    def _populate_table(self, workflow: WorkflowDocument) -> None:
        self.step_table.setRowCount(0)
        for step in workflow.steps:
            self._append_step_row(step)

    @Slot(object)
    def _append_step_row(self, step: WorkflowStep) -> None:
        row = self.step_table.rowCount()
        self.step_table.insertRow(row)

        if step.channel == "gui" and step.gui:
            action_desc = step.gui.action_type
            if step.gui.text_input:
                action_desc += f" ('{step.gui.text_input}')"
            coords_desc = f"Rel:{step.gui.relative_coords} | Scr:{step.gui.screen_coords}"
            tpl_desc = f"{step.gui.template_image_path} (>={step.gui.confidence_threshold:.2f})"
        else:
            cmd = step.bash.command if step.bash and step.bash.command else "<python_script>"
            action_desc = cmd[:35]
            coords_desc = "N/A (Background)"
            tpl_desc = step.bash.verify_condition if step.bash else ""

        values = [
            str(step.step_id),
            step.description,
            step.channel.upper(),
            action_desc,
            coords_desc,
            tpl_desc,
            "READY",
        ]
        for col, val in enumerate(values):
            item = QTableWidgetItem(val)
            item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.step_table.setItem(row, col, item)

    @Slot(int, str, str)
    def _update_step_row_status(self, step_id: int, status: str, detail: str) -> None:
        for row in range(self.step_table.rowCount()):
            id_item = self.step_table.item(row, 0)
            if id_item and int(id_item.text()) == step_id:
                status_item = QTableWidgetItem(f"{status} — {detail}")
                if "Fast" in status or "Bash" in status:
                    status_item.setForeground(QColor("#34d399"))
                elif "Mage" in status:
                    status_item.setForeground(QColor("#c084fc"))
                elif "FAILED" in status:
                    status_item.setForeground(QColor("#f87171"))
                self.step_table.setItem(row, 6, status_item)

                # Cập nhật lại cột tọa độ nếu vừa được Mage chữa lành
                if self.current_workflow:
                    st = self.current_workflow.get_step(step_id)
                    if st and st.channel == "gui" and st.gui:
                        coords_desc = f"Rel:{st.gui.relative_coords} | Scr:{st.gui.screen_coords}"
                        self.step_table.setItem(row, 4, QTableWidgetItem(coords_desc))
                break

    def _sync_healer_config(self) -> None:
        self.healer.api_base_url = self.api_input.text().strip().rstrip("/")
        self.healer.model_name = self.model_input.text().strip()
        self.healer.api_key = self.api_key_input.text().strip() or "EMPTY"

    def _set_controls_running(self, running: bool) -> None:
        self.btn_generate.setEnabled(not running)
        self.btn_fast_run.setEnabled(not running)
        self.btn_open_json.setEnabled(not running)

    def start_ai_generation(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            return

        goal = self.prompt_input.text().strip()
        if not goal:
            QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng nhập Prompt yêu cầu công việc!")
            return

        self._sync_healer_config()
        self.stop_event.clear()
        self.step_table.setRowCount(0)
        self._set_controls_running(True)

        def worker():
            try:
                doc = self.healer.generate_workflow_from_prompt(
                    goal_prompt=goal,
                    workflow_path=self.workflow_path,
                    step_added_callback=lambda step: self.bridge.step_added_signal.emit(step),
                )
                self.current_workflow = doc
                self.bridge.worker_finished_signal.emit("AI Generation hoàn tất!")
            except Exception as exc:
                self.emit_log(f"Lỗi khi tạo quy trình bằng AI: {exc}", "ERROR")
                self.bridge.worker_finished_signal.emit(f"Lỗi: {exc}")

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def start_fast_runner(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            return

        if not self.workflow_path.exists():
            QMessageBox.warning(self, "Chưa có kịch bản", "Không tìm thấy file workflow.json!")
            return

        self._sync_healer_config()
        self.stop_event.clear()
        self.current_workflow = WorkflowDocument.load(self.workflow_path)
        self._populate_table(self.current_workflow)
        iterations = self.loop_spin.value()
        self._set_controls_running(True)

        def on_ui_shift(ui_exc: UIShiftException, wf: WorkflowDocument, wf_path: Path) -> bool:
            return self.healer.heal_ui_shift(ui_exc, wf, wf_path)

        def worker():
            try:
                assert self.current_workflow is not None
                self.runner.run_workflow_iterations(
                    workflow=self.current_workflow,
                    workflow_path=self.workflow_path,
                    iterations=iterations,
                    healer_callback=on_ui_shift,
                    step_status_callback=lambda sid, st, dt: self.bridge.step_status_signal.emit(sid, st, dt),
                )
                self.bridge.worker_finished_signal.emit("Hoàn thành vòng lặp kịch bản!")
            except Exception as exc:
                self.emit_log(f"Vòng lặp dừng do lỗi: {exc}", "ERROR")
                self.bridge.worker_finished_signal.emit(f"Dừng: {exc}")

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def trigger_emergency_stop(self) -> None:
        self.stop_event.set()
        self.emit_log("ĐÃ KÍCH HOẠT TÍN HIỆU DỪNG KHẨN CẤP (F12)!", "ERROR")

    @Slot(str)
    def _on_worker_finished(self, summary: str) -> None:
        self._set_controls_running(False)
        self.emit_log(summary, "INFO")


def main() -> None:
    app = QApplication(sys.argv)
    base_dir = Path(__file__).resolve().parent
    win = MageFlowMainWindow(base_dir=base_dir)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
