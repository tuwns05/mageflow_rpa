from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple


ChannelType = Literal["gui", "bash"]
GUIActionType = Literal["click", "double_click", "right_click", "type", "hotkey", "scroll"]


@dataclass
class GUIStepPayload:
    action_type: GUIActionType
    window_title: str = ""
    relative_coords: Tuple[int, int] = (0, 0)
    screen_coords: Tuple[int, int] = (0, 0)
    template_image_path: str = ""
    confidence_threshold: float = 0.85
    text_input: Optional[str] = None
    keys: List[str] = field(default_factory=list)
    search_margin: int = 120  # Bán kính vùng tìm kiếm quanh tọa độ dự kiến (px)

    @classmethod
    def from_dict(cls, data: Dict) -> "GUIStepPayload":
        return cls(
            action_type=data.get("action_type", "click"),
            window_title=data.get("window_title", ""),
            relative_coords=tuple(data.get("relative_coords", (0, 0))),  # type: ignore
            screen_coords=tuple(data.get("screen_coords", (0, 0))),  # type: ignore
            template_image_path=data.get("template_image_path", ""),
            confidence_threshold=float(data.get("confidence_threshold", 0.85)),
            text_input=data.get("text_input"),
            keys=list(data.get("keys", [])),
            search_margin=int(data.get("search_margin", 120)),
        )


@dataclass
class BashStepPayload:
    command: str = ""
    python_script: str = ""
    verify_condition: str = "exit_code == 0"
    expected_output_contains: Optional[str] = None
    timeout_seconds: int = 30
    working_dir: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "BashStepPayload":
        return cls(
            command=data.get("command", ""),
            python_script=data.get("python_script", ""),
            verify_condition=data.get("verify_condition", "exit_code == 0"),
            expected_output_contains=data.get("expected_output_contains"),
            timeout_seconds=int(data.get("timeout_seconds", 30)),
            working_dir=data.get("working_dir"),
        )


@dataclass
class WorkflowStep:
    step_id: int
    description: str
    channel: ChannelType
    gui: Optional[GUIStepPayload] = None
    bash: Optional[BashStepPayload] = None
    wait_after_ms: int = 300
    timeout_seconds: float = 5.0
    healed_count: int = 0
    last_healed_at: Optional[str] = None

    def validate(self) -> None:
        if self.channel == "gui" and self.gui is None:
            raise ValueError(f"Step #{self.step_id}: channel='gui' yêu cầu trường 'gui'.")
        if self.channel == "bash" and self.bash is None:
            raise ValueError(f"Step #{self.step_id}: channel='bash' yêu cầu trường 'bash'.")

    def to_dict(self) -> Dict:
        data = asdict(self)
        if self.channel == "gui":
            data.pop("bash", None)
        elif self.channel == "bash":
            data.pop("gui", None)
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "WorkflowStep":
        channel: ChannelType = data["channel"]
        gui_payload = GUIStepPayload.from_dict(data["gui"]) if channel == "gui" and data.get("gui") else None
        bash_payload = BashStepPayload.from_dict(data["bash"]) if channel == "bash" and data.get("bash") else None
        step = cls(
            step_id=int(data["step_id"]),
            description=str(data.get("description", "")),
            channel=channel,
            gui=gui_payload,
            bash=bash_payload,
            wait_after_ms=int(data.get("wait_after_ms", 300)),
            timeout_seconds=float(data.get("timeout_seconds", 5.0)),
            healed_count=int(data.get("healed_count", 0)),
            last_healed_at=data.get("last_healed_at"),
        )
        step.validate()
        return step


@dataclass
class WorkflowDocument:
    name: str
    goal_prompt: str
    created_at: str
    updated_at: str
    steps: List[WorkflowStep] = field(default_factory=list)
    version: str = "1.0.0"

    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "name": self.name,
            "goal_prompt": self.goal_prompt,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "steps": [step.to_dict() for step in self.steps],
        }

    def save(self, file_path: str | Path) -> Path:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, file_path: str | Path) -> "WorkflowDocument":
        path = Path(file_path)
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        steps = [WorkflowStep.from_dict(item) for item in raw.get("steps", [])]
        return cls(
            version=str(raw.get("version", "1.0.0")),
            name=str(raw.get("name", "Untitled Workflow")),
            goal_prompt=str(raw.get("goal_prompt", "")),
            created_at=str(raw.get("created_at", "")),
            updated_at=str(raw.get("updated_at", "")),
            steps=steps,
        )

    def get_step(self, step_id: int) -> Optional[WorkflowStep]:
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None
