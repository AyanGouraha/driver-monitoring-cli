from dataclasses import dataclass







@dataclass
class FrameResult:
    frame_number: int
    avg_ear: float = 0.0
    pitch: float = 0.0          # degrees; + = looking down
    yaw: float = 0.0            # degrees; + = turning right
    status: str = "NO_FACE"
    face_detected: bool = False