import cv2
import numpy as np
import mediapipe as mp
from src.geometry import calculate_ear, get_3d_face_model, build_camera_matrix, normalize_pitch
from src.models import FrameResult
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
POSE_IDX      = [1, 152, 33, 263, 61, 291]

class DriverDetector:
    """ Encapsulates MediaPipe FaceMesh setup and per-frame driver state analysis.
    Args:
        ear_threshold:   EAR value below which the driver is flagged DROWSY.
        pitch_threshold: Absolute pitch angle (degrees) above which DISTRACTED is flagged.
        yaw_threshold:   Absolute yaw angle (degrees) above which DISTRACTED is flagged."""

    def __init__(
        self,
        ear_threshold: float = 0.22,
        pitch_threshold: float = 15.0,
        yaw_threshold: float = 20.0,
    ) -> None:
        self.ear_threshold = ear_threshold
        self.pitch_threshold = pitch_threshold
        self.yaw_threshold = yaw_threshold
        self._face_3d = get_3d_face_model()
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    def process_frame(self, frame: np.ndarray, frame_number: int) -> FrameResult:
        res = FrameResult(frame_number=frame_number)
        image_h, image_w = frame.shape[:2]
        cam_matrix = build_camera_matrix(image_w, image_h)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_res = self._face_mesh.process(image_rgb)
        if not mp_res.multi_face_landmarks:
            res.status = "NO_FACE"
            return res

        res.face_detected = True
        face_landmarks = mp_res.multi_face_landmarks[0]

        # 1. DROWSINESS — Eye Aspect Ratio
        left_eye  = self._extract_landmarks(face_landmarks, LEFT_EYE_IDX,  image_w, image_h)
        right_eye = self._extract_landmarks(face_landmarks, RIGHT_EYE_IDX, image_w, image_h)

        res.avg_ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
        res.status  = "DROWSY" if res.avg_ear < self.ear_threshold else "ALERTT"

        # 2. DISTRACTION — Head Pose via solvePnP
        face_2d = np.array(
            self._extract_landmarks(face_landmarks, POSE_IDX, image_w, image_h),
            dtype=np.float64,
        )

        success_pnp, rot_vec, _ = cv2.solvePnP(
            self._face_3d, face_2d, cam_matrix, self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if success_pnp:
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, *_ = cv2.RQDecomp3x3(rmat)   # returns degrees directly
            res.pitch = normalize_pitch(float(angles[0]))
            res.yaw   = float(angles[1])

            if res.status == "ALERTT":
                if abs(res.pitch) > self.pitch_threshold or abs(res.yaw) > self.yaw_threshold:
                    res.status = "DISTRACTED"

        return res

    def release(self) -> None:
        self._face_mesh.close()

    @staticmethod
    def _extract_landmarks(face_landmarks, indices, image_w, image_h):
        return [
            (face_landmarks.landmark[i].x * image_w,
             face_landmarks.landmark[i].y * image_h)
            for i in indices
        ]