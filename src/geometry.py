"""The EAR is defined as:
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)"""

import numpy as np
def calculate_ear(eye_landmarks):
    p = [np.array(pt) for pt in eye_landmarks]
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3])
    ear = (v1 + v2) / (2.0 * h)
    return ear





def get_3d_face_model():
    return np.array([
        (0.0,    0.0,    0.0),      # Nose tip
        (0.0,   -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),    # Left eye, outer corner
        (225.0,  170.0, -135.0),    # Right eye, outer corner
        (-150.0, -150.0, -125.0),   # Left mouth corner
        (150.0,  -150.0, -125.0),   # Right mouth corner
    ], dtype=np.float64)





def build_camera_matrix(image_w, image_h):
    focal_length = float(image_w)
    cx = image_w / 2.0
    cy = image_h / 2.0
    return np.array([
        [focal_length, 0.0,          cx],
        [0.0,          focal_length, cy],
        [0.0,          0.0,          1.0],
    ], dtype=np.float64)




def rotation_vector_to_euler(rot_vec):
    import cv2
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = angles[0], angles[1], angles[2]
    return pitch, yaw, roll




def normalize_pitch(pitch: float) -> float:
    if pitch > 90:
        return pitch - 180.0
    elif pitch < -90:
        return pitch + 180.0
    return pitch