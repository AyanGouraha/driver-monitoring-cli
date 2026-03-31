# * *Driver Monitoring System (CLI)**

A command-line tool that analyses dashcam or recorded video to detect drowsiness and distraction in a driver using facial geometry (no GUI required).  

## **Technique:-**  
 Eye Aspect Ratio (EAR) --> Detects Drowsiness  
 Head Pose via solvePnP --> Distraction

 ## Requirements
 Python  ≥ 3.9   
 opencv-python  ≥ 4.8.0   
 mediapipe  ≥ 0.10.0   
 numpy  ≥ 1.24.0   

 ## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/driver-monitoring-cli.git
cd driver-monitoring-cli
```

### 2. Create and activate a virtual environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activat
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```
## Project Structure

```
driver-monitoring-cli/
├── data/
│   └── sample_drive.mp4      # Test video
├── results/                  # Output logs are written here (auto-created)
├── src/
│   ├── __init__.py
│   ├── geometry.py           # EAR calculation, 3D face model, camera matrix
│   ├── detector.py           # MediaPipe init and per-frame processing
│   └── logger.py             # CSV and JSON result logging
├── monitor.py                # Main CLI entrypoint
├── requirements.txt
└── README.md
```
### Basic run (defaults)

```bash
python monitor.py --video data/sample_drive.mp4
```

## Output Format

### CSV (`driver_log.csv`)

| Column | Description |
|---|---|
| `frame_number` | 1-based frame index |
| `avg_ear` | Average Eye Aspect Ratio (both eyes) |
| `pitch` | Head pitch angle in degrees |
| `yaw` | Head yaw angle in degrees |
| `status` | `ALERT`, `DROWSY`, `DISTRACTED`, or `NO_FACE` |
| `face_detected` | `True` / `False` |

### JSON (`driver_log.json`)

A companion `.json` file is always written alongside the CSV containing:
- **`summary`** — aggregate stats: drowsy%, distracted%, avg EAR, min EAR, frame counts
- **`frames`** — full per-frame records

---

## Status Logic
`ALERT` -> Face detected, EAR ≥ threshold, head pose within limits   
 `DROWSY`-> EAR < `--ear-thresh` (takes priority over DISTRACTED)   
`DISTRACTED` -> Pitch > `--pitch-thresh` OR Yaw > `--yaw-thresh` (only if not DROWSY)   
`NO_FACE` -> MediaPipe could not detect a face in the frame   

## How It Works

### Drowsiness — Eye Aspect Ratio (EAR)

EAR measures the ratio of the eye's height to its width using six MediaPipe landmark points. When a driver's eyes close or droop, EAR drops below the threshold.

```
EAR = (||p2−p6|| + ||p3−p5||) / (2 × ||p1−p4||)
```

### Distraction — Head Pose Estimation

Six 2D facial landmarks are matched against a generic 3D face model using `cv2.solvePnP`. The resulting rotation vector is decomposed into pitch (vertical tilt) and yaw (horizontal turn) via `cv2.RQDecomp3x3`, which returns angles directly in degrees.

---
