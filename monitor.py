
import argparse
import sys
import cv2
from src.detector import DriverDetector
from src.logger import ResultLogger
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="monitor.py",
        description="CLI Driver Monitoring System — drowsiness & distraction detection via facial geometry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the input video file (e.g., data/sample_drive.mp4).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/driver_log.csv",
        help="Path for the output log file. Use .csv or .json extension.",
    )
    parser.add_argument(
        "--ear-thresh",
        type=float,
        default=0.22,
        help="EAR value below which the driver is flagged as DROWSY.",
    )
    parser.add_argument(
        "--pitch-thresh",
        type=float,
        default=15.0,
        help="Absolute pitch angle (degrees) above which driver is flagged as DISTRACTED.",
    )
    parser.add_argument(
        "--yaw-thresh",
        type=float,
        default=20.0,
        help="Absolute yaw angle (degrees) above which driver is flagged as DISTRACTED.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=30,
        help="Print a progress line every N frames (set 0 to suppress).",
    )
    return parser.parse_args()
def main() -> None:
    args = parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {args.video}", file=sys.stderr)
        sys.exit(1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[INFO] Video     : {args.video}")
    print(f"[INFO] Frames    : {total_frames}  |  FPS: {fps:.1f}")
    print(f"[INFO] EAR thresh: {args.ear_thresh}  |  Pitch: {args.pitch_thresh}°  |  Yaw: {args.yaw_thresh}°")
    print(f"[INFO] Output    : {args.output}\n")
    detector = DriverDetector(
        ear_threshold=args.ear_thresh,
        pitch_threshold=args.pitch_thresh,
        yaw_threshold=args.yaw_thresh,
    )
    frame_count = 0
    with ResultLogger(args.output) as logger:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame_count += 1
            result = detector.process_frame(frame, frame_count)
            logger.log(result)
            if args.progress_interval and frame_count % args.progress_interval == 0:
                pct = (frame_count / total_frames * 100) if total_frames > 0 else 0
                print(
                    f"  Frame {frame_count:05d}/{total_frames} ({pct:5.1f}%) | "
                    f"EAR: {result.avg_ear:.3f} | "
                    f"Pitch: {result.pitch:6.1f}° | "
                    f"Yaw: {result.yaw:6.1f}° | "
                    f"State: {result.status}"
                )
        logger.print_summary()
    cap.release()
    detector.release()
    print(f"[SUCCESS] Results saved to: {args.output}")





    
if __name__ == "__main__":
    main()