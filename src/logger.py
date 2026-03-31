
import csv
import json
import os
from datetime import datetime
from typing import List
from src.models import FrameResult
CSV_COLUMNS = ["frame_number", "avg_ear", "pitch", "yaw", "status", "face_detected"]
class ResultLogger:
    def __init__(self, output_path: str) -> None:
        self.output_path = output_path
        self._records: List[FrameResult] = []
        self._start_time = datetime.now()

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        self._is_json_only = output_path.lower().endswith(".json")
        self._csv_file   = None
        self._csv_writer = None
    def __enter__(self) -> "ResultLogger":
        if not self._is_json_only:
            self._csv_file   = open(self.output_path, mode="w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=CSV_COLUMNS)
            self._csv_writer.writeheader()
        return self
    def log(self, result: FrameResult) -> None:
        self._records.append(result)
        if self._csv_writer:
            self._csv_writer.writerow({
                "frame_number":  result.frame_number,
                "avg_ear":       round(result.avg_ear, 4),
                "pitch":         round(result.pitch, 2),
                "yaw":           round(result.yaw, 2),
                "status":        result.status,
                "face_detected": result.face_detected,
            })
    def __exit__(self, *_) -> None:
        if self._csv_file:
            self._csv_file.close()
        self._write_json_summary()
    def _compute_summary(self) -> dict:
        total = len(self._records)
        if total == 0:
            return {"total_frames": 0}
        ear_values        = [r.avg_ear for r in self._records if r.face_detected]
        drowsy_frames     = sum(1 for r in self._records if r.status == "DROWSY")
        distracted_frames = sum(1 for r in self._records if r.status == "DISTRACTED")
        no_face_frames    = sum(1 for r in self._records if r.status == "NO_FACE")
        alert_frames      = sum(1 for r in self._records if r.status == "ALERTT")
        return {
            "session_start":        self._start_time.isoformat(),
            "session_end":          datetime.now().isoformat(),
            "total_frames":         total,
            "face_detected_frames": sum(1 for r in self._records if r.face_detected),
            "alert_frames":         alert_frames,
            "drowsy_frames":        drowsy_frames,
            "distracted_frames":    distracted_frames,
            "no_face_frames":       no_face_frames,
            "drowsy_pct":           round(100 * drowsy_frames / total, 2),
            "distracted_pct":       round(100 * distracted_frames / total, 2),
            "avg_ear_overall":      round(float(sum(ear_values) / len(ear_values)), 4) if ear_values else None,
            "min_ear":              round(float(min(ear_values)), 4) if ear_values else None,
        }
    def _write_json_summary(self) -> None:
        if self._is_json_only:
            json_path = self.output_path
        else:
            base, _ = os.path.splitext(self.output_path)
            json_path = base + ".json"

        payload = {
            "summary": self._compute_summary(),
            "frames": [
                {
                    "frame":         r.frame_number,
                    "ear":           round(r.avg_ear, 4),
                    "pitch":         round(r.pitch, 2),
                    "yaw":           round(r.yaw, 2),
                    "status":        r.status,
                    "face_detected": r.face_detected,
                }
                for r in self._records
            ],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    def print_summary(self) -> None:
        s = self._compute_summary()
        print("\n" + "=" * 50)
        print("  SESSION SUMMARY")
        print("=" * 50)
        print(f"  Total frames processed : {s.get('total_frames', 0)}")
        print(f"  Frames with face       : {s.get('face_detected_frames', 0)}")
        print(f"  ALERT frames           : {s.get('alert_frames', 0)}")
        print(f"  DROWSY frames          : {s.get('drowsy_frames', 0)}  ({s.get('drowsy_pct', 0):.1f}%)")
        print(f"  DISTRACTED frames      : {s.get('distracted_frames', 0)}  ({s.get('distracted_pct', 0):.1f}%)")
        if s.get("avg_ear_overall") is not None:
            print(f"  Avg EAR (face frames)  : {s['avg_ear_overall']:.4f}")
            print(f"  Min EAR observed       : {s['min_ear']:.4f}")
        print("=" * 50 + "\n")