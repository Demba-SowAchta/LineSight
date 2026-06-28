"""
Live webcam demo -- the most impressive thing to show in a 10-minute presentation.

It opens your webcam, runs every frame through the SAME pipeline as the rest of the
system, and overlays the verdict live. Hold a good part, then a defective one, and
the banner flips PASS/FAIL in real time.

RUN:  python -m scripts.webcam_demo
      press 'q' to quit.

Needs OpenCV:  pip install opencv-python
"""

from __future__ import annotations

import cv2

from src.agents import Orchestrator
from src.utils.images import overlay_heatmap


def main(camera_index: int = 0) -> None:
    orch = Orchestrator(model_version="webcam-demo")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam.")

    print("Webcam demo running. Press 'q' to quit.")
    frame_no = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_no += 1

            # Convert BGR (OpenCV) -> RGB (our pipeline), inspect, then draw.
            rgb = frame_bgr[:, :, ::-1].copy()
            result = orch.inspect_one(f"frame-{frame_no:06d}", rgb)

            display = frame_bgr
            if result.get("heatmap") is not None:
                overlay_rgb = overlay_heatmap(rgb, result["heatmap"])
                display = overlay_rgb[:, :, ::-1].copy()  # back to BGR for cv2

            color = (0, 0, 255) if result["verdict"] == "FAIL" else (0, 180, 0)
            label = f"{result['verdict']}  score={result['score']:.2f}"
            cv2.rectangle(display, (0, 0), (display.shape[1], 40), color, -1)
            cv2.putText(display, label, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.imshow("Industrial Vision Platform - live", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
