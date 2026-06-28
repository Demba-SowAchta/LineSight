"""
AGENT 1 of 6 -- AcquisitionAgent

ROLE:   Get images INTO the system. It is the only agent that knows where pixels
        come from. Everything downstream receives a clean numpy RGB array and does
        not care whether it came from a webcam, a folder, or an industrial camera.

WHY A SEPARATE AGENT: the source changes between development (a folder of test
images), the demo (a file upload or webcam), and production (a GigE industrial
camera triggered by a PLC). By isolating "where pixels come from" in one agent,
the rest of the pipeline never changes when the source changes. To switch source
you swap this ONE agent -- see `from_folder`, `from_webcam`, `from_array`.

INTERACTS WITH: feeds the Orchestrator, which passes frames to the InferenceAgent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from src.utils.images import load_image


class AcquisitionAgent:
    name = "acquisition"

    # --- Source A: a folder of images (development / batch inspection) --------
    def from_folder(
        self, folder: str | Path, exts=(".png", ".jpg", ".jpeg", ".bmp")
    ) -> Iterator[tuple[str, np.ndarray]]:
        """
        Yield (part_id, image) for every image file in a folder.
        We use the filename as a stand-in part_id so each record is traceable.
        """
        folder = Path(folder)
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() in exts:
                yield path.stem, load_image(path)

    # --- Source B: a single in-memory array (Streamlit upload / API call) -----
    def from_array(
        self, image: np.ndarray, part_id: str = "upload"
    ) -> tuple[str, np.ndarray]:
        """Wrap an already-loaded array so it matches the (part_id, image) shape."""
        return part_id, np.asarray(image, dtype=np.uint8)

    # --- Source C: a live webcam (impressive live demo) -----------------------
    def from_webcam(self, camera_index: int = 0) -> Iterator[tuple[str, np.ndarray]]:
        """
        Yield frames from a webcam. Needs OpenCV (`pip install opencv-python`).
        In production this method is replaced by the industrial camera SDK
        (e.g. Basler pypylon), but the rest of the pipeline stays identical.
        """
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Webcam source needs OpenCV: pip install opencv-python"
            ) from exc

        cap = cv2.VideoCapture(camera_index)
        frame_no = 0
        try:
            while cap.isOpened():
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                # OpenCV gives BGR; the rest of the system expects RGB.
                frame_rgb = frame_bgr[:, :, ::-1].copy()
                frame_no += 1
                yield f"frame-{frame_no:06d}", frame_rgb
        finally:
            cap.release()
