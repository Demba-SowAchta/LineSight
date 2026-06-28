"""
DummyDetector -- a numpy-only baseline that needs NO PyTorch.

WHY THIS EXISTS:
1. Tests and CI must run fast and without GPUs or 2 GB of PyTorch wheels.
2. New contributors can run the WHOLE system end-to-end (agents -> DB -> dashboard)
   in seconds, before downloading the dataset or training anything.
3. It is a real, if weak, anomaly heuristic -- a sane baseline to beat.

HOW IT WORKS (simple but not arbitrary):
A manufacturing "good" part, photographed under fixed lighting, tends to have a
stable overall brightness and texture. Many defects (missing part, wrong colour,
big scratch) shift the image statistics. So we score how far an image's mean and
texture (edge energy) sit from a reference learned on good images. It will never
match a trained CNN -- that is the point: it is the floor, not the ceiling.
"""

from __future__ import annotations

import numpy as np

from src.models.base import BaseDetector, DetectionResult


class DummyDetector(BaseDetector):
    name = "dummy-baseline-v1"

    def __init__(
        self, threshold: float = 0.5, ref_mean: float = 0.5, ref_edges: float = 0.08
    ):
        # Reference statistics of "good" parts. `fit()` can refine these.
        self.threshold = threshold
        self.ref_mean = ref_mean  # expected average brightness (0..1)
        self.ref_edges = ref_edges  # expected edge density (0..1)

    def fit(self, good_images: list[np.ndarray]) -> "DummyDetector":
        """
        Learn the reference statistics from a handful of good images.
        Optional -- the defaults already work for a quick demo.
        """
        means, edges = [], []
        for img in good_images:
            g = self._gray(img)
            means.append(g.mean())
            edges.append(self._edge_density(g))
        if means:
            self.ref_mean = float(np.mean(means))
            self.ref_edges = float(np.mean(edges))
        return self

    def predict(self, image: np.ndarray) -> DetectionResult:
        g = self._gray(image)  # 1) to grayscale, 0..1
        mean_dev = abs(g.mean() - self.ref_mean)  # 2) global brightness deviation
        edge_dev = abs(self._edge_density(g) - self.ref_edges)  # 3) texture deviation

        # 4) LOCAL outlier signal: a missing part or scratch is usually a small
        #    region that is much brighter/darker than the rest. We measure the
        #    largest per-block brightness deviation from the image's own mean.
        dev_map = self._block_deviation_map(g)  # RAW per-block deviations
        local_dev = float(dev_map.max())  # strongest local anomaly (raw)
        m = dev_map.max()
        heatmap = dev_map / m if m > 0 else dev_map  # normalised 0..1 for display

        # 5) Combine into a single 0..1 anomaly score. Weights are a simple,
        #    explainable choice; the local term dominates so obvious localized
        #    defects are caught. tanh keeps the score bounded in 0..1.
        raw = 1.5 * mean_dev + 4.0 * edge_dev + 6.0 * local_dev
        score = float(np.tanh(raw))

        return DetectionResult(
            score=score,
            is_anomaly=score >= self.threshold,
            threshold=self.threshold,
            heatmap=heatmap,
            extra={"mean_dev": mean_dev, "edge_dev": edge_dev},
        )

    # -- small image helpers (pure numpy, no external libraries) --------------
    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        """Convert HxWx3 uint8 RGB to HxW float grayscale in 0..1."""
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., :3].mean(axis=2)
        return arr / 255.0 if arr.max() > 1.0 else arr

    @staticmethod
    def _edge_density(gray: np.ndarray) -> float:
        """Average magnitude of horizontal+vertical gradients = how 'busy' the image is."""
        gx = np.abs(np.diff(gray, axis=1)).mean()
        gy = np.abs(np.diff(gray, axis=0)).mean()
        return float((gx + gy) / 2.0)

    def _block_deviation_map(self, gray: np.ndarray, blocks: int = 16) -> np.ndarray:
        """
        RAW per-block brightness deviation from the image's own mean.
        A bright defect patch produces a large value; a uniform good part stays
        near zero everywhere. Used both for scoring (max) and, once normalised,
        for the display heatmap.
        """
        h, w = gray.shape
        bh, bw = max(1, h // blocks), max(1, w // blocks)
        global_mean = float(gray.mean())
        heat = np.zeros((blocks, blocks), dtype=np.float32)
        for i in range(blocks):
            for j in range(blocks):
                block = gray[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw]
                if block.size:
                    heat[i, j] = abs(float(block.mean()) - global_mean)
        return heat
