"""Arc-length parameterized 2D reference path."""
import numpy as np


class ReferencePath:
    def __init__(self, waypoints):
        wp = np.asarray(waypoints, dtype=float)
        if wp.ndim != 2 or wp.shape[1] != 2 or wp.shape[0] < 2:
            raise ValueError("waypoints must be (N>=2, 2)")
        self.xy = wp
        diffs = np.diff(wp, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg_len)])
        self.headings = np.arctan2(diffs[:, 1], diffs[:, 0])
        self.headings = np.concatenate([self.headings, self.headings[-1:]])
        self.total_length = self.s[-1]

    def nearest_point(self, pos):
        """Return (point_xy, s, heading, signed_lateral_error d) for pos=(x,y).

        Uses projection onto each segment and picks the closest.
        """
        p = np.asarray(pos, dtype=float)
        a = self.xy[:-1]
        b = self.xy[1:]
        ab = b - a
        ap = p - a
        ab_len2 = np.sum(ab * ab, axis=1)
        ab_len2 = np.where(ab_len2 < 1e-12, 1e-12, ab_len2)
        t = np.clip(np.sum(ap * ab, axis=1) / ab_len2, 0.0, 1.0)
        proj = a + (t[:, None] * ab)
        d2 = np.sum((proj - p) ** 2, axis=1)
        i = int(np.argmin(d2))

        point = proj[i]
        seg_s = self.s[i] + t[i] * np.linalg.norm(ab[i])
        heading = np.arctan2(ab[i, 1], ab[i, 0])
        # signed lateral error: left of path positive
        nx, ny = -np.sin(heading), np.cos(heading)
        d = (p[0] - point[0]) * nx + (p[1] - point[1]) * ny
        return point, seg_s, heading, d
