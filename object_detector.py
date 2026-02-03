import cv2
import numpy as np


class HomogeneousBgDetector:
    def __init__(self):
        pass

    def detect_objects(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold for homogeneous background
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            19,
            5
        )

        # Remove noise
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter small contours (noise)
        filtered = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:
                filtered.append(cnt)

        return filtered
