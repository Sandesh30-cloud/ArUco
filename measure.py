# import cv2
# import numpy as np
# import math

# # ================= CONFIG =================
# IMAGE_PATH = "marker29.png"
# MARKER_SIZE_MM = 50.0   # REAL printed marker size (black square only)
# # =========================================

# # Load image
# image = cv2.imread(IMAGE_PATH)
# if image is None:
#     print("❌ Image not found")
#     exit()


# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.equalizeHist(gray)   # improves contrast

# # All supported ArUco dictionaries
# ARUCO_DICTS = {
#     "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
#     "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
#     "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
#     "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
#     "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
#     "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
#     "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
#     "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
#     "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
#     "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
#     "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
#     "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
#     "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
#     "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
#     "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
#     "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
# }

# found = False

# for dict_name, dict_id in ARUCO_DICTS.items():
#     aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

#     params = cv2.aruco.DetectorParameters()
#     # params.adaptiveThreshWinSizeMin = 3
#     # params.adaptiveThreshWinSizeMax = 23
#     # params.adaptiveThreshWinSizeStep = 10
#     # params.minMarkerPerimeterRate = 0.03
#     # params.maxMarkerPerimeterRate = 4.0
#     # params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

#     detector = cv2.aruco.ArucoDetector(aruco_dict, params)
#     corners, ids, rejected = detector.detectMarkers(gray)

#     if ids is not None:
#         print(f"✅ Detected marker using {dict_name}")
#         found = True
#         break

# if not found:
#     print("❌ No ArUco marker detected in any dictionary")
#     exit()

# # Draw detected marker
# cv2.aruco.drawDetectedMarkers(image, corners, ids)

# # Take first detected marker
# marker_corners = corners[0][0]
# marker_id = ids[0][0]

# print(f"📌 Marker ID: {marker_id}")

# # Compute average marker side length (BEST PRACTICE)
# sides = [
#     np.linalg.norm(marker_corners[i] - marker_corners[(i + 1) % 4])
#     for i in range(4)
# ]
# marker_pixel_size = sum(sides) / 4

# pixel_to_mm = MARKER_SIZE_MM / marker_pixel_size

# print(f"📏 Marker pixel size: {marker_pixel_size:.2f}px")
# print(f"📐 Pixel-to-mm ratio: {pixel_to_mm:.4f}")

# # ================= OBJECT MEASUREMENT =================
# points = []

# def mouse_callback(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         points.append((x, y))
#         cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

# cv2.namedWindow("Measurement")
# cv2.setMouseCallback("Measurement", mouse_callback)

# print("👉 Click TWO points on the object")

# while True:
#     cv2.imshow("Measurement", image)
#     key = cv2.waitKey(1)

#     if len(points) == 2:
#         pixel_dist = math.dist(points[0], points[1])
#         real_length_mm = pixel_dist * pixel_to_mm

#         cv2.line(image, points[0], points[1], (255, 0, 0), 2)
#         cv2.putText(
#             image,
#             f"{real_length_mm:.2f} mm",
#             (points[0][0], points[0][1] - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             (0, 255, 0),
#             2
#         )

#         print(f"✅ Object Length: {real_length_mm:.2f} mm")
#         cv2.imshow("Measurement", image)
#         cv2.waitKey(0)
#         break

#     if key == 27:
#         break

# cv2.destroyAllWindows()


































import cv2
import numpy as np
from pupil_apriltags import Detector

IMAGE_PATH = "april3.jpeg"

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError("Image not found")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 🔹 Improve contrast (IMPORTANT)
gray = cv2.equalizeHist(gray)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# 🔹 Robust AprilTag detector
detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.5
)

detections = detector.detect(gray)

print(f"Detected {len(detections)} AprilTags")

if len(detections) == 0:
    print("❌ No AprilTags detected")
else:
    for tag in detections:
        corners = tag.corners.astype(int)
        tag_id = tag.tag_id

        cv2.polylines(image, [corners], True, (0, 255, 0), 2)
        cv2.putText(
            image,
            f"ID {tag_id}",
            (corners[0][0], corners[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

cv2.imshow("AprilTag Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
