import cv2
import numpy as np
import math
from pupil_apriltags import Detector


IMAGE_PATH = "april14.jpeg"
TAG_SIZE_MM = 100.0        # REAL printed AprilTag size (black square only)
APRILTAG_FAMILY = "tag36h11"



image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError("Image not found")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 🔹 Improve contrast (IMPORTANT)
gray = cv2.equalizeHist(gray)
gray = cv2.GaussianBlur(gray, (5, 5), 0)


detector = Detector(
    families=APRILTAG_FAMILY,
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.5
)

detections = detector.detect(gray)

if len(detections) == 0:
    raise RuntimeError("No AprilTag detected")

# Use the first detected tag
tag = detections[0]
corners = tag.corners.astype(np.float32)
tag_id = tag.tag_id

print(f"✅ AprilTag detected | ID = {tag_id}")

# Draw tag
cv2.polylines(image, [corners.astype(int)], True, (0, 255, 0), 2)

# ================= SCALE COMPUTATION =================
# Robust average of all 4 sides
side_lengths = [
    np.linalg.norm(corners[i] - corners[(i + 1) % 4])
    for i in range(4)
]

avg_tag_side_px = np.mean(side_lengths)
mm_per_pixel = TAG_SIZE_MM / avg_tag_side_px

print(f"📐 Tag size (px): {avg_tag_side_px:.2f}")
print(f"📏 mm per pixel: {mm_per_pixel:.6f}")

# ================= OBJECT MEASUREMENT =================
points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

cv2.namedWindow("Measurement")
cv2.setMouseCallback("Measurement", mouse_callback)

print("👉 Click TWO points on the object (start → end)")

while True:
    cv2.imshow("Measurement", image)
    key = cv2.waitKey(1)

    if len(points) == 2:
        pixel_distance = math.dist(points[0], points[1])
        real_length_mm = pixel_distance * mm_per_pixel

        # Draw measurement
        cv2.line(image, points[0], points[1], (255, 0, 0), 2)
        cv2.putText(
            image,
            f"{real_length_mm:.2f} mm",
            (points[0][0], points[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        print(f"✅ Object length: {real_length_mm:.2f} mm")

        cv2.imshow("Measurement", image)
        cv2.waitKey(0)
        break

    if key == 27:  # ESC to exit
        break

cv2.destroyAllWindows()

























# import cv2
# import numpy as np
# import math
# from pupil_apriltags import Detector

# IMAGE_PATH = "april11.jpeg"
# TAG_SIZE_MM = 53.0
# APRILTAG_FAMILY = "tag36h11"

# # ================= LOAD IMAGE =================
# image = cv2.imread(IMAGE_PATH)
# if image is None:
#     raise FileNotFoundError("Image not found")

# # ================= TRY MULTIPLE ROTATIONS =================
# def detect_with_rotations(img):
#     detector = Detector(
#         families=APRILTAG_FAMILY,
#         nthreads=4,
#         quad_decimate=2.0,     # IMPORTANT
#         quad_sigma=0.8,        # helps noisy backgrounds
#         refine_edges=1,
#         decode_sharpening=0.25
#     )

#     for angle in [0, 90, 180, 270]:
#         rotated = img if angle == 0 else cv2.rotate(
#             img,
#             {
#                 90: cv2.ROTATE_90_CLOCKWISE,
#                 180: cv2.ROTATE_180,
#                 270: cv2.ROTATE_90_COUNTERCLOCKWISE
#             }[angle]
#         )

#         gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
#         gray = cv2.GaussianBlur(gray, (5, 5), 0)

#         detections = detector.detect(gray)
#         if detections:
#             return detections, rotated, angle

#     return None, None, None

# detections, used_image, angle = detect_with_rotations(image)

# if detections is None:
#     raise RuntimeError("❌ No AprilTag detected in any orientation")

# print(f"✅ Tag detected after rotating image by {angle}°")

# # ================= PROCESS DETECTION =================
# tag = detections[0]
# corners = tag.corners.astype(np.float32)
# tag_id = tag.tag_id

# print(f"🆔 Tag ID: {tag_id}")

# cv2.polylines(used_image, [corners.astype(int)], True, (0, 255, 0), 2)

# # ================= SCALE COMPUTATION =================
# side_lengths = [
#     np.linalg.norm(corners[i] - corners[(i + 1) % 4])
#     for i in range(4)
# ]

# avg_side_px = np.mean(side_lengths)
# mm_per_pixel = TAG_SIZE_MM / avg_side_px

# print(f"📐 Tag side (px): {avg_side_px:.2f}")
# print(f"📏 Scale: {mm_per_pixel:.6f} mm/px")

# # ================= OBJECT MEASUREMENT =================
# points = []

# def mouse_callback(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         points.append((x, y))
#         cv2.circle(used_image, (x, y), 5, (0, 0, 255), -1)

# cv2.namedWindow("Measurement")
# cv2.setMouseCallback("Measurement", mouse_callback)

# print("👉 Click TWO points on the object")

# while True:
#     cv2.imshow("Measurement", used_image)
#     key = cv2.waitKey(1)

#     if len(points) == 2:
#         px_dist = math.dist(points[0], points[1])
#         real_mm = px_dist * mm_per_pixel

#         cv2.line(used_image, points[0], points[1], (255, 0, 0), 2)
#         cv2.putText(
#             used_image,
#             f"{real_mm:.2f} mm",
#             (points[0][0], points[0][1] - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             (0, 255, 0),
#             2
#         )

#         print(f"✅ Object length: {real_mm:.2f} mm")
#         cv2.imshow("Measurement", used_image)
#         cv2.waitKey(0)
#         break

#     if key == 27:
#         break

# cv2.destroyAllWindows()
