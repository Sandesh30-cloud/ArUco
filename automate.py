GPT



import cv2
import numpy as np
import math

# ================= CONFIG =================
IMAGE_PATH = "marker24.jpeg"
MARKER_SIZE_MM = 50.0
MIN_OBJECT_AREA = 1500     # reject small noise
# =========================================

image = cv2.imread(IMAGE_PATH)
if image is None:
    print("❌ Image not found")
    exit()

display = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

# ================= ARUCO DETECTION (ROBUST) =================
aruco_dicts = [
    cv2.aruco.DICT_4X4_50, cv2.aruco.DICT_4X4_100, cv2.aruco.DICT_4X4_250,
    cv2.aruco.DICT_5X5_50, cv2.aruco.DICT_6X6_50, cv2.aruco.DICT_7X7_50
]

params = cv2.aruco.DetectorParameters()
params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeMax = 35
params.adaptiveThreshWinSizeStep = 4
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
params.minMarkerPerimeterRate = 0.02
params.maxMarkerPerimeterRate = 4.0

found = False

for d in aruco_dicts:
    aruco_dict = cv2.aruco.getPredefinedDictionary(d)
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        found = True
        break

if not found:
    print("❌ ArUco not detected — measurement aborted")
    cv2.imshow("Image", display)
    cv2.waitKey(0)
    exit()

# ================= SCALE =================
marker = corners[0][0]

sides = [
    np.linalg.norm(marker[i] - marker[(i + 1) % 4])
    for i in range(4)
]
marker_px = sum(sides) / 4
pixel_to_mm = MARKER_SIZE_MM / marker_px

print(f"📐 Pixel → mm = {pixel_to_mm:.5f}")

# ================= STRONG MARKER MASK =================
mask = np.ones(gray.shape, dtype="uint8") * 255

# expand bounding box (important!)
x, y, w, h = cv2.boundingRect(marker.astype(int))
pad = 15
x, y = max(0, x-pad), max(0, y-pad)
w, h = w + 2*pad, h + 2*pad

cv2.rectangle(mask, (x, y), (x+w, y+h), 0, -1)

no_marker = cv2.bitwise_and(image, image, mask=mask)

# ================= OBJECT DETECTION =================
g = cv2.cvtColor(no_marker, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(g, (5, 5), 0)
edges = cv2.Canny(blur, 60, 160)

kernel = np.ones((5, 5), np.uint8)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# ================= CONTOURS =================
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

valid_objects = []

for c in contours:
    area = cv2.contourArea(c)
    if area < MIN_OBJECT_AREA:
        continue

    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = max(w, h) / (min(w, h) + 1e-5)

    # Reject square-like (marker-ish) shapes
    if 0.8 < aspect_ratio < 1.25:
        continue

    valid_objects.append(c)

if not valid_objects:
    print("❌ Object not detected")
    exit()

# choose largest valid object
obj = max(valid_objects, key=cv2.contourArea)
cv2.drawContours(display, [obj], -1, (0, 255, 255), 2)

# ================= ENDPOINTS =================
left = tuple(obj[obj[:, :, 0].argmin()][0])
right = tuple(obj[obj[:, :, 0].argmax()][0])
top = tuple(obj[obj[:, :, 1].argmin()][0])
bottom = tuple(obj[obj[:, :, 1].argmax()][0])

pairs = [(left, right), (top, bottom)]
px_dist, (p1, p2) = max(
    [(math.dist(a, b), (a, b)) for a, b in pairs],
    key=lambda x: x[0]
)

length_mm = px_dist * pixel_to_mm

# ================= DRAW =================
cv2.circle(display, p1, 6, (0, 0, 255), -1)
cv2.circle(display, p2, 6, (0, 0, 255), -1)
cv2.line(display, p1, p2, (255, 0, 0), 2)

cv2.putText(
    display,
    f"{length_mm:.2f} mm",
    (p1[0], p1[1] - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2
)

print(f"✅ Object Length: {length_mm:.2f} mm")

cv2.imshow("Edges", edges)
cv2.imshow("Final Measurement", display)
cv2.waitKey(0)
cv2.destroyAllWindows()










