import cv2

img = cv2.imread("ex.jpeg", cv2.IMREAD_GRAYSCALE)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, params)

corners, ids, rejected = detector.detectMarkers(img)

print("IDs:", ids)
print("Rejected:", len(rejected))

cv2.imshow("img", img)
cv2.imshow("rejected", cv2.aruco.drawDetectedMarkers(img.copy(), rejected))
cv2.waitKey(0)