import argparse
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pupil_apriltags import Detector


ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


def detect_aruco_markers(
    gray: np.ndarray, marker_size_mm: float, aruco_dict_name: str
) -> Tuple[List[dict], Optional[str]]:
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 35
    params.adaptiveThreshWinSizeStep = 4
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 4.0

    if aruco_dict_name != "auto":
        dict_names = [aruco_dict_name]
    else:
        dict_names = list(ARUCO_DICT_MAP.keys())

    for dict_name in dict_names:
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[dict_name])
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            continue

        markers = []
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            points = marker_corners[0].astype(np.float32)
            side_lengths = [
                np.linalg.norm(points[i] - points[(i + 1) % 4]) for i in range(4)
            ]
            side_px = float(np.mean(side_lengths))
            if side_px <= 0.0:
                continue
            center = np.mean(points, axis=0)
            markers.append(
                {
                    "id": int(marker_id),
                    "corners": points,
                    "center": center,
                    "side_px": side_px,
                    "mm_per_px": float(marker_size_mm / side_px),
                    "dict_name": dict_name,
                }
            )
        if markers:
            return markers, dict_name
    return [], None


def detect_apriltag_markers(
    gray: np.ndarray, marker_size_mm: float, apriltag_family: str
) -> Tuple[List[dict], Optional[str]]:
    families = [apriltag_family]
    if apriltag_family == "auto":
        families = ["tag36h11", "tag25h9", "tag16h5"]

    detector_setups = [(1.0, 0.0, 0.5), (1.5, 0.8, 0.25), (2.0, 0.8, 0.25)]
    for family in families:
        for decimate, sigma, sharpen in detector_setups:
            detector = Detector(
                families=family,
                nthreads=4,
                quad_decimate=decimate,
                quad_sigma=sigma,
                refine_edges=1,
                decode_sharpening=sharpen,
            )
            detections = detector.detect(gray)
            if not detections:
                continue

            markers = []
            for tag in detections:
                points = tag.corners.astype(np.float32)
                side_lengths = [
                    np.linalg.norm(points[i] - points[(i + 1) % 4]) for i in range(4)
                ]
                side_px = float(np.mean(side_lengths))
                if side_px <= 0.0:
                    continue
                center = np.mean(points, axis=0)
                markers.append(
                    {
                        "id": int(tag.tag_id),
                        "corners": points,
                        "center": center,
                        "side_px": side_px,
                        "mm_per_px": float(marker_size_mm / side_px),
                        "dict_name": family,
                    }
                )
            if markers:
                return markers, family
    return [], None


def build_marker_ignore_mask(
    shape: Tuple[int, int], markers: List[dict], pad_px: int, pad_factor: float
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for marker in markers:
        poly = marker["corners"].astype(np.int32)
        marker_mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(marker_mask, [poly], 255)

        dynamic_pad = int(marker["side_px"] * pad_factor)
        k = max(pad_px, dynamic_pad)
        if k > 0:
            kernel = np.ones((k, k), np.uint8)
            marker_mask = cv2.dilate(marker_mask, kernel, iterations=1)
        mask = cv2.bitwise_or(mask, marker_mask)
    return mask


def detect_object_contours(
    frame: np.ndarray, ignore_mask: Optional[np.ndarray]
) -> List[np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 60, 170)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5
    )
    combined = cv2.bitwise_or(edges, thresh)
    combined = cv2.morphologyEx(
        combined, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2
    )
    combined = cv2.morphologyEx(
        combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
    )

    if ignore_mask is not None:
        combined[ignore_mask > 0] = 0

    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def contour_centroid(contour: np.ndarray) -> Optional[np.ndarray]:
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    return np.array(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        dtype=np.float32,
    )


def contour_long_axis_px(
    contour: np.ndarray,
) -> Tuple[float, Tuple[int, int], Tuple[int, int]]:
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect
    length_px = max(float(w), float(h))
    theta = np.deg2rad(angle)
    if h > w:
        theta += np.pi / 2.0
    direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    p1 = np.array([cx, cy], dtype=np.float32) - direction * (length_px / 2.0)
    p2 = np.array([cx, cy], dtype=np.float32) + direction * (length_px / 2.0)
    return length_px, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))


def contour_overlap_ratio_with_mask(contour: np.ndarray, mask: np.ndarray) -> float:
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return 0.0
    roi_mask = mask[y : y + h, x : x + w]
    if roi_mask.size == 0:
        return 0.0

    cnt_mask = np.zeros((h, w), dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, 0, 0] -= x
    shifted[:, 0, 1] -= y
    cv2.drawContours(cnt_mask, [shifted], -1, 255, -1)

    overlap = cv2.bitwise_and(cnt_mask, roi_mask)
    denom = float(np.count_nonzero(cnt_mask))
    if denom <= 0:
        return 0.0
    return float(np.count_nonzero(overlap)) / denom


def select_nearby_objects(
    contours: List[np.ndarray],
    markers: List[dict],
    marker_exclusion_mask: np.ndarray,
    min_object_area: float,
    min_distance_factor: float,
    min_point_distance_factor: float,
    max_distance_factor: float,
    max_marker_overlap: float,
) -> List[dict]:
    results = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_object_area:
            continue

        centroid = contour_centroid(cnt)
        if centroid is None:
            continue

        nearest = None
        min_dist = float("inf")
        for marker in markers:
            dist = float(np.linalg.norm(centroid - marker["center"]))
            if dist < min_dist:
                min_dist = dist
                nearest = marker

        if nearest is None:
            continue

        max_dist = max_distance_factor * nearest["side_px"]
        min_dist_allowed = min_distance_factor * nearest["side_px"]
        if min_dist < min_dist_allowed:
            continue
        if min_dist > max_dist:
            continue

        marker_center = (float(nearest["center"][0]), float(nearest["center"][1]))
        if cv2.pointPolygonTest(cnt, marker_center, False) >= 0:
            continue

        contour_pts = cnt.reshape(-1, 2).astype(np.float32)
        distances = np.linalg.norm(contour_pts - nearest["center"], axis=1)
        min_point_dist = float(np.min(distances)) if distances.size else float("inf")
        if min_point_dist < (min_point_distance_factor * nearest["side_px"]):
            continue

        overlap_ratio = contour_overlap_ratio_with_mask(cnt, marker_exclusion_mask)
        if overlap_ratio > max_marker_overlap:
            continue

        length_px, p1, p2 = contour_long_axis_px(cnt)
        length_mm = length_px * nearest["mm_per_px"]

        proximity = max(0.0, 1.0 - (min_dist / (max_dist + 1e-6)))
        area_score = min(1.0, area / (min_object_area * 8.0))
        score = (0.65 * proximity) + (0.35 * area_score)

        results.append(
            {
                "contour": cnt,
                "marker": nearest,
                "centroid": centroid,
                "distance_px": min_dist,
                "min_point_distance_px": min_point_dist,
                "length_px": length_px,
                "length_mm": length_mm,
                "marker_overlap": overlap_ratio,
                "score": score,
                "p1": p1,
                "p2": p2,
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results


def annotate(
    frame: np.ndarray,
    markers: List[dict],
    results: List[dict],
    min_marker_side_px: float,
) -> np.ndarray:
    out = frame.copy()
    for marker in markers:
        corners_i = marker["corners"].astype(np.int32)
        cv2.polylines(out, [corners_i], True, (0, 255, 0), 2)
        cv2.putText(
            out,
            f"Marker {marker['id']} ({marker['dict_name']})",
            (int(marker["corners"][0][0]), int(marker["corners"][0][1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    if not results:
        cv2.putText(
            out,
            "No nearby objects found",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return out

    for idx, result in enumerate(results, start=1):
        marker = result["marker"]
        cv2.drawContours(out, [result["contour"]], -1, (0, 255, 255), 2)
        cv2.line(out, result["p1"], result["p2"], (255, 0, 0), 2)
        cv2.circle(out, result["p1"], 5, (0, 0, 255), -1)
        cv2.circle(out, result["p2"], 5, (0, 0, 255), -1)
        cv2.line(
            out,
            (int(result["centroid"][0]), int(result["centroid"][1])),
            (int(marker["center"][0]), int(marker["center"][1])),
            (255, 200, 0),
            1,
        )
        label_anchor_x = min(result["p1"][0], result["p2"][0])
        label_anchor_y = min(result["p1"][1], result["p2"][1]) - 8
        cv2.putText(
            out,
            f"Obj{idx}: {result['length_mm']:.2f} mm",
            (label_anchor_x, label_anchor_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    if any(r["marker"]["side_px"] < min_marker_side_px for r in results):
        cv2.putText(
            out,
            "Low confidence: marker too small",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
        )
    return out


def process_frame(
    frame: np.ndarray,
    marker_size_mm: float,
    marker_type: str,
    aruco_dict_name: str,
    apriltag_family: str,
    min_object_area: float,
    min_distance_factor: float,
    min_point_distance_factor: float,
    max_distance_factor: float,
    marker_pad_px: int,
    marker_pad_factor: float,
    max_marker_overlap: float,
    min_marker_side_px: float,
) -> Tuple[np.ndarray, List[dict], List[dict]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    markers = []
    if marker_type in ("aruco", "auto"):
        markers, _ = detect_aruco_markers(gray, marker_size_mm, aruco_dict_name)
    if not markers and marker_type in ("apriltag", "auto"):
        markers, _ = detect_apriltag_markers(gray, marker_size_mm, apriltag_family)

    if not markers:
        out = frame.copy()
        cv2.putText(
            out,
            "No marker detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return out, [], markers

    ignore_mask = build_marker_ignore_mask(
        gray.shape, markers, marker_pad_px, marker_pad_factor
    )
    contours = detect_object_contours(frame, ignore_mask)
    results = select_nearby_objects(
        contours,
        markers,
        ignore_mask,
        min_object_area,
        min_distance_factor,
        min_point_distance_factor,
        max_distance_factor,
        max_marker_overlap,
    )
    out = annotate(frame, markers, results, min_marker_side_px)
    return out, results, markers


def resolve_image_inputs(source: str) -> List[str]:
    is_glob = any(token in source for token in ["*", "?", "["])
    is_dir = os.path.isdir(source)
    is_image = source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))

    if is_dir:
        return sorted(
            [
                os.path.join(source, f)
                for f in os.listdir(source)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]
        )
    if is_glob:
        import glob

        return sorted(glob.glob(source))
    if is_image:
        return [source]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect only marker and measure nearby object length."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Image path, folder, glob, video path, or camera index.",
    )
    parser.add_argument(
        "--marker-size-mm",
        "--tag-size-mm",
        dest="marker_size_mm",
        type=float,
        required=True,
        help="Printed marker size in mm (black square only).",
    )
    parser.add_argument(
        "--marker-type",
        default="auto",
        choices=["auto", "aruco", "apriltag"],
        help="Marker detector type. auto tries ArUco then AprilTag.",
    )
    parser.add_argument(
        "--aruco-dict",
        default="auto",
        choices=["auto"] + list(ARUCO_DICT_MAP.keys()),
        help="ArUco dictionary to use (auto tries common dicts).",
    )
    parser.add_argument(
        "--apriltag-family",
        default="auto",
        help="AprilTag family when marker type uses apriltag (or auto).",
    )
    parser.add_argument("--min-object-area", type=float, default=1500.0)
    parser.add_argument(
        "--min-distance-factor",
        type=float,
        default=1.2,
        help="Ignore contours too close to marker center (in marker side units).",
    )
    parser.add_argument(
        "--max-distance-factor",
        type=float,
        default=5.0,
        help="Object must be within this multiple of marker side length.",
    )
    parser.add_argument(
        "--min-point-distance-factor",
        type=float,
        default=0.95,
        help="Reject contours that come too close to marker border.",
    )
    parser.add_argument("--marker-pad-px", type=int, default=15)
    parser.add_argument(
        "--marker-pad-factor",
        type=float,
        default=0.55,
        help="Extra exclusion around marker in marker side units.",
    )
    parser.add_argument(
        "--max-marker-overlap",
        type=float,
        default=0.08,
        help="Reject contour if overlap with marker exclusion area is above this ratio.",
    )
    parser.add_argument("--min-marker-side-px", type=float, default=25.0)
    parser.add_argument("--output-video", default=None)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    image_paths = resolve_image_inputs(args.source)
    if image_paths:
        for path in image_paths:
            frame = cv2.imread(path)
            if frame is None:
                print(f"Skipping unreadable image: {path}")
                continue

            out, results, markers = process_frame(
                frame=frame,
                marker_size_mm=args.marker_size_mm,
                marker_type=args.marker_type,
                aruco_dict_name=args.aruco_dict,
                apriltag_family=args.apriltag_family,
                min_object_area=args.min_object_area,
                min_distance_factor=args.min_distance_factor,
                min_point_distance_factor=args.min_point_distance_factor,
                max_distance_factor=args.max_distance_factor,
                marker_pad_px=args.marker_pad_px,
                marker_pad_factor=args.marker_pad_factor,
                max_marker_overlap=args.max_marker_overlap,
                min_marker_side_px=args.min_marker_side_px,
            )

            if not markers:
                print(f"{os.path.basename(path)} -> no marker")
            elif not results:
                print(f"{os.path.basename(path)} -> no nearby objects")
            else:
                labels = [
                    f"obj{i + 1}: {r['length_mm']:.2f} mm (marker {r['marker']['id']})"
                    for i, r in enumerate(results)
                ]
                print(f"{os.path.basename(path)} -> " + " | ".join(labels))

            if not args.no_display:
                cv2.imshow("Marker Nearby Measurement", out)
                if cv2.waitKey(0) == 27:
                    break

        if not args.no_display:
            cv2.destroyAllWindows()
        return 0

    try:
        source_index = int(args.source)
        cap = cv2.VideoCapture(source_index)
    except ValueError:
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        print("Failed to open source.")
        return 1

    writer = None
    if args.output_video:
        os.makedirs(os.path.dirname(args.output_video) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        out, results, markers = process_frame(
            frame=frame,
            marker_size_mm=args.marker_size_mm,
            marker_type=args.marker_type,
            aruco_dict_name=args.aruco_dict,
            apriltag_family=args.apriltag_family,
            min_object_area=args.min_object_area,
            min_distance_factor=args.min_distance_factor,
            min_point_distance_factor=args.min_point_distance_factor,
            max_distance_factor=args.max_distance_factor,
            marker_pad_px=args.marker_pad_px,
            marker_pad_factor=args.marker_pad_factor,
            max_marker_overlap=args.max_marker_overlap,
            min_marker_side_px=args.min_marker_side_px,
        )

        if markers and results:
            labels = [
                f"obj{i + 1}: {r['length_mm']:.2f} mm"
                for i, r in enumerate(results)
            ]
            print("Lengths: " + " | ".join(labels))

        if writer is not None:
            writer.write(out)

        if not args.no_display:
            cv2.imshow("Marker Nearby Measurement", out)
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
