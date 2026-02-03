import argparse
import os
from typing import Optional, Tuple

import cv2
import numpy as np
from pupil_apriltags import Detector

from object_detector import HomogeneousBgDetector


def build_apriltag_detector(family: str) -> Detector:
    return Detector(
        families=family,
        nthreads=2,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.5,
    )


def detect_apriltag_scale_mm_per_px(
    gray: np.ndarray,
    detector: Detector,
    tag_size_mm: float,
) -> Tuple[Optional[float], Optional[np.ndarray], Optional[int], Optional[float]]:
    detections = detector.detect(gray)
    if not detections:
        return None, None, None, None

    tag = detections[0]
    corners = tag.corners.astype(np.float32)
    side_lengths = [
        np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)
    ]
    avg_side_px = float(np.mean(side_lengths))
    if avg_side_px <= 0.0:
        return None, None, None, None

    mm_per_px = tag_size_mm / avg_side_px
    return mm_per_px, corners, int(tag.tag_id), avg_side_px


def pca_length_px(contour: np.ndarray) -> Tuple[float, Tuple[int, int], Tuple[int, int]]:
    pts = contour.reshape(-1, 2).astype(np.float32)
    mean = np.mean(pts, axis=0)
    cov = np.cov(pts.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    principal = eigenvectors[:, int(np.argmax(eigenvalues))]

    projections = (pts - mean) @ principal
    min_proj = float(np.min(projections))
    max_proj = float(np.max(projections))
    length_px = max_proj - min_proj

    p1 = (mean + principal * min_proj).astype(int)
    p2 = (mean + principal * max_proj).astype(int)
    return float(length_px), (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))


def classic_fish_contour(image: np.ndarray, ignore_mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    detector = HomogeneousBgDetector()
    contours = detector.detect_objects(image)
    if contours:
        contours = filter_contours_by_mask(contours, ignore_mask)
        if contours:
            return max(contours, key=cv2.contourArea)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = filter_contours_by_mask(contours, ignore_mask)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def filter_contours_by_mask(
    contours: list, ignore_mask: Optional[np.ndarray]
) -> list:
    if ignore_mask is None:
        return contours
    filtered = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            continue
        roi = ignore_mask[y : y + h, x : x + w]
        if roi.size == 0:
            filtered.append(cnt)
            continue
        cnt_mask = np.zeros((h, w), dtype=np.uint8)
        shifted = cnt.copy()
        shifted[:, 0, 0] -= x
        shifted[:, 0, 1] -= y
        cv2.drawContours(cnt_mask, [shifted], -1, 255, -1)
        overlap = cv2.bitwise_and(cnt_mask, roi)
        overlap_ratio = float(np.count_nonzero(overlap)) / float(np.count_nonzero(cnt_mask) + 1)
        if overlap_ratio < 0.2:
            filtered.append(cnt)
    return filtered


def load_dnn(
    weights: Optional[str],
    config: Optional[str],
    classes_path: Optional[str],
    input_size: Tuple[int, int],
) -> Tuple[Optional[cv2.dnn_DetectionModel], Optional[list]]:
    if not weights or not config:
        return None, None
    if not os.path.exists(weights) or not os.path.exists(config):
        return None, None

    net = cv2.dnn.readNet(weights, config)
    model = cv2.dnn.DetectionModel(net)
    model.setInputSize(input_size)
    model.setInputScale(1.0 / 255.0)
    model.setInputSwapRB(True)

    class_names = None
    if classes_path and os.path.exists(classes_path):
        with open(classes_path, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]
    return model, class_names


def dnn_detect_fish_box(
    model: cv2.dnn_DetectionModel,
    class_names: Optional[list],
    image: np.ndarray,
    class_name: str,
    score_thresh: float,
    nms_thresh: float,
) -> Optional[Tuple[int, int, int, int]]:
    classes, scores, boxes = model.detect(image, score_thresh, nms_thresh)
    if len(boxes) == 0:
        return None

    best_idx = None
    best_score = -1.0
    for idx, (cls_id, score) in enumerate(zip(classes.flatten(), scores.flatten())):
        name_match = True
        if class_names and cls_id < len(class_names):
            name_match = class_names[cls_id].lower() == class_name.lower()
        if name_match and score > best_score:
            best_score = float(score)
            best_idx = idx

    if best_idx is None:
        return None
    return tuple(int(x) for x in boxes[best_idx])


def crop_contour_to_box(contour: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, 0, 0] -= x
    shifted[:, 0, 1] -= y
    cv2.drawContours(mask, [shifted], -1, 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    largest[:, 0, 0] += x
    largest[:, 0, 1] += y
    return largest


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid fish length measurement with AprilTag scale.")
    parser.add_argument("--source", default="0", help="Image path, video path, folder, glob, or camera index (default: 0).")
    parser.add_argument("--tag-size-mm", type=float, required=True, help="Printed AprilTag size in mm (black square only).")
    parser.add_argument("--tag-family", default="tag36h11", help="AprilTag family.")
    parser.add_argument("--min-tag-side-px", type=float, default=20.0, help="Min tag side length in pixels.")
    parser.add_argument("--min-fish-area", type=float, default=800.0, help="Min fish contour area.")
    parser.add_argument("--dnn-weights", default=None, help="DNN weights path (Darknet weights).")
    parser.add_argument("--dnn-config", default=None, help="DNN config path (Darknet cfg).")
    parser.add_argument("--dnn-classes", default=None, help="Class names path.")
    parser.add_argument("--dnn-class-name", default="fish", help="Class name to use from the model.")
    parser.add_argument("--dnn-input", type=int, default=320, help="DNN input size (square).")
    parser.add_argument("--score-thresh", type=float, default=0.4, help="DNN score threshold.")
    parser.add_argument("--nms-thresh", type=float, default=0.4, help="DNN NMS threshold.")
    parser.add_argument("--output", default=None, help="Optional output video path.")
    parser.add_argument("--output-dir", default="outputs", help="Output dir for annotated images.")
    parser.add_argument("--no-display", action="store_true", help="Disable display windows.")
    parser.add_argument("--manual", action="store_true", help="Manually click two points to measure.")
    args = parser.parse_args()

    detector = build_apriltag_detector(args.tag_family)

    # If source is a file/folder/glob of images, process as batch.
    is_glob = isinstance(args.source, str) and any(token in args.source for token in ["*", "?", "["])
    is_dir = isinstance(args.source, str) and os.path.isdir(args.source)
    is_image_file = isinstance(args.source, str) and args.source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))

    if is_glob or is_dir or is_image_file:
        if os.path.isdir(args.source):
            image_paths = sorted(
                [
                    os.path.join(args.source, f)
                    for f in os.listdir(args.source)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                ]
            )
        elif is_glob:
            import glob

            image_paths = sorted(glob.glob(args.source))
        else:
            image_paths = [args.source]

        if not image_paths:
            print("No images found for the given source.")
            return 1

        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Saving outputs to: {args.output_dir}")
        dnn_model, class_names = load_dnn(
            args.dnn_weights,
            args.dnn_config,
            args.dnn_classes,
            (args.dnn_input, args.dnn_input),
        )

        for image_path in image_paths:
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Skipping unreadable image: {image_path}")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            mm_per_px, tag_corners, tag_id, tag_side_px = detect_apriltag_scale_mm_per_px(
                gray, detector, args.tag_size_mm
            )

            ignore_mask = None
            if tag_corners is not None:
                cv2.polylines(frame, [tag_corners.astype(int)], True, (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Tag {tag_id}",
                    (int(tag_corners[0][0]), int(tag_corners[0][1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                ignore_mask = np.zeros(gray.shape[:2], dtype=np.uint8)
                cv2.fillPoly(ignore_mask, [tag_corners.astype(int)], 255)
                ignore_mask = cv2.dilate(ignore_mask, np.ones((9, 9), np.uint8), iterations=1)

            fish_contour = None if args.manual else classic_fish_contour(frame, ignore_mask)
            if dnn_model is not None:
                box = dnn_detect_fish_box(
                    dnn_model,
                    class_names,
                    frame,
                    args.dnn_class_name,
                    args.score_thresh,
                    args.nms_thresh,
                )
                if box:
                    x, y, w, h = box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                    if fish_contour is not None:
                        refined = crop_contour_to_box(fish_contour, box)
                        if refined is not None:
                            fish_contour = refined

            measured = False
            if args.manual and not args.no_display:
                points = []

                def click_cb(event, x, y, flags, param):
                    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
                        points.append((x, y))

                cv2.namedWindow("Fish Length (Hybrid)")
                cv2.setMouseCallback("Fish Length (Hybrid)", click_cb)

                while True:
                    preview = frame.copy()
                    if len(points) >= 1:
                        cv2.circle(preview, points[0], 5, (0, 0, 255), -1)
                    if len(points) == 2:
                        cv2.circle(preview, points[1], 5, (0, 0, 255), -1)
                        cv2.line(preview, points[0], points[1], (0, 0, 255), 2)
                    cv2.imshow("Fish Length (Hybrid)", preview)
                    key = cv2.waitKey(20)
                    if key == 27:
                        break
                    if len(points) == 2:
                        p1, p2 = points
                        length_px = float(np.linalg.norm(np.array(p1) - np.array(p2)))
                        if mm_per_px is not None:
                            length_mm = length_px * mm_per_px
                            label = f"{length_mm:.1f} mm"
                        else:
                            label = f"{length_px:.1f} px"
                        cv2.putText(
                            frame,
                            label,
                            (p1[0], p1[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )
                        cv2.line(frame, p1, p2, (0, 0, 255), 2)
                        measured = True
                        if mm_per_px is not None:
                            print(f"{os.path.basename(image_path)} -> {length_mm:.1f} mm")
                        else:
                            print(f"{os.path.basename(image_path)} -> {length_px:.1f} px (no tag)")
                        break
            elif fish_contour is not None and cv2.contourArea(fish_contour) >= args.min_fish_area:
                length_px, p1, p2 = pca_length_px(fish_contour)
                cv2.line(frame, p1, p2, (0, 0, 255), 2)

                if mm_per_px is not None:
                    length_mm = length_px * mm_per_px
                    label = f"{length_mm:.1f} mm"
                else:
                    label = f"{length_px:.1f} px"

                cv2.putText(
                    frame,
                    label,
                    (p1[0], p1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                measured = True
                if mm_per_px is not None:
                    print(f"{os.path.basename(image_path)} -> {length_mm:.1f} mm")
                else:
                    print(f"{os.path.basename(image_path)} -> {length_px:.1f} px (no tag)")

            if mm_per_px is None or (tag_side_px is not None and tag_side_px < args.min_tag_side_px):
                cv2.putText(
                    frame,
                    "Low confidence: tag too small or missing",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
            if args.manual and not measured:
                cv2.putText(
                    frame,
                    "Manual measurement skipped",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            out_name = f"out_{os.path.basename(image_path)}"
            out_path = os.path.join(args.output_dir, out_name)
            cv2.imwrite(out_path, frame)

            if not args.no_display and not args.manual:
                cv2.imshow("Fish Length (Hybrid)", frame)
                key = cv2.waitKey(0)
                if key == 27:
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
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    dnn_model, class_names = load_dnn(
        args.dnn_weights,
        args.dnn_config,
        args.dnn_classes,
        (args.dnn_input, args.dnn_input),
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        mm_per_px, tag_corners, tag_id, tag_side_px = detect_apriltag_scale_mm_per_px(
            gray, detector, args.tag_size_mm
        )

        ignore_mask = None
        if tag_corners is not None:
            cv2.polylines(frame, [tag_corners.astype(int)], True, (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Tag {tag_id}",
                (int(tag_corners[0][0]), int(tag_corners[0][1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            ignore_mask = np.zeros(gray.shape[:2], dtype=np.uint8)
            cv2.fillPoly(ignore_mask, [tag_corners.astype(int)], 255)
            ignore_mask = cv2.dilate(ignore_mask, np.ones((9, 9), np.uint8), iterations=1)

        fish_contour = classic_fish_contour(frame, ignore_mask)
        fish_box = None

        if dnn_model is not None:
            box = dnn_detect_fish_box(
                dnn_model,
                class_names,
                frame,
                args.dnn_class_name,
                args.score_thresh,
                args.nms_thresh,
            )
            if box:
                fish_box = box
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                if fish_contour is not None:
                    refined = crop_contour_to_box(fish_contour, box)
                    if refined is not None:
                        fish_contour = refined

        if fish_contour is not None and cv2.contourArea(fish_contour) >= args.min_fish_area:
            length_px, p1, p2 = pca_length_px(fish_contour)
            cv2.line(frame, p1, p2, (0, 0, 255), 2)

            if mm_per_px is not None:
                length_mm = length_px * mm_per_px
                label = f"{length_mm:.1f} mm"
            else:
                label = f"{length_px:.1f} px"

            cv2.putText(
                frame,
                label,
                (p1[0], p1[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        if mm_per_px is None or (tag_side_px is not None and tag_side_px < args.min_tag_side_px):
            cv2.putText(
                frame,
                "Low confidence: tag too small or missing",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Fish Length (Hybrid)", frame)
        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
