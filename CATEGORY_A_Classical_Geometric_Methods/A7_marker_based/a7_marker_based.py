import argparse
import cv2
import cv2.aruco as aruco
import numpy as np


def detect_marker_scale(image, marker_size_real):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(image)

    if ids is None or len(ids) == 0:
        raise ValueError("Marker not found. I need the marker to set the scale!")

    c = corners[0][0]
    marker_width_px = np.linalg.norm(c[0] - c[1])
    if marker_width_px <= 0:
        raise ValueError("Detected marker width is invalid.")

    pixels_per_metric = marker_width_px / marker_size_real
    print(f"Scale established: {pixels_per_metric:.2f} pixels per metric")
    return pixels_per_metric


def measure_object_from_pixel_coords(pixel_coords, pixels_per_metric, unit="cm"):
    if len(pixel_coords) < 3:
        raise ValueError("At least 3 pixel coordinates are required.")

    contour = np.array(pixel_coords, dtype=np.int32).reshape((-1, 1, 2))
    area_px = cv2.contourArea(contour)

    if area_px <= 0:
        raise ValueError("The provided coordinates do not form a valid shape.")

    rect = cv2.minAreaRect(contour)
    width_px, height_px = rect[1]
    if width_px < height_px:
        width_px, height_px = height_px, width_px

    length = width_px / pixels_per_metric
    breadth = height_px / pixels_per_metric
    area = area_px / (pixels_per_metric ** 2)

    return {
        "length": length,
        "breadth": breadth,
        "area": area,
        "unit": unit,
        "length_px": width_px,
        "breadth_px": height_px,
        "area_px": area_px,
    }


def measure_scene(image_path, marker_size_real, object_pixel_coords=None, output_path="a7_measured_scene.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    pixels_per_metric = detect_marker_scale(img, marker_size_real)

    if object_pixel_coords is not None:
        result = measure_object_from_pixel_coords(object_pixel_coords, pixels_per_metric)
        annotated = img.copy()
        pts = np.array(object_pixel_coords, dtype=np.int32)
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(
            annotated,
            f"L={result['length']:.2f}{result['unit']} B={result['breadth']:.2f}{result['unit']} A={result['area']:.2f}{result['unit']}^2",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imwrite(output_path, annotated)
        print(f"Measurement complete. Saved to {output_path}")
        return result

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = img.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        obj_width = w / pixels_per_metric
        obj_height = h / pixels_per_metric
        obj_area = cv2.contourArea(cnt) / (pixels_per_metric ** 2)

        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{obj_width:.1f}x{obj_height:.1f}cm A={obj_area:.1f}cm^2",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    cv2.imwrite(output_path, annotated)
    print(f"Measurement complete. Saved to {output_path}")
    return {"pixels_per_metric": pixels_per_metric}


def parse_pixel_coords(raw_coords):
    if not raw_coords:
        return None
    pairs = raw_coords.split()
    coords = []
    for pair in pairs:
        x_str, y_str = pair.split(",")
        coords.append((int(float(x_str)), int(float(y_str))))
    return coords


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure an object's dimensions using an ArUco marker as scale.")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--size", type=float, required=True, help="Real-world size of the marker in cm")
    parser.add_argument("--coords", help="Space-separated pixel coordinates, e.g. '10,20 100,20 100,80 10,80'")
    parser.add_argument("--output", default="a7_measured_scene.jpg", help="Path to save the annotated image")
    args = parser.parse_args()

    measure_scene(args.image, args.size, object_pixel_coords=parse_pixel_coords(args.coords), output_path=args.output)