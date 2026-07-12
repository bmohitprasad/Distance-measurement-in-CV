Category A7: ArUco Marker Based Measurement

ArUco markers are the industry standard for precise measurement and robot localization. They provide a high-contrast, error-correcting binary pattern that OpenCV can identify and measure with sub-pixel accuracy.

How to Run

Get a Marker: Print a standard ArUco marker (DICT_4X4_50).

Measure it: Use a ruler to find the physical side length (e.g., 5.0 cm).

Run the script:

python a7_marker_based.py --image "images/a7_marker.jpg" --size 5.0 --output "a7_result.jpg"


Why use A7 over others?

Zero Ambiguity: Unlike regular objects, the computer knows this is a marker. It won't mistake a leaf for a coin.

Orientation: ArUco markers are rotation-invariant. The script will detect them even if you hold them at a 45-degree angle.

Sub-pixel Accuracy: The corners are defined by 4 sharp binary squares, making the coordinate detection much more stable than detecting the blurry edge of an irregular object.

Calibration Warning

This script currently uses a "Scale Ratio" approach. If you want to calculate distance from the camera (instead of just marker size), you must use the cv2.solvePnP function combined with your camera's intrinsic matrix (obtained from a chessboard calibration).