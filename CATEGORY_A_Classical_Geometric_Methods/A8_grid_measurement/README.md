Category A8: Grid-Based Measurement

This method is best when you have a structured background (graph paper, a cutting mat with a grid, or tiled floor) where the distance between grid lines is constant.

How to Run

python a8_grid_measurement.py --image "images/a8_grid.jpg" --cell-size 1.0 --output "a8_result.jpg"


Why use A8?

Calibration Stability: The grid serves as a permanent, non-moving reference.

Accuracy: Because the reference is spread across the entire image (not just one point), you can account for slight camera lens distortions that might warp the edges of the image.

Usage Tips:

Ensure the grid lines are clearly visible. If they are too faint, use a photo editor to increase the contrast before running the script.

Ensure the camera is parallel to the grid for the most accurate results. If the camera is tilted, the grid will appear distorted (trapezoidal), and you should use the Homography (A4) method instead.