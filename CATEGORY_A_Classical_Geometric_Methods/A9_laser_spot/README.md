Category A9: Laser Triangulation

This method is best for non-contact distance estimation. By mounting two laser pointers exactly parallel (or at a known angle) to each other, you create a projection on any surface. The distance between those dots on the surface changes based on the distance of the surface from the camera.

How to Run

Mounting: Attach two laser pointers to a fixed rig. Measure the physical distance between them (the baseline).

Calibration: Calculate your camera's focal length in pixels.

Run the script:

python a9_laser_spot.py \
  --image "images/a9_lasers.jpg" \
  --baseline 10.0 \
  --focal 800.0 \
  --output "a9_result.jpg"


Why use A9?

Zero Calibration Targets: Unlike methods A7 or A8, you don't need to put a marker or grid on the surface. You only need the lasers.

Long Range: This works well in low-light environments where you can easily isolate the laser dots from the background noise.

Setup Requirement

Ensure the lasers are bright enough to show up clearly on the surface but not so bright that they saturate the camera sensor (which makes the dots appear as giant blurry white blobs).

If the camera sensor saturates, turn down your camera's exposure settings.