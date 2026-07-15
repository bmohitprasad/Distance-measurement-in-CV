Category A5: CCTV Ground-Plane Geometry

This method estimates the real-world distance from a fixed camera to an object by projecting a ray from the camera lens to the ground contact point of the object.

How to Run

Because this method relies on the camera's physical mounting properties, you must know your camera's height and tilt.

python3 a5_cctv_geometry.py --pixel-coords '301,405 295,300 281,395 320,394' --height 1.16 --tilt 30.0 --focal-mm 3.67 --sensor-width-mm 4.0 --image-width-px 640 --cy 240 --image images/a5_object.jpg --output output/a5_measurement_result.jpg




Arguments:

--row: The pixel row (y-coordinate) where the object touches the floor.

--height: The height of the camera above the floor (e.g., 3.0 meters).

--tilt: The camera's depression angle (e.g., 20 degrees).

--focal: Your camera's focal length in pixels.

--cy: The vertical center of your image (usually image_height / 2).

Why this works for the bottle height:

If you want the height of your bottle, you can calculate the distance to the base using pixel_row_base and calculate the distance to the top using pixel_row_top.

Since the base and top are at different "distances" from the camera (because the top is farther away in your tilted view), you can use the intercept theorem or compute the real-world Y-coordinate of both points using this method. The difference between the two is the bottle's height!