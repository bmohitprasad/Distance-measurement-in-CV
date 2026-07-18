### 3. README for C3
```markdown:C3_Depth_from_Defocus/README.md
# Category C3: Depth from Defocus

## Theory
Depth from Defocus (DFD) assumes that the "blurriness" of an image is a function of the distance from the lens's focal plane. By convolving the image with a Laplacian operator, we quantify the "edginess." Areas with high edge strength (high variance) are in focus; areas with low edge strength are defocused.

## How to use images
*   **The "Focus Sweep" Technique:** To get a real depth map, you need a "focal stack." Take 5-10 photos of the same scene, but manually turn the focus ring on your lens slightly for each shot.
*   **The Script:** The provided script analyzes the blurriness of a single image. If you run it on a series of photos where the focus moves from the front to the back of the room, you will see the "hot spot" (the red area in the heatmap) travel across the objects in your room.

## Requirements
*   **Wide Aperture:** This method works best with a camera that has a "shallow depth of field" (e.g., a DSLR with a prime lens at f/1.8). If you use a phone camera with a massive depth of field (where everything is always in focus), the math becomes very noisy.
```eof

### Suggestions:
* **Why it matters:** This is the math behind "Portrait Mode" on smartphones. The phone detects the depth using a combination of stereo cameras or laser sensors, then applies a Gaussian blur to the "out-of-focus" areas to simulate this DFD effect.
* **Next Step:** Try taking 3 photos of your car: one focused on the front, one on the middle, and one on the back. Run the script on all three and compare the `c3_sharpness_map.jpg` outputs!