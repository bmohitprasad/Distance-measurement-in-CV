Category C2: Structure from Motion (SfM)

This module moves beyond single-image geometry into Multi-View Geometry. By analyzing how a scene changes between two photographs taken from different positions, we can extract the exact 3D coordinates of the objects in the scene.

1. How to Capture Images for SfM

Structure from Motion algorithms are incredibly powerful, but they will instantly fail if your photos violate the rules of Epipolar Geometry.

Rule 1: You MUST Translate (Move), not just Rotate.

BAD: Standing in one spot and turning your camera (panning/tilting) like a panorama. SfM relies on parallax. If the camera doesn't physically move its position in space, the math results in a division-by-zero error.

GOOD: Take a photo, step 1 foot to the right, and take another photo looking at the same object.

Rule 2: The Overlap Rule (70% - 80%)

The script needs to find the exact same pixels in both images. Make sure the object you want to reconstruct is clearly visible in the center of both shots.

Rule 3: Texture is King

SfM cannot reconstruct a blank white wall or a completely smooth, featureless surface (like a shiny mirror or a clear glass bottle). It needs "texture" (wood grain, text, patterns, corners) to extract SIFT features.

2. Mathematical Intuition & Working Principle

The c2_sfm_twoview.py script executes a 5-step mathematical pipeline.

Step 1: Feature Extraction (SIFT)

We cannot compare every pixel. Instead, the algorithm finds "Keypoints"—corners and edges that are highly unique. For every keypoint, it generates a 128-number vector called a "Descriptor" that acts like a fingerprint for that pixel.

Step 2: Feature Matching

We compare the fingerprints in Image 1 to Image 2. If a pixel fingerprint on a desk in Image 1 matches a pixel fingerprint in Image 2, we assume they are the same physical 3D point.

Step 3: Epipolar Geometry & The Essential Matrix ($E$)

This is the core of the physics. Because we calibrated the camera (giving us matrix $K$), the algorithm uses the matched pixels to calculate the Essential Matrix ($E$).

The equation is:  $x'^T E x = 0$
(where $x$ is a pixel in Image 1, and $x'$ is the same pixel in Image 2).

The Essential Matrix mathematically describes the physical relationship between the two camera lenses in 3D space.

Step 4: Pose Recovery (Camera Movement)

The Essential Matrix $E$ is actually composed of two physical things:

$$E = [t]_{\times} R$$

$R$ (Rotation): How the camera rotated between shot 1 and shot 2.

$t$ (Translation): The 3D vector representing how the camera moved (up/down, left/right, forward/back).

Using cv2.recoverPose, we mathematically rip $E$ apart to get exactly how you moved your hand between taking the two photos!

Step 5: Triangulation

Now we have:

Where Camera 1 is.

Where Camera 2 is.

A pixel ray shooting out from Camera 1.

A pixel ray shooting out from Camera 2.

We use cv2.triangulatePoints to find exactly where those two rays intersect in 3D space. That intersection is the true 3D coordinate $(X, Y, Z)$ of the object.

3. How to Run the Code

Now that you have your calibration values from Category B (e.g., your 1600x1204 focal lengths and principal points), you can run this script.

Take two overlapping photos of an object (e.g., your desk or a toy) by stepping slightly to the side between shots. Let's call them img_left.jpg and img_right.jpg.

Run the command:

python c2_sfm_twoview.py \
  --img1 "img_left.jpg" \
  --img2 "img_right.jpg" \
  --fx 1471.0 \
  --fy 1471.0 \
  --cx 800.0 \
  --cy 602.0 \
  --output "my_desk.ply"



(Note: Replace --fx, --fy, --cx, and --cy with your actual calibration output!).

Viewing the Output

The script generates a file named my_desk.ply. This is a standard 3D point cloud file.

Windows: You can literally double-click it to open it in "3D Viewer".

Mac/Linux: Download MeshLab (free, open-source) to open it, rotate it, and view the actual 3D geometry of the scene you just photographed!