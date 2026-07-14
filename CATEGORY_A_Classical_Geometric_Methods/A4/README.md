Category A4: Homography Method

This module calculates real-world distances between objects lying on a flat plane (like the floor or a table) using a Homography perspective transformation matrix.

Folder Structure

A4_Homography/
│
├── a4_homography.py          # The main execution script
├── README.md                 # This file
└── utils/
    ├── __init__.py           # (Empty file to allow importing)
    └── camera_utils.py       # Shared drawing and loading logic


How to Run

Unlike A1 and A2, Homography requires you to know the $(X, Y)$ pixel coordinates of 4 reference corners in your photo, and their true real-world dimensions.

The Coordinate System Argument Format

You must pass a string of pairs separated by a comma (no spaces), with each pair separated by a space.

Format: "X1,Y1 X2,Y2 X3,Y3 X4,Y4"

Example Execution

Let's assume you have a photo (floor.jpg) of a rug on the ground.

The four corners of the rug in the photo are at pixels: 120,400, 520,400, 520,100, and 120,100.

You know the real-world rug is exactly 2 meters wide and 1 meter tall. So the true coordinates of those corners are: 0,0, 2,0, 2,1, and 0,1.

You want to measure the distance between two random points on the floor (e.g., dropped keys at 200,350 and a dropped coin at 450,150).

Run this command from inside the A4_Homography folder:

python a4_homography.py \
  --image "floor.jpg" \
  --img-pts "120,400 520,400 520,100 120,100" \
  --world-pts "0,0 2,0 2,1 0,1" \
  --measure-pts "200,350 450,150" \
  --output "homography_result.jpg"


The script will instantly reverse the perspective tilt of the camera, calculate the straight-line physical distance between the keys and the coin, and output a visualized image with a connecting line!