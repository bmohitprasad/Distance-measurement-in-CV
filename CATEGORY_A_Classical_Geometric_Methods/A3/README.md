Category A3: Stereo Photogrammetry

This module calculates a dense depth map using two images taken simultaneously from horizontally offset cameras (a stereo pair).

Folder Structure

A3_Stereo_Photogrammetry/
│
├── a3_stereo_photogrammetry.py     # The main execution script
└── README.md                       # This file


How to Run

You must provide two "rectified" stereo images. Do not attempt this with two photos taken by moving a single phone by hand; micro-rotations will cause the block-matching algorithm to fail. Use sample images from datasets like KITTI or Middlebury.

Run this command from inside the A3_Stereo_Photogrammetry folder:

python a3_stereo_photogrammetry.py --left "images/left.jpg" --right "images/right.jpg" --focal 718.856 --baseline 0.54 --output "heatmap.jpg"


Arguments:

--left: Path to the left camera image.

--right: Path to the right camera image.

--focal: Focal length in pixels. Defaults to 718.856 (Standard KITTI dataset focal length).

--baseline: Physical distance between the two cameras. Defaults to 0.54 meters (Standard KITTI rig).

--output: Saves a color-coded "thermal" heatmap where red/warm colors mean the object is close, and blue/cool colors mean the object is far away.