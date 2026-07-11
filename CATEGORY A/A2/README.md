Category A2: Single-Image Photogrammetry

This module contains a fully standalone implementation of the A2 classical geometric method, determining the distance of an object using similar triangles.

Folder Structure

A2_Single_Image_Photogrammetry/
│
├── a2_single_image_photogrammetry.py     # The main execution script
├── README.md                             # This file
└── utils/
    ├── __init__.py                       # (Empty file to allow importing)
    └── camera_utils.py                   # Shared OpenCV contour logic


How to Run

Method A2 works in two steps. First, you calculate the focal length of your camera (calibrate). Then, you reuse that focal length to calculate the distance of the object in new images (measure).

Navigate into the A2_Single_Image_Photogrammetry directory and use the commands below.

Step 1: Calibrate

Take a photo of an object of known width at a specific, known distance. Ensure the object is the largest item in the frame and does not touch the edges.

python a2_single_image_photogrammetry.py calibrate --image "path_to_calib_photo.jpg" --distance 50.0 --width 15.0 --output "calib_result.jpg"


Note the "focal_length_px" returned in your terminal!

Step 2: Measure

Take a new photo of that same object (or another object with the same known width) at a new, unknown distance. Feed the focal length from Step 1 back into the algorithm.

python a2_single_image_photogrammetry.py measure --image "path_to_new_photo.jpg" --focal 850.5 --width 15.0 --output "measure_result.jpg"


Debug Mode

If the algorithm calculates a completely wrong focal length or distance (e.g., < 20 cm), it means the algorithm boxed the background instead of the object. Add --debug to your command to print out exactly what shapes the algorithm is considering!

python a2_single_image_photogrammetry.py measure --image "photo.jpg" --focal 850.5 --width 15.0 --debug
