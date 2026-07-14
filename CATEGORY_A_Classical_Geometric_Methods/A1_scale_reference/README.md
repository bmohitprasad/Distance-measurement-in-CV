Category A1: Scale Reference Method

This module contains a fully standalone implementation of the A1 classical geometric method. It calculates the real-world dimensions of objects based on a known reference object in the same image frame.

Folder Structure

A1_Scale_Reference/
│
├── a1_scale_reference.py     # The main execution script
├── README.md                 # This file
└── utils/
    ├── __init__.py           # (Empty file to allow importing)
    └── camera_utils.py       # Shared OpenCV contour logic



How to Run

Because this folder is completely isolated, you can run the script directly from the terminal.

Navigate into the A1_Scale_Reference directory and run:

python a1_scale_reference.py --image "path_to_your_image.jpg" --ref-width 21.0 --output "result.jpg"



Arguments:

--image: (Required) Path to the photo containing your reference and target objects.

--ref-width: (Required) The physical width of your reference object (e.g., 21.0 if using A4 paper).

--output: (Optional) Path to save the beautifully annotated image with colored bounding boxes. Defaults to a1_output_annotated.jpg.

--ref-index: (Optional) Defaults to 0. If your reference object is not the left-most object in the photo, change this index.