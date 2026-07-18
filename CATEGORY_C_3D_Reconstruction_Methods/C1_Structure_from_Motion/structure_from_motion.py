"""
c2_sfm_twoview.py
-----------------
A 2-View Structure from Motion pipeline.
Takes two overlapping images and your camera's intrinsic matrix, and generates
a 3D point cloud (.ply) of the scene.
"""

import cv2
import numpy as np
import argparse
import os

def export_point_cloud(filename, points_3d, colors):
    """
    Saves a 3D point cloud to a standard .ply file format.
    You can open this file in MeshLab, Blender, or Windows 3D Viewer.
    """
    print(f"Exporting {len(points_3d)} points to {filename}...")
    
    # Clean out any infinities or NaNs that OpenCV triangulation might produce
    mask = np.all(np.isfinite(points_3d), axis=1)
    points_3d = points_3d[mask]
    colors = colors[mask]

    # Create the PLY header
    header = f"""ply
format ascii 1.0
element vertex {len(points_3d)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    # Write to file
    with open(filename, 'w') as f:
        f.write(header)
        for i in range(len(points_3d)):
            x, y, z = points_3d[i]
            # OpenCV colors are BGR, PLY needs RGB
            b, g, r = colors[i]
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")
            
    print(f"Saved successfully to {filename}")

def reconstruct_3d(img1_path, img2_path, K):
    """
    The core SfM mathematical pipeline.
    1. Feature Extraction
    2. Feature Matching
    3. Essential Matrix Calculation
    4. Pose Recovery (Camera Movement)
    5. Triangulation (3D Math)
    """
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        raise ValueError("Could not load images. Check the paths.")

    # Convert to grayscale for feature extraction
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    print("1. Extracting SIFT Features...")
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    print(f"   Found {len(kp1)} keypoints in Image 1 and {len(kp2)} in Image 2.")

    print("2. Matching Features across images...")
    # FLANN is a fast nearest-neighbor matching algorithm
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's Ratio Test to keep only the highest quality matches
    good_matches = []
    pts1 = []
    pts2 = []
    colors = [] # We extract color from Image 1 for the 3D point cloud
    
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
            
            # Extract color at this pixel coordinate from img1
            x, y = int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])
            colors.append(img1[y, x])
            
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    colors = np.array(colors)
    print(f"   Kept {len(good_matches)} high-quality matches.")

    print("3. Computing Essential Matrix (E)...")
    # This solves x'^T E x = 0 to find the geometric relationship between the views
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    # Filter out outliers identified by RANSAC
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    colors = colors[mask.ravel() == 1]
    print(f"   {len(pts1)} matches survived the geometric RANSAC check.")

    print("4. Recovering Camera Pose (R, t)...")
    # This extracts the Rotation (R) and Translation (t) of Camera 2 relative to Camera 1
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    print("5. Triangulating 3D Points...")
    # Projection Matrix for Camera 1 (Assuming it is at the origin: 0,0,0)
    # [I | 0] means no rotation, no translation
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    
    # Projection Matrix for Camera 2 (Moved by R and t)
    P2 = K @ np.hstack((R, t))

    # OpenCV triangulates in Homogeneous Coordinates (X, Y, Z, W)
    # We must transpose the points to match the expected shape (2, N)
    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    
    # Convert Homogeneous back to 3D Cartesian coordinates (X, Y, Z)
    points_3d = points_4d_hom[:3, :] / points_4d_hom[3, :]
    points_3d = points_3d.T # Transpose back to (N, 3)

    return points_3d, colors, pts1, pts2, img1, img2, good_matches, kp1, kp2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-View Structure from Motion")
    parser.add_argument("--img1", required=True, help="Path to Left/First image")
    parser.add_argument("--img2", required=True, help="Path to Right/Second image")
    # Your Calibration values go here!
    parser.add_argument("--fx", type=float, required=True, help="Focal length X (px)")
    parser.add_argument("--fy", type=float, required=True, help="Focal length Y (px)")
    parser.add_argument("--cx", type=float, required=True, help="Principal point X (px)")
    parser.add_argument("--cy", type=float, required=True, help="Principal point Y (px)")
    parser.add_argument("--output", default="c2_point_cloud.ply", help="Output .ply file")
    
    args = parser.parse_args()
    
    # Build your Intrinsic Matrix
    K = np.array([
        [args.fx, 0, args.cx],
        [0, args.fy, args.cy],
        [0, 0, 1]
    ])
    
    try:
        points_3d, colors, pts1, pts2, img1, img2, matches, kp1, kp2 = reconstruct_3d(args.img1, args.img2, K)
        export_point_cloud(args.output, points_3d, colors)
        
        # Optional: Save a visualization of the matched features
        viz_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite("c2_feature_matches.jpg", viz_img)
        print("Saved feature matching visualization to c2_feature_matches.jpg")
        
    except Exception as e:
        print(f"Error during SfM pipeline: {e}")