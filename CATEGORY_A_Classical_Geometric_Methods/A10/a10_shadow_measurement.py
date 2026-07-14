"""
A10 - Shadow Geometry (Similar Triangles)
===========================================
Math:
    H_target = H_ref * (S_target / S_ref)

    This relies on the sun's rays being parallel.
    We measure the shadow of a known reference (e.g., a 1m stick)
    and the shadow of a target (e.g., a tree) at the exact same time.
"""

import argparse

def calculate_height(ref_height, ref_shadow, target_shadow):
    """
    Computes target height using the ratio of shadows.
    """
    if ref_shadow == 0:
        raise ValueError("Reference shadow cannot be zero!")
        
    ratio = target_shadow / ref_shadow
    target_height = ref_height * ratio
    return target_height

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A10 - Shadow Geometry")
    parser.add_argument("--ref-height", type=float, required=True, help="Height of reference object (m)")
    parser.add_argument("--ref-shadow", type=float, required=True, help="Shadow length of reference (m)")
    parser.add_argument("--target-shadow", type=float, required=True, help="Shadow length of target (m)")
    
    args = parser.parse_args()
    
    try:
        height = calculate_height(args.ref_height, args.ref_shadow, args.target_shadow)
        print(f"Calculated Target Height: {height:.2f} meters")
    except Exception as e:
        print(f"Error: {e}")