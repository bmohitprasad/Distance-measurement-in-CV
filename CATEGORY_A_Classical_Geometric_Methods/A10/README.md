Category A10: Shadow Geometry

This method uses the fact that the sun's rays arrive at the Earth essentially parallel. This creates Similar Triangles between objects and their shadows. By measuring the shadow of a known reference object (a stick of known height), you can calculate the height of any other object in the scene.

How to Run

You don't need a computer to do the math, but this script will handle the ratio calculations for you.

python a10_shadow_measurement.py \
  --ref-height 1.0 \
  --ref-shadow 1.4 \
  --target-shadow 8.4


The Math:

Because the triangle formed by the reference stick is similar to the triangle formed by the target object:

$$\frac{Height_{target}}{Shadow_{target}} = \frac{Height_{reference}}{Shadow_{reference}}$$

Solving for the target height:

$$Height_{target} = Height_{reference} \times \left( \frac{Shadow_{target}}{Shadow_{reference}} \right)$$

Usage Tips:

Ground: This only works on flat, level ground. If the shadow falls on a hill or a curb, the ratio will be broken.

Timing: Measure both shadows at the same time. Shadows change length rapidly as the sun moves!