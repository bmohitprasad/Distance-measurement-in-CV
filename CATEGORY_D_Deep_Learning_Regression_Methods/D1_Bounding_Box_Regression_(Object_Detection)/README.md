# METHOD D1 — Bounding Box Regression (Object Detection)
### Measuring real-world object size using a trained deep neural network

---

## 1. What Problem This Solves

Every method before this one (A1–A10 classical, B1–B4 calibration) required
**you** to locate the object in the image — click points, threshold pixels,
find contours by hand. D1 is the first method where a neural network finds
the object automatically. You show it any photo; it tells you *what* is in
the photo and *where* (as a bounding box), for objects it has learned to
recognize.

This is the conceptual bridge from classical computer vision into deep
learning for your PhD project. Everything from here (E — depth estimation,
F — keypoints) builds on this same idea: a network trained on millions of
labelled examples generalizes to images it has never seen.

---Goodfellow

## 2. The Core Idea, in One Sentence

> A convolutional network looks at an image, proposes thousands of candidate
> boxes, scores each one ("is there an object here, and what class is it?"),
> then **regresses** (adjusts) each box's coordinates to tightly fit the
> real object boundary.

The word "regression" here means exactly what it means in statistics: the
network predicts continuous numbers — the four box coordinates
`[x1, y1, x2, y2]` — rather than a discrete label. This is why it's called
**bounding box regression**.

---

## 3. Mathematical Foundation

### 3.1 The Two-Stage Detector Pipeline (Faster R-CNN)

```
Input image
     |
     v
[Backbone CNN: ResNet-50]  -->  extracts feature maps at multiple scales
     |
     v
[Feature Pyramid Network]  -->  combines scales so small AND large
     |                          objects are both detectable
     v
[Region Proposal Network]  -->  slides "anchor boxes" over the feature
  (RPN)                         map, proposes ~2000 candidate regions
     |                          that MIGHT contain an object
     v
[RoI Align]                -->  crops and resizes each proposed region
     |                          to a fixed size for the next stage
     v
[Classification head]      -->  "what class is this?" (91 COCO classes)
[Box regression head]      -->  "adjust this box to fit the object exactly"
     |
     v
[Non-Max Suppression]      -->  removes duplicate overlapping boxes
     |
     v
Final boxes + class labels + confidence scores
```

### 3.2 Anchor Boxes — the geometric starting point

At every spatial location on the feature map, the RPN places **k** anchor
boxes of different sizes and aspect ratios (typically k=9: 3 scales × 3
ratios). Each anchor is a hypothesis: "maybe there's an object roughly this
shape, centred here."

```
Anchor definition:
    center = (x_a, y_a)      — feature map grid location
    scale  = {128, 256, 512} — pixels (relative to input image)
    ratio  = {1:2, 1:1, 2:1} — width:height

For a feature map of size H x W, total anchors = H x W x k
```

### 3.3 The Regression Target — how "adjust the box" is expressed

The network doesn't directly predict `[x1,y1,x2,y2]`. It predicts small
**offsets** relative to the anchor box — this is far easier to learn than
predicting raw pixel coordinates from scratch.

```
Given anchor box: (x_a, y_a, w_a, h_a)   [center-x, center-y, width, height]
Given ground truth box: (x, y, w, h)

The network predicts 4 numbers (t_x, t_y, t_w, t_h) such that:

    t_x = (x - x_a) / w_a          <- normalized center shift
    t_y = (y - y_a) / h_a
    t_w = log(w / w_a)             <- log-scale size ratio
    t_h = log(h / h_a)

To recover the actual box at inference time:
    x_pred = t_x * w_a + x_a
    y_pred = t_y * h_a + y_a
    w_pred = w_a * exp(t_w)
    h_pred = h_a * exp(t_h)
```

**Why log-scale for width/height?** Object sizes vary over a huge range
(a person vs. a bottle cap). Predicting `log(w/w_a)` compresses that range
so the network doesn't need to output huge or tiny raw numbers — it just
predicts a reasonable multiplicative adjustment.

### 3.4 The Loss Functions (this is your Prof. Sa deliverable)

Faster R-CNN trains with a **multi-task loss** — two losses added together,
computed at TWO stages (RPN and final detection head):

```
L_total = L_cls + λ · L_box

L_cls  = classification loss  (is there an object? which class?)
L_box  = box regression loss  (how far off is the predicted box?)
λ      = balancing weight, typically 1.0
```

**Classification loss — Cross-Entropy:**
```
L_cls = -log(p_class)

Where p_class = predicted softmax probability of the correct class
```

**Box regression loss — Smooth L1 (Huber-style):**
```
smooth_L1(x) = 0.5x²        if |x| < 1
             = |x| - 0.5    otherwise

L_box = Σ smooth_L1(t_pred - t_target)   summed over the 4 coordinates
        (x, y, w, h) described in section 3.3
```

**Why Smooth L1 instead of plain MSE?** Early in training, predicted boxes
can be wildly wrong. MSE (`x²`) would produce enormous gradients for large
errors, destabilizing training. Smooth L1 behaves like L2 (smooth,
well-behaved) for small errors but switches to L1 (constant gradient, no
explosion) for large errors — exactly the Huber loss you saw in earlier
methods, applied here to box coordinates specifically.

**Only used for POSITIVE anchors:** Box regression loss is only computed
for anchors that overlap a ground-truth object above a threshold (IoU >
0.7 = positive, IoU < 0.3 = negative/background, in between = ignored).
Background anchors don't need box regression — there's nothing to fit.

### 3.5 Non-Max Suppression (NMS) — removing duplicates

The RPN proposes ~2000 overlapping boxes for a single object. NMS keeps
only the best one:

```
1. Sort all boxes by confidence score, descending
2. Take the highest-scoring box, add to keep-list
3. Remove all remaining boxes with IoU > threshold (e.g. 0.5)
   against the box just kept
4. Repeat from step 2 until no boxes remain
```

```python
def iou(box1, box2):
    xa = max(box1[0], box2[0]); ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2]); yb = min(box1[3], box2[3])
    inter = max(0, xb-xa) * max(0, yb-ya)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    return inter / (area1 + area2 - inter + 1e-7)
```

### 3.6 From Bounding Box to Real-World Size — the part D1 adds on top

Once the network outputs `[x1,y1,x2,y2]` in pixels, converting to real
units uses the **exact same scale-factor principle as Method A1**:

```
width_px  = x2 - x1
height_px = y2 - y1

width_cm  = width_px  × scale     where scale = cm/pixel
height_cm = height_px × scale
```

The deep learning part (everything in 3.1–3.5) is entirely about finding
`[x1,y1,x2,y2]` automatically. The measurement conversion at the end is
identical, simple geometry — this is an important point to understand:
**deep learning replaces manual localization, not the physics of
measurement.**

---

## 4. Why Faster R-CNN Specifically (vs YOLO, SSD)

| Detector | Speed | Accuracy | Why/why not here |
|---|---|---|---|
| Faster R-CNN | Slower (~5 FPS CPU) | Highest | Used here — best accuracy for research measurement, no real-time need |
| YOLO | Fast (30+ FPS) | Good | Better for real-time video, covered separately if needed |
| SSD | Fast | Moderate | Rarely used now, superseded by YOLO variants |

For your PhD work — measuring produce quality, not tracking video in real
time — accuracy matters more than speed. Faster R-CNN is the right choice
for this method.

---

## 5. What Images to Capture

### 5.1 Two things every image needs

1. **A reference object of known real size** somewhere in frame (ruler,
   A4 sheet, coin, credit card) — used to compute `scale_cm_per_px`.
2. **The object(s) you want measured**, visible and not too occluded.

### 5.2 Important limitation — read this before capturing

The pretrained model recognizes **91 COCO categories**: person, bicycle,
car, dog, cat, bottle, cup, apple, orange, banana, chair, laptop, etc.
**It does NOT know "rice grain" or "mango" or "goat" out of the box** —
these are not COCO classes.

```
This means for your actual PhD project (rice grading), D1 as shown here
will NOT detect rice grains by class name. It demonstrates the FULL
detection pipeline correctly, but for rice-specific detection you would
need to either:

  (a) Fine-tune this same Faster R-CNN on your own labelled rice images
      (transfer learning — swap the final classification layer,
       retrain on your dataset), or

  (b) Use the closest COCO categories the demo DOES support to verify
      the pipeline works (banana, apple, orange — for produce testing)
```

For learning and testing the pipeline right now, capture photos of:
- **A banana, apple, or orange** next to a ruler or A4 sheet
- **A bottle or cup** next to a reference
- **Any COCO object** — this validates your code works correctly

### 5.3 Capture checklist

```
[ ] Reference object clearly visible, not blurry, flat orientation
[ ] Target object (COCO class) clearly visible, minimal occlusion
[ ] Good lighting — avoid harsh shadows or backlighting
[ ] Camera roughly perpendicular to the scene (top-down or straight-on)
[ ] Object not touching image edges (full box must be visible)
[ ] Resolution at least 640x480, ideally higher for small objects
```

### 5.4 Example capture setup

```
Photograph an apple next to a printed A4 sheet edge:
  1. Place A4 sheet flat on table
  2. Place apple on/near the sheet
  3. Photograph from directly above, sheet fully in frame
  4. Note: A4 width = 21cm (use this as your reference length)
```

---

## 6. How To Run

### 6.1 Install (one time)

```bash
pip install torch torchvision opencv-python numpy
```

First real run downloads Faster R-CNN's pretrained weights (~160MB) from
PyTorch's servers — needs internet once, then cached locally forever.

### 6.2 Demo mode — verify pipeline works, no real image needed

```bash
python method_d1_bbox.py --demo
```

### 6.3 Real image — interactive reference (recommended)

```bash
python method_d1_bbox.py \
    --image your_photo.jpg \
    --interactive_ref --ref_cm 21.0 \
    --conf 0.5
```

Click two points spanning your known 21cm reference (e.g. A4 sheet width)
when the window opens.

### 6.4 Real image — scale already known

```bash
python method_d1_bbox.py --image your_photo.jpg --scale 0.05 --conf 0.5
```

### 6.5 Only detect specific classes

```bash
python method_d1_bbox.py --image fruit_bowl.jpg \
    --interactive_ref --ref_cm 21.0 \
    --classes apple orange banana
```

### 6.6 Parameters explained

| Flag | Meaning |
|---|---|
| `--conf` | Minimum confidence to keep a detection (0–1). Lower = more (noisier) detections. Default 0.5. |
| `--interactive_ref` | Click two points on a known-length reference to compute scale automatically |
| `--ref_cm` | Real length of that reference, in cm |
| `--scale` | Skip reference-clicking, directly give cm/pixel if already known |
| `--classes` | Restrict output to specific COCO class names |

---

## 7. Reading the Output

```
[1] apple (conf 0.94)   8.2 x 7.9 cm  | area=64.78cm2  diag=11.4cm
```

- **conf 0.94** — the classification head is 94% confident this is an apple
- **8.2 x 7.9 cm** — bounding box width × height converted via your scale
- **area** — width × height (box area, not the object's actual silhouette
  area — a limitation discussed below)

### 7.1 Important accuracy note

A bounding box is a rectangle drawn around the object — for round or
irregular objects (apple, mango, rice grain) the box will always be
**slightly larger** than the object's true extent, especially along the
diagonal. For precise dimension measurement, combine D1 (to locate the
object automatically) with Method A6 (contour geometry) applied **within**
the detected box — this gets you automatic localization AND precise
silhouette-based measurement. That combination is a natural next
implementation step for your project.

---

## 8. Connecting This Back to Your Rice Project

For rice quality grading specifically, this exact architecture (Faster
R-CNN, or its faster cousin YOLOv8) is what you'll eventually **fine-tune**
on your own labelled rice grain images — replacing "apple/orange/banana"
with "whole grain / broken grain / chalky grain / foreign matter". The
code structure stays identical; only the training data and final
classification layer change. Understanding D1 deeply now is the
foundation for that fine-tuning work later in your PhD.