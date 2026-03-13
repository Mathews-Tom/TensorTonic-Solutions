"""
## Anchor Box Generation

Object detectors like Faster R-CNN and SSD generate a dense set of predefined bounding boxes called anchors at every position on a feature grid. Each anchor serves as an initial guess that the network refines during training.

Given a square feature grid size, the original image size, a list of scales, and a list of aspect ratios, generate all anchor boxes in image coordinates.

### Algorithm

1. Compute the stride (spacing between grid cells in image space):

$$\text{stride}=\frac{\text{image\_size}}{\text{feature\_size}}$$
 
2. For each grid cell $(i, j)$, compute the center in image coordinates:

$$cx=(j+0.5)\times \text{stride}\:\:cy=(1+0.5)\times \text{stride}$$

3. For each combination of scale s and aspect ratio r, compute the box width and height:

$$w=s\sqrt{r}\:\:h\frac{s}{\sqrt{r}}$$
 
The anchor box is [cx - w/2, cy - h/2, cx + w/2, cy + h/2].
Iterate over grid cells in row-major order (i then j), and for each cell iterate over scales then aspect ratios.

### Examples

Input: feature_size = 1, image_size = 8, scales = [4], aspect_ratios = [1.0]

Output: [[2.0, 2.0, 6.0, 6.0]]

stride = 8, center = (4, 4), w = 4, h = 4. Box = [4-2, 4-2, 4+2, 4+2].

Input: feature_size = 2, image_size = 8, scales = [2], aspect_ratios = [1.0]

Output: [[1.0, 1.0, 3.0, 3.0], [5.0, 1.0, 7.0, 3.0], [1.0, 5.0, 3.0, 7.0], [5.0, 5.0, 7.0, 7.0]]

stride = 4. The four centers are (2,2), (6,2), (2,6), (6,6). Each box has w = 2, h = 2.

### Hints
Hint 1: The stride tells you how many image pixels each feature cell spans. The center of cell (i, j) is at ((j + 0.5) * stride, (i + 0.5) * stride).

Hint 2: For a given scale s and aspect ratio r, width = s * sqrt(r) and height = s / sqrt(r). This keeps the anchor area close to s * s regardless of aspect ratio.

### Requirements

- Map each grid cell center to image coordinates using the 0.5 offset
- Generate one anchor per (scale, aspect_ratio) pair at each grid position
- Return anchors in row-major grid order, then by scale, then by ratio

### Constraints

- feature_size >= 1, image_size >= 1
- scales and aspect_ratios are non-empty lists of positive floats
- Return a list of [x1, y1, x2, y2] boxes as floats
- Time limit: 300 ms
"""

import math

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    anchors = []
    stride = image_size / feature_size

    for i in range(feature_size):
        for j in range(feature_size):
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride

            for s in scales:
                for r in aspect_ratios:
                    w = s * math.sqrt(r)
                    h = s / math.sqrt(r)

                    x1 = cx - w / 2.0
                    y1 = cy - h / 2.0
                    x2 = cx + w / 2.0
                    y2 = cy + h / 2.0

                    anchors.append([x1, y1, x2, y2])

    return anchors