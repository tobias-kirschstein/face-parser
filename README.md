# face-parser

Usage:
```python
from face_parser.bisenet import BiSeNetFaceParser
from face_parser.visualize import apply_colormap

img = load_img()  # torch.Tensor [3, H, W]
face_parser = BiSeNetFaceParser()
segmentation_mask = face_parser.parse(img)

segmentation_mask_colored = apply_colormap(segmentation_mask)  # Colorizes each class with a distinct color for better viewing
plt.imshow(segmentation_mask_colored)

```
