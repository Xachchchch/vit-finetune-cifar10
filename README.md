# Vision Transformer (ViT) on CIFAR-10

This project demonstrates training and fine-tuning a Vision Transformer (ViT) on the CIFAR-10 dataset using PyTorch. A custom `Patching` module is used to tokenize the input image before feeding it into a pre-trained ViT model from the `timm` library.

---

## Features

-  Custom patch embedding module (`Patching`)
-  Pretrained ViT backbone (`timm`)
-  Selective fine-tuning (only head and positional embeddings are trained)
-  CIFAR-10 classification with data augmentation
-  Training metrics logged to Weights & Biases (W&B)

---

## Results (After 5 Epochs)

| Metric           | Value   |
|------------------|---------|
| Train Accuracy    | 47.3%   |
| Test Accuracy     | 51.0%   |
| Train Loss        | 1.45    |
| Test Loss         | 1.38    |

🔗 [View W&B Run](https://wandb.ai/khachblb06-polytechnic-of-a/VIT%20tuning/runs/v9obdhwo)

---

## Model Architecture

1. The image is split into non-overlapping 16×16 patches using a custom `Patching` class.
2. Patches are flattened and embedded.
3. The embeddings are passed to a pre-trained ViT from `timm`.
4. All transformer blocks are frozen.
5. Only the classification head and positional embeddings are trainable.
6. The `[CLS]` token is used for the final classification output.

---

## Using the Model

```python
from PIL import Image
import torchvision.transforms as transforms
import torch

# Load and preprocess image
image = Image.open("your_image.png").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
image = transform(image).unsqueeze(0).to(device)  # shape: [1, 3, 224, 224]

# Load model
model.eval()
model.to(device)

# Inference
with torch.no_grad():
    output = model(image)
    prediction = output.argmax(dim=1).item()

print("Predicted class:", prediction)

```
## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
   ```
2. Run the notebook or script.
