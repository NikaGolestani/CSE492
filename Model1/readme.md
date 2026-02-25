# Gaze Prediction Model – CNN 

## Model

We used MobileNetV2 pre-trained on ImageNet as the backbone and converted it into a U-Net style encoder–decoder architecture. Skip connections from Blocks 3, 6, and 13 are fused into the decoder to preserve high-resolution spatial information, ensuring that predicted saliency peaks align precisely with real-world object locations instead of becoming blurry or misaligned.

---

## Data Strategy

The model was trained on the AVIMOS dataset (1000 videos). To reduce redundancy, we sampled every 10th frame and resized all frames to 224×224 to match the backbone input size. Raw gaze coordinates were mapped into this space, rendered as circular points, and smoothed using a 7×7 Gaussian filter to produce continuous heatmaps. Finally, heatmaps were normalized to [0,1] to stabilize training.

---

## Training Strategy

Because the full dataset exceeds RAM limits, videos were loaded in buffers of 10 at a time, with an 8:2 train-validation split per buffer. Each buffer was trained for 3 epochs to avoid overfitting and prevent the model from producing overly confident, binary-like outputs. We used a hybrid BCE + Dice loss, combining stable gradient descent with better shape refinement of saliency regions.

---

## Deployment

After training, the model was compressed using float16 post-training quantization, reducing its size and computational cost while maintaining performance. This enables efficient deployment on mobile devices and edge hardware.
It was reduced to 9MB from 54MB, enabling efficient deployment on mobile and edge devices without significant performance degradation.
