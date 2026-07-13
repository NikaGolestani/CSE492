# Saliency Model

### Why saliency prediction (not face/gaze/motion detection separately)

An early design considered stitching together face detection, gaze tracking,
and motion detection. The problem: the list of "things that might matter"
never ends (animals, objects a child fixates on for other reasons, etc.), and
each detector has its own failure modes and needs its own weighting logic.
Saliency prediction sidesteps this by learning directly from where humans
actually look — one model, one input, one output, and it's what makes
real-time mobile deployment feasible.

### Architecture

- **Backbone:** MobileNetV2 (ImageNet pretrained, not frozen — fine-tuned for
  the gaze-prediction task), chosen over EfficientNet specifically for mobile
  deployment efficiency.
- **Structure:** U-Net-style encoder–decoder with skip connections. Three
  intermediate backbone layers are tapped:
  - Layer 3 — low-level spatial detail (edges, corners, textures)
  - Layer 6 — intermediate semantic structure (object parts, shapes)
  - Layer 13 — high-level semantic context (full object/scene representations)
- **Regularization:** SpatialDropout2D (channel-wise, preserves spatial
  consistency — chosen over standard dropout because nearby pixels are highly
  correlated).
- **Input:** 224×224×3 (RGB)
- **Output:** 224×224×1, sigmoid activation — per-pixel probability of gaze
  presence.
- **Parameters:** 2.39M (float32 checkpoint)

### Loss function

Standard saliency losses (KLD / SIM / CC — jointly referred to as the "KSS"
set) were tried but produced faint, low-confidence outputs unsuitable for
this application's needs. **Binary Cross-Entropy** was used instead, treating
the task as per-pixel classification:

```
L_BCE = -1/N * Σ [ y_i·log(ŷ_i) + (1-y_i)·log(1-ŷ_i) ]
```

BCE gives stable, aggressive pixel-wise gradients, which matters given the
class imbalance in gaze data (~10% salient pixels vs. ~90% background), and
forces the model to output distinct high-confidence blobs rather than
ambiguous smoothed regions.

### Training data

- **Dataset:** AVIMOS (1500 videos total; 1000 train / 500 test; of the
  training set, 800 train / 200 validation)
- Ground-truth gaze was collected by showing blurred videos to 70+
  participants who moved a mouse cursor to indicate gaze location.
- Frames sampled every 10 frames to avoid overfitting to near-duplicate
  consecutive frames.
- Point-based gaze labels were converted to a distribution via 7×7 Gaussian
  smoothing (so training targets are regions of probability, not single
  pixels; overlapping gaze points combine additively).
- Frames resized to 224×224, BGR→RGB.

### Training details

- Batched in groups of 16 videos (full dataset didn't fit in RAM).
- 3 epochs (vs. 15 tested) — more epochs produced smoother but less
  confident/less accurate region highlighting; 3 gave the best balance of
  consistency and generalization.
- ~10 hours total training time; checkpointing used to guard against loss of
  progress.
- Loss dropped from >0.5 to <0.2 within the first 10 batches; train/val loss
  stayed within 0.1–0.2 of each other afterward. Occasional spikes were
  traced to certain video types (notably vertical video) acting as outliers.

### Compression / deployment

- Saved as Keras `.h5` (54MB) → converted to **TFLite with float16
  quantization** → **9MB** `.tflite` model.
- The size reduction is larger than the theoretical 2× from float16 alone
  because TFLite conversion also strips training-only nodes and fuses
  operations.
- INT8 quantization was evaluated and rejected — required a calibration
  dataset and produced a noticeable accuracy drop; float16 gave a better
  size/accuracy tradeoff for this use case.
- Runtime: `org.tensorflow.lite.Interpreter` (CPU inference).

### Evaluation

Evaluated on the AVIMOS held-out test set (500 videos, 5 frames each, none
seen during training/validation):

| Metric | Mean | Std Dev | Interpretation |
|---|---|---|---|
| CC (Correlation Coefficient) | 0.52 | 0.10 | Moderate-to-strong structural correlation with human fixations |
| SIM (Similarity) | 0.32 | 0.05 | Limited but consistent distributional overlap |
| NSS (Normalized Scanpath Saliency) | 2.51 | 0.85 | Reliably assigns above-average saliency at actual fixation points |
| Recall | 0.69 | 0.15 | Captures most fixation regions |
| Area Percentage | 0.10 | — | Confirms recall isn't inflated by marking the whole frame salient |

**Benchmark comparison** (AIM 2024 challenge leaderboard, same 500-video test
set):

| Model | Backbone | SIM | CC | NSS | Params (M) |
|---|---|---|---|---|---|
| CV_MM | UMT | 0.635 | 0.774 | 3.464 | 420.5 |
| VistaHL | Dual-Stream HiSal | 0.623 | 0.769 | 3.352 | 187.7 |
| PeRCeiVe Lab | Multi-view Transformer | 0.610 | 0.766 | 3.422 | 402.9 |
| SJTU-MML | Audio-Visual U-Net | 0.615 | 0.760 | 3.356 | 1288.7 |
| MVP | Video Swin Transformer | 0.587 | 0.749 | 3.404 | 99.6 |
| ZenithChaser | Lightweight CNN | 0.517 | 0.606 | 2.482 | 20.19 |
| Exodus | Audio-Visual Transformer | 0.510 | 0.599 | 2.491 | 169.7 |
| **This model** | **Lightweight CNN (Optimized)** | **0.320** | **0.522** | **2.510** | **2.39** |

This model trails top leaderboard entries on CC/SIM (which use 8–540× more
parameters, often with audio-visual or transformer architectures), but
matches or marginally exceeds NSS against the next-smallest lightweight
competitors while using far fewer parameters (8.4× fewer than ZenithChaser,
71× fewer than Exodus) — the relevant tradeoff for real-time on-device
deployment.

**Performance by content type:** best on Social & Personal and Information &
Education content (stable regions — faces, slides). Weaker on Music &
Entertainment and Action/Gaming/Sports content (high motion, frequent scene
changes, competing objects). Note: this weaker category is *also* the
category most likely to be flagged as overstimulating by the sensory scoring
engine below — so degraded saliency accuracy on this content is a lesser
practical concern than it might first appear.




