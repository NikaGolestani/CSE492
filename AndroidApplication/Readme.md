# Android App

The Android application that ties the saliency model and sensory scoring engine into a usable tool. Built in Java, runs entirely on-device.

---

## Architecture at a glance

Three main components:
- A **configuration settings module** (caregiver-only, passcode-protected)
- A **video analysis dashboard** (caregiver-only)
- A **simplified viewing interface** (child-facing, minimal)

---

## Dual-user design

The app defines two roles with separate interaction paths, because the
person configuring the app is never the same person using it.

**Primary user (the child):**
- Sees only a distraction-free video gallery and a fullscreen player.
- No settings, no extra navigation, no prompts.
- Gallery items are laid out with near-square aspect ratios for consistent
  visual processing across phones and tablets.
- **Why:** Children have zero tolerance for complex interfaces. Any menu or
  prompt risks immediate abandonment.

**Secondary user (the caregiver):**
- Imports videos, reviews sensory analysis results, decides which videos
  are shown, and configures attentional-guidance parameters.
- All caregiver-facing screens are behind a **local passcode**.
- **Why (passcode choice):** Cloud auth was ruled out for setup complexity.
  Math challenges were avoided because caregivers may be using this in
  high-stress moments. The passcode is a mode separator, not a security
  boundary in the traditional sense.

---

## Video lifecycle

Each imported video moves through a defined state machine:

```
Unprocessed → Processing → Analyzed → Cached → Visible (in gallery)
                   │
                   └──(decode/read error)──→ Failed (retry or discard)

Visible → UnderReview → Hidden (excluded by caregiver)
              └──(re-enabled by caregiver)──→ Visible
```

- Analysis runs automatically and asynchronously when a video is added, and
  does not block the UI.
- Extracted sensory features are cached to a local JSON store (see
  `VideoScoreCache`) keyed by a SHA-256 hash of the video file, so re-adding
  the same file doesn't trigger redundant re-analysis.
- Intermediate state is preserved so interrupted processing can resume
  instead of restarting from scratch.
- A caregiver can toggle **Exclude from Gallery** to hide a video from the
  child without deleting the underlying file.
- **Why:** This lifecycle gives caregivers control over content *before* it
  reaches the child. It also respects limited device resources by avoiding
  redundant work.

---

## Structural design (core components)

Local storage only — lightweight structured (JSON) files, no SQL/NoSQL
database, to keep things suitable for edge deployment.

| Component | Purpose |
| :--- | :--- |
| **`Video`** | File reference, identity, usage stats; composed with a `VideoFeatures` entity that only exists once analysis is complete (so "analyzed" vs. "unanalyzed" is a clean distinction, not a null-check). |
| **`VideoFeatures`** | Raw sensory signals, normalized penalties, safety score, and derived classification/recommendation. |
| **`VideoAnalyzer`** | Extracts sensory features from raw frames. |
| **`VideoScoreCache`** | Persistent read/write layer; hashes video content to prevent duplicate analysis entries. |
| **`VideoAdapter` / `VideoViewHolder`** | Connects the stored video list to the caregiver-facing UI. |
| **`AnalysisConfig`** | Runtime FPS, input size, thresholds, smoothing alpha, brightness/contrast, filter chain. |
| **`AppConfig`** | Gallery-level settings (pagination). |
| **`SettingsManager`** | Passcode verification, config load/save. |
| **`SaliencyFilter`** | Gain/bias applied to raw saliency output before rendering, so overlay intensity is adjustable without retraining the model. |

---

## Overlay calculation engine

Converts the raw saliency model output into a visual guide, applied on a
background thread so playback is never blocked:

```
Frame → Preprocess (resize/normalize) → Model → Raw saliency map
      → Threshold (drop low-confidence regions)
      → Four Circle Method (cluster focal points, prevent distant
        stimuli from merging into one blob)
      → Filter translation (map clusters → color/transparency)
      → Temporal smoothing (blend across frames, reduce jitter)
      → Composite onto display surface (UI thread)
```

### Overlay mode: Dynamic vs. Static

| Mode | Behavior | When to use |
| :--- | :--- | :--- |
| **Dynamic** | Region adapts to saliency heatmap variance; more accurate, but the added motion can itself be a sensory trigger. | Children who tolerate (or need) motion and benefit from precise tracking. |
| **Static** | Region size is fixed; less accurate, less motion. | Motion-sensitive children, or as a starting point for ASD children who are overwhelmed by dynamic elements. |

### Overlay filters (5, selectable per child)

| Filter | Effect | Use case |
| :--- | :--- | :--- |
| **Glow Neon** | High-contrast boundary line. | Primarily for debugging; visually harsh, rarely used with children. |
| **Biological Fovea** | Sharp salient region, heavily masked periphery (mimics foveal vision). | Simulates natural vision; good for transitioning to real-world habits. |
| **Blackout** | Darkens non-salient regions significantly. | ASD children overwhelmed by background clutter; reduces overall screen brightness. |
| **Vignette Shadow** | Smooth radial gradient. | Gentle, non-distracting; good for children sensitive to harsh edges. |
| **Mist Focus** | Soft white fade-out. | Users sensitive to harsh dark edges; creates a gentle, dreamlike transition. |



## Performance / inference rate

- Default inference rate: **5 FPS** (≈200ms budget per cycle on a
  mid-range device running the 2.39M-parameter float16 TFLite model),
  configurable up to 20 FPS on capable hardware.
- Inference is **fully suspended when video is paused** — no frames
  decoded, no model calls, background thread idles. This matters for
  battery/thermal behavior during long sessions.
- Emulator profiling (AVD, 4 vCPU, 2GB RAM): active playback at 5 FPS ≈
  303% aggregate CPU across 4 cores, ~4.88GB resident memory; paused ≈ 47%
  CPU, ~4.79GB resident memory. Real hardware is expected to do better due
  to hardware media codecs / GPU availability not present in the emulator.

---

## Settings exposed to the caregiver

| Setting | Options | Why |
| :--- | :--- | :--- |
| **Child Profile** | ASD / ADHD / Asperger's / Baby Digital | Determines whether the system tightens or widens during performance dips (see above). |
| **Overlay Filter Style** | Glow Neon, Biological Fovea, Blackout, Vignette Shadow, Mist Focus | Matches the child's visual comfort level. |
| **Radius Mode** | Fixed vs. Dynamic | Static for motion-sensitive children; dynamic for precise tracking. |
| **Inference FPS** | 1–20 | Balances precision and battery life. |
| **Videos Per Page** | Configurable pagination | Caregiver preference; reduces cognitive load when browsing. |

---

## Screens

| Screen | User | Description |
| :--- | :--- | :--- |
| **Passcode Entry** | Caregiver | Local passcode to access settings; mode separator, not security boundary. |
| **General Settings** | Caregiver | Live preview window, profile selection, filter style, radius mode, FPS. |
| **Videos Tab** | Caregiver | Per-video card with thumbnail, Safety Score, total views, watch time, flagged "Primary Concern" (e.g., Flicker, Rapid Motion), Exclude-from-Gallery toggle, "Show Analysis" dataset-level dashboard. |
| **Primary Gallery** | Child | Grid of approved videos only, near-square thumbnails, tap to launch fullscreen overlay playback. |

---

## Requirements compliance

| Requirement | Where it's implemented |
| :--- | :--- |
| **FR-01** Video import | Videos tab |
| **FR-02** Video deletion/exclusion | Videos tab (exclude toggle) |
| **FR-03** Sensory attribute display | `VideoAnalyzer` + Videos tab card |
| **FR-04** Filter configuration | General settings |
| **FR-05** FPS configuration | General settings |
| **NFR-01** Real-time performance | Background inference pipeline |
| **NFR-02** Smooth playback | TFLite on-device execution, dedicated thread |
| **NFR-03** Privacy / offline | On-device-only architecture, no network calls |

## Acknowledgements

Built with academic supervision, and with generous input from special
education specialists consulted during development on sensory heterogeneity,
caregiver access design, and per-feature sensory reporting.
