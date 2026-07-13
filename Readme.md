# Attentional Guidance for ASD — Edge AI Android System

A free, offline, real-time assistive tool for children with Autism Spectrum
Disorder (ASD), built as a standalone Android application. It does two things:

1. **AI Attentional Guidance** — an on-device saliency model highlights the
   regions of a video most likely to carry social/linguistic meaning (faces,
   relevant objects), reducing the cognitive load of figuring out where to
   look.
2. **Sensory / Overstimulation Scoring** — an on-device analysis engine scores
   each video for overstimulation risk (flicker, hard cuts, motion, color
   variance, visual complexity) *before* it's shown to the child, so a
   caregiver can screen content in advance.

Everything runs locally on the device. No video data leaves the phone, no
internet connection is required after install, and there is no subscription
or recurring cost.

This grew out of an engineering project report written for a university
course, *Edge AI for Attentional Guidance: A Localized Android System for
Sensory-Aware ASD Support* (Yeditepe University, 2026) — not a published or
peer-reviewed paper. It's included in this repo (`/report`) for reference,
and the project itself also grew out of conversations with a
special education specialist during development. **Before contributing,
please read [`ProblemDefinition.md`](ProblemDefinition.md)** — it
explains *why* the system is built the way it is, including some tradeoffs
that aren't obvious from the code alone (e.g. why more visual detail is
usually worse for ASD, but why ADHD requires the opposite).

---

## Why this exists

Specialist interventions for ASD (e.g. Applied Behavior Analysis) work, but are expensive and require one-on-one delivery, putting them out of reach for many families. Research-grade solutions that guide a child's attention during video playback exist, but rely on dedicated eye-tracking hardware and a clinical setting. This project asks a narrower, practical question: can a comparable effect be delivered on a phone the family already owns, for any video, for free, entirely offline?

### Why this is harder than it looks (The Spectrum Nuance)

A crucial insight from special education is that **one size does not fit all**:
- A child with **ASD** is often *overwhelmed* by too much input. More detail = more overwhelm and less eye contact. They benefit from the viewing area being **tightened** (reduced) to block background noise.
- A child with **ADHD** is often *under-stimulated*. They lose focus without explicit visual anchors. They benefit from the viewing area being **widened** or made more dynamic to maintain engagement.

This means the system cannot rely on a single rule (e.g., "always shrink the screen when the child struggles"). It must adapt directionally based on the child's specific neurotype—a requirement the technical solution must explicitly address.

---

## Requirements this project was built around

- **Free** — no subscription, no paid tiers
- **Offline** — no cloud dependency, no video data transmitted anywhere
- **Fast** — real-time inference on standard (not flagship) Android hardware
- **Configurable by Profile, not fixed** — because ASD is a spectrum, no single overlay or sensory-filtering configuration is "correct" for every child; the caregiver configures it per child (e.g., setting the child's profile to ASD vs. ADHD so the system knows whether to tighten or widen).
- **Implicit Tracking (No Tests)** — children cannot take tests. The system must infer progress and adaptation purely from passive viewing behavior (e.g., gaze alignment, re-engagement speed).

---

## Status

This is a thesis-derived, cold-start version of the system — it works and has
been evaluated, but it's an early foundation rather than a finished clinical
product. Known limitations are listed at the end of
`ProblemDefinition.md`, and include:

- **Visual-only input** (no audio processing); sound-driven attention is ignored.
- **Animated / Cartoon Data Gap**: Currently, eye-tracking datasets contain
  almost exclusively real-world footage. This means the system performs
  *better on live-action videos* (documentaries, educational clips with real
  humans) than on cartoons/CGI. Face isolation, frequently requested by
  caregivers for Asperger's, is not reliable on animated media without
  dedicated new data collection.
- **No automatic personalization yet** — the system relies on manual caregiver
  profile selection to adapt (ASD/ADHD).

---


## Acknowledgements

Built with academic supervision, and with generous input from special
education specialists consulted during development on sensory heterogeneity,
caregiver access design, and per-feature sensory reporting.
