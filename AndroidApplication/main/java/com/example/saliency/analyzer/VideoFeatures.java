package com.example.saliency.analyzer;

public class VideoFeatures {

    // --------------------
    // RAW SIGNALS
    // --------------------
    public final float rawFlicker;
    public final float cutsPerSec;

    // motion + visual signals (pre-normalization optional)
    public final float motionSignal;
    public final float colorSignal;

    // --------------------
    // NORMALIZED PENALTIES (0–1)
    // --------------------
    public final float flash;
    public final float motion;
    public final float cuts;
    public final float color;
    public final float complexity;

    // --------------------
    // EMERGENT METRICS
    // --------------------
    public final float chaos;
    public final float shock;

    // --------------------
    // OUTPUT
    // --------------------
    public final int score;

    public VideoFeatures(
            float flash,
            float motion,
            float cuts,
            float color,
            float complexity,
            float chaos,
            float rawFlicker,
            float cutsPerSec,
            float score
    ) {
        this.flash = flash;
        this.motion = motion;
        this.cuts = cuts;
        this.color = color;
        this.complexity = complexity;
        this.chaos = chaos;

        this.rawFlicker = rawFlicker;
        this.cutsPerSec = cutsPerSec;

        this.motionSignal = motion;   // optional alias
        this.colorSignal = color;

        this.shock = 0f; // if you compute it, store it properly

        this.score = (int) score;
    }

    public String getPrimaryConcern() {
        float max = Math.max(
                Math.max(flash, motion),
                Math.max(color, complexity)
        );

        if (max < 0.3f) return "None";
        if (flash == max) return "Flicker/Flash";
        if (motion == max) return "Rapid Motion";
        if (color == max) return "Color Saturation";
        return "Visual Complexity";
    }

    public static int pct(float v) {
        return Math.max(0, Math.min(100, Math.round(v * 100f)));
    }
    public String getClassification() {
        if (score >= 80) return "COMFORTABLE";
        if (score >= 55) return "MODERATE";
        return "OVERLOADING";
    }
    public String getRecommendation() {
        if (score >= 80) return "Safe for prolonged viewing.";
        if (score >= 55) return "Use with supervision.";
        return "High sensory load. Limit exposure.";
    }
}