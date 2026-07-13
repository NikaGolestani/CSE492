package com.example.saliency.filter;

/**
 * Biologically-inspired Thresholding.
 * Instead of a hard cut to 0 (which causes flickering), this uses a "Soft-Knee"
 * to dim background noise while keeping the normative regions clear.
 */
public class ThresholdFilter implements SaliencyFilter {

    private final float staticThreshold;
    private final float dynamicFallbackPercentile;

    public ThresholdFilter(float staticThreshold, float dynamicFallbackPercentile) {
        this.staticThreshold           = staticThreshold;
        this.dynamicFallbackPercentile = dynamicFallbackPercentile;
    }

    @Override
    public float[][] apply(float[][] saliency) {
        int h = saliency.length;
        int w = saliency[0].length;

        float min = Float.MAX_VALUE;
        float max = -Float.MAX_VALUE;

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (saliency[y][x] < min) min = saliency[y][x];
                if (saliency[y][x] > max) max = saliency[y][x];
            }
        }

        float threshold = staticThreshold;
        if (max < staticThreshold) {
            // Fallback logic remains: ensures we always find a "target" in low-confidence frames
            threshold = min + (max - min) * dynamicFallbackPercentile;
        }

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float value = saliency[y][x];

                if (value < threshold) {
                    /* * BIOLOGICAL ADJUSTMENT: Soft-Knee
                     * Instead of snapping to 0.0f, we compress the background.
                     * This prevents "pixel popping" and mimics peripheral vision.
                     */
                    saliency[y][x] = value * 0.15f;
                } else {
                    /*
                     * NORMALIZATION:
                     * Ensure the salient region is boosted toward 1.0 (clear vision).
                     */
                    saliency[y][x] = Math.min(1.0f, value / (max + 0.0001f));
                }
            }
        }
        return saliency;
    }
}