package com.example.saliency.filter;

/**
 * Implements the enhancement pipeline from the technical notes:
 * I_total = I_base + I_sharpened
 */
public class SaliencyEnhancementFilter implements SaliencyFilter {
    private final float gain; // 'B' in notes - Brightness scaling
    private final float bias; // 'A' in notes - Contrast/Offset scaling

    private static final float[][] SHARPEN_KERNEL = {
            {  0f, -1f,  0f },
            { -1f,  5f, -1f },
            {  0f, -1f,  0f }
    };

    public SaliencyEnhancementFilter(float gain, float bias) {
        this.gain = gain;
        this.bias = bias;
    }

    @Override
    public float[][] apply(float[][] inputMap) {
        int h = inputMap.length;
        int w = inputMap[0].length;

        // Step 1: I_base = gain * inputMap
        float[][] baseLayer = new float[h][w];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                baseLayer[y][x] = clamp(gain * inputMap[y][x]);

        // Step 2: I_contrast = bias * baseLayer (centered)
        float[][] contrastLayer = new float[h][w];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                contrastLayer[y][x] = clamp(0.5f + bias * (baseLayer[y][x] - 0.5f));

        // Step 3: I_edges = convolve(I_contrast, SHARPEN_KERNEL)
        float[][] edgeLayer = convolve(contrastLayer, SHARPEN_KERNEL);

        // Step 4: I_total = I_base + I_edges
        float[][] enhancedMap = new float[h][w];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                enhancedMap[y][x] = clamp(baseLayer[y][x] + edgeLayer[y][x]);

        return enhancedMap;
    }

    private float[][] convolve(float[][] src, float[][] kernel) {
        int h = src.length; int w = src[0].length;
        float[][] out = new float[h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float sum = 0f;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int sy = y + ky; int sx = x + kx;
                        float val = (sy >= 0 && sy < h && sx >= 0 && sx < w) ? src[sy][sx] : 0f;
                        sum += val * kernel[ky + 1][kx + 1];
                    }
                }
                out[y][x] = clamp(sum);
            }
        }
        return out;
    }

    private static float clamp(float v) { return Math.max(0f, Math.min(1f, v)); }
}