package com.example.saliency.filter;

public class SmoothingFilter implements SaliencyFilter {
    private final float baseAlpha;
    private float[][] buffer;

    public SmoothingFilter(float baseAlpha) {
        this.baseAlpha = baseAlpha;
    }

    @Override
    public float[][] apply(float[][] current) {
        int h = current.length;
        int w = current[0].length;

        if (buffer == null || buffer.length != h) {
            buffer = deepCopy(current);
            return current;
        }

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float diff = Math.abs(current[y][x] - buffer[y][x]);

                // Adaptive Smoothing Logic:
                // High difference = Saccade/Movement (low smoothing)
                // Low difference = Jitter/Noise (high smoothing)
                float adaptiveAlpha = Math.max(baseAlpha, Math.min(1.0f, diff * 1.8f));

                buffer[y][x] = (adaptiveAlpha * current[y][x]) +
                        ((1.0f - adaptiveAlpha) * buffer[y][x]);

                current[y][x] = buffer[y][x];
            }
        }
        return current;
    }

    private float[][] deepCopy(float[][] original) {
        float[][] copy = new float[original.length][original[0].length];
        for (int i = 0; i < original.length; i++) {
            System.arraycopy(original[i], 0, copy[i], 0, original[i].length);
        }
        return copy;
    }
}