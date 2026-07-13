package com.example.saliency.filter;

/**
 * A post-processing step applied to the raw saliency map produced by TFLite.
 *
 * <p>Implementations must be stateless <em>or</em> document their internal state
 * clearly (e.g. {@link SmoothingFilter} keeps a rolling average per pixel).
 *
 * <p>Filters receive a <strong>copy</strong> of the saliency values and may
 * modify the array in-place — callers should not rely on the input surviving
 * the call unchanged.
 */
public interface SaliencyFilter {

    /**
     * Apply this filter to a raw saliency map.
     *
     * @param saliency 2-D array [height][width] of sigmoid values in [0, 1].
     *                 May be modified in-place.
     * @return the filtered saliency map (may be the same array or a new one).
     */
    float[][] apply(float[][] saliency);
}
