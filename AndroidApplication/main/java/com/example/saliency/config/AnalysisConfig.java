package com.example.saliency.config;

import com.example.saliency.filter.SaliencyEnhancementFilter;
import com.example.saliency.filter.SaliencyFilter;
import com.example.saliency.filter.SmoothingFilter;
import com.example.saliency.filter.ThresholdFilter;
import com.example.saliency.overlay.OverlayStyle;

import java.util.Arrays;
import java.util.List;

/**
 * Central configuration for saliency analysis.
 *
 * Compile-time constants (ALL_CAPS finals) are the factory defaults.
 * RUNTIME_ fields hold the live values used by the app; they are populated
 * by SettingsManager.loadAll() on start and written back by SettingsManager.saveAll().
 *
 * Call rebuildFilterChain() after changing any filter parameter so the
 * active FILTER_CHAIN picks up the new values immediately.
 */
public class AnalysisConfig {

    // -----------------------------------------------------------------------
    // Compile-time defaults
    // -----------------------------------------------------------------------

    public static final int ANALYSIS_FPS    = 5;
    public static final long ANALYSIS_INTERVAL_MS = 1000L / ANALYSIS_FPS;
    public static final int INPUT_SIZE      = 224;
    public static final int OVERLAY_PADDING = 50;

    // -----------------------------------------------------------------------
    // Runtime-mutable values (initialised from defaults, overwritten by SettingsManager)
    // -----------------------------------------------------------------------

    public static int  RUNTIME_FPS         = ANALYSIS_FPS;
    public static int  RUNTIME_INPUT_SIZE  = INPUT_SIZE;
    public static int  RUNTIME_PADDING     = OVERLAY_PADDING;
    public static long RUNTIME_INTERVAL_MS = 1000L / RUNTIME_FPS;

    // Overlay style — also written directly by SettingsBottomSheet
    public static OverlayStyle OVERLAY_STYLE = OverlayStyle.ATMOSPHERIC_MIST;

    // Filter parameters
    public static float THRESHOLD_STATIC    = 0.2f;
    public static float THRESHOLD_DYNAMIC   = 0.7f;
    public static float SMOOTHING_ALPHA     = 0.4f;
    public static float BRIGHTNESS = 1.2f;
    public static float CONTRAST   = 1.5f;
    public static boolean USE_FIXED_RADIUS = true; // Toggle this in Settings
    public static float FIXED_RADIUS_VALUE = 250f;

    // -----------------------------------------------------------------------
    // Live filter chain — rebuilt whenever parameters change
    // -----------------------------------------------------------------------

    public static List<SaliencyFilter> FILTER_CHAIN = buildChain();


    /**
     * Rebuilds the filter chain from the current runtime parameter values.
     * Call this after any filter parameter is changed (e.g. from SettingsActivity).
     */
    public static void rebuildFilterChain() {
        RUNTIME_INTERVAL_MS = 1000L / RUNTIME_FPS;
        FILTER_CHAIN = buildChain();
    }
    private static List<SaliencyFilter> buildChain() {
        return Arrays.asList(
                new ThresholdFilter(THRESHOLD_STATIC, THRESHOLD_DYNAMIC),
                new SmoothingFilter(SMOOTHING_ALPHA),
                new SaliencyEnhancementFilter(BRIGHTNESS, CONTRAST)
        );
    }
}
