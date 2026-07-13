package com.example.saliency.config;

import android.content.Context;
import android.content.SharedPreferences;

import com.example.saliency.overlay.OverlayStyle;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Persists all user-configurable settings to SharedPreferences and loads
 * them back into the live config fields on app start.
 *
 * Password is stored as plain text. For higher security, replace
 * with a BCrypt/PBKDF2 hash in a future version.
 *
 * Usage:
 *   // On app start (e.g. in GalleryActivity.onCreate):
 *   SettingsManager.loadAll(context);
 *
 *   // After the user saves a setting:
 *   SettingsManager.saveAll(context);
 */
public class SettingsManager {

    private static final String PREFS_NAME = "saliency_settings";

    // Keys
    private static final String KEY_PASSWORD             = "password";
    private static final String KEY_ANALYSIS_FPS         = "analysis_fps";
    private static final String KEY_INPUT_SIZE           = "input_size";
    private static final String KEY_OVERLAY_PADDING      = "overlay_padding";
    private static final String KEY_OVERLAY_STYLE        = "overlay_style";
    private static final String KEY_THRESHOLD_STATIC     = "threshold_static";
    private static final String KEY_THRESHOLD_DYNAMIC    = "threshold_dynamic";
    private static final String KEY_SMOOTHING_ALPHA      = "smoothing_alpha";
    private static final String KEY_NOTEBOOK_BRIGHTNESS  = "notebook_brightness";
    private static final String KEY_NOTEBOOK_CONTRAST    = "notebook_contrast";
    private static final String KEY_VIDEOS_PER_PAGE      = "videos_per_page";
    private static final String KEY_INCLUDE_FOLDERS      = "include_folders";
    private static final String KEY_EXCLUDE_PATHS        = "exclude_paths";
    private static final String KEY_MANUAL_PATHS         = "manual_paths";

    // -----------------------------------------------------------------------
    // Password
    // -----------------------------------------------------------------------

    public static boolean hasPassword(Context ctx) {
        return !getPrefs(ctx).getString(KEY_PASSWORD, "").isEmpty();
    }

    public static void setPassword(Context ctx, String password) {
        getPrefs(ctx).edit().putString(KEY_PASSWORD, password).apply();
    }

    public static boolean checkPassword(Context ctx, String input) {
        return getPrefs(ctx).getString(KEY_PASSWORD, "").equals(input);
    }

    // -----------------------------------------------------------------------
    // Load all settings → live config fields
    // -----------------------------------------------------------------------

    public static void loadAll(Context ctx) {
        SharedPreferences p = getPrefs(ctx);

        // AnalysisConfig
        AnalysisConfig.OVERLAY_STYLE     = OverlayStyle.fromString(p.getString("overlay_style", "NEON_GLOW"));
        AnalysisConfig.RUNTIME_FPS       = p.getInt(KEY_ANALYSIS_FPS,    AnalysisConfig.ANALYSIS_FPS);
        AnalysisConfig.RUNTIME_INPUT_SIZE= p.getInt(KEY_INPUT_SIZE,       AnalysisConfig.INPUT_SIZE);
        AnalysisConfig.RUNTIME_PADDING   = p.getInt(KEY_OVERLAY_PADDING,  AnalysisConfig.OVERLAY_PADDING);

        // Filter params
        AnalysisConfig.THRESHOLD_STATIC  = p.getFloat(KEY_THRESHOLD_STATIC,  0.2f);
        AnalysisConfig.THRESHOLD_DYNAMIC = p.getFloat(KEY_THRESHOLD_DYNAMIC, 0.7f);
        AnalysisConfig.SMOOTHING_ALPHA   = p.getFloat(KEY_SMOOTHING_ALPHA,   0.4f);
        AnalysisConfig.BRIGHTNESS = p.getFloat(KEY_NOTEBOOK_BRIGHTNESS, 1.2f);
        AnalysisConfig.CONTRAST   = p.getFloat(KEY_NOTEBOOK_CONTRAST,   1.5f);

        // AppConfig
        AppConfig.RUNTIME_VIDEOS_PER_PAGE = p.getInt(KEY_VIDEOS_PER_PAGE, AppConfig.VIDEOS_PER_PAGE);




        // Rebuild filter chain with loaded params
        AnalysisConfig.rebuildFilterChain();
    }

    // -----------------------------------------------------------------------
    // Save all live config fields → SharedPreferences
    // -----------------------------------------------------------------------

    public static void saveAll(Context ctx) {
        SharedPreferences.Editor e = getPrefs(ctx).edit();

        e.putString(KEY_OVERLAY_STYLE,       AnalysisConfig.OVERLAY_STYLE.name());
        e.putInt   (KEY_ANALYSIS_FPS,        AnalysisConfig.RUNTIME_FPS);
        e.putInt   (KEY_INPUT_SIZE,          AnalysisConfig.RUNTIME_INPUT_SIZE);
        e.putInt   (KEY_OVERLAY_PADDING,     AnalysisConfig.RUNTIME_PADDING);

        e.putFloat (KEY_THRESHOLD_STATIC,    AnalysisConfig.THRESHOLD_STATIC);
        e.putFloat (KEY_THRESHOLD_DYNAMIC,   AnalysisConfig.THRESHOLD_DYNAMIC);
        e.putFloat (KEY_SMOOTHING_ALPHA,     AnalysisConfig.SMOOTHING_ALPHA);
        e.putFloat (KEY_NOTEBOOK_BRIGHTNESS, AnalysisConfig.BRIGHTNESS);
        e.putFloat (KEY_NOTEBOOK_CONTRAST,   AnalysisConfig.CONTRAST);

        e.putInt   (KEY_VIDEOS_PER_PAGE,     AppConfig.RUNTIME_VIDEOS_PER_PAGE);


        e.apply();
    }

    // -----------------------------------------------------------------------

    private static SharedPreferences getPrefs(Context ctx) {
        return ctx.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
    }

    private static List<String> toList(Set<String> set) {
        return Arrays.asList(set.toArray(new String[0]));
    }
}
