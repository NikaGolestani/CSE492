package com.example.saliency.config;

import java.util.Arrays;
import java.util.List;

/**
 * Static configuration for the gallery / video scanner.
 *
 * Compile-time constants are factory defaults.
 * RUNTIME_ fields are what the app actually uses — populated by
 * SettingsManager.loadAll() on start and written back by SettingsManager.saveAll().
 */
public class AppConfig {

    // -----------------------------------------------------------------------
    // Compile-time defaults
    // -----------------------------------------------------------------------

    /** Number of videos displayed per swipe page in GalleryActivity. */
    public static final int VIDEOS_PER_PAGE = 3;



    // -----------------------------------------------------------------------
    // Runtime-mutable values (overwritten by SettingsManager on load)
    // -----------------------------------------------------------------------

    public static int          RUNTIME_VIDEOS_PER_PAGE = VIDEOS_PER_PAGE;

}
