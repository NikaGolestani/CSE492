package com.example.saliency.analyzer;

import org.json.JSONObject;

public class Video {
    public String name;
    public String path;
    public String hash;
    public VideoFeatures features;
    public boolean excluded = false;

    // The two numeric "conclusions"
    public int watchedPeriodSeconds = 0;
    public int watchedTimesCount = 0;

    public Video(String name, String path, String hash, JSONObject fullCache) {
        this.name = name;
        this.path = path;
        this.hash = hash;

        if (fullCache != null && fullCache.has(hash)) {
            JSONObject entry = fullCache.optJSONObject(hash);

            // Load Analysis Features
            if (entry.has("score")) {
                this.features = new VideoFeatures(
                        (float) entry.optDouble("flash", 0),
                        (float) entry.optDouble("motion", 0),
                        (float) entry.optDouble("cuts", 0),
                        (float) entry.optDouble("color", 0),
                        (float) entry.optDouble("clutter", 0),
                        (float) entry.optDouble("chaos", 0),
                        (float) entry.optDouble("rawFlicker", 0),
                        (float) entry.optDouble("cutsPerSec", 0),
                        entry.optInt("score", 0)
                );
            }

            // Load Numeric Counters
            this.watchedPeriodSeconds = entry.optInt("watched_period", 0);
            this.watchedTimesCount = entry.optInt("watched_times", 0);
            this.excluded = entry.optBoolean("excluded", false);
        }
    }


    public boolean isAnalyzed() { return features != null; }
}