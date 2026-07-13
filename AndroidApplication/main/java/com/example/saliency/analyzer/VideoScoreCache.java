package com.example.saliency.analyzer;

import android.content.Context;
import android.util.Log;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.*;
import java.security.MessageDigest;

public class VideoScoreCache {

    private static final String TAG = "VideoScoreCache";
    private static final String CACHE_FILENAME = "video_analysis_cache.json";

    private static JSONObject cacheData = new JSONObject();
    private static boolean isLoaded = false;

    /**
     * Standardized Hashing: SHA-256.
     */
    public static String computeFileHash(String videoPath) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        File file = new File(videoPath);
        if (!file.exists()) return "unknown_" + videoPath.hashCode();

        try (FileInputStream fis = new FileInputStream(file)) {
            byte[] buffer = new byte[8192];
            int read;
            while ((read = fis.read(buffer)) != -1) {
                digest.update(buffer, 0, read);
            }
        }
        byte[] hashBytes = digest.digest();
        StringBuilder hex = new StringBuilder();
        for (byte b : hashBytes) hex.append(String.format("%02x", b));
        return hex.toString();
    }

    /**
     * RE-ADDED: This method creates a placeholder entry if one doesn't exist,
     * ensuring we use the SHA-256 hash as the key.
     */
    public static synchronized void initPlaceholderIfMissing(Context context, String videoPath) {
        try {
            ensureLoaded(context);
            String fileHash = computeFileHash(videoPath);

            if (!cacheData.has(fileHash)) {
                JSONObject entry = createBaseEntry(videoPath);
                cacheData.put(fileHash, entry);
                persistCache(context);
                Log.d(TAG, "Initialized placeholder for: " + videoPath);
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to init placeholder", e);
        }
    }

    /**
     * Updates numeric watch stats.
     */
    public static synchronized void incrementWatchMetrics(Context context, String videoPath, int sessionSeconds) {
        try {
            ensureLoaded(context);
            String fileHash = computeFileHash(videoPath);

            JSONObject entry = cacheData.optJSONObject(fileHash);
            if (entry == null) {
                entry = createBaseEntry(videoPath);
            }

            entry.put("watched_period", entry.optInt("watched_period", 0) + sessionSeconds);
            entry.put("watched_times", entry.optInt("watched_times", 0) + 1);

            cacheData.put(fileHash, entry);
            persistCache(context);
        } catch (Exception e) {
            Log.e(TAG, "Error updating numeric logs", e);
        }
    }

    public static synchronized void addWatchLog(Context context, String videoPath, String openedAt, long durationSec) {
        try {
            ensureLoaded(context);
            String fileHash = computeFileHash(videoPath);

            JSONObject entry = cacheData.optJSONObject(fileHash);
            if (entry == null) entry = createBaseEntry(videoPath);

            JSONObject logEntry = new JSONObject();
            logEntry.put("opened_at", openedAt);
            logEntry.put("watched_seconds", durationSec);

            if (!entry.has("watch_logs")) entry.put("watch_logs", new JSONArray());
            entry.getJSONArray("watch_logs").put(logEntry);

            cacheData.put(fileHash, entry);
            persistCache(context);
        } catch (Exception e) {
            Log.e(TAG, "Failed to add watch log", e);
        }
    }

    public static synchronized void put(Context context, String videoPath, VideoFeatures features) {
        try {
            ensureLoaded(context);
            String fileHash = computeFileHash(videoPath);

            JSONObject entry = cacheData.optJSONObject(fileHash);
            if (entry == null) entry = createBaseEntry(videoPath);

            entry.put("score", features.score);
            entry.put("motion", features.motion);
            entry.put("flash", features.flash);
            entry.put("color", features.color);
            entry.put("clutter", features.complexity);
            entry.put("cached_at", System.currentTimeMillis());

            cacheData.put(fileHash, entry);
            persistCache(context);
        } catch (Exception e) {
            Log.e(TAG, "Failed to cache results", e);
        }
    }

    public static synchronized VideoFeatures get(Context context, String videoPath) {
        try {
            ensureLoaded(context);
            String fileHash = computeFileHash(videoPath);
            if (!cacheData.has(fileHash)) return null;
            JSONObject entry = cacheData.getJSONObject(fileHash);
            if (!entry.has("score")) return null;
            return parseVideoFeatures(entry);
        } catch (Exception e) {
            return null;
        }
    }

    private static JSONObject createBaseEntry(String videoPath) throws JSONException {
        JSONObject entry = new JSONObject();
        entry.put("video_name", new File(videoPath).getName());
        entry.put("video_path", videoPath);
        entry.put("watched_period", 0);
        entry.put("watched_times", 0);
        entry.put("excluded", false);
        return entry;
    }

    private static void ensureLoaded(Context context) throws IOException, JSONException {
        if (isLoaded) return;
        File cacheFile = new File(context.getFilesDir(), CACHE_FILENAME);
        if (!cacheFile.exists()) { isLoaded = true; return; }

        StringBuilder sb = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new FileReader(cacheFile))) {
            String line;
            while ((line = reader.readLine()) != null) sb.append(line);
        }
        if (sb.length() > 0) cacheData = new JSONObject(sb.toString());
        isLoaded = true;
    }

    private static void persistCache(Context context) {
        try {
            File cacheFile = new File(context.getFilesDir(), CACHE_FILENAME);
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(cacheFile))) {
                writer.write(cacheData.toString(4));
            }
        } catch (Exception e) {
            Log.e(TAG, "Persist failed", e);
        }
    }

    private static VideoFeatures parseVideoFeatures(JSONObject obj) throws JSONException {
        return new VideoFeatures(
                (float) obj.optDouble("flash", 0),
                (float) obj.optDouble("motion", 0),
                (float) obj.optDouble("cuts", 0),
                (float) obj.optDouble("color", 0),
                (float) obj.optDouble("clutter", 0),
                (float) obj.optDouble("chaos", 0),
                (float) obj.optDouble("rawFlicker", 0),
                (float) obj.optDouble("cutsPerSec", 0),
                obj.optInt("score", 0)
        );
    }

    public static synchronized void clear(Context context) {
        cacheData = new JSONObject();
        persistCache(context);
    }
}