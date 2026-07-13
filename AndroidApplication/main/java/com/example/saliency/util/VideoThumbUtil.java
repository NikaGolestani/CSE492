package com.example.saliency.util;

import android.graphics.Bitmap;
import android.media.MediaMetadataRetriever;

import java.util.Random;

/**
 * Utility for extracting a thumbnail from a local video file.
 */
public class VideoThumbUtil {

    private VideoThumbUtil() { }

    public static Bitmap getThumb(String path) {
        MediaMetadataRetriever r = new MediaMetadataRetriever();
        try {
            r.setDataSource(path);

            // get duration in microseconds
            String durationStr = r.extractMetadata(
                    MediaMetadataRetriever.METADATA_KEY_DURATION
            );

            long durationUs = 0;
            if (durationStr != null) {
                durationUs = Long.parseLong(durationStr) * 1000;
            }

            // pick random frame (avoid always first frame)
            long timeUs = 0;
            if (durationUs > 0) {
                timeUs = (long) (new Random().nextDouble() * durationUs);
            }

            Bitmap bmp = r.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC);

            // fallback if frame is still null
            if (bmp == null) {
                bmp = r.getFrameAtTime(0);
            }

            return bmp;

        } catch (Exception e) {
            return null; // keeping your original behavior intact
        } finally {
            try {
                r.release();
            } catch (Exception ignored) {}
        }
    }
}