package com.example.saliency.analyzer;

import android.content.Context;
import android.graphics.Bitmap;
import android.media.MediaMetadataRetriever;
import android.util.Log;

import java.util.ArrayList;
import java.util.List;

/**
 * VideoAnalyzer — ASD/SPD-aware video chaos scorer.
 *
 * Scoring model:
 *   score = clamp(100 - penalty, 0, 100)
 *   penalty = flicker*W_FLASH + cuts*W_CUT + motion*W_MOTION + color*W_COLOR + complexity*W_COMPLEX
 *
 * All metric values are in [0, 1].
 * Motion is NEVER rewarded — it is always a stressor for SPD/ASD.
 *
 * Weights and thresholds are kept in exact sync with the Python reference.
 */
public class VideoAnalyzer {

    private static final String TAG = "VideoAnalyzer";


    private static final int W      = 96;
    private static final int H      = 54;
    private static final int PIXELS = W * H;

    // -------------------------------------------------------
    // SAMPLING
    // -------------------------------------------------------
    private static final int MIN_FRAMES = 30;
    private static final int MAX_FRAMES = 120;

    /** 50 ms inter-frame offset — the clinically relevant flash window */
    private static final long OFFSET_US = 50_000L;

    // -------------------------------------------------------
    // WEIGHTS  — must stay in sync with Python
    // -------------------------------------------------------
    private static final float W_FLASH    = 50f;
    private static final float W_CUT      = 30f;
    private static final float W_MOTION   = 15f;
    private static final float W_COLOR    = 10f;
    private static final float W_COMPLEX  =  5f;
    // NO ASMR_REWARD — motion is always a stressor

    // -------------------------------------------------------
    // FLICKER THRESHOLDS  — tuned for ASD/SPD sensitivity
    // -------------------------------------------------------
    private static final float FLICKER_NOISE_FLOOR  = 1.5f;  // ignore sub-noise deltas
    private static final float FLICKER_SPIKE_THRESH = 8.0f;  // SPD spike level (~3-5x below NT)
    private static final float FLICKER_NORM         = 18.0f;

    // -------------------------------------------------------
    // CUT THRESHOLDS
    // -------------------------------------------------------
    private static final float CUT_HARD_THRESH = 10.0f;  // mean luma diff that counts as hard cut
    private static final float CUT_NORM        = 255.0f;

    // -------------------------------------------------------
    // OTHER NORMALISATION CEILINGS
    // -------------------------------------------------------
    private static final float MOTION_NORM  = 40.0f;
    private static final float COLOR_NORM   = 60.0f;
    private static final float EDGE_NORM    = 80.0f;

    // -------------------------------------------------------
    // PUBLIC ENTRY
    // -------------------------------------------------------
    public static VideoFeatures analyze(Context context, String videoPath) {

        VideoFeatures cached = VideoScoreCache.get(context, videoPath);
        if (cached != null && cached.score != 0) return cached;

        MediaMetadataRetriever r = new MediaMetadataRetriever();
        try {
            r.setDataSource(videoPath);

            long durationMs = Long.parseLong(
                    r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION));
            float durationSec = durationMs / 1000f;

            int frameCount = Math.min(MAX_FRAMES,
                    Math.max(MIN_FRAMES, (int)(durationSec * 10)));
            long stepUs = (durationMs * 1000L) / frameCount;

            // Per-pair accumulators
            // Each iteration = one (frameA, frameB) pair 50ms apart
            List<Float> lumA_list = new ArrayList<>();  // mean lum of frame A
            List<Float> lumB_list = new ArrayList<>();  // mean lum of frame B
            List<Float> motionList = new ArrayList<>();
            List<Float> cutDiffList = new ArrayList<>();
            List<Float> satStdList = new ArrayList<>();
            List<Float> edgeList = new ArrayList<>();

            int validPairs = 0;

            for (int i = 0; i < frameCount; i++) {
                long baseUs = i * stepUs;

                Bitmap bA = extractFrame(r, baseUs);
                Bitmap bB = extractFrame(r, baseUs + OFFSET_US);
                if (bA == null || bB == null) {
                    if (bA != null) bA.recycle();
                    if (bB != null) bB.recycle();
                    continue;
                }

                int[] pixA = new int[PIXELS];
                int[] pixB = new int[PIXELS];
                bA.getPixels(pixA, 0, W, 0, 0, W, H);
                bB.getPixels(pixB, 0, W, 0, 0, W, H);
                bA.recycle();
                bB.recycle();

                // ----- per-frame metrics -----
                FrameStats sA = frameStats(pixA);
                FrameStats sB = frameStats(pixB);

                lumA_list.add(sA.lum);
                lumB_list.add(sB.lum);

                // average saturation std across both frames
                satStdList.add((sA.satStd + sB.satStd) / 2f);

                // average spatial edge density across both frames
                edgeList.add((sA.edgeDensity + sB.edgeDensity) / 2f);

                // ----- pair metrics -----
                motionList.add(pairMotion(pixA, pixB));
                cutDiffList.add(pairMeanLumDiff(sA.lum, sB.lum));   // scalar diff of means

                validPairs++;
            }

            if (validPairs == 0) {
                return defaultFeatures();
            }

            // -------------------------------------------------------
            // FLICKER  — mirrors Python compute_flicker exactly
            //   signal = interleaved [lumA0, lumB0, lumA1, lumB1, ...]
            //   diff   = |A_i - B_i| for each pair
            //   energy = mean(diff[diff > NOISE_FLOOR]) / FLICKER_NORM
            //   spikes = count(diff > SPIKE_THRESH) / n_pairs
            //   result = clip(energy*0.7 + spikes*0.3, 0, 1)
            // -------------------------------------------------------
            float flicker = computeFlicker(lumA_list, lumB_list);

            // -------------------------------------------------------
            // MOTION  — mean per-pixel lum diff normalised to MOTION_NORM
            // -------------------------------------------------------
            float motion = mean(motionList);   // already normalised inside pairMotion()

            // -------------------------------------------------------
            // CUTS  — hard-cut ratio + soft mean-diff component
            //   hard_score = count(diff > CUT_HARD_THRESH) / n_pairs
            //   soft_score = mean(diff) / CUT_NORM
            //   result     = clip(hard_score + soft_score * 0.4, 0, 1)
            // -------------------------------------------------------
            float cuts = computeCuts(cutDiffList);

            // -------------------------------------------------------
            // COLOR  — mean saturation std normalised to COLOR_NORM
            // -------------------------------------------------------
            float color = Math.min(1f, mean(satStdList) / COLOR_NORM);

            // -------------------------------------------------------
            // COMPLEXITY  — mean spatial edge density normalised
            // -------------------------------------------------------
            float complexity = Math.min(1f, mean(edgeList) / EDGE_NORM);

            // -------------------------------------------------------
            // SCORE  — lower penalty = calmer = higher score
            // -------------------------------------------------------
            float penalty = flicker   * W_FLASH   +
                    cuts      * W_CUT     +
                    motion    * W_MOTION  +
                    color     * W_COLOR   +
                    complexity * W_COMPLEX;

            float score = Math.max(0f, Math.min(100f, 100f - penalty));

            // cutsPerSec for the features struct (informational)
            int cutCount = 0;
            for (float d : cutDiffList) if (d > CUT_HARD_THRESH) cutCount++;
            float cutsPerSec = cutCount / Math.max(durationSec, 0.1f);

            VideoFeatures result = new VideoFeatures(
                    flicker,
                    motion,
                    cuts,
                    color,
                    complexity,
                    complexity,          // base_complexity (kept for struct compat)
                    flicker * 10f,       // raw_flicker (informational)
                    cutsPerSec,
                    Math.round(score)
            );

            VideoScoreCache.put(context, videoPath, result);
            return result;

        } catch (Exception e) {
            Log.e(TAG, "analyze failed", e);
            return defaultFeatures();
        } finally {
            try { r.release(); } catch (Exception ignored) {}
        }
    }

    // -------------------------------------------------------
    // METRICS
    // -------------------------------------------------------

    /**
     * Python compute_flicker():
     *   diff = |A[i] - B[i]| for every pair
     *   keep only diff > NOISE_FLOOR
     *   energy = mean(kept) / FLICKER_NORM
     *   spikes = count(kept > SPIKE_THRESH) / n_pairs
     *   return clip(energy*0.7 + spikes*0.3, 0, 1)
     */
    private static float computeFlicker(List<Float> lumA, List<Float> lumB) {
        int n = Math.min(lumA.size(), lumB.size());
        if (n < 3) return 0f;

        float energySum = 0f;
        int   energyCnt = 0;
        int   spikeCnt  = 0;

        for (int i = 0; i < n; i++) {
            float d = Math.abs(lumA.get(i) - lumB.get(i));
            if (d > FLICKER_NOISE_FLOOR) {
                energySum += d;
                energyCnt++;
                if (d > FLICKER_SPIKE_THRESH) spikeCnt++;
            }
        }

        if (energyCnt == 0) return 0f;

        float energy = (energySum / energyCnt) / FLICKER_NORM;
        float spikes = (float) spikeCnt / n;

        return Math.min(1f, energy * 0.7f + spikes * 0.3f);
    }

    /**
     * Python compute_cuts():
     *   pair_diffs  = mean |A - B| per pixel per pair  (scalar per pair)
     *   hard_score  = count(pair_diffs > CUT_HARD_THRESH) / n
     *   soft_score  = mean(pair_diffs) / CUT_NORM
     *   return clip(hard_score + soft_score * 0.4, 0, 1)
     *
     * cutDiffList already holds per-pair scalar mean-lum-diff values.
     */
    private static float computeCuts(List<Float> cutDiffList) {
        int n = cutDiffList.size();
        if (n == 0) return 0f;

        int   hardCount = 0;
        float diffSum   = 0f;

        for (float d : cutDiffList) {
            if (d > CUT_HARD_THRESH) hardCount++;
            diffSum += d;
        }

        float hardScore = (float) hardCount / n;
        float softScore = (diffSum / n) / CUT_NORM;

        return Math.min(1f, hardScore + softScore * 0.4f);
    }

    // -------------------------------------------------------
    // FRAME-LEVEL HELPERS
    // -------------------------------------------------------

    private static class FrameStats {
        float lum;          // mean luminance
        float satStd;       // saturation standard deviation
        float edgeDensity;  // mean vertical edge magnitude
    }

    /**
     * Compute per-frame statistics in a single pixel pass.
     * Sampling stride = 6 pixels (matches original perf approach).
     */
    private static FrameStats frameStats(int[] pixels) {
        FrameStats s = new FrameStats();

        int sampled = 0;
        float lumSum   = 0f;
        float satSum   = 0f;
        float satSumSq = 0f;
        float edgeSum  = 0f;
        int   edgeCnt  = 0;

        for (int p = 0; p < PIXELS; p += 6) {
            int px = pixels[p];
            int rr = (px >> 16) & 0xFF;
            int gg = (px >> 8)  & 0xFF;
            int bb =  px        & 0xFF;

            float l   = rr * 0.299f + gg * 0.587f + bb * 0.114f;
            float max = Math.max(rr, Math.max(gg, bb));
            float min = Math.min(rr, Math.min(gg, bb));
            float sat = (max == 0f) ? 0f : (max - min) / max;

            lumSum   += l;
            satSum   += sat;
            satSumSq += sat * sat;

            // vertical edge: compare with pixel one row below
            if (p + W < PIXELS) {
                int below = pixels[p + W];
                float lb = ((below >> 16) & 0xFF) * 0.299f
                        + ((below >>  8) & 0xFF) * 0.587f
                        + ( below        & 0xFF) * 0.114f;
                edgeSum += Math.abs(l - lb);
                edgeCnt++;
            }

            sampled++;
        }

        // mean luminance
        s.lum = lumSum / sampled;

        // saturation std  = sqrt(E[x^2] - E[x]^2)
        float satMean = satSum / sampled;
        float satVar  = satSumSq / sampled - satMean * satMean;
        s.satStd = (satVar > 0f) ? (float) Math.sqrt(satVar) : 0f;
        // satStd is in [0,1]; scale to [0,255] range to match Python's cv2 HSV [0-255] space
        s.satStd *= 255f;

        s.edgeDensity = (edgeCnt > 0) ? (edgeSum / edgeCnt) : 0f;

        return s;
    }

    /**
     * Per-pair motion: mean absolute lum diff over all pixels.
     * Normalised to MOTION_NORM, clamped to [0,1].
     * No band filter — mirrors Python exactly.
     */
    private static float pairMotion(int[] a, int[] b) {
        float sum = 0f;
        int   cnt = 0;

        for (int i = 0; i < PIXELS; i += 6) {
            sum += Math.abs(lum(a[i]) - lum(b[i]));
            cnt++;
        }

        return Math.min(1f, (sum / cnt) / MOTION_NORM);
    }

    /**
     * Scalar mean-lum diff for a pair (used by cut scoring).
     * Python uses per-pixel mean across the spatial plane; here we
     * use the mean-lum scalars as an efficient approximation.
     */
    private static float pairMeanLumDiff(float lumA, float lumB) {
        return Math.abs(lumA - lumB);
    }

    // -------------------------------------------------------
    // UTILITIES
    // -------------------------------------------------------

    private static float lum(int px) {
        return ((px >> 16) & 0xFF) * 0.299f
                + ((px >> 8)  & 0xFF) * 0.587f
                + ( px        & 0xFF) * 0.114f;
    }

    private static float mean(List<Float> list) {
        if (list.isEmpty()) return 0f;
        float s = 0f;
        for (float v : list) s += v;
        return s / list.size();
    }

    private static Bitmap extractFrame(MediaMetadataRetriever r, long us) {
        Bitmap b = r.getFrameAtTime(us, MediaMetadataRetriever.OPTION_CLOSEST_SYNC);
        if (b == null) return null;
        return Bitmap.createScaledBitmap(b, W, H, true);
    }

    private static VideoFeatures defaultFeatures() {
        return new VideoFeatures(0.3f, 0.3f, 0.2f, 0.2f, 0.3f, 0.3f, 0f, 0f, 60);
    }
}