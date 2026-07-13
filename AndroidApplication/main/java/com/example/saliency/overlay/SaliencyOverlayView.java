package com.example.saliency.overlay;

import android.content.Context;
import android.graphics.*;
import android.util.AttributeSet;
import android.view.View;
import android.util.Log;
import com.example.saliency.config.AnalysisConfig;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

/**
 * Multi-blob saliency overlay.
 *
 * WHY PLAIN BFS FAILED:
 *   Saliency models output smooth Gaussian-like heatmaps, not discrete islands.
 *   At threshold=0.15 the whole heatmap is one connected region — BFS always
 *   returns a single blob no matter how many subjects are in the frame.
 *
 * THE FIX — peak-seeded region growing:
 *   1. Find all local maxima above a HIGH threshold (peaks only).
 *   2. Suppress peaks that are too close together (non-maximum suppression).
 *   3. Grow each peak's region outward via BFS but stop at a LOWER threshold
 *      AND stop when hitting another peak's territory (watershed boundary).
 *   This correctly separates two faces even when their halos overlap.
 */
public class SaliencyOverlayView extends View {

    private static final String TAG = "SaliencyOverlay";

    // -----------------------------------------------------------------------
    // Animated blob state
    // -----------------------------------------------------------------------
    private static final class BlobState {
        float cx, cy, radX, radY;
        boolean alive;
        BlobState(float cx, float cy, float radX, float radY) {
            this.cx = cx; this.cy = cy; this.radX = radX; this.radY = radY; this.alive = true;
        }
    }

    // -----------------------------------------------------------------------
    // Tuning — adjust these to taste
    // -----------------------------------------------------------------------

    /** Minimum value to be considered a peak (local maximum). 0.6–0.8 is typical. */
    private static final float PEAK_THRESHOLD    = 0.5f;

    /** Region-grow continues down to this value (should be < PEAK_THRESHOLD). */
    private static final float GROW_THRESHOLD    = 0.20f;

    /**
     * Half-size of the window used to detect local maxima.
     * Larger = peaks must be further apart to both count.
     * In map pixels — 224×224 map → 15px means peaks must be ~7% of width apart.
     */
    private static final int   PEAK_WINDOW       = 15;

    /**
     * Minimum normalised distance (0–1) between two peaks for both to survive
     * non-maximum suppression. Peaks closer than this are merged to the stronger one.
     */
    private static final float NMS_DIST_NORM     = 0.15f;

    /** Blobs below this fraction of the strongest blob's weight are discarded. */
    private static final float MIN_WEIGHT_RATIO  = 0.12f;

    /** Max blobs rendered at once. */
    private static final int   MAX_BLOBS         = 4;

    /** Lerp speed for position and radius animation. */
    private static final float LERP_POS          = 0.28f;
    private static final float LERP_RADIUS       = 0.28f;

    // -----------------------------------------------------------------------
    // Runtime state
    // -----------------------------------------------------------------------
    private float[][] map;
    private float     threshold = GROW_THRESHOLD; // setSaliency threshold is ignored; we use constants

    private final List<BlobState> blobs = new ArrayList<>();

    private int   videoWidth = 1, videoHeight = 1;
    private float scaleX = 1f, scaleY = 1f, offsetX = 0f, offsetY = 0f;

    // -----------------------------------------------------------------------
    // Pre-allocated render tools
    // -----------------------------------------------------------------------
    private final Paint  maskPaint  = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint  clearPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint  neonPaint  = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final RectF  oval       = new RectF();
    private final Rect   screen     = new Rect();
    private final Matrix gradMatrix = new Matrix();

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------
    public SaliencyOverlayView(Context ctx, AttributeSet attrs) {
        super(ctx, attrs);
        setLayerType(LAYER_TYPE_SOFTWARE, null);
        clearPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.CLEAR));
        neonPaint.setStyle(Paint.Style.STROKE);
        neonPaint.setStrokeWidth(10f);
        neonPaint.setColor(Color.CYAN);
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------
    public void setSaliency(float[][] m, float ignoredThreshold) {
        // We ignore the passed threshold and use our own constants
        // so the caller doesn't need to change anything
        if (m == null) return;
        this.map = m;
        invalidate();
    }

    public void clear() {
        map = null;
        blobs.clear();
        invalidate();
    }

    public void setVideoSize(int width, int height) {
        if (width <= 0 || height <= 0) return;
        videoWidth = width; videoHeight = height;
        float viewW = getWidth(), viewH = getHeight();
        if (viewW == 0 || viewH == 0) return;
        if ((float) width / height > viewW / viewH) {
            scaleX = viewW / width;  scaleY = scaleX;
            offsetX = 0f;            offsetY = (viewH - height * scaleY) / 2f;
        } else {
            scaleY = viewH / height; scaleX = scaleY;
            offsetY = 0f;            offsetX = (viewW - width * scaleX) / 2f;
        }
        invalidate();
    }

    // -----------------------------------------------------------------------
    // Draw
    // -----------------------------------------------------------------------
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (map == null || getWidth() == 0) return;

        screen.set(0, 0, getWidth(), getHeight());

        List<RawBlob> detected = detectBlobsByPeaks(map);

        reconcileBlobs(detected);

        try {
            switch (AnalysisConfig.OVERLAY_STYLE) {
                case BLACKOUT:
                    canvas.drawColor(Color.argb(220, 0, 0, 0));
                    for (BlobState b : blobs) drawHole(canvas, b, 0.85f);
                    break;
                case VIGNETTE_SHADOW:
                    for (BlobState b : blobs) drawEllipticalGradient(canvas, b, Color.BLACK, 255, 0.2f);
                    break;
                case ATMOSPHERIC_MIST:
                    for (BlobState b : blobs) drawEllipticalGradient(canvas, b, Color.WHITE, 190, 0.35f);
                    break;
                case FOVEATED_EYE:
                    drawBlurBackground(canvas, 80f);
                    for (BlobState b : blobs) drawHole(canvas, b, 0.15f);
                    break;
                case NEON_GLOW:
                    for (BlobState b : blobs) {
                        float r = resolveRadX(b);
                        oval.set(b.cx - r, b.cy - r, b.cx + r, b.cy + r);
                        canvas.drawOval(oval, neonPaint);
                    }
                    break;
            }
        } catch (Exception e) {
            Log.e(TAG, "Draw error", e);
        }

        if (!blobs.isEmpty()) postInvalidateOnAnimation();
    }

    // -----------------------------------------------------------------------
    // Peak-seeded blob detection
    // -----------------------------------------------------------------------

    private static final class RawBlob {
        float cx, cy, radX, radY, weight;
    }

    private List<RawBlob> detectBlobsByPeaks(float[][] m) {
        int rows = m.length, cols = m[0].length;

        // STEP 1: Find all local maxima above PEAK_THRESHOLD.
        // A pixel is a local maximum if it is >= all neighbours in a PEAK_WINDOW×PEAK_WINDOW area.
        List<int[]> peaks = new ArrayList<>(); // [col, row, value*1000]
        for (int y = PEAK_WINDOW; y < rows - PEAK_WINDOW; y++) {
            for (int x = PEAK_WINDOW; x < cols - PEAK_WINDOW; x++) {
                float v = m[y][x];
                if (v < PEAK_THRESHOLD) continue;
                boolean isPeak = true;
                outer:
                for (int dy = -PEAK_WINDOW; dy <= PEAK_WINDOW; dy++) {
                    for (int dx = -PEAK_WINDOW; dx <= PEAK_WINDOW; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        if (m[y + dy][x + dx] > v) { isPeak = false; break outer; }
                    }
                }
                if (isPeak) peaks.add(new int[]{x, y, (int)(v * 1000)});
            }
        }


        // STEP 2: Non-maximum suppression — remove weaker peaks too close to a stronger one.
        peaks.sort((a, b) -> b[2] - a[2]); // sort by value descending
        List<int[]> suppressed = new ArrayList<>();
        float nmsDistPx = NMS_DIST_NORM * cols;
        for (int[] p : peaks) {
            boolean blocked = false;
            for (int[] kept : suppressed) {
                float dx = p[0] - kept[0], dy = p[1] - kept[1];
                if (Math.sqrt(dx * dx + dy * dy) < nmsDistPx) { blocked = true; break; }
            }
            if (!blocked) suppressed.add(p);
            if (suppressed.size() >= MAX_BLOBS) break;
        }


        if (suppressed.isEmpty()) {
            // Fallback: no peaks above PEAK_THRESHOLD — lower bar and take single global max
            float maxV = 0; int maxX = cols/2, maxY = rows/2;
            for (int y = 0; y < rows; y++)
                for (int x = 0; x < cols; x++)
                    if (m[y][x] > maxV) { maxV = m[y][x]; maxX = x; maxY = y; }
            suppressed.add(new int[]{maxX, maxY, (int)(maxV * 1000)});
        }

        // STEP 3: Region grow from each peak down to GROW_THRESHOLD.
        // Label map: -1=unvisited, 0=below threshold, i=owned by peak i.
        int[][] label = new int[rows][cols];
        for (int[] row : label) java.util.Arrays.fill(row, -1);

        // Seed queues — one per peak
        @SuppressWarnings("unchecked")
        Queue<int[]>[] queues = new Queue[suppressed.size()];
        for (int i = 0; i < suppressed.size(); i++) {
            queues[i] = new LinkedList<>();
            int[] p = suppressed.get(i);
            label[p[1]][p[0]] = i;
            queues[i].add(new int[]{p[0], p[1]});
        }

        // Interleaved BFS — each peak grows one pixel at a time, preventing any one
        // from swamping the others (this is the key watershed separation step).
        boolean anyActive = true;
        while (anyActive) {
            anyActive = false;
            for (int i = 0; i < queues.length; i++) {
                Queue<int[]> q = queues[i];
                if (q.isEmpty()) continue;
                int[] cur = q.poll();
                anyActive = true;
                int[][] neighbours = {{cur[0]-1,cur[1]},{cur[0]+1,cur[1]},{cur[0],cur[1]-1},{cur[0],cur[1]+1}};
                for (int[] n : neighbours) {
                    int nx = n[0], ny = n[1];
                    if (nx < 0 || nx >= cols || ny < 0 || ny >= rows) continue;
                    if (label[ny][nx] != -1) continue;
                    if (m[ny][nx] < GROW_THRESHOLD) { label[ny][nx] = -2; continue; } // below threshold
                    label[ny][nx] = i;
                    q.add(new int[]{nx, ny});
                }
            }
        }

        // STEP 4: Collect stats per labelled region → RawBlob
        float[] wX    = new float[suppressed.size()];
        float[] wY    = new float[suppressed.size()];
        float[] sumW  = new float[suppressed.size()];
        float[] minX  = new float[suppressed.size()];
        float[] maxX  = new float[suppressed.size()];
        float[] minY_ = new float[suppressed.size()];
        float[] maxY_ = new float[suppressed.size()];
        for (int i = 0; i < suppressed.size(); i++) {
            minX[i] = cols; maxX[i] = 0; minY_[i] = rows; maxY_[i] = 0;
        }

        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                int lbl = label[y][x];
                if (lbl < 0) continue;
                float w = m[y][x] * m[y][x];
                wX[lbl]   += x * w;
                wY[lbl]   += y * w;
                sumW[lbl] += w;
                if (x < minX[lbl])  minX[lbl]  = x;
                if (x > maxX[lbl])  maxX[lbl]  = x;
                if (y < minY_[lbl]) minY_[lbl] = y;
                if (y > maxY_[lbl]) maxY_[lbl] = y;
            }
        }

        // STEP 5: Build RawBlob list, filter weak blobs
        List<RawBlob> result = new ArrayList<>();
        float maxBlobWeight = 0;
        for (float w : sumW) if (w > maxBlobWeight) maxBlobWeight = w;

        for (int i = 0; i < suppressed.size(); i++) {
            if (sumW[i] == 0) continue;
            if (sumW[i] < maxBlobWeight * MIN_WEIGHT_RATIO) {
                continue;
            }
            RawBlob rb  = new RawBlob();
            rb.cx       = wX[i] / sumW[i];
            rb.cy       = wY[i] / sumW[i];
            rb.radX     = (maxX[i] - minX[i]) / 2f;
            rb.radY     = (maxY_[i] - minY_[i]) / 2f;
            rb.weight   = sumW[i];
            result.add(rb);
        }

        return result;
    }

    // -----------------------------------------------------------------------
    // Reconcile detected blobs ↔ animated BlobStates
    // -----------------------------------------------------------------------
    private void reconcileBlobs(List<RawBlob> detected) {
        int rows = map.length, cols = map[0].length;
        float pad = AnalysisConfig.OVERLAY_PADDING;

        // Convert map-space → screen-space
        float[][] targets = new float[detected.size()][4];
        for (int i = 0; i < detected.size(); i++) {
            RawBlob d = detected.get(i);
            float tcx  = (d.cx / cols) * videoWidth  * scaleX + offsetX;
            float tcy  = (d.cy / rows) * videoHeight * scaleY + offsetY;
            float tradX, tradY;
            if (AnalysisConfig.USE_FIXED_RADIUS) {
                tradX = tradY = AnalysisConfig.FIXED_RADIUS_VALUE;
            } else {
                tradX = (d.radX / cols) * videoWidth  * scaleX * 0.65f + pad;
                tradY = (d.radY / rows) * videoHeight * scaleY * 0.65f + pad * 1.3f;
                tradX = Math.max(pad * 3f, Math.min(getWidth()  / 2.5f, tradX));
                tradY = Math.max(pad * 4f, Math.min(getHeight() / 2.5f, tradY));
            }
            targets[i] = new float[]{tcx, tcy, tradX, tradY};
        }

        for (BlobState b : blobs) b.alive = false;
        boolean[] matched = new boolean[targets.length];
        float snapThresh  = getWidth() * 0.45f;

        for (int ti = 0; ti < targets.length; ti++) {
            float[] t = targets[ti];
            BlobState best = null; float bestDist = Float.MAX_VALUE;
            for (BlobState b : blobs) {
                if (b.alive) continue;
                float dx = b.cx - t[0], dy = b.cy - t[1];
                float dist = (float) Math.sqrt(dx*dx + dy*dy);
                if (dist < bestDist) { bestDist = dist; best = b; }
            }
            if (best != null && bestDist < snapThresh) {
                best.cx   += (t[0] - best.cx)   * LERP_POS;
                best.cy   += (t[1] - best.cy)   * LERP_POS;
                best.radX += (t[2] - best.radX) * LERP_RADIUS;
                best.radY += (t[3] - best.radY) * LERP_RADIUS;
                best.alive = true; matched[ti] = true;
            }
        }

        blobs.removeIf(b -> !b.alive);
        for (int ti = 0; ti < targets.length; ti++) {
            if (!matched[ti]) {
                float[] t = targets[ti];
                blobs.add(new BlobState(t[0], t[1], t[2], t[3]));
            }
        }
    }

    // -----------------------------------------------------------------------
    // Render helpers
    // -----------------------------------------------------------------------
    private float resolveRadX(BlobState b) {
        return AnalysisConfig.USE_FIXED_RADIUS ? AnalysisConfig.FIXED_RADIUS_VALUE : b.radX;
    }
    private float resolveRadY(BlobState b) {
        return AnalysisConfig.USE_FIXED_RADIUS ? AnalysisConfig.FIXED_RADIUS_VALUE : b.radY;
    }

    private void drawHole(Canvas canvas, BlobState b, float feather) {
        float radX = resolveRadX(b), radY = resolveRadY(b);
        float maxR = Math.max(radX, radY);
        if (maxR <= 0) return;
        RadialGradient grad = new RadialGradient(b.cx, b.cy, maxR,
                new int[]{Color.BLACK, Color.TRANSPARENT},
                new float[]{feather, 1.0f}, Shader.TileMode.CLAMP);
        gradMatrix.reset();
        gradMatrix.postScale(radX / maxR, radY / maxR, b.cx, b.cy);
        grad.setLocalMatrix(gradMatrix);
        oval.set(b.cx - radX, b.cy - radY, b.cx + radX, b.cy + radY);
        clearPaint.setShader(grad);
        canvas.drawOval(oval, clearPaint);
        clearPaint.setShader(null);
    }

    private void drawEllipticalGradient(Canvas canvas, BlobState b,
                                        int color, int maxAlpha, float t0) {
        float radX = resolveRadX(b), radY = resolveRadY(b);
        float maxR = Math.max(radX, radY) * 2.5f;
        if (maxR <= 0) return;
        RadialGradient grad = new RadialGradient(b.cx, b.cy, maxR,
                new int[]{Color.TRANSPARENT, Color.argb(maxAlpha,
                        Color.red(color), Color.green(color), Color.blue(color))},
                new float[]{t0, 1.0f}, Shader.TileMode.CLAMP);
        float aspect = Math.max(radX, radY);
        gradMatrix.reset();
        gradMatrix.postScale(radX / aspect, radY / aspect, b.cx, b.cy);
        grad.setLocalMatrix(gradMatrix);
        maskPaint.setShader(grad);
        canvas.drawRect(screen, maskPaint);
        maskPaint.setShader(null);
    }

    private void drawBlurBackground(Canvas canvas, float sigma) {
        maskPaint.reset();
        maskPaint.setAntiAlias(true);
        maskPaint.setMaskFilter(new BlurMaskFilter(sigma, BlurMaskFilter.Blur.NORMAL));
        maskPaint.setColor(Color.WHITE);
        maskPaint.setAlpha(120);
        canvas.drawRect(screen, maskPaint);
        maskPaint.setMaskFilter(null);
    }
}