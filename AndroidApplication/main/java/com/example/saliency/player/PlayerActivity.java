package com.example.saliency.player;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Rect;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.TextureView;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.OptIn;
import androidx.appcompat.app.AppCompatActivity;
import androidx.media3.common.MediaItem;
import androidx.media3.common.Player;
import androidx.media3.common.VideoSize;
import androidx.media3.common.util.UnstableApi;
import androidx.media3.exoplayer.ExoPlayer;
import androidx.media3.ui.PlayerView;

import com.example.saliency.R;
import com.example.saliency.analyzer.VideoScoreCache;
import com.example.saliency.config.AnalysisConfig;
import com.example.saliency.filter.SaliencyFilter;
import com.example.saliency.overlay.SaliencyOverlayView;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

public class PlayerActivity extends AppCompatActivity {

    private static final String TAG = "SALIENCY_DEBUG";

    private PlayerView playerView;
    private SaliencyOverlayView overlayView;

    private ExoPlayer exoPlayer;
    private Interpreter interpreter;

    private int inputSize = AnalysisConfig.RUNTIME_INPUT_SIZE;

    private final ExecutorService inferenceExecutor = Executors.newSingleThreadExecutor();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    private final AtomicBoolean running = new AtomicBoolean(false);
    private final AtomicBoolean inferring = new AtomicBoolean(false);
    private final AtomicBoolean activityActive = new AtomicBoolean(true);

    private Runnable loop;

    private ByteBuffer inputBuffer;
    private int[] pixelBuffer;
    private float[][] saliencyBuffer;
    private float[][][][] outputBuffer;
    private Bitmap scaledBitmap;

    private String videoPath;
    private String timeOpened;
    private long lastStartTime = 0;
    private long totalWatchedMs = 0;

    // ------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_player);

        videoPath = getIntent().getStringExtra("video_path");
        String title = getIntent().getStringExtra("video_title");

        timeOpened = new SimpleDateFormat(
                "yyyy-MM-dd HH:mm:ss",
                Locale.getDefault()
        ).format(new Date());

        if (videoPath == null) {
            Toast.makeText(this, "Invalid video path", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        playerView = findViewById(R.id.player_view);
        overlayView = findViewById(R.id.overlay_view);

        TextView txtTitle = findViewById(R.id.txt_video_title);
        if (txtTitle != null) {
            txtTitle.setText(title);
        }

        View btnBack = findViewById(R.id.btn_back);
        if (btnBack != null) {
            btnBack.setOnClickListener(v -> finish());
        }

        loop = new Runnable() {
            @Override
            public void run() {
                if (!running.get() || !activityActive.get()) {
                    return;
                }

                scheduleInference();

                mainHandler.postDelayed(
                        this,
                        AnalysisConfig.RUNTIME_INTERVAL_MS
                );
            }
        };

        initModel();
        initPlayer(videoPath);
    }

    @Override
    protected void onPause() {
        super.onPause();

        activityActive.set(false);

        updateWatchedDuration();

        stopAnalysis();

        // IMPORTANT:
        // Clear overlay and NEVER update it again while paused
        overlayView.clear();

        mainHandler.removeCallbacksAndMessages(null);
    }

    @Override
    protected void onUserLeaveHint() {
        super.onUserLeaveHint();

        closePlayer();
    }

    @Override
    protected void onDestroy() {

        activityActive.set(false);

        updateWatchedDuration();

        VideoScoreCache.incrementWatchMetrics(
                this,
                videoPath,
                (int) (totalWatchedMs / 1000)
        );

        stopAnalysis();

        inferenceExecutor.shutdownNow();

        if (exoPlayer != null) {
            exoPlayer.release();
            exoPlayer = null;
        }

        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }

        if (scaledBitmap != null && !scaledBitmap.isRecycled()) {
            scaledBitmap.recycle();
        }

        super.onDestroy();
    }

    // ------------------------------------------------------------
    // Close player completely
    // ------------------------------------------------------------

    private void closePlayer() {

        activityActive.set(false);

        stopAnalysis();

        updateWatchedDuration();

        if (exoPlayer != null) {
            exoPlayer.setPlayWhenReady(false);
            exoPlayer.stop();
            exoPlayer.release();
            exoPlayer = null;
        }

        overlayView.clear();

        finish();
    }

    // ------------------------------------------------------------
    // Model
    // ------------------------------------------------------------

    private void initModel() {
        try {

            MappedByteBuffer model = loadModel("saliency_float16.tflite");

            interpreter = new Interpreter(model);

            Tensor inputTensor = interpreter.getInputTensor(0);

            if (inputTensor.shape().length == 4) {

                inputSize = inputTensor.shape()[1];

                AnalysisConfig.RUNTIME_INPUT_SIZE = inputSize;
            }

            allocateBuffers(inputSize);

            Log.d(
                    TAG,
                    "Model ready — inputSize=" + inputSize
            );

        } catch (Exception e) {

            Log.e(TAG, "Model load failed", e);
        }
    }

    private void allocateBuffers(int size) {

        int pixels = size * size;

        inputBuffer = ByteBuffer
                .allocateDirect(pixels * 3 * 4)
                .order(ByteOrder.nativeOrder());

        pixelBuffer = new int[pixels];

        saliencyBuffer = new float[size][size];

        outputBuffer = new float[1][size][size][1];

        if (scaledBitmap != null && !scaledBitmap.isRecycled()) {
            scaledBitmap.recycle();
        }

        scaledBitmap = Bitmap.createBitmap(
                size,
                size,
                Bitmap.Config.ARGB_8888
        );
    }

    private MappedByteBuffer loadModel(String file) throws IOException {

        android.content.res.AssetFileDescriptor afd =
                getAssets().openFd(file);

        return new FileInputStream(afd.getFileDescriptor())
                .getChannel()
                .map(
                        FileChannel.MapMode.READ_ONLY,
                        afd.getStartOffset(),
                        afd.getDeclaredLength()
                );
    }

    // ------------------------------------------------------------
    // Player
    // ------------------------------------------------------------

    private void initPlayer(String path) {

        exoPlayer = new ExoPlayer.Builder(this).build();

        playerView.setPlayer(exoPlayer);

        exoPlayer.setMediaItem(MediaItem.fromUri(path));

        exoPlayer.prepare();

        exoPlayer.setPlayWhenReady(true);

        exoPlayer.addListener(new Player.Listener() {

            @Override
            public void onVideoSizeChanged(VideoSize videoSize) {

                if (!activityActive.get()) {
                    return;
                }

                overlayView.setVideoSize(
                        videoSize.width,
                        videoSize.height
                );
            }

            @Override
            public void onIsPlayingChanged(boolean isPlaying) {

                if (!activityActive.get()) {
                    return;
                }

                if (isPlaying) {

                    lastStartTime = System.currentTimeMillis();

                    startAnalysis();

                } else {

                    updateWatchedDuration();

                    stopAnalysis();
                }
            }
        });
    }

    // ------------------------------------------------------------
    // Analysis
    // ------------------------------------------------------------

    private void startAnalysis() {

        if (!activityActive.get()) {
            return;
        }

        if (running.compareAndSet(false, true)) {

            mainHandler.post(loop);

            Log.d(TAG, "Analysis started");
        }
    }

    private void stopAnalysis() {

        running.set(false);

        mainHandler.removeCallbacks(loop);

        overlayView.clear();

        Log.d(TAG, "Analysis stopped");
    }

    // ------------------------------------------------------------
    // Inference
    // ------------------------------------------------------------

    @OptIn(markerClass = UnstableApi.class)
    private void scheduleInference() {

        if (!activityActive.get()) {
            return;
        }

        if (interpreter == null) {
            return;
        }

        if (!inferring.compareAndSet(false, true)) {
            return;
        }

        View v = playerView.getVideoSurfaceView();

        if (!(v instanceof TextureView)) {
            inferring.set(false);
            return;
        }

        TextureView tv = (TextureView) v;

        if (!tv.isAvailable()) {
            inferring.set(false);
            return;
        }

        Bitmap frame = tv.getBitmap();

        if (frame == null) {
            inferring.set(false);
            return;
        }

        inferenceExecutor.execute(() -> runInferenceOnBackground(frame));
    }

    private void runInferenceOnBackground(Bitmap frame) {

        try {

            if (!activityActive.get()) {
                frame.recycle();
                return;
            }

            Canvas c = new Canvas(scaledBitmap);

            c.drawBitmap(
                    frame,
                    null,
                    new Rect(0, 0, inputSize, inputSize),
                    null
            );

            frame.recycle();

            scaledBitmap.getPixels(
                    pixelBuffer,
                    0,
                    inputSize,
                    0,
                    0,
                    inputSize,
                    inputSize
            );

            inputBuffer.rewind();

            for (int px : pixelBuffer) {

                inputBuffer.putFloat((px >> 16) & 0xFF);
                inputBuffer.putFloat((px >> 8) & 0xFF);
                inputBuffer.putFloat(px & 0xFF);
            }

            inputBuffer.rewind();

            interpreter.run(inputBuffer, outputBuffer);

            float max = 0.0001f;

            for (int y = 0; y < inputSize; y++) {

                for (int x = 0; x < inputSize; x++) {

                    float val = outputBuffer[0][y][x][0];

                    saliencyBuffer[y][x] = val;

                    if (val > max) {
                        max = val;
                    }
                }
            }

            for (int y = 0; y < inputSize; y++) {

                for (int x = 0; x < inputSize; x++) {

                    saliencyBuffer[y][x] /= max;
                }
            }

            float[][] result = saliencyBuffer;

            List<SaliencyFilter> chain = AnalysisConfig.FILTER_CHAIN;

            for (SaliencyFilter filter : chain) {
                result = filter.apply(result);
            }

            final float[][] snapshot = deepCopy(result);

            final float threshold =
                    AnalysisConfig.THRESHOLD_STATIC;

            // IMPORTANT:
            // NEVER post overlay update if paused/backgrounded
            if (activityActive.get()) {

                mainHandler.post(() -> {

                    if (!activityActive.get()) {
                        return;
                    }

                    overlayView.setSaliency(
                            snapshot,
                            threshold
                    );
                });
            }

        } catch (Exception e) {

            Log.e(TAG, "Inference error", e);

        } finally {

            inferring.set(false);
        }
    }

    private float[][] deepCopy(float[][] src) {

        float[][] dst = new float[src.length][];

        for (int i = 0; i < src.length; i++) {
            dst[i] = src[i].clone();
        }

        return dst;
    }

    // ------------------------------------------------------------
    // Watch time
    // ------------------------------------------------------------

    private void updateWatchedDuration() {

        if (lastStartTime > 0) {

            totalWatchedMs +=
                    System.currentTimeMillis() - lastStartTime;

            lastStartTime = 0;
        }
    }
}