package com.example.saliency.settings;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Color;
import android.net.Uri;
import android.os.*;
import android.provider.MediaStore;
import android.util.Log;
import android.view.*;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.widget.*;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.example.saliency.analyzer.Video;
import com.example.saliency.analyzer.VideoAdapter;
import com.example.saliency.analyzer.VideoAnalyzer;
import com.example.saliency.analyzer.VideoFeatures;
import com.example.saliency.analyzer.VideoScoreCache;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class VideoAnalysisSettingsFragment extends Fragment {

    private static final String TAG = "SETTINGS_LOGS";
    private RecyclerView rv;
    private VideoAdapter adapter;
    private final List<Video> videoList = new ArrayList<>();
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    private final ActivityResultLauncher<Intent> filePickerLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == android.app.Activity.RESULT_OK && result.getData() != null) {
                    Uri selectedUri = result.getData().getData();
                    handlePickedVideo(selectedUri);
                }
            }
    );

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        LinearLayout root = new LinearLayout(requireContext());
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.parseColor("#0D0D0D"));

        Button btnAdd = new Button(requireContext());
        btnAdd.setText("Add Video");
        btnAdd.setBackgroundColor(Color.parseColor("#a6b1e1"));
        btnAdd.setTextColor(Color.parseColor("#0D0D0D"));
        btnAdd.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Video.Media.EXTERNAL_CONTENT_URI);
            filePickerLauncher.launch(intent);
        });

        LinearLayout.LayoutParams btnParams = new LinearLayout.LayoutParams(-1, -2);
        btnParams.setMargins(40, 40, 40, 20);
        root.addView(btnAdd, btnParams);

        Button btnPlots = new Button(requireContext());
        btnPlots.setText("Show Analysis");
        btnPlots.setBackgroundColor(Color.parseColor("#2a2a3a"));
        btnPlots.setTextColor(Color.parseColor("#a6b1e1"));
        btnPlots.setOnClickListener(v -> showPlotsDialog());

        LinearLayout.LayoutParams plotsBtnParams = new LinearLayout.LayoutParams(-1, -2);
        plotsBtnParams.setMargins(40, 0, 40, 20);
        root.addView(btnPlots, plotsBtnParams);

        rv = new RecyclerView(requireContext());
        rv.setLayoutManager(new LinearLayoutManager(requireContext()));
        root.addView(rv, new LinearLayout.LayoutParams(-1, -1));

        return root;
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        adapter = new VideoAdapter(videoList);
        adapter.setOnVideoInteractionListener(new VideoAdapter.OnVideoInteractionListener() {
            @Override
            public void onRemoveRequested(Video video) {
                confirmAndRemove(video);
            }

            @Override
            public void onExclusionToggled(Video video, boolean isExcluded) {
                updateExclusionInCache(video, isExcluded);
            }
        });
        rv.setAdapter(adapter);
        checkAndLoad();
    }

    // ═══════════════════════════════════════════════════════════════
    // PLOTS DIALOG WITH FIXED DATA EXTRACTION
    // ═══════════════════════════════════════════════════════════════
    @SuppressLint("SetJavaScriptEnabled")
    private void showPlotsDialog() {
        executor.execute(() -> {
            StringBuilder labels      = new StringBuilder();
            StringBuilder watchTimes  = new StringBuilder();
            StringBuilder motionArr   = new StringBuilder();
            StringBuilder flashArr    = new StringBuilder();
            StringBuilder colorArr    = new StringBuilder();
            StringBuilder clutterArr  = new StringBuilder();
            StringBuilder scoreArr    = new StringBuilder();
            StringBuilder tapsArr     = new StringBuilder();

            try {
                File cacheFile = new File(requireContext().getFilesDir(), "video_analysis_cache.json");
                JSONObject fullCache = getCacheObject(cacheFile);
                Iterator<String> keys = fullCache.keys();

                while (keys.hasNext()) {
                    String hash  = keys.next();
                    JSONObject e = fullCache.getJSONObject(hash);

                    String name = e.optString("video_name", hash.substring(0, 8));
                    if (name.contains(".")) name = name.substring(0, name.lastIndexOf('.'));

                    if (labels.length() > 0) {
                        labels.append(",");
                        watchTimes.append(",");
                        motionArr.append(",");
                        flashArr.append(",");
                        colorArr.append(",");
                        clutterArr.append(",");
                        scoreArr.append(",");
                        tapsArr.append(",");
                    }

                    labels.append("\"").append(name.replace("\"", "")).append("\"");
                    watchTimes.append(e.optInt("watched_period", 0));
                    motionArr.append(String.format("%.4f", e.optDouble("motion", 0)));
                    flashArr.append(String.format("%.4f",  e.optDouble("flash",  0)));
                    colorArr.append(String.format("%.4f",  e.optDouble("color",  0)));
                    clutterArr.append(String.format("%.4f",e.optDouble("clutter",0)));
                    scoreArr.append(e.optInt("score", 0));
                    tapsArr.append(e.optInt("watched_times", 0));
                }
            } catch (Exception ex) {
                Log.e(TAG, "Plot data build error", ex);
            }

            final String html = buildPlotsHtml(
                    labels.toString(),
                    watchTimes.toString(),
                    motionArr.toString(),
                    flashArr.toString(),
                    colorArr.toString(),
                    clutterArr.toString(),
                    scoreArr.toString(),
                    tapsArr.toString()
            );

            new Handler(Looper.getMainLooper()).post(() -> {
                WebView webView = new WebView(requireContext());
                WebSettings ws = webView.getSettings();
                ws.setJavaScriptEnabled(true);
                ws.setDomStorageEnabled(true);
                webView.setBackgroundColor(Color.parseColor("#0D0D0D"));

                webView.loadDataWithBaseURL(
                        "https://cdnjs.cloudflare.com",
                        html,
                        "text/html",
                        "UTF-8",
                        null
                );

                AlertDialog dialog = new AlertDialog.Builder(requireContext())
                        .setView(webView)
                        .setPositiveButton("Close", null)
                        .create();

                dialog.show();
                Window window = dialog.getWindow();
                if (window != null) {
                    window.setBackgroundDrawable(new android.graphics.drawable.ColorDrawable(Color.parseColor("#0D0D0D")));
                }
                dialog.getButton(AlertDialog.BUTTON_POSITIVE)
                        .setTextColor(Color.parseColor("#a6b1e1"));
            });
        });
    }

    private String buildPlotsHtml(
            String labels, String watchTimes,
            String motion, String flash,
            String color,  String clutter,
            String scores, String taps
    ) {
        return "<!DOCTYPE html><html><head>"
                + "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                + "<style>"
                + "body{background:#0d0d0d;color:#ccc;font-family:sans-serif;margin:0;padding:12px;}"
                + "h3{color:#a6b1e1;font-size:13px;margin:16px 0 6px;font-weight:500;}"
                + ".legend{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:8px;font-size:11px;color:#aaa;}"
                + ".dot{width:10px;height:10px;border-radius:2px;display:inline-block;margin-right:4px;}"
                + ".scroll-container{width:100%;overflow-x:auto;overflow-y:hidden;margin-bottom:24px;"
                + "  scrollbar-width:thin;scrollbar-color:#333 #0d0d0d;}"
                + ".scroll-container::-webkit-scrollbar{height:6px;}"
                + ".scroll-container::-webkit-scrollbar-thumb{background:#333;border-radius:3px;}"
                + ".wrap{position:relative;height:220px;min-width:100%;}"
                + "</style></head><body>"

                + "<div class='legend'>"
                + "<span><span class='dot' style='background:#378ADD'></span>Motion</span>"
                + "<span><span class='dot' style='background:#D85A30'></span>Flash</span>"
                + "<span><span class='dot' style='background:#1D9E75'></span>Color</span>"
                + "<span><span class='dot' style='background:#7F77DD'></span>Clutter</span>"
                + "<span><span class='dot' style='background:#FFC107'></span>Taps</span>"
                + "</div>"

                + "<h3>Feature scores by video (watch time in seconds)</h3>"
                + "<div class='scroll-container'><div class='wrap' id='barWrap'><canvas id='barChart'></canvas></div></div>"

                + "<h3>Watch time (s) vs feature score</h3>"
                + "<div class='scroll-container'><div class='wrap' id='scatterWrap'><canvas id='scatterChart'></canvas></div></div>"

                + "<h3>Tap Numbers Frequency</h3>"
                + "<div class='scroll-container'><div class='wrap' id='tapWrap'><canvas id='tapChart'></canvas></div></div>"

                + "<script src='https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js'></script>"
                + "<script>"

                + "var rawLabels=[" + labels + "];"
                + "var watchTimes=[" + watchTimes + "];"
                + "var motion=[" + motion + "];"
                + "var flash=[" + flash + "];"
                + "var colorD=[" + color + "];"
                + "var clutter=[" + clutter + "];"
                + "var scores=[" + scores + "];"
                + "var taps=[" + (taps != null ? taps : "") + "];"

                + "var minWidth = Math.max(window.innerWidth - 24, rawLabels.length * 55);"
                + "document.getElementById('barWrap').style.width = minWidth + 'px';"
                + "document.getElementById('tapWrap').style.width = minWidth + 'px';"
                + "document.getElementById('scatterWrap').style.width = '100%';"

                + "var cleanBarLabels=rawLabels.map(function(n,i){return 'Vid '+(i+1)+' ('+watchTimes[i]+'s)';});"
                + "var axCfg={grid:{color:'rgba(255,255,255,0.08)'},ticks:{color:'#aaa',font:{size:10}}};"

                + "new Chart(document.getElementById('barChart'),{"
                + "type:'bar',"
                + "data:{labels:cleanBarLabels,datasets:["
                + "{label:'Motion', data:motion,  backgroundColor:'#378ADD',borderRadius:3},"
                + "{label:'Flash',  data:flash,   backgroundColor:'#D85A30',borderRadius:3},"
                + "{label:'Color',  data:colorD,  backgroundColor:'#1D9E75',borderRadius:3},"
                + "{label:'Clutter',data:clutter, backgroundColor:'#7F77DD',borderRadius:3}"
                + "]},"
                + "options:{responsive:true,maintainAspectRatio:false,"
                + "plugins:{"
                + "legend:{display:false},"
                + "tooltip:{callbacks:{title:function(items){var idx=items[0].dataIndex; return rawLabels[idx];}}}"
                + "},"
                + "scales:{x:Object.assign({},axCfg,{grid:{display:false}}),y:Object.assign({},axCfg,{min:0,max:0.55})}}"
                + "});"

                + "var feats=["
                + "{key:'Motion', data:motion,  color:'#378ADD'},"
                + "{key:'Flash',  data:flash,   color:'#D85A30'},"
                + "{key:'Color',  data:colorD,  color:'#1D9E75'},"
                + "{key:'Clutter',data:clutter, color:'#7F77DD'}"
                + "];"
                + "var scatterDs=feats.map(function(f){"
                + "return{label:f.key,"
                + "data:f.data.map(function(v,i){return{x:watchTimes[i],y:v};}),"
                + "backgroundColor:f.color,pointRadius:7,pointHoverRadius:9};"
                + "});"
                + "new Chart(document.getElementById('scatterChart'),{"
                + "type:'scatter',"
                + "data:{datasets:scatterDs},"
                + "options:{responsive:true,maintainAspectRatio:false,"
                + "plugins:{legend:{display:false},"
                + "tooltip:{callbacks:{label:function(c){return c.dataset.label+': '+c.parsed.y.toFixed(3)+' @ '+c.parsed.x+'s';}}}},"
                + "scales:{"
                + "x:Object.assign({},axCfg,{title:{display:true,text:'Watch time (s)',color:'#aaa',font:{size:11}},min:-1}),"
                + "y:Object.assign({},axCfg,{title:{display:true,text:'Feature score',color:'#aaa',font:{size:11}},min:0,max:0.55})"
                + "}}"
                + "});"

                + "new Chart(document.getElementById('tapChart'),{"
                + "type:'line',"
                + "data:{"
                + "labels:cleanBarLabels,"
                + "datasets:[{label:'Taps Count',data:taps,borderColor:'#FFC107',backgroundColor:'rgba(255,193,7,0.15)',fill:true,tension:0.2,pointRadius:4}]"
                + "},"
                + "options:{responsive:true,maintainAspectRatio:false,"
                + "plugins:{"
                + "legend:{display:false},"
                + "tooltip:{callbacks:{title:function(items){var idx=items[0].dataIndex; return rawLabels[idx];}}}"
                + "},"
                + "scales:{x:Object.assign({},axCfg,{grid:{display:false}}),y:Object.assign({},axCfg,{beginAtZero:true})}}"
                + "});"

                + "</script></body></html>";
    }

    private void handlePickedVideo(Uri uri) {
        String realPath = null;
        String[] projection = { MediaStore.Video.Media.DATA };

        try (Cursor cursor = requireContext().getContentResolver().query(uri, projection, null, null, null)) {
            if (cursor != null && cursor.moveToFirst()) {
                int colIndex = cursor.getColumnIndexOrThrow(MediaStore.Video.Media.DATA);
                realPath = cursor.getString(colIndex);
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to resolve real path", e);
        }

        if (realPath != null && new File(realPath).exists()) {
            addNewVideo(realPath);
        } else {
            Toast.makeText(getContext(), "Error: Could not find real file location.", Toast.LENGTH_SHORT).show();
        }
    }

    private void addNewVideo(String path) {
        executor.execute(() -> {
            try {
                VideoScoreCache.initPlaceholderIfMissing(requireContext(), path);
                VideoFeatures existing = VideoScoreCache.get(requireContext(), path);
                if (existing != null) {
                    new Handler(Looper.getMainLooper()).post(() ->
                            Toast.makeText(getContext(), "Already analyzed!", Toast.LENGTH_SHORT).show());
                    return;
                }

                new Handler(Looper.getMainLooper()).post(() ->
                        Toast.makeText(getContext(), "Analyzing...", Toast.LENGTH_SHORT).show());

                VideoFeatures features = VideoAnalyzer.analyze(requireContext(), path);
                VideoScoreCache.put(requireContext(), path, features);
                loadFromCache();

                new Handler(Looper.getMainLooper()).post(() ->
                        Toast.makeText(getContext(), "Analysis Complete!", Toast.LENGTH_SHORT).show());

            } catch (Exception e) {
                Log.e(TAG, "Analysis error", e);
                new Handler(Looper.getMainLooper()).post(() ->
                        Toast.makeText(getContext(), "Analysis failed.", Toast.LENGTH_SHORT).show());
            }
        });
    }

    private void loadFromCache() {
        executor.execute(() -> {
            List<Video> results = new ArrayList<>();
            File cacheFile = new File(requireContext().getFilesDir(), "video_analysis_cache.json");

            if (!cacheFile.exists()) return;

            try {
                JSONObject fullCache = getCacheObject(cacheFile);
                Iterator<String> hashes = fullCache.keys();

                while (hashes.hasNext()) {
                    String hash = hashes.next();
                    JSONObject entry = fullCache.getJSONObject(hash);
                    String path = entry.optString("video_path", "");

                    if (new File(path).exists() || path.contains("storage")) {
                        results.add(new Video(entry.optString("video_name", "Video"), path, hash, fullCache));
                    }
                }
            } catch (Exception e) {
                Log.e(TAG, "Load Error", e);
            }

            new Handler(Looper.getMainLooper()).post(() -> {
                videoList.clear();
                videoList.addAll(results);
                adapter.notifyDataSetChanged();
            });
        });
    }

    private void confirmAndRemove(Video video) {
        new AlertDialog.Builder(requireContext())
                .setTitle("Delete?")
                .setMessage("Remove from analysis cache?")
                .setPositiveButton("Remove", (d, w) -> {
                    executor.execute(() -> {
                        try {
                            File cacheFile = new File(requireContext().getFilesDir(), "video_analysis_cache.json");
                            JSONObject cache = getCacheObject(cacheFile);
                            cache.remove(video.hash);
                            saveCacheObject(cacheFile, cache);
                            loadFromCache();
                        } catch (Exception e) { Log.e(TAG, "Delete error", e); }
                    });
                })
                .setNegativeButton("Cancel", null)
                .show();
    }

    private void updateExclusionInCache(Video video, boolean isExcluded) {
        executor.execute(() -> {
            try {
                File cacheFile = new File(requireContext().getFilesDir(), "video_analysis_cache.json");
                JSONObject cache = getCacheObject(cacheFile);
                if (cache.has(video.hash)) {
                    cache.getJSONObject(video.hash).put("excluded", isExcluded);
                    saveCacheObject(cacheFile, cache);
                }
            } catch (Exception e) { Log.e(TAG, "Exclusion error", e); }
        });
    }

    private JSONObject getCacheObject(File file) throws Exception {
        if (!file.exists()) return new JSONObject();
        StringBuilder sb = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = reader.readLine()) != null) sb.append(line);
        }
        return new JSONObject(sb.toString());
    }

    private void saveCacheObject(File file, JSONObject json) throws Exception {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
            writer.write(json.toString(4));
        }
    }

    private void checkAndLoad() {
        String perm = Build.VERSION.SDK_INT >= 33 ? Manifest.permission.READ_MEDIA_VIDEO : Manifest.permission.READ_EXTERNAL_STORAGE;
        if (ContextCompat.checkSelfPermission(requireContext(), perm) == PackageManager.PERMISSION_GRANTED) {
            loadFromCache();
        } else {
            requestPermissions(new String[]{perm}, 100);
        }
    }

    public static int getScoreColor(int score) {
        if (score >= 80) return Color.parseColor("#4CAF50");
        if (score >= 45) return Color.parseColor("#FFC107");
        return Color.parseColor("#F44336");
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        executor.shutdownNow();
    }
}