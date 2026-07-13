package com.example.saliency.gallery;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewpager2.widget.ViewPager2;

import com.example.saliency.R;
import com.example.saliency.config.AppConfig;
import com.example.saliency.config.SettingsManager;
import com.example.saliency.analyzer.Video;
import com.example.saliency.player.PlayerActivity;
import com.example.saliency.settings.PasswordActivity;
import com.example.saliency.util.VideoThumbUtil;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class GalleryActivity extends AppCompatActivity {

    private static final int REQUEST_PERMISSION = 1;

    final ExecutorService thumbExecutor = Executors.newFixedThreadPool(2);
    private final ExecutorService dataExecutor = Executors.newSingleThreadExecutor();

    private final List<Video> allVideos = new ArrayList<>();

    private ViewPager2 viewPager;
    private PageAdapter pageAdapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_gallery_pager);

        SettingsManager.loadAll(this);

        viewPager   = findViewById(R.id.view_pager_gallery);
        pageAdapter = new PageAdapter();

        viewPager.setOrientation(ViewPager2.ORIENTATION_VERTICAL);
        viewPager.setAdapter(pageAdapter);

        View btnSettings = findViewById(R.id.btn_settings);
        if (btnSettings != null) {
            btnSettings.setOnClickListener(v ->
                    startActivity(new Intent(this, PasswordActivity.class)));
        }

        checkPermissionAndLoad();
    }

    @Override
    protected void onResume() {
        super.onResume();
        SettingsManager.loadAll(this);

        if (!allVideos.isEmpty()) {
            pageAdapter.refresh();
        }
    }

    // -----------------------------------------------------------------------
    // Permission Management
    // -----------------------------------------------------------------------

    private void checkPermissionAndLoad() {
        String perm = Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU
                ? Manifest.permission.READ_MEDIA_VIDEO
                : Manifest.permission.READ_EXTERNAL_STORAGE;

        if (ContextCompat.checkSelfPermission(this, perm) == PackageManager.PERMISSION_GRANTED)
            loadData();
        else
            ActivityCompat.requestPermissions(this, new String[]{perm}, REQUEST_PERMISSION);
    }

    @Override
    public void onRequestPermissionsResult(int req, @NonNull String[] perms, @NonNull int[] results) {
        super.onRequestPermissionsResult(req, perms, results);
        if (req == REQUEST_PERMISSION && results.length > 0
                && results[0] == PackageManager.PERMISSION_GRANTED)
            loadData();
    }

    // -----------------------------------------------------------------------
    // Data Loading Pipeline
    // -----------------------------------------------------------------------

    private void loadData() {
        dataExecutor.execute(() -> {
            List<Video> loaded = readVideosFromCache();
            runOnUiThread(() -> {
                allVideos.clear();
                allVideos.addAll(loaded);
                pageAdapter.refresh();
            });
        });
    }

    private List<Video> readVideosFromCache() {
        List<Video> result = new ArrayList<>();
        File cacheFile = new File(getFilesDir(), "video_analysis_cache.json");
        if (!cacheFile.exists()) return result;
        try {
            StringBuilder sb = new StringBuilder();
            try (BufferedReader r = new BufferedReader(new FileReader(cacheFile))) {
                String line;
                while ((line = r.readLine()) != null) sb.append(line);
            }
            JSONObject full = new JSONObject(sb.toString());
            Iterator<String> keys = full.keys();
            while (keys.hasNext()) {
                String hash  = keys.next();
                JSONObject e = full.getJSONObject(hash);
                if (e.optBoolean("excluded", false)) continue;
                String path = e.optString("video_path", "").replace("\\/", "/");
                String name = e.optString("video_name", "Unknown Video");
                if (new File(path).exists())
                    result.add(new Video(name, path, hash, full));
            }
        } catch (Exception e) {
            android.util.Log.e("GALLERY_DEBUG", "JSON load error", e);
        }
        return result;
    }

    // -----------------------------------------------------------------------
    // Pagination Context Calculations
    // -----------------------------------------------------------------------

    private int pageCount() {
        int vpp = Math.max(1, AppConfig.RUNTIME_VIDEOS_PER_PAGE);
        return (int) Math.ceil((double) allVideos.size() / vpp);
    }

    List<Video> getPageVideos(int pageIndex) {
        int vpp   = Math.max(1, AppConfig.RUNTIME_VIDEOS_PER_PAGE);
        int start = pageIndex * vpp;
        int end   = Math.min(start + vpp, allVideos.size());
        if (start >= allVideos.size()) return new ArrayList<>();
        return new ArrayList<>(allVideos.subList(start, end));
    }

    // -----------------------------------------------------------------------
    // PageAdapter
    // -----------------------------------------------------------------------
    class PageAdapter extends RecyclerView.Adapter<PageAdapter.PageVH> {

        PageAdapter() {
            setHasStableIds(false);
        }

        void refresh() {
            notifyDataSetChanged();
        }

        @NonNull
        @Override
        public PageVH onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View v = LayoutInflater.from(parent.getContext())
                    .inflate(R.layout.item_page_container, parent, false);

            v.setLayoutParams(new ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT));

            RecyclerView rv = v.findViewById(R.id.inner_recycler);
            return new PageVH(v, rv);
        }

        @Override
        public void onBindViewHolder(@NonNull PageVH holder, int position) {
            List<Video> videos = getPageVideos(position);
            int itemCount = videos.size();

            // Sane baseline defaults
            int columns = 2;

            if (itemCount == 1) {
                columns = 1;
            } else {
                // ── SQUARE ASPECT RATIO GRID SOLVER ──
                // Dynamically splits columns to keep cells as close to square (1:1) as possible
                int screenWidth = GalleryActivity.this.getResources().getDisplayMetrics().widthPixels;
                int screenHeight = GalleryActivity.this.getResources().getDisplayMetrics().heightPixels;

                // Add a bottom padding offset calculation to clean up the screen edge area
                float density = GalleryActivity.this.getResources().getDisplayMetrics().density;
                int bottomPadding = (int) (64 * density); // Sane 64dp safety margin
                int workingHeight = Math.max(100, screenHeight - bottomPadding);

                double bestSquareDiff = Double.MAX_VALUE;

                // Loop to evaluate structural layouts that fit perfectly
                for (int c = 1; c <= itemCount; c++) {
                    int r = (int) Math.ceil((double) itemCount / c);
                    double cellWidth = (double) screenWidth / c;
                    double cellHeight = (double) workingHeight / r;

                    // We check your constraint: divide longer axis until it is shorter than the short one
                    // We track the configuration that creates the minimal deviation from a true square ratio
                    double ratio = cellWidth / cellHeight;
                    double diff = Math.abs(1.0 - ratio);

                    if (diff < bestSquareDiff) {
                        bestSquareDiff = diff;
                        columns = c;
                    }
                }
            }

            if (columns < 1) columns = 1;

            final int finalColumns = columns;
            GridLayoutManager gridLayoutManager = new GridLayoutManager(GalleryActivity.this, finalColumns) {
                @Override
                public boolean canScrollVertically() { return false; }
                @Override
                public boolean canScrollHorizontally() { return false; }
            };

            holder.rv.setLayoutManager(gridLayoutManager);
            holder.rv.setHasFixedSize(true);
            holder.rv.setAdapter(new VideoAdapter(GalleryActivity.this, videos, finalColumns));
        }

        @Override
        public int getItemCount() {
            return pageCount();
        }

        class PageVH extends RecyclerView.ViewHolder {
            final RecyclerView rv;
            PageVH(View v, RecyclerView rv) {
                super(v);
                this.rv = rv;
            }
        }
    }

    // -----------------------------------------------------------------------
    // VideoAdapter
    // -----------------------------------------------------------------------
    static class VideoAdapter extends RecyclerView.Adapter<VideoAdapter.VH> {
        private final GalleryActivity activity;
        private final List<Video>     items;
        private final int             columns;

        VideoAdapter(GalleryActivity activity, List<Video> items, int columns) {
            this.activity = activity;
            this.items    = items;
            this.columns  = columns;
        }

        @NonNull
        @Override
        public VideoAdapter.VH onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View v = LayoutInflater.from(parent.getContext())
                    .inflate(R.layout.item_video_card, parent, false);

            int totalItems = getItemCount();
            int estimatedRows = (int) Math.ceil((double) totalItems / columns);
            if (estimatedRows < 1) estimatedRows = 1;

            int parentHeight = parent.getHeight();
            ViewGroup.LayoutParams lp = v.getLayoutParams();

            if (parentHeight > 0) {
                // Deduct extra layout padding from the total height computation
                float density = parent.getContext().getResources().getDisplayMetrics().density;
                int bottomPaddingPadding = (int) (64 * density); // Matching bottom margin rule

                int usableHeight = parentHeight - bottomPaddingPadding;
                if (usableHeight < 100) usableHeight = parentHeight;

                lp.height = usableHeight / estimatedRows;
            } else {
                lp.height = ViewGroup.LayoutParams.MATCH_PARENT;
            }

            v.setLayoutParams(lp);
            return new VideoAdapter.VH(v);
        }

        @Override
        public void onBindViewHolder(@NonNull VideoAdapter.VH holder, int position) {
            Video video = items.get(position);

            holder.title.setText(video.name);
            if (video.watchedTimesCount > 0) {
                holder.title.append(" (" + video.watchedTimesCount + ")");
            }

            if (items.size() > 15) {
                holder.title.setVisibility(View.GONE);
            } else {
                holder.title.setVisibility(View.VISIBLE);
            }

            holder.thumb.setImageBitmap(null);
            holder.thumb.setTag(video.path);

            activity.thumbExecutor.execute(() -> {
                Bitmap thumb = VideoThumbUtil.getThumb(video.path);
                activity.runOnUiThread(() -> {
                    if (video.path.equals(holder.thumb.getTag())) {
                        holder.thumb.setImageBitmap(thumb != null ? thumb : null);
                        if (thumb == null) {
                            holder.thumb.setImageResource(android.R.drawable.ic_media_play);
                        }
                    }
                });
            });

            holder.itemView.setOnClickListener(v -> {
                Intent intent = new Intent(activity, PlayerActivity.class);
                intent.putExtra("video_title", video.name);
                intent.putExtra("video_path",  video.path);
                activity.startActivity(intent);
            });
        }

        @Override
        public int getItemCount() { return items.size(); }

        static class VH extends RecyclerView.ViewHolder {
            final ImageView thumb;
            final TextView  title;
            VH(View v) {
                super(v);
                thumb = v.findViewById(R.id.img_thumb);
                title = v.findViewById(R.id.txt_title);
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        thumbExecutor.shutdownNow();
        dataExecutor.shutdownNow();
    }
}