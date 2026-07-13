package com.example.saliency.analyzer;

import android.graphics.*;
import android.graphics.drawable.GradientDrawable;
import android.os.Handler;
import android.os.Looper;
import android.view.*;
import android.widget.*;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.recyclerview.widget.RecyclerView;

import com.example.saliency.settings.VideoAnalysisSettingsFragment;
import com.example.saliency.util.VideoThumbUtil;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class VideoAdapter extends RecyclerView.Adapter<VideoAdapter.VideoViewHolder> {

    private final ExecutorService executor = Executors.newFixedThreadPool(2);
    private final Handler handler = new Handler(Looper.getMainLooper());
    private final List<Video> videos;

    // Communication listener for the Fragment
    private OnVideoInteractionListener interactionListener;

    public interface OnVideoInteractionListener {
        void onRemoveRequested(Video video);
        void onExclusionToggled(Video video, boolean isExcluded);
    }

    public void setOnVideoInteractionListener(OnVideoInteractionListener listener) {
        this.interactionListener = listener;
    }

    public VideoAdapter(List<Video> videos) {
        this.videos = videos;
    }

    @NonNull
    @Override
    public VideoViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        // --- Manual Layout Construction ---
        LinearLayout root = new LinearLayout(parent.getContext());
        root.setOrientation(LinearLayout.HORIZONTAL);
        root.setPadding(36, 36, 36, 36);
        root.setGravity(Gravity.CENTER_VERTICAL);

        RecyclerView.LayoutParams params = new RecyclerView.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        params.setMargins(18, 18, 18, 18);
        root.setLayoutParams(params);

        ImageView thumb = new ImageView(parent.getContext());
        thumb.setId(ViewIds.THUMB);
        thumb.setLayoutParams(new LinearLayout.LayoutParams(220, 220));
        thumb.setScaleType(ImageView.ScaleType.CENTER_CROP);

        LinearLayout textLayout = new LinearLayout(parent.getContext());
        textLayout.setOrientation(LinearLayout.VERTICAL);
        textLayout.setPadding(35, 0, 0, 0);

        TextView title = new TextView(parent.getContext());
        title.setId(ViewIds.TITLE);
        title.setTextSize(17);
        title.setTypeface(null, Typeface.BOLD);
        title.setTextColor(Color.BLACK);

        TextView details = new TextView(parent.getContext());
        details.setId(ViewIds.DETAILS);
        details.setVisibility(View.GONE);
        details.setTextSize(13);
        details.setPadding(0, 15, 0, 15);

        CheckBox exclude = new CheckBox(parent.getContext());
        exclude.setId(ViewIds.EXCLUDE);
        exclude.setText("Exclude from Gallery");
        exclude.setTextSize(12);

        textLayout.addView(title);
        textLayout.addView(details);
        textLayout.addView(exclude);

        root.addView(thumb);
        root.addView(textLayout);

        return new VideoViewHolder(root);
    }

    @Override
    public void onBindViewHolder(@NonNull VideoViewHolder h, int pos) {
        Video v = videos.get(pos);

        h.title.setText(v.name);
        h.thumb.setImageResource(android.R.drawable.ic_media_play);
        loadThumb(h.thumb, v.path);

        // Expand/Collapse analysis and logs
        h.details.setText(buildDetailsText(v));
        h.itemView.setOnClickListener(v1 -> {
            boolean isVisible = h.details.getVisibility() == View.VISIBLE;
            h.details.setVisibility(isVisible ? View.GONE : View.VISIBLE);
        });

        // LONG CLICK: Remove from JSON
        h.itemView.setOnLongClickListener(v1 -> {
            showDeleteDialog(v1.getContext(), v);
            return true;
        });

        // Border color based on score
        int color = Color.LTGRAY;
        if (v.features != null) {
            color = VideoAnalysisSettingsFragment.getScoreColor(v.features.score);
        }
        setBorder(h.itemView, color);

        // Exclude Toggle Logic
        h.exclude.setOnCheckedChangeListener(null); // prevent recursive trigger
        h.exclude.setChecked(v.excluded);
        h.exclude.setOnCheckedChangeListener((cb, checked) -> {
            v.excluded = checked;
            if (interactionListener != null) {
                interactionListener.onExclusionToggled(v, checked);
            }
        });
    }

    private String buildDetailsText(Video v) {
        StringBuilder sb = new StringBuilder();

        sb.append("📊 USAGE STATS\n");
        sb.append("Total Views: ").append(v.watchedTimesCount).append("\n");
        sb.append("Time Watched: ").append(formatTime(v.watchedPeriodSeconds)).append("\n");
        sb.append("---------------------------\n");

        if (v.isAnalyzed()) {
            sb.append("Safety Score: ").append(v.features.score).append("/100\n");
            sb.append("Main Issue: ").append(v.features.getPrimaryConcern());
        } else {
            sb.append("⚠️ Analysis: Processing...\n");
            sb.append("Safety data will appear shortly.");
        }

        return sb.toString();
    }

    private String formatTime(int totalSecs) {
        int hours = totalSecs / 3600;
        int minutes = (totalSecs % 3600) / 60;
        int seconds = totalSecs % 60;

        if (hours > 0) return String.format("%dh %dm %ds", hours, minutes, seconds);
        if (minutes > 0) return String.format("%dm %ds", minutes, seconds);
        return seconds + "s";
    }

    private void showDeleteDialog(android.content.Context context, Video v) {
        new AlertDialog.Builder(context)
                .setTitle("Delete Analysis?")
                .setMessage("This will remove this video's scores and logs from the cache.")
                .setPositiveButton("Delete", (d, w) -> {
                    if (interactionListener != null) interactionListener.onRemoveRequested(v);
                })
                .setNegativeButton("Cancel", null)
                .show();
    }

    private void loadThumb(ImageView v, String path) {
        executor.execute(() -> {
            Bitmap b = VideoThumbUtil.getThumb(path);
            if (b != null) handler.post(() -> v.setImageBitmap(b));
        });
    }

    private void setBorder(View v, int color) {
        GradientDrawable d = new GradientDrawable();
        d.setColor(Color.WHITE);
        d.setStroke(8, color);
        d.setCornerRadius(24);
        v.setBackground(d);
    }

    @Override
    public int getItemCount() {
        return videos.size();
    }

    // FIX: Add shutdown method to prevent memory leak
    public void shutdown() {
        executor.shutdownNow();
    }

    public static class VideoViewHolder extends RecyclerView.ViewHolder {
        TextView title, details;
        ImageView thumb;
        CheckBox exclude;

        public VideoViewHolder(@NonNull View itemView) {
            super(itemView);
            title = itemView.findViewById(ViewIds.TITLE);
            details = itemView.findViewById(ViewIds.DETAILS);
            thumb = itemView.findViewById(ViewIds.THUMB);
            exclude = itemView.findViewById(ViewIds.EXCLUDE);
        }
    }

    private static class ViewIds {
        static final int TITLE = 101;
        static final int DETAILS = 102;
        static final int EXCLUDE = 103;
        static final int THUMB = 104;
    }
}