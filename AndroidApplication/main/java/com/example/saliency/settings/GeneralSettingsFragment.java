package com.example.saliency.settings;

import android.graphics.Color;
import android.graphics.Typeface;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.*;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import com.example.saliency.config.AnalysisConfig;
import com.example.saliency.config.AppConfig;
import com.example.saliency.config.SettingsManager;
import com.example.saliency.overlay.OverlayStyle;
import com.example.saliency.overlay.SaliencyOverlayView;

public class GeneralSettingsFragment extends Fragment {

    private EditText etFps, etPadding, etVideosPerPage;
    private Spinner spinnerFilter;
    private SeekBar seekBarFilterValue, seekBarRadius;
    private TextView tvFilterValue, tvRadiusValue;
    private RadioGroup rgOverlayStyle;
    private Switch swFixedRadius;

    // Live Preview
    private SaliencyOverlayView previewView;

    private String selectedFilter = "Threshold Static";
    private static final String[] FILTERS = {
            "Threshold Static",
            "Threshold Dynamic",
            "Smoothing Alpha",
            "Notebook Brightness",
            "Notebook Contrast"
    };

    @Nullable
    @Override
    public View onCreateView(@NonNull android.view.LayoutInflater inflater,
                             @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {

        ScrollView scroll = new ScrollView(requireContext());
        scroll.setBackgroundColor(Color.parseColor("#0D0D0D"));

        LinearLayout root = new LinearLayout(requireContext());
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(20), dp(24), dp(20), dp(40));
        scroll.addView(root);



        // ================= LIVE PREVIEW (NEW) =================
        root.addView(section("👁 Live Preview"));
        FrameLayout previewContainer = new FrameLayout(requireContext());
        previewContainer.setBackgroundColor(Color.BLUE); // Dummy black window
        LinearLayout.LayoutParams previewParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(250) // Fixed height for preview
        );
        previewParams.setMargins(0, dp(10), 0, dp(20));
        previewContainer.setLayoutParams(previewParams);

        previewView = new SaliencyOverlayView(requireContext(), null);
        previewContainer.addView(previewView);
        root.addView(previewContainer);

        // Feed dummy data centered in the middle
        previewView.setSaliency(generateDummyMap(), 0.15f);

        // ================= OVERLAY STYLE (UPDATED) =================
        root.addView(section("🎨 Overlay Style"));
        rgOverlayStyle = new RadioGroup(requireContext());

        // Research Styles
        rgOverlayStyle.addView(radio("Biological Fovea", OverlayStyle.FOVEATED_EYE));
        rgOverlayStyle.addView(radio("Mist Focus", OverlayStyle.ATMOSPHERIC_MIST));
        rgOverlayStyle.addView(radio("Blackout", OverlayStyle.BLACKOUT));
        rgOverlayStyle.addView(radio("Vignette Shadow", OverlayStyle.VIGNETTE_SHADOW));
        rgOverlayStyle.addView(radio("Neon Glow", OverlayStyle.NEON_GLOW));

        setCheckedStyle(AnalysisConfig.OVERLAY_STYLE);

        // Trigger live update on style change
        rgOverlayStyle.setOnCheckedChangeListener((group, checkedId) -> {
            AnalysisConfig.OVERLAY_STYLE = getSelectedStyle();
            previewView.invalidate(); // Force redraw the preview
        });

        root.addView(rgOverlayStyle);

        // ================= RADIUS TUNING =================
        root.addView(section("Radius Control"));

        swFixedRadius = new Switch(requireContext());
        swFixedRadius.setText("Use Fixed Radius (Spotlight Mode)");
        swFixedRadius.setTextColor(Color.WHITE);
        swFixedRadius.setChecked(AnalysisConfig.USE_FIXED_RADIUS);
        swFixedRadius.setOnCheckedChangeListener((v, isChecked) -> {
            AnalysisConfig.USE_FIXED_RADIUS = isChecked;
            seekBarRadius.setEnabled(isChecked);
            previewView.invalidate(); // Force redraw
        });
        root.addView(swFixedRadius);

        tvRadiusValue = new TextView(requireContext());
        tvRadiusValue.setTextColor(Color.CYAN);
        tvRadiusValue.setPadding(0, dp(10), 0, 0);
        root.addView(tvRadiusValue);

        seekBarRadius = new SeekBar(requireContext());
        seekBarRadius.setMax(600); // 0 to 600 pixels
        seekBarRadius.setProgress((int) AnalysisConfig.FIXED_RADIUS_VALUE);
        seekBarRadius.setEnabled(AnalysisConfig.USE_FIXED_RADIUS);
        seekBarRadius.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar s, int p, boolean f) {
                AnalysisConfig.FIXED_RADIUS_VALUE = (float) p;
                tvRadiusValue.setText("Fixed Radius: " + p + " px");
                previewView.invalidate(); // Force redraw to see radius grow/shrink
            }
            @Override public void onStartTrackingTouch(SeekBar s) {}
            @Override public void onStopTrackingTouch(SeekBar s) {}
        });
        root.addView(seekBarRadius);
        tvRadiusValue.setText("Fixed Radius: " + (int)AnalysisConfig.FIXED_RADIUS_VALUE + " px");

        // ================= SYSTEM SETTINGS =================
        root.addView(section("📊 System"));
        root.addView(label("FPS"));
        etFps = field(String.valueOf(AnalysisConfig.RUNTIME_FPS));
        root.addView(etFps);

        root.addView(label("Videos per page"));
        etVideosPerPage = field(String.valueOf(AppConfig.RUNTIME_VIDEOS_PER_PAGE));
        root.addView(etVideosPerPage);

        updateSeek();

        Button save = new Button(requireContext());
        save.setText("SAVE CHANGES");
        save.setBackgroundColor(Color.parseColor("#a6b1e1"));
        save.setTextColor(Color.parseColor("#0D0D0D"));
        save.setAllCaps(true);
        save.setOnClickListener(v -> save());
        root.addView(save);

        return scroll;
    }

    // Generates a dummy saliency map with a high-attention spot right in the center
    private float[][] generateDummyMap() {
        int size = 20; // 20x20 grid is enough for the dynamic logic to scale
        float[][] map = new float[size][size];
        float cx = size / 2f;
        float cy = size / 2f;

        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                // Calculate distance to center
                float dist = (float) Math.sqrt(Math.pow(x - cx, 2) + Math.pow(y - cy, 2));
                // Creates a gradient peaking at 1.0 in the center, dropping off
                map[y][x] = Math.max(0f, 1.0f - (dist / (size / 3f)));
            }
        }
        return map;
    }

    private void save() {
        try {
            AnalysisConfig.OVERLAY_STYLE = getSelectedStyle();
            AnalysisConfig.RUNTIME_FPS = Integer.parseInt(etFps.getText().toString());
            // Added null check/catch for padding field since it wasn't instantiated in UI code
            if(etPadding != null && etPadding.getText() != null) {
                AnalysisConfig.RUNTIME_PADDING = Integer.parseInt(etPadding.getText().toString());
            }
            AppConfig.RUNTIME_VIDEOS_PER_PAGE = Integer.parseInt(etVideosPerPage.getText().toString());

            AnalysisConfig.rebuildFilterChain();
            SettingsManager.saveAll(requireContext());

            Toast.makeText(requireContext(), "Settings Updated", Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            Toast.makeText(requireContext(), "Error: Check your numbers", Toast.LENGTH_SHORT).show();
        }
    }

    private OverlayStyle getSelectedStyle() {
        int id = rgOverlayStyle.getCheckedRadioButtonId();
        int ordinal = id - 1001;
        if (ordinal >= 0 && ordinal < OverlayStyle.values().length) {
            return OverlayStyle.values()[ordinal];
        }
        return OverlayStyle.FOVEATED_EYE;
    }

    private void setCheckedStyle(OverlayStyle style) {
        rgOverlayStyle.check(style.ordinal() + 1001);
    }

    private void updateConfigValue(float val) {
        switch (selectedFilter) {
            case "Threshold Static": AnalysisConfig.THRESHOLD_STATIC = val; break;
            case "Threshold Dynamic": AnalysisConfig.THRESHOLD_DYNAMIC = val; break;
            case "Smoothing Alpha": AnalysisConfig.SMOOTHING_ALPHA = val; break;
            case "Notebook Brightness": AnalysisConfig.BRIGHTNESS = val; break;
            case "Notebook Contrast": AnalysisConfig.CONTRAST = val; break;
        }
    }

    private void updateSeek() {
        if(seekBarFilterValue == null || tvFilterValue == null) return; // Prevent crash if missing

        float val = 0;
        switch (selectedFilter) {
            case "Threshold Static": val = AnalysisConfig.THRESHOLD_STATIC; break;
            case "Threshold Dynamic": val = AnalysisConfig.THRESHOLD_DYNAMIC; break;
            case "Smoothing Alpha": val = AnalysisConfig.SMOOTHING_ALPHA; break;
            case "Notebook Brightness": val = AnalysisConfig.BRIGHTNESS; break;
            case "Notebook Contrast": val = AnalysisConfig.CONTRAST; break;
        }
        seekBarFilterValue.setProgress((int)(val * 100));
        updateDisplay(seekBarFilterValue.getProgress());
    }

    private void updateDisplay(int progress) {
        tvFilterValue.setText(selectedFilter + ": " + String.format("%.2f", progress / 100f));
    }

    private RadioButton radio(String text, OverlayStyle style) {
        RadioButton rb = new RadioButton(requireContext());
        rb.setText(text);
        rb.setId(style.ordinal() + 1001);
        rb.setTextColor(Color.WHITE);
        return rb;
    }

    private EditText field(String val) {
        EditText e = new EditText(requireContext());
        e.setText(val);
        e.setTextColor(Color.WHITE);
        e.setInputType(android.text.InputType.TYPE_CLASS_NUMBER);
        return e;
    }

    private TextView label(String t) {
        TextView tv = new TextView(requireContext());
        tv.setText(t);
        tv.setPadding(0, dp(10), 0, 0);
        tv.setTextColor(Color.GRAY);
        return tv;
    }

    private TextView section(String t) {
        TextView tv = new TextView(requireContext());
        tv.setText(t);
        tv.setPadding(0, dp(20), 0, dp(5));
        tv.setTextColor(Color.parseColor("#a6b1e1"));
        tv.setTypeface(null, Typeface.BOLD);
        return tv;
    }

    private TextView title(String t) {
        TextView tv = new TextView(requireContext());
        tv.setText(t);
        tv.setTextColor(Color.WHITE);
        tv.setTextSize(22);
        tv.setPadding(0, 0, 0, dp(15));
        tv.setGravity(Gravity.CENTER);
        return tv;
    }

    private int dp(int v) {
        return (int)(v * getResources().getDisplayMetrics().density);
    }
}