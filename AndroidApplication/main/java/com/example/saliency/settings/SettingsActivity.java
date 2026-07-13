package com.example.saliency.settings;

import android.os.Bundle;
import com.example.saliency.R;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.viewpager2.widget.ViewPager2;

import com.google.android.material.tabs.TabLayout;
import com.google.android.material.tabs.TabLayoutMediator;

public class SettingsActivity extends AppCompatActivity {

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_settings);

        ViewPager2 pager = findViewById(R.id.viewPager);
        TabLayout tabs = findViewById(R.id.tabs);

        pager.setAdapter(new SettingsPagerAdapter(this));

        new TabLayoutMediator(tabs, pager, (tab, position) -> {
            if (position == 0) tab.setText("General");
            else tab.setText("Videos");
        }).attach();
    }
}