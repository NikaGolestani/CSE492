package com.example.saliency.settings;

import android.content.Intent;
import android.graphics.Color;
import android.graphics.Typeface;
import android.os.Bundle;
import android.text.InputType;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.example.saliency.config.SettingsManager;

/**
 * Gate screen shown when the user taps the Settings lock icon in GalleryActivity.
 *
 * Two modes, determined automatically:
 *   • SETUP  — no password has been saved yet (first launch of settings).
 *              Shows two fields: "New password" + "Confirm password".
 *   • UNLOCK — a password already exists. Shows one field: "Enter password".
 *
 * On success, launches SettingsActivity and finishes itself.
 */
public class PasswordActivity extends AppCompatActivity {

    private boolean isSetupMode;
    private EditText fieldPassword;
    private EditText fieldConfirm;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        isSetupMode = !SettingsManager.hasPassword(this);

        // -----------------------------------------------------------------------
        // Root layout
        // -----------------------------------------------------------------------
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.parseColor("#0D0D0D"));
        root.setGravity(Gravity.CENTER);
        root.setPadding(dp(40), dp(60), dp(40), dp(60));

        // -----------------------------------------------------------------------
        // Icon / title
        // -----------------------------------------------------------------------
        TextView icon = new TextView(this);
        icon.setText("⚙");
        icon.setTextSize(TypedValue.COMPLEX_UNIT_SP, 52);
        icon.setGravity(Gravity.CENTER);
        icon.setPadding(0, 0, 0, dp(8));
        root.addView(icon, centredParams(dp(16)));

        TextView title = new TextView(this);
        title.setText(isSetupMode ? "Create Settings Password" : "Settings");
        title.setTextColor(Color.WHITE);
        title.setTextSize(TypedValue.COMPLEX_UNIT_SP, 24);
        title.setTypeface(null, Typeface.BOLD);
        title.setGravity(Gravity.CENTER);
        root.addView(title, centredParams(dp(6)));

        TextView subtitle = new TextView(this);
        subtitle.setText(isSetupMode
                ? "Choose a password to protect settings"
                : "Enter your password to continue");
        subtitle.setTextColor(Color.parseColor("#888888"));
        subtitle.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14);
        subtitle.setGravity(Gravity.CENTER);
        root.addView(subtitle, centredParams(dp(40)));

        // -----------------------------------------------------------------------
        // Password field
        // -----------------------------------------------------------------------
        fieldPassword = styledEditText(isSetupMode ? "New password" : "Password");
        root.addView(fieldPassword, fullWidthParams(dp(16)));

        // -----------------------------------------------------------------------
        // Confirm field (setup mode only)
        // -----------------------------------------------------------------------
        if (isSetupMode) {
            fieldConfirm = styledEditText("Confirm password");
            root.addView(fieldConfirm, fullWidthParams(dp(32)));
        } else {
            // spacer
            View spacer = new View(this);
            root.addView(spacer, new LinearLayout.LayoutParams(1, dp(32)));
        }

        // -----------------------------------------------------------------------
        // Action button
        // -----------------------------------------------------------------------
        Button btnAction = new Button(this);
        btnAction.setText(isSetupMode ? "SET PASSWORD" : "UNLOCK");
        btnAction.setAllCaps(true);
        btnAction.setTextColor(Color.parseColor("#0D0D0D"));
        btnAction.setBackgroundColor(Color.parseColor("#a6b1e1"));
        btnAction.setTextSize(TypedValue.COMPLEX_UNIT_SP, 15);
        btnAction.setTypeface(null, Typeface.BOLD);
        btnAction.setPadding(dp(24), dp(16), dp(24), dp(16));
        btnAction.setOnClickListener(v -> handleAction());

        LinearLayout.LayoutParams btnParams =
                new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,
                        LinearLayout.LayoutParams.WRAP_CONTENT);
        root.addView(btnAction, btnParams);

        // Cancel
        TextView cancel = new TextView(this);
        cancel.setText("Cancel");
        cancel.setTextColor(Color.parseColor("#555555"));
        cancel.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14);
        cancel.setGravity(Gravity.CENTER);
        cancel.setPadding(0, dp(24), 0, 0);
        cancel.setOnClickListener(v -> finish());
        root.addView(cancel, centredParams(0));

        setContentView(root);
    }

    // -----------------------------------------------------------------------
    // Logic
    // -----------------------------------------------------------------------

    private void handleAction() {
        String password = fieldPassword.getText().toString().trim();

        if (password.isEmpty()) {
            toast("Password cannot be empty");
            return;
        }

        if (isSetupMode) {
            String confirm = fieldConfirm.getText().toString().trim();
            if (!password.equals(confirm)) {
                toast("Passwords do not match");
                return;
            }
            SettingsManager.setPassword(this, password);
            openSettings();
        } else {
            if (SettingsManager.checkPassword(this, password)) {
                openSettings();
            } else {
                toast("Incorrect password");
                fieldPassword.setText("");
            }
        }
    }

    private void openSettings() {
        startActivity(new Intent(this, SettingsActivity.class));
        finish();
    }

    // -----------------------------------------------------------------------
    // View helpers
    // -----------------------------------------------------------------------

    private EditText styledEditText(String hint) {
        EditText et = new EditText(this);
        et.setHint(hint);
        et.setHintTextColor(Color.parseColor("#555555"));
        et.setTextColor(Color.WHITE);
        et.setTextSize(TypedValue.COMPLEX_UNIT_SP, 16);
        et.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_PASSWORD);
        et.setBackgroundColor(Color.parseColor("#1A1A1A"));
        et.setPadding(dp(16), dp(14), dp(16), dp(14));
        return et;
    }

    private LinearLayout.LayoutParams centredParams(int bottomMarginPx) {
        LinearLayout.LayoutParams p =
                new LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.WRAP_CONTENT,
                        LinearLayout.LayoutParams.WRAP_CONTENT);
        p.gravity = Gravity.CENTER_HORIZONTAL;
        p.bottomMargin = bottomMarginPx;
        return p;
    }

    private LinearLayout.LayoutParams fullWidthParams(int bottomMarginPx) {
        LinearLayout.LayoutParams p =
                new LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT,
                        LinearLayout.LayoutParams.WRAP_CONTENT);
        p.bottomMargin = bottomMarginPx;
        return p;
    }

    private int dp(int value) {
        return Math.round(TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP, value,
                getResources().getDisplayMetrics()));
    }

    private void toast(String msg) {
        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show();
    }
}
