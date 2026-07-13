package com.example.saliency.overlay;

public enum OverlayStyle {
    BLACKOUT,//EVERYWHERE A BLACK SEMI TRANSPARENT BUT CENTER HAS NON
    VIGNETTE_SHADOW,//in the center we have transparent as we go toward edge it will become more dark
    NEON_GLOW,//only a circle no edit
    ATMOSPHERIC_MIST,//SAME AS VIGNETT SHADOW BUT WITH MIST
    //NEEDS BLUR
    FOVEATED_EYE;// BIOLOGICAL HOW WE SEE SIMULATED

    public static OverlayStyle fromString(String v) {
        try { return valueOf(v); }
        catch (Exception e) { return FOVEATED_EYE; }
    }
}