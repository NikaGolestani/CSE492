package com.example.saliency.model;

/**
 * Lightweight data class representing a single video entry in the gallery.
 *
 * In v1 this is backed by a local file path.
 * Swap {@code path} for a URL string in v2 when streaming is introduced.
 */
public class Video {

    private final String title;
    private final String path; // local file path — swap for URL in v2

    public Video(String title, String path) {
        this.title = title;
        this.path  = path;
    }

    public String getTitle() { return title; }
    public String getPath()  { return path;  }
}
