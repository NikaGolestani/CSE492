"""
Video Saliency Evaluation Script
--------------------------------
Evaluates a TFLite saliency model against human fixation data using academic metrics:
MSE (Mean Squared Error), SIM (Similarity), CC (Correlation Coefficient), and NSS (Normalized Scanpath Saliency).
"""

import os
import csv
import json
import logging
import argparse
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def calculate_metrics(target: np.ndarray, pred: np.ndarray, fixation_pts: List[List[float]]) -> Tuple[float, float, float, float]:
    """
    Calculates standard saliency metrics between the ground truth and prediction.
    
    Args:
        target: Ground truth heatmap (normalized 0-1).
        pred: Predicted saliency map (normalized 0-1).
        fixation_pts: List of [y, x] fixation coordinates.
        
    Returns:
        tuple: (MSE, SIM, CC, NSS)
    """
    # 1. MSE (Mean Squared Error)
    mse = np.mean((pred - target)**2)

    # 2. SIM (Similarity - Distribution based)
    t_sum, p_sum = np.sum(target), np.sum(pred)
    t_dist = target / (t_sum + 1e-7)
    p_dist = pred / (p_sum + 1e-7)
    sim = np.sum(np.minimum(t_dist, p_dist))

    # 3. CC (Linear Correlation Coefficient)
    t_flat, p_flat = target.flatten(), pred.flatten()
    if np.std(t_flat) == 0 or np.std(p_flat) == 0:
        cc = 0.0
    else:
        cc = np.corrcoef(t_flat, p_flat)[0, 1]

    # 4. NSS (Normalized Scanpath Saliency)
    eps = 1e-7
    p_mean, p_std = np.mean(pred), np.std(pred)
    p_zscore = (pred - p_mean) / (p_std + eps)
    
    nss_scores = []
    for pt in fixation_pts:
        # Map original 1080x1920 coordinates to 224x224
        ty, tx = int(pt[0] * 224 / 1080), int(pt[1] * 224 / 1920)
        nss_scores.append(p_zscore[np.clip(ty, 0, 223), np.clip(tx, 0, 223)])
    nss = np.mean(nss_scores) if nss_scores else 0.0

    return mse, sim, cc, nss

def run_evaluation(args):
    # Initialize TFLite Interpreter
    try:
        interpreter = tf.lite.Interpreter(model_path=args.model)
        interpreter.allocate_tensors()
    except Exception as e:
        logging.error(f"Failed to load TFLite model: {e}")
        return

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    video_files = [f for f in os.listdir(args.video_dir) if f.endswith(".mp4")]
    if not video_files:
        logging.warning(f"No MP4 files found in {args.video_dir}")
        return

    logging.info(f"Starting evaluation on {len(video_files)} videos...")

    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["video", "MSE", "SIM", "CC", "NSS"])

        for i, v_file in enumerate(video_files, 1):
            v_name = os.path.splitext(v_file)[0]
            fix_path = os.path.join(args.fix_dir, v_name, "fixations.json")
            
            if not os.path.exists(fix_path):
                logging.debug(f"Skipping {v_file}: Fixation JSON not found.")
                continue
            
            cap = cv2.VideoCapture(os.path.join(args.video_dir, v_file))
            with open(fix_path, 'r') as fj:
                fix_data = json.load(fj)

            # Determine frame indices (5 random frames)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            max_idx = min(total_frames, len(fix_data))
            indices = sorted(np.random.choice(max_idx, min(max_idx, 5), replace=False))

            v_metrics = []
            for f_idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret: break

                # Preprocess & Inference
                img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (args.size, args.size))
                img_input = np.expand_dims(img, 0).astype(np.float32)
                
                interpreter.set_tensor(input_details['index'], img_input)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details['index'])[0]

                # Generate GT Heatmap
                hm = np.zeros((args.size, args.size, 1), dtype=np.float32)
                curr_fixations = fix_data[f_idx]
                for pt in curr_fixations:
                    ty, tx = int(pt[0] * args.size / 1080), int(pt[1] * args.size / 1920)
                    cv2.circle(hm, (np.clip(tx, 0, args.size-1), np.clip(ty, 0, args.size-1)), 4, 1.0, -1)
                hm = cv2.GaussianBlur(hm, (7, 7), 0)
                if hm.max() > 0: hm /= hm.max()

                v_metrics.append(calculate_metrics(hm, pred, curr_fixations))

            cap.release()
            
            if v_metrics:
                avg_m = np.mean(v_metrics, axis=0)
                writer.writerow([v_file] + [f"{m:.6f}" for m in avg_m])

            if i % 50 == 0 or i == len(video_files):
                f.flush()
                logging.info(f"Processed [{i}/{len(video_files)}] | Last NSS: {avg_m[3]:.3f}")

    logging.info(f"Evaluation complete. Results saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Video Saliency Model")
    parser.add_argument("--video_dir", type=str, default="AVIMOS/Test", help="Path to videos")
    parser.add_argument("--fix_dir", type=str, default="Avimos/FixationsTest/Test", help="Path to fixation JSONs")
    parser.add_argument("--model", type=str, default="saliency_float16.tflite", help="TFLite model path")
    parser.add_argument("--output", type=str, default="Saliency_result.csv", help="Output CSV filename")
    parser.add_argument("--size", type=int, default=224, help="Model input size")
    
    args = parser.parse_args()
    run_evaluation(args)
