import os
import cv2
import json
import numpy as np


class VideoDatasetLoader:

    def __init__(self, config):
        self.cfg = config

    def load_video(self, video_file):

        X, Y = [], []

        v_path = os.path.join(
            self.cfg.VIDEO_DIR, video_file
        )

        v_name = os.path.splitext(video_file)[0]

        fix_path = os.path.join(
            self.cfg.FIX_ROOT,
            v_name,
            "fixations.json"
        )

        if not os.path.exists(fix_path):
            return None, None

        with open(fix_path, 'r') as f:
            fix_data = json.load(f)

        cap = cv2.VideoCapture(v_path)
        f_idx = 0

        while True:

            if f_idx % self.cfg.SAMPLING_RATE == 0:
                ret, frame = cap.read()
                if not ret:
                    break

                img = cv2.resize(
                    cv2.cvtColor(frame,
                                 cv2.COLOR_BGR2RGB),
                    (self.cfg.INPUT_SIZE,
                     self.cfg.INPUT_SIZE)
                )

                hm = np.zeros(
                    (self.cfg.INPUT_SIZE,
                     self.cfg.INPUT_SIZE),
                    dtype=np.float32
                )

                if f_idx < len(fix_data):
                    for pt in fix_data[f_idx]:

                        tx = int(
                            (pt[1] / 1920)
                            * self.cfg.INPUT_SIZE
                        )

                        ty = int(
                            (pt[0] / 1080)
                            * self.cfg.INPUT_SIZE
                        )

                        cv2.circle(
                            hm,
                            (
                                np.clip(tx, 0,
                                        self.cfg.INPUT_SIZE - 1),
                                np.clip(ty, 0,
                                        self.cfg.INPUT_SIZE - 1)
                            ),
                            4,
                            1.0,
                            -1
                        )

                hm = cv2.GaussianBlur(hm, (7, 7), 0)

                if hm.max() > 0:
                    hm /= hm.max()

                X.append(img)
                Y.append(hm[..., np.newaxis])

            else:
                ret = cap.grab()
                if not ret:
                    break

            f_idx += 1

        cap.release()

        if len(X) == 0:
            return None, None

        return (
            np.array(X, dtype=np.float32),
            np.array(Y, dtype=np.float32)
        )