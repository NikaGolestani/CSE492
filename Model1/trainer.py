import os
import numpy as np
import tensorflow as tf

from model_builder import SaliencyModelBuilder
from data_loader import VideoDatasetLoader


class SaliencyTrainer:

    def __init__(self, config):
        self.cfg = config
        self.loader = VideoDatasetLoader(config)
        self.model = self._initialize_model()
        self.val_losses_running = []

    def _initialize_model(self):

        if os.path.exists(self.cfg.MODEL_SAVE_NAME):
            print(">> Resuming existing model")
            return tf.keras.models.load_model(
                self.cfg.MODEL_SAVE_NAME
            )

        print(">> Building new model")
        builder = SaliencyModelBuilder(self.cfg)
        return builder.build()

    def _prepare_data(self, files):

        X_list, Y_list = [], []

        for f in files:
            x, y = self.loader.load_video(f)
            if x is not None:
                X_list.append(x)
                Y_list.append(y)

        if not X_list:
            return None, None

        X = np.concatenate(X_list, axis=0)
        Y = np.concatenate(Y_list, axis=0)

        idx = np.random.permutation(len(X))
        return X[idx], Y[idx]

    def train(self):

        video_files = [
            f for f in os.listdir(self.cfg.VIDEO_DIR)
            if f.endswith(".mp4")
        ]

        np.random.shuffle(video_files)

        total_buffers = len(video_files) // self.cfg.BUFFER_SIZE

        for i in range(
                0,
                len(video_files),
                self.cfg.BUFFER_SIZE):

            buffer_files = video_files[
                i:i + self.cfg.BUFFER_SIZE
            ]

            if len(buffer_files) < 3:
                continue

            buffer_id = i // self.cfg.BUFFER_SIZE + 1

            print("\n" + "=" * 60)
            print(f"BUFFER {buffer_id}/{total_buffers}")
            print("=" * 60)

            train_files = buffer_files[:-2]
            val_files = buffer_files[-2:]

            X_train, Y_train = self._prepare_data(train_files)

            if X_train is None:
                continue

            print(f"Train Frames : {X_train.shape[0]}")

            X_val, Y_val = self._prepare_data(val_files)
            val_data = None

            if X_val is not None:
                val_data = (X_val, Y_val)
                print(f"Val Frames   : {X_val.shape[0]}")

            history = self.model.fit(
                X_train,
                Y_train,
                epochs=self.cfg.EPOCHS_PER_BUFFER,
                batch_size=self.cfg.BATCH_SIZE,
                validation_data=val_data,
                shuffle=True
            )

            if val_data is not None:
                last_val = history.history['val_loss'][-1]
                self.val_losses_running.append(last_val)

                print(f"Buffer Val Loss : {last_val:.5f}")
                print(f"Running Val Avg : "
                      f"{np.mean(self.val_losses_running):.5f}")

            self.model.save(self.cfg.MODEL_SAVE_NAME)

            del X_train, Y_train
            if val_data is not None:
                del X_val, Y_val

        print("\nTraining Complete.")