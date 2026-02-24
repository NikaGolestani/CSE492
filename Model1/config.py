class Configs:
    VIDEO_DIR = "../../Videos/Train"
    FIX_ROOT = "../../FixationsTrain/Train"

    INPUT_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS_PER_BUFFER = 3
    SAMPLING_RATE = 10
    BUFFER_SIZE = 10

    MODEL_SAVE_NAME = "saliency_x10_x3_final.keras"
    def __init__(self):
       return None