from config import Configs
from trainer import SaliencyTrainer


def main():
    cfg = Configs()
    trainer = SaliencyTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()