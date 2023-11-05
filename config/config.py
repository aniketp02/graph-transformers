''' File having all the config for all the models to experiment with '''

import argparse

class CNNTrainingOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def parse_args(self):
        self.parser.add_argument('--filelists', type=str, help='train filelists path')
        self.parser.add_argument('--data_root', type=str, help='train data path')
        self.parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
        self.parser.add_argument('--epochs', type=int, default=100, help='No. of epoch to train')
        self.parser.add_argument('--checkpoint', type=int, default=2, help='Checkpoint to save')
        self.parser.add_argument('--checkpoint_path', type=str, default="./checkpoints",
                            help='path to store the checkpoints')
        self.parser.add_argument('--result_path', type=str, default="./results",
                            help='path store the results generated')
        self.parser.add_argument('--dataset_name', type=str, required=True, help="Dataset Name (EuroSAT, PatternNet)")

        return self.parser.parse_args()