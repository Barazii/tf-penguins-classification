import argparse
import os

def train(train_data_dir, model_dir, hp_epochs, hp_batch_size):
    pass

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--epochs", type=int, default=50)
    arg_parser.add_argument("--batch_size", type=int, default=32)
    arg_parser.add_argument("--channel_train", type=str,)
    arg_parser.add_argument("--model_dir", type=str,)
    args = arg_parser.parse_args()
    train(
        hp_epochs=args.epochs,
        hp_batch_size=args.batch_size,
        model_dir=args.model_dir,
        train_data_dir=args.channel_train,
    )