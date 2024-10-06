import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras import Input
from sklearn.metrics import accuracy_score

def train(train_data_dir, valid_data_dir, hp_epochs=20, hp_batch_size=32):
    # read the processed/transformed data
    train_data = pd.read_csv(train_data_dir / "t_train_data.csv", header=None)
    y_train = train_data.iloc[:,-3:]
    X_train = train_data.iloc[:,:-3]
    valid_data = pd.read_csv(valid_data_dir / "t_validation_data.csv")
    y_valid = valid_data.iloc[:,-3:]
    X_valid = valid_data.iloc[:,:-3]

    # retrieve or build the training model
    model = Sequential([
        Input(shape=(X_train.shape[1])),
        Dense(10, activation="relu"),
        Dense(8, activation="relu"),
        Dense(3, activation="softmax"),
    ])

    model.compile(
        optimizer=SGD(learning_rate=0.01),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        X_train,
        y_train,
        epochs=hp_epochs,
        batch_size=hp_batch_size,
        verbose=2,
        validation_data=(X_valid, y_valid)
    )

    ## only for debugging (not necessary)
    # predicted_class_probabilities = model.predict(X_train)
    # predicted_labels = np.argmax(predicted_class_probabilities, axis=1)
    # print(f"predicted_labels: {len(predicted_labels)}")
    # true_labels = np.argmax(y_train, axis=1)
    # print(f"true_labels: {len(true_labels)}")
    # accuracy = accuracy_score(true_labels, predicted_labels)
    # print(f"Training accuracy: {accuracy}")

    # save the trained model into the container (and sm will transfer it to s3)
    model_dir = os.environ["SM_MODEL_DIR"]
    model.save(Path(model_dir) / "001")


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--epochs", type=int, default=50)
    arg_parser.add_argument("--batch_size", type=int, default=32)
    arg_parser.add_argument("--channel_train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    arg_parser.add_argument("--channel_validation", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])

    args, _ = arg_parser.parse_known_args()
    train(
        hp_epochs=args.epochs,
        hp_batch_size=args.batch_size,
        train_data_dir=Path(args.channel_train),
        valid_data_dir=Path(args.channel_validation),
    )