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
    train_data_1 = pd.read_csv(train_data_dir / "t_train_data_1.csv", header=None)
    train_data_2 = pd.read_csv(train_data_dir / "t_train_data_2.csv", header=None)
    train_data_3 = pd.read_csv(train_data_dir / "t_train_data_3.csv", header=None)
    y_train_1 = train_data_1.iloc[:,-3:]
    X_train_1 = train_data_1.iloc[:,:-3]
    y_train_2 = train_data_2.iloc[:,-3:]
    X_train_2 = train_data_2.iloc[:,:-3]
    y_train_3 = train_data_3.iloc[:,-3:]
    X_train_3 = train_data_3.iloc[:,:-3]
    train_data = [(X_train_1, y_train_1), (X_train_2, y_train_2), (X_train_3, y_train_3)]

    valid_data_1 = pd.read_csv(valid_data_dir / "t_validation_data_1.csv", header=None)
    valid_data_2 = pd.read_csv(valid_data_dir / "t_validation_data_2.csv", header=None)
    valid_data_3 = pd.read_csv(valid_data_dir / "t_validation_data_3.csv", header=None)
    y_valid_1 = valid_data_1.iloc[:,-3:]
    X_valid_1 = valid_data_1.iloc[:,:-3]
    y_valid_2 = valid_data_2.iloc[:,-3:]
    X_valid_2 = valid_data_2.iloc[:,:-3]
    y_valid_3 = valid_data_3.iloc[:,-3:]
    X_valid_3 = valid_data_3.iloc[:,:-3]
    valid_data = [(X_valid_1, y_valid_1), (X_valid_2, y_valid_2), (X_valid_3, y_valid_3)]

    # retrieve or build the training model
    shape = train_data[0][0].shape
    model = Sequential([
        Input(shape=shape[1]),
        Dense(10, activation="relu"),
        Dense(8, activation="relu"),
        Dense(3, activation="softmax"),
    ])

    model.compile(
        optimizer=SGD(learning_rate=0.01),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    for (X_train, y_train), (X_valid, y_valid), i in zip(train_data, valid_data, range(3)):
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

        # save the trained model into the container (and sm will transfer it to s3).
        # model number and version number in this way required by sagemaker for
        # handling multiple models. 
        model_dir = os.environ["SM_MODEL_DIR"]
        model.save(Path(model_dir) / f"model{i+1}" / f"00{i+1}")


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