import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def create_model():
    """Create a neural network model for attack detection"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(17,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )
    return model

def load_data():
    """Load and combine all training data"""
    features_list = []
    labels_list = []
    
    # Load data from all clients
    for client_id in range(3):
        features = pd.read_csv(f'data/client_{client_id}_features.csv')
        labels = pd.read_csv(f'data/client_{client_id}_labels.csv')
        features_list.append(features.values)
        labels_list.append(labels.values.ravel())
    
    # Combine all data
    x_train = np.concatenate(features_list)
    y_train = np.concatenate(labels_list)
    
    return x_train, y_train

def main():
    try:
        # Create models directory
        Path("models").mkdir(exist_ok=True)
        logging.info("Created models directory")
        
        # Load all training data
        x_train, y_train = load_data()
        logging.info(f"Loaded training data: {x_train.shape} samples")
        
        # Create and train model
        model = create_model()
        logging.info("Created model")
        
        history = model.fit(
            x_train, y_train,
            epochs=20,
            batch_size=64,
            validation_split=0.2,
            verbose=1
        )
        
        # Save the trained model
        model.save('models/trained_model')
        logging.info("Saved trained model")
        
        # Save training history
        history_file = Path("models") / "training_history.json"
        with open(history_file, "w") as f:
            history_dict = {
                "accuracy": [float(x) for x in history.history['accuracy']],
                "val_accuracy": [float(x) for x in history.history['val_accuracy']],
                "precision": [float(x) for x in history.history['precision']],
                "val_precision": [float(x) for x in history.history['val_precision']],
                "recall": [float(x) for x in history.history['recall']],
                "val_recall": [float(x) for x in history.history['val_recall']],
                "auc": [float(x) for x in history.history['auc']],
                "val_auc": [float(x) for x in history.history['val_auc']]
            }
            json.dump(history_dict, f)
        logging.info("Saved training history")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
