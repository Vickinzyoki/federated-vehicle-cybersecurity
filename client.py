import flwr as fl
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def load_data(client_id):
    """Load training data for a specific client"""
    features = pd.read_csv(f'data/client_{client_id}_features.csv')
    labels = pd.read_csv(f'data/client_{client_id}_labels.csv')
    return features.values, labels.values.ravel()

def create_model():
    """Create a neural network model for attack detection"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(17,)),  # Input shape based on feature extraction
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

class CyberattackClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.x_train, self.y_train = load_data(client_id)
        self.model = create_model()
    
    def get_parameters(self, config):
        return [np.array(x) for x in self.model.get_weights()]
    
    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        
        # Get training config
        batch_size = config.get("batch_size", 64)
        epochs = config.get("local_epochs", 3)
        
        # Train the model
        history = self.model.fit(
            self.x_train, 
            self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            shuffle=True,
            verbose=0
        )
        
        # Return updated model parameters and number of examples used
        parameters_prime = [np.array(x) for x in self.model.get_weights()]
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][-1],
            "accuracy": history.history["accuracy"][-1],
            "precision": history.history["precision"][-1],
            "recall": history.history["recall"][-1],
            "auc": history.history["auc"][-1]
        }
        
        return parameters_prime, num_examples_train, results
    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        
        # Evaluate the model
        loss, accuracy, precision, recall, auc = self.model.evaluate(
            self.x_train,
            self.y_train,
            verbose=0
        )
        
        return loss, len(self.x_train), {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc": auc
        }

def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Flower client for cyberattack detection')
        parser.add_argument('--client_id', type=int, required=True, help='Client ID')
        args = parser.parse_args()
        
        # Initialize client
        logging.info(f"Initializing client {args.client_id}")
        client = CyberattackClient(client_id=args.client_id)
        logging.info(f"Client {args.client_id} initialized with {len(client.x_train)} samples")
        
        # Start Flower client
        logging.info(f"Starting client {args.client_id}...")
        fl.client.start_numpy_client(
            server_address="0.0.0.0:8081",
            client=client,
            grpc_max_message_length=1024*1024*1024  # 1GB max message size
        )
    except Exception as e:
        logging.error(f"Client error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
