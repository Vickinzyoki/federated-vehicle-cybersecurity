import flwr as fl
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
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

def load_test_data():
    """Load test data for evaluation by combining a portion of each client's data"""
    features_list = []
    labels_list = []
    
    # Load 20% of each client's data for testing
    for client_id in range(3):
        features = pd.read_csv(f'data/client_{client_id}_features.csv')
        labels = pd.read_csv(f'data/client_{client_id}_labels.csv')
        
        # Take 20% of each client's data
        n_samples = len(features)
        n_test = int(n_samples * 0.2)
        
        features_list.append(features.values[-n_test:])
        labels_list.append(labels.values.ravel()[-n_test:])
    
    x_test = np.concatenate(features_list)
    y_test = np.concatenate(labels_list)
    
    return x_test, y_test

def get_evaluate_fn():
    """Return an evaluation function for server-side evaluation."""
    
    # Load test data
    x_test, y_test = load_test_data()
    
    # Create model for server-side evaluation
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
    
    # The `evaluate` function will be called after every round
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, float]]]:
        model.set_weights(parameters)
        loss, accuracy, precision, recall, auc = model.evaluate(x_test, y_test, verbose=0)
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        
        # Save metrics for this round
        metrics = {
            "round": server_round,
            "loss": float(loss),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc": float(auc)
        }
        
        # Save metrics to a file
        metrics_file = Path("models") / f"metrics_round_{server_round}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f)
        
        return loss, metrics
    
    return evaluate

def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "batch_size": 64,
        "local_epochs": 3,
        "round": server_round,
    }
    return config

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

def main():
    try:
        # Create models directory
        Path("models").mkdir(exist_ok=True)
        logging.info("Created models directory")
        
        # Clear existing metrics
        for f in Path("models").glob("metrics_round_*.json"):
            f.unlink()
        logging.info("Cleared existing metrics")
        
        # Load test data to verify it's available
        x_test, y_test = load_test_data()
        logging.info(f"Loaded test data: {x_test.shape} samples")
        
        # Create and verify model
        model = create_model()
        logging.info("Created model successfully")
        
        # Define strategy with more conservative settings
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=1,  # More lenient - start with 1 client
            min_evaluate_clients=1,
            min_available_clients=1,
            evaluate_fn=get_evaluate_fn(),
            on_fit_config_fn=fit_config,
            initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
        )
        logging.info("Defined federated strategy")
        
        # Start server with more detailed logging
        logging.info("Starting FL server...")
        fl.server.start_server(
            server_address="0.0.0.0:8081",
            config=fl.server.ServerConfig(
                num_rounds=10,
                round_timeout=600.0
            ),
            strategy=strategy,
            grpc_max_message_length=1024*1024*1024
        )
    except Exception as e:
        logging.error(f"Server error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("\nServer shutdown complete")
