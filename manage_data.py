#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from utils.preprocess import preprocess_can_data, split_data_for_clients, save_client_data
import logging

logging.basicConfig(level=logging.INFO)

def clear_data():
    """Clear existing client data files"""
    data_dir = Path('data')
    for file in data_dir.glob('client_*'):
        file.unlink()
    logging.info("Cleared existing client data files")

def process_new_data(input_files, clear_existing=False):
    """Process new CAN bus data files and prepare them for analysis
    
    Args:
        input_files (list): List of paths to raw CAN bus data files
        clear_existing (bool): Whether to clear existing client data before processing
    """
    if clear_existing:
        clear_data()
    
    features_list = []
    labels_list = []
    
    for i, file_path in enumerate(input_files):
        logging.info(f"Processing file {file_path}...")
        features, labels = preprocess_can_data(file_path)
        features_list.append(features)
        labels_list.append(labels)
    
    # Combine all features and labels
    all_features = np.concatenate(features_list)
    all_labels = np.concatenate(labels_list)
    
    # Split data for clients
    client_data = split_data_for_clients(all_features, all_labels)
    save_client_data(client_data, base_path='data')
    logging.info("Data processing complete!")

def main():
    parser = argparse.ArgumentParser(description='Manage CAN bus datasets')
    parser.add_argument('--input', '-i', nargs='+', help='Input raw CAN bus data files')
    parser.add_argument('--clear', '-c', action='store_true', help='Clear existing data before processing')
    args = parser.parse_args()
    
    if args.input:
        process_new_data(args.input, args.clear)
    elif args.clear:
        clear_data()

if __name__ == '__main__':
    main()
