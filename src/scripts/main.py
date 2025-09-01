#!/usr/bin/env python3
"""
Main entry point for CTRvision project.
This script orchestrates the full pipeline: download -> train -> evaluate -> plot.
"""

import os
import sys
import argparse
from src.utils.config_parser import ConfigParser
from src import path_to_config
from src.utils.custom_logging import get_logger

logger = get_logger(__name__)

def run_download():
    """Download dataset."""
    logger.info("Starting dataset download...")
    from src.scripts.download import main as download_main
    download_main()
    logger.info("Dataset download completed.")

def run_training():
    """Run model training."""
    logger.info("Starting model training...")
    from src.scripts.train import train
    train()
    logger.info("Model training completed.")

def run_plotting():
    """Generate plots and evaluation reports."""
    logger.info("Starting result plotting...")
    from src.scripts.plot import plot_results
    plot_results()
    logger.info("Result plotting completed.")

def run_full_pipeline():
    """Run the complete pipeline."""
    logger.info("Starting full CTRvision pipeline...")
    
    try:
        # Step 1: Download data
        run_download()
        
        # Step 2: Train model
        run_training()
        
        # Step 3: Generate plots and reports
        run_plotting()
        
        logger.info("Full pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="CTRvision ML Pipeline")
    parser.add_argument('--step', choices=['download', 'train', 'plot', 'all'], 
                       default='all', help='Pipeline step to run')
    parser.add_argument('--config', default=None, 
                       help='Path to config file (default: use project config)')
    
    args = parser.parse_args()
    
    # Load and validate config
    config_path = args.config if args.config else path_to_config()
    config = ConfigParser().parse(config_path)
    logger.info(f"Using config: {config_path}")
    
    # Run requested step(s)
    if args.step == 'download':
        run_download()
    elif args.step == 'train':
        run_training()
    elif args.step == 'plot':
        run_plotting()
    elif args.step == 'all':
        run_full_pipeline()

if __name__ == '__main__':
    main()