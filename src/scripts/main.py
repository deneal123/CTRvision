#!/usr/bin/env python3
"""
Main entry point for CTRvision project.
This script orchestrates the full pipeline: download -> train -> evaluate -> plot.
"""

import os
import sys
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

def get_simple_logger():
    """Simple logger that doesn't require external dependencies."""
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    return logging.getLogger(__name__)

# Try to import advanced logger, fallback to simple one
try:
    from utils.custom_logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = get_simple_logger()

def run_download():
    """Download dataset."""
    logger.info("Starting dataset download...")
    try:
        from scripts.download import main as download_main
        download_main()
        logger.info("Dataset download completed.")
    except ImportError as e:
        logger.error(f"Cannot import download module: {e}")
        logger.info("Please ensure all dependencies are installed.")

def run_training():
    """Run model training."""
    logger.info("Starting model training...")
    try:
        from scripts.train import train
        train()
        logger.info("Model training completed.")
    except ImportError as e:
        logger.error(f"Cannot import training module: {e}")
        logger.info("Please ensure all dependencies are installed.")

def run_plotting():
    """Generate plots and evaluation reports."""
    logger.info("Starting result plotting...")
    try:
        from scripts.plot import plot_results
        plot_results()
        logger.info("Result plotting completed.")
    except ImportError as e:
        logger.error(f"Cannot import plotting module: {e}")
        logger.info("Please ensure all dependencies are installed.")

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
    try:
        from utils.config_parser import ConfigParser
        from src import path_to_config
        config_path = args.config if args.config else path_to_config()
        config = ConfigParser().parse(config_path)
        logger.info(f"Using config: {config_path}")
    except ImportError:
        logger.warning("Config parser not available. Please ensure dependencies are installed.")
        return
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return
    
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