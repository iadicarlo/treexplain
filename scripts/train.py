#!/usr/bin/env python3
"""Command-line training script for TreeXplain."""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from treexplain.model.trainer import ModelTrainer, train_with_sample_data
from treexplain.data.pipeline import DataPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Train TreeXplain deforestation detection model"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--patience", 
        type=int, 
        default=5,
        help="Patience for early stopping"
    )
    parser.add_argument(
        "--sample-data", 
        action="store_true",
        help="Use sample data for quick testing"
    )
    parser.add_argument(
        "--num-samples", 
        type=int, 
        default=20,
        help="Number of sample images to generate"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Device to use (cuda, cpu, mps)"
    )
    
    args = parser.parse_args()
    
    print("TreeXplain Model Training")
    print("=" * 50)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Patience: {args.patience}")
    print(f"Device: {args.device or 'auto'}")
    print()
    
    if args.sample_data:
        print("Using sample data for training...")
        trainer, results = train_with_sample_data()
    else:
        print("Setting up data pipeline...")
        pipeline = DataPipeline(
            batch_size=args.batch_size
        )
        
        # For now, use sample data as we don't have real STAC access
        print("Generating sample data...")
        images, labels = pipeline.get_sample_data(num_samples=args.num_samples)
        
        print(f"Created {len(images)} samples")
        pipeline.create_datasets(images, labels)
        dataloaders = pipeline.get_dataloaders()
        
        print("Initializing trainer...")
        trainer = ModelTrainer(
            num_epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            device=args.device
        )
        
        print("Starting training...")
        results = trainer.train(
            dataloaders['train'],
            dataloaders['val'],
            dataloaders['test']
        )
        
        # Plot training history
        trainer.plot_training_history()
    
    print("\nTraining completed successfully!")
    print(f"Best model saved to: {results['best_model_path']}")
    print(f"Final model saved to: {results['final_model_path']}")
    if results['test_results']:
        print(f"Test accuracy: {results['test_results']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
