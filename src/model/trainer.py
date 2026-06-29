"""Training script for deforestation detection model."""
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

from treexplain.config import MODELS_DIR, OUTPUT_DIR
from treexplain.model.deforestation import DeforestationModel
from treexplain.data.pipeline import DataPipeline, DeforestationDataset


class ModelTrainer:
    """Train deforestation detection model."""
    
    def __init__(
        self,
        model: Optional[DeforestationModel] = None,
        device: str = None,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        num_epochs: int = 50,
        patience: int = 5
    ):
        """Initialize trainer.
        
        Args:
            model: Deforestation model to train
            device: Device to use (cuda, cpu, etc.)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            num_epochs: Maximum number of epochs
            patience: Patience for early stopping
        """
        self.model = model or DeforestationModel()
        self.device = device or self._get_device()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.patience = patience
        
        # Move model to device
        self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=patience // 2,
            verbose=True
        )
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_model_path = None
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def _get_device(self) -> str:
        """Get available device."""
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def _calculate_accuracy(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """Calculate accuracy."""
        predictions = (outputs > 0.5).float()
        correct = (predictions == labels.float()).float()
        return correct.mean().item()
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            dataloader: Training DataLoader
            
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        
        running_loss = 0.0
        running_acc = 0.0
        num_batches = 0
        
        for images, labels in dataloader:
            # Move to device
            images = images.to(self.device)
            labels = labels.float().unsqueeze(1).to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            acc = self._calculate_accuracy(outputs, labels)
            
            running_loss += loss.item()
            running_acc += acc
            num_batches += 1
        
        epoch_loss = running_loss / num_batches
        epoch_acc = running_acc / num_batches
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model.
        
        Args:
            dataloader: Validation DataLoader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        running_loss = 0.0
        running_acc = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                # Move to device
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Calculate metrics
                acc = self._calculate_accuracy(outputs, labels)
                
                running_loss += loss.item()
                running_acc += acc
                num_batches += 1
        
        epoch_loss = running_loss / num_batches
        epoch_acc = running_acc / num_batches
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc
        }
    
    def train(
        self, 
        train_dataloader: DataLoader, 
        val_dataloader: DataLoader,
        test_dataloader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """Train model with early stopping.
        
        Args:
            train_dataloader: Training DataLoader
            val_dataloader: Validation DataLoader
            test_dataloader: Optional test DataLoader
            
        Returns:
            Dictionary with training results
        """
        print(f"Starting training on {self.device}")
        print(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        start_time = time.time()
        
        # Early stopping counter
        epochs_without_improvement = 0
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_dataloader)
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
            
            # Validate
            val_metrics = self.validate(val_dataloader)
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Store history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            # Check for improvement
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                epochs_without_improvement = 0
                
                # Save best model
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = f"deforestation_model_{timestamp}.pt"
                self.best_model_path = MODELS_DIR / model_name
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"  New best model saved to {self.best_model_path}")
            else:
                epochs_without_improvement += 1
                print(f"  No improvement for {epochs_without_improvement} epochs")
            
            # Early stopping
            if epochs_without_improvement >= self.patience:
                print(f"  Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Test on test set if provided
        test_results = None
        if test_dataloader:
            test_metrics = self.validate(test_dataloader)
            test_results = {
                'loss': test_metrics['loss'],
                'accuracy': test_metrics['accuracy']
            }
            print(f"\nTest Loss: {test_results['loss']:.4f}, Accuracy: {test_results['accuracy']:.4f}")
        
        # Save final model
        final_model_path = MODELS_DIR / "deforestation_model_final.pt"
        torch.save(self.model.state_dict(), final_model_path)
        
        # Save training history
        history_path = OUTPUT_DIR / "training_history.npz"
        np.savez(
            history_path,
            train_loss=np.array(self.history['train_loss']),
            val_loss=np.array(self.history['val_loss']),
            train_acc=np.array(self.history['train_acc']),
            val_acc=np.array(self.history['val_acc'])
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        return {
            'best_model_path': self.best_model_path,
            'final_model_path': final_model_path,
            'history_path': history_path,
            'best_val_loss': self.best_val_loss,
            'training_time': training_time,
            'test_results': test_results,
            'history': self.history
        }
    
    def plot_training_history(self, save_path: Optional[Path] = None) -> None:
        """Plot training history.
        
        Args:
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.history['train_acc'], label='Train Accuracy')
        ax2.plot(self.history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training history plot saved to {save_path}")
        else:
            plt.show()
    
    def load_model(self, model_path: Path) -> None:
        """Load model weights.
        
        Args:
            model_path: Path to model weights
        """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {model_path}")


def train_with_sample_data():
    """Train model with sample data for demonstration."""
    from treexplain.data.pipeline import DataPipeline
    
    print("Creating sample data...")
    pipeline = DataPipeline(batch_size=4)
    images, labels = pipeline.get_sample_data(num_samples=20)
    
    print(f"Created {len(images)} samples")
    pipeline.create_datasets(images, labels)
    dataloaders = pipeline.get_dataloaders()
    
    print("Initializing trainer...")
    trainer = ModelTrainer(
        num_epochs=10,
        patience=3,
        learning_rate=0.001
    )
    
    print("Starting training...")
    results = trainer.train(
        dataloaders['train'],
        dataloaders['val'],
        dataloaders['test']
    )
    
    print("\nTraining Results:")
    print(f"  Best validation loss: {results['best_val_loss']:.4f}")
    print(f"  Test accuracy: {results['test_results']['accuracy']:.4f}")
    print(f"  Model saved to: {results['best_model_path']}")
    
    # Plot training history
    trainer.plot_training_history(save_path=OUTPUT_DIR / "training_history.png")
    
    return trainer, results


if __name__ == "__main__":
    train_with_sample_data()
