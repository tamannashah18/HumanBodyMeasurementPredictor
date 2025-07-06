import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from model_architecture import create_body_measurement_model, compile_model, get_measurement_labels
from data_generator import create_training_data
import json

class ModelTrainer:
    """Train the body measurement prediction model"""
    
    def __init__(self, model_save_path="final_model.h5"):
        self.model_save_path = model_save_path
        self.model = None
        self.history = None
        
    def create_model(self):
        """Create and compile the model"""
        print("Creating model architecture...")
        self.model = create_body_measurement_model()
        self.model = compile_model(self.model)
        
        print("Model created successfully!")
        print(f"Total parameters: {self.model.count_params():,}")
        return self.model
    
    def train(self, epochs=50, num_samples=2000, validation_split=0.2):
        """Train the model with synthetic data"""
        
        if self.model is None:
            self.create_model()
        
        print(f"Generating {num_samples} synthetic training samples...")
        train_generator, val_generator = create_training_data(
            num_samples=num_samples, 
            validation_split=validation_split
        )
        
        # Define callbacks
        callbacks = [
            ModelCheckpoint(
                self.model_save_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print("Starting training...")
        print(f"Training samples: {len(train_generator) * train_generator.batch_size}")
        print(f"Validation samples: {len(val_generator) * val_generator.batch_size}")
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"Training completed! Model saved to {self.model_save_path}")
        
        # Save training history
        self._save_training_history()
        
        # Plot training history
        self._plot_training_history()
        
        return self.history
    
    def _save_training_history(self):
        """Save training history to JSON file"""
        if self.history is None:
            return
        
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        with open('training_history.json', 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print("Training history saved to 'training_history.json'")
    
    def _plot_training_history(self):
        """Plot and save training history"""
        if self.history is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Mean Absolute Error
        axes[0, 1].plot(self.history.history['mean_absolute_error'], label='Training MAE')
        axes[0, 1].plot(self.history.history['val_mean_absolute_error'], label='Validation MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Mean Squared Error
        axes[1, 0].plot(self.history.history['mean_squared_error'], label='Training MSE')
        axes[1, 0].plot(self.history.history['val_mean_squared_error'], label='Validation MSE')
        axes[1, 0].set_title('Mean Squared Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate (if available)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Training history plot saved as 'training_history.png'")
    
    def evaluate_model(self, num_test_samples=200):
        """Evaluate the trained model"""
        if self.model is None:
            print("No model to evaluate. Please train the model first.")
            return
        
        print(f"Evaluating model on {num_test_samples} test samples...")
        
        # Generate test data
        from data_generator import SyntheticDataGenerator
        test_generator = SyntheticDataGenerator(
            batch_size=32, 
            num_samples=num_test_samples
        )
        
        # Evaluate
        results = self.model.evaluate(test_generator, verbose=1)
        
        print("\nEvaluation Results:")
        for i, metric_name in enumerate(self.model.metrics_names):
            print(f"{metric_name}: {results[i]:.4f}")
        
        # Generate predictions for analysis
        test_images, test_measurements = test_generator[0]
        predictions = self.model.predict(test_images)
        
        # Calculate per-measurement accuracy
        measurement_labels = get_measurement_labels()
        print("\nPer-measurement Mean Absolute Error:")
        for i, label in enumerate(measurement_labels):
            mae = np.mean(np.abs(predictions[:, i] - test_measurements[:, i]))
            print(f"{label}: {mae:.2f} cm")
        
        return results

def main():
    """Main training function"""
    print("=== Body Measurement Model Training ===")
    
    # Check if model already exists
    model_path = "final_model.h5"
    if os.path.exists(model_path):
        response = input(f"Model '{model_path}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return
    
    # Create trainer
    trainer = ModelTrainer(model_save_path=model_path)
    
    # Get training parameters
    try:
        epochs = int(input("Enter number of epochs (default 50): ") or "50")
        num_samples = int(input("Enter number of training samples (default 2000): ") or "2000")
    except ValueError:
        print("Invalid input. Using default values.")
        epochs = 50
        num_samples = 2000
    
    # Train the model
    try:
        history = trainer.train(epochs=epochs, num_samples=num_samples)
        
        # Evaluate the model
        trainer.evaluate_model()
        
        print(f"\nTraining completed successfully!")
        print(f"Model saved as: {model_path}")
        print("You can now use 'webcam_predictor.py' to make predictions!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main()