import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from model_architecture import create_body_measurement_model, compile_model, get_measurement_labels
from aws_data_loader import AWSBodyMDataLoader
from real_data_generator import create_real_data_generators, test_data_generator
import json
import time

class RealModelTrainer:
    """Train the body measurement prediction model with real Amazon BodyM data"""
    
    def __init__(self, model_save_path="bodym_real_model.h5", data_dir="bodym_dataset"):
        self.model_save_path = model_save_path
        self.data_dir = data_dir
        self.model = None
        self.history = None
        self.data_loader = AWSBodyMDataLoader(data_dir)
        
    def prepare_data(self, max_images=1000, force_download=False):
        """Prepare training data from AWS S3"""
        print("=== Preparing Real Dataset ===")
        
        training_data = self.data_loader.prepare_dataset(
            max_images=max_images, 
            force_download=force_download
        )
        
        if not training_data:
            print("Failed to prepare training data!")
            return None
        
        print(f"✓ Prepared {len(training_data)} training samples")
        
        # Verify we have enough data
        if len(training_data) < 50:
            print(f"⚠️  Warning: Only {len(training_data)} samples available. Consider:")
            print("   - Increasing max_images parameter")
            print("   - Checking internet connection")
            print("   - Using force_download=True")
        
        # Test the data generator with a small subset
        test_data_generator(training_data[:min(10, len(training_data))])
        
        return training_data
    
    def create_model(self):
        """Create and compile the model"""
        print("Creating model architecture...")
        self.model = create_body_measurement_model()
        self.model = compile_model(self.model)
        
        print("Model created successfully!")
        print(f"Total parameters: {self.model.count_params():,}")
        return self.model
    
    def train(self, training_data, epochs=100, batch_size=16, validation_split=0.2):
        """Train the model with real data"""
        
        if self.model is None:
            self.create_model()
        
        if not training_data:
            print("No training data available!")
            return None
        
        print(f"Training with {len(training_data)} samples...")
        
        # Adjust batch size if we have limited data
        if len(training_data) < 100:
            batch_size = min(batch_size, max(2, len(training_data) // 10))
            print(f"Adjusted batch size to {batch_size} due to limited data")
        
        # Create data generators
        try:
            train_generator, val_generator = create_real_data_generators(
                training_data, 
                validation_split=validation_split, 
                batch_size=batch_size
            )
            
            print(f"Training batches: {len(train_generator)}")
            print(f"Validation batches: {len(val_generator)}")
            
            # Verify generators work
            try:
                test_batch = train_generator[0]
                print(f"Batch test successful - Images: {test_batch[0].shape}, Measurements: {test_batch[1].shape}")
            except Exception as e:
                print(f"Warning: Batch test failed: {e}")
                
        except Exception as e:
            print(f"Error creating data generators: {e}")
            return None
        
        # Adjust training parameters for small datasets
        if len(training_data) < 200:
            epochs = min(epochs, 50)  # Reduce epochs for small datasets
            print(f"Adjusted epochs to {epochs} due to limited data")
        
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
                patience=min(15, epochs // 3),  # Adjust patience based on epochs
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=min(8, epochs // 6),  # Adjust patience based on epochs
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print("Starting training...")
        
        # Train the model
        try:
            start_time = time.time()
            
            self.history = self.model.fit(
                train_generator,
                epochs=epochs,
                validation_data=val_generator,
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f} seconds!")
            print(f"Model saved to {self.model_save_path}")
            
            # Save training history
            self._save_training_history()
            
            # Plot training history
            self._plot_training_history()
            
            return self.history
            
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_training_history(self):
        """Save training history to JSON file"""
        if self.history is None:
            return
        
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        history_file = 'real_training_history.json'
        with open(history_file, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"Training history saved to '{history_file}'")
    
    def _plot_training_history(self):
        """Plot and save training history"""
        if self.history is None:
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss
            axes[0, 0].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
            axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
            axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Mean Absolute Error
            axes[0, 1].plot(self.history.history['mean_absolute_error'], label='Training MAE', linewidth=2)
            axes[0, 1].plot(self.history.history['val_mean_absolute_error'], label='Validation MAE', linewidth=2)
            axes[0, 1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE (cm)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Mean Squared Error
            axes[1, 0].plot(self.history.history['mean_squared_error'], label='Training MSE', linewidth=2)
            axes[1, 0].plot(self.history.history['val_mean_squared_error'], label='Validation MSE', linewidth=2)
            axes[1, 0].set_title('Mean Squared Error', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('MSE')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Learning Rate (if available)
            if 'lr' in self.history.history:
                axes[1, 1].plot(self.history.history['lr'], linewidth=2, color='orange')
                axes[1, 1].set_title('Learning Rate', fontsize=14, fontweight='bold')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].set_yscale('log')
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes,
                               fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('real_training_history.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("Training history plot saved as 'real_training_history.png'")
            
        except Exception as e:
            print(f"Error plotting training history: {e}")
    
    def evaluate_model(self, training_data, num_test_samples=100):
        """Evaluate the trained model"""
        if self.model is None:
            print("No model to evaluate. Please train the model first.")
            return
        
        # Adjust test samples based on available data
        num_test_samples = min(num_test_samples, len(training_data) // 4, 50)
        print(f"Evaluating model on {num_test_samples} test samples...")
        
        # Use last samples as test data
        test_data = training_data[-num_test_samples:] if len(training_data) >= num_test_samples else training_data
        
        try:
            # Create test generator
            from real_data_generator import RealDataGenerator
            test_generator = RealDataGenerator(
                test_data, 
                batch_size=min(8, len(test_data)), 
                shuffle=False, 
                augment=False
            )
            
            # Evaluate
            results = self.model.evaluate(test_generator, verbose=1)
            
            print("\nEvaluation Results:")
            for i, metric_name in enumerate(self.model.metrics_names):
                print(f"{metric_name}: {results[i]:.4f}")
            
            # Generate predictions for analysis
            test_images, test_measurements = test_generator[0]
            predictions = self.model.predict(test_images, verbose=0)
            
            # Calculate per-measurement accuracy
            measurement_labels = get_measurement_labels()
            print("\nPer-measurement Mean Absolute Error:")
            for i, label in enumerate(measurement_labels):
                mae = np.mean(np.abs(predictions[:, i] - test_measurements[:, i]))
                print(f"{label}: {mae:.2f} cm")
            
            return results
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main training function"""
    print("=== Real Amazon BodyM Dataset Training ===")
    
    # Configuration
    MAX_IMAGES = 1000  # Adjust based on your needs and resources
    EPOCHS = 50  # Reduced default for faster testing
    BATCH_SIZE = 20
    
    # Check if model already exists
    model_path = "bodym_real_model.h5"
    if os.path.exists(model_path):
        response = input(f"Model '{model_path}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return
    
    # Create trainer
    trainer = RealModelTrainer(model_save_path=model_path)
    
    # Test AWS connection first
    print("Testing AWS S3 connection...")
    try:
        objects = trainer.data_loader.list_s3_contents(max_keys=100)
        if not objects:
            print("Cannot access AWS S3 bucket. Please check your internet connection.")
            return
        print(f"✓ Successfully connected to S3. Found {len(objects)} objects.")
    except Exception as e:
        print(f"✗ AWS S3 connection failed: {e}")
        return
    
    # Get user preferences
    try:
        max_images = int(input(f"Enter max images to download (default {MAX_IMAGES}): ") or str(MAX_IMAGES))
        epochs = int(input(f"Enter number of epochs (default {EPOCHS}): ") or str(EPOCHS))
        batch_size = int(input(f"Enter batch size (default {BATCH_SIZE}): ") or str(BATCH_SIZE))
        
        force_download = input("Force re-download data? (y/n): ").lower() == 'y'
        
    except ValueError:
        print("Invalid input. Using default values.")
        max_images = MAX_IMAGES
        epochs = EPOCHS
        batch_size = BATCH_SIZE
        force_download = False
    
    # Prepare data
    print(f"\nPreparing dataset with max {max_images} images...")
    training_data = trainer.prepare_data(max_images=max_images, force_download=force_download)
    
    if not training_data:
        print("Failed to prepare training data. Exiting.")
        return
    
    if len(training_data) < 50:
        print(f"⚠️  Warning: Only {len(training_data)} samples available.")
        proceed = input("Continue with limited data? (y/n): ").lower()
        if proceed != 'y':
            print("Training cancelled. Try increasing max_images or force re-download.")
            return
    
    # Train the model
    try:
        print(f"\nStarting training with {len(training_data)} samples...")
        history = trainer.train(
            training_data, 
            epochs=epochs, 
            batch_size=batch_size
        )
        
        if history is not None:
            # Evaluate the model
            trainer.evaluate_model(training_data)
            
            print(f"\n✓ Training completed successfully!")
            print(f"✓ Model saved as: {model_path}")
            print("✓ You can now use 'update_webcam_predictor.py' to make predictions!")
            print("✓ The new model will be automatically detected and used")
        else:
            print("✗ Training failed!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()