#!/bin/bash
# Body Measurement Predictor - Virtual Environment Activation
echo "Activating Body Measurement Predictor virtual environment..."
source venv/bin/activate
echo "Virtual environment activated!"
echo "Available commands:"
echo "  python train_model.py     - Train the AI model"
echo "  python webcam_predictor.py - Start prediction session"
echo "  deactivate                - Exit virtual environment"
