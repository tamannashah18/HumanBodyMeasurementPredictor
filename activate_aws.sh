#!/bin/bash
# AWS BodyM Training Environment
echo "Activating AWS BodyM training environment..."
source venv_aws/bin/activate
echo "AWS training environment activated!"
echo ""
echo "Available commands:"
echo "  python aws_data_loader.py     - Test AWS data loading"
echo "  python train_real_model.py    - Train with real BodyM data"
echo "  python update_webcam_predictor.py - Test predictions with both models"
echo "  deactivate                    - Exit environment"
echo ""
echo "Dataset info:"
echo "  Bucket: s3://amazon-bodym"
echo "  Access: Public (no credentials needed)"
