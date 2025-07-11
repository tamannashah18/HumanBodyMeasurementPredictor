#!/bin/bash
# 3D Body Measurement App - Virtual Environment Activation
echo "Activating 3D Body Measurement App environment..."
source venv_3d/bin/activate
echo "3D app environment activated!"
echo ""
echo "Available commands:"
echo "  python main_app.py              - Start the main application"
echo "  python multi_view_capture.py    - Capture multi-view images"
echo "  python pose_estimation_3d.py    - Process pose estimation"
echo "  python body_measurements_3d.py  - Calculate measurements"
echo "  python mesh_generation.py       - Generate 3D mesh"
echo "  python report_generator.py      - Generate reports"
echo "  deactivate                      - Exit environment"
echo ""
echo "🎯 Start with: python main_app.py"
