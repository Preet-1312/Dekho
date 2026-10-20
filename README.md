Grounding DINO Visual Navigation System
Complete visual navigation system using Grounding DINO for object detection, Depth-Anything-V2 for depth estimation, with object tracking and path planning capabilities.

🌟 Features
Open Vocabulary Object Detection - Detects ANY object based on text prompts (not limited to fixed classes)
Depth Estimation - Real-time depth mapping using Depth-Anything-V2
Object Tracking - Multi-object tracking across frames with velocity estimation
Path Planning - Real-time navigation guidance with risk assessment and directional arrows
Video Processing - Process video files and save annotated output
Live Camera Support - Works with webcam, DroidCam, IP cameras
📋 Requirements
System Requirements
Python 3.8+
CUDA-capable GPU (recommended for real-time performance)
8GB+ RAM
Ubuntu/Linux (tested), Windows, or macOS
Python Dependencies
pip install torch torchvision transformers opencv-python numpy scipy pillow
Required Models
Models will be downloaded automatically on first run:

IDEA-Research/grounding-dino-tiny (~400MB)
depth-anything/Depth-Anything-V2-Small-hf (~100MB)
🎥 DroidCam Setup
DroidCam allows you to use your smartphone as a wireless camera.

Option 1: DroidCam App (Easiest)
On Android/iOS:
Download DroidCam from Play Store or App Store
Install and open the app
Note the IP address and port shown in the app (e.g., 192.168.1.100:4747)
Ensure phone and computer are on the same WiFi network
On Your Computer:
# Test the connection first
firefox http://192.168.1.100:4747/video
# Replace IP with your phone's IP from the app

# If you see the camera feed, you're ready!
🚀 Usage
1. Test on Image
Process a single image with detection, depth, and path planning:

python3 final.py image.jpg
Output:

Terminal: List of detected objects with depth and confidence
File: output_gdino.jpg with visualizations
2. Process Video File
Process an entire video with frame-by-frame analysis:

python3 final.py --video input_video.mp4
Output:

results/output_input_video.mp4 - Annotated video
results/frames/ - Individual processed frames
Features:

Progress bar with ETA
Object tracking across frames
Navigation guidance overlaid
Frame-by-frame saving
3. Live Camera (Webcam)
Use built-in webcam:

python3 final.py --live
4. Live Camera (DroidCam via WiFi)
Using DroidCam over WiFi:

# Method 1: Specify IP directly
python3 final.py --live --ip 192.168.1.100 --port 4747

# Method 2: Use full URL
python3 final.py --live --url http://192.168.1.100:4747/video

# For IP Webcam app (default port 8080)
python3 final.py --live --ip 192.168.1.100 --port 8080
5. Custom Text Prompts
Test detection with specific object categories:

python3 final.py --custom image.jpg
This will test multiple prompts:

"person. human. people."
"furniture. chair. table. bed. couch."
"luggage. suitcase. bag. backpack. box."
"all objects. everything. anything visible."
⌨️ Interactive Controls (Live Mode)
When running in live mode, use these keyboard shortcuts:

Key	Action
q	Quit
t	Toggle depth estimation on/off
p	Toggle path planning on/off
🎛️ Configuration Options
Performance Tuning
Edit these parameters in the code for your hardware:

detector = GroundingDinoDetector(
    model_size='tiny',      # 'tiny' (faster) or 'base' (more accurate)
    depth_size='small',     # 'small', 'base', or 'large'
    use_fp16=True,         # Use mixed precision (2x faster on GPU)
    max_resolution=512     # Lower = faster (384, 448, 512, 640)
)
For slower systems:

max_resolution=384 - Faster processing
model_size='tiny' - Lightweight model
Process every 4th frame instead of 3rd
For better accuracy:

max_resolution=640 or higher
model_size='base' - More accurate
depth_size='base' - Better depth
Detection Threshold
Adjust sensitivity:

# Lower threshold = more detections (may include false positives)
detections = detector.detect(frame, score_threshold=0.20)

# Higher threshold = fewer, more confident detections
detections = detector.detect(frame, score_threshold=0.35)
📊 Output Interpretation
Visual Indicators
Bounding Boxes:

🔴 Red - Object very close (depth > 0.7) - CRITICAL
🟠 Orange - Object close (depth 0.5-0.7) - HIGH RISK
🟡 Yellow - Object medium distance (depth 0.3-0.5) - MEDIUM RISK
🟢 Green - Object far (depth < 0.3) - LOW RISK
Navigation Arrow:

Points in recommended direction to avoid obstacles
Color matches highest risk level
Angle shown in degrees (0° = straight)
Instructions:

"STOP! person ahead!" - Critical danger
"Move left, chair approaching" - High risk
"Bear right, obstacle ahead" - Medium risk
"Path clear" - Safe to proceed
Terminal Output
Found 5 objects:

  1. suitcase            | conf=0.852 | d=0.784  ← Very close!
  2. person              | conf=0.921 | d=0.645  ← Close
  3. chair               | conf=0.743 | d=0.412  ← Medium
  4. table               | conf=0.681 | d=0.298  ← Far
  5. backpack            | conf=0.567 | d=0.156  ← Very far

Path Planning:
  Risk Level: HIGH
  Direction: -25.0° (left)
  Instruction: Move left, suitcase approaching
🔧 Troubleshooting
DroidCam Connection Issues
Problem: Camera feed not accessible

Solutions:

Verify phone and PC on same WiFi network
Check firewall isn't blocking the connection
Try accessing http://PHONE_IP:PORT/video in browser first
Restart DroidCam app on phone
Use USB connection as fallback
Low FPS / Slow Performance
Problem: Getting < 5 FPS

Solutions:

Lower resolution: max_resolution=384
Disable depth: Use with_depth=False
Increase frame skip: Process every 5th frame
Use CPU if GPU memory limited: Comment out use_fp16=True
Close other GPU-intensive applications
Model Download Fails
Problem: "Connection timeout" or "Failed to download"

Solutions:

Check internet connection
Try with VPN if huggingface.co is blocked
Manually download models:
# Download from https://huggingface.co/IDEA-Research/grounding-dino-tiny
# Place in ~/.cache/huggingface/hub/
CUDA Out of Memory
Problem: "RuntimeError: CUDA out of memory"

Solutions:

Lower resolution: max_resolution=320
Use smaller models: depth_size='small'
Reduce batch processing
Clear GPU cache more frequently
Restart Python script
No Objects Detected
Problem: Not detecting objects that are clearly visible

Solutions:

Lower threshold: score_threshold=0.20
Add specific keywords to prompt (see Custom Prompts)
Improve lighting conditions
Check if object is in the default prompt list
Try model_size='base' for better accuracy
🎯 Use Cases
Visual Navigation for Visually Impaired - Real-time obstacle detection and guidance
Robotics - Autonomous navigation in indoor environments
Safety Systems - Detect hazards in work environments
Video Analysis - Process surveillance or drone footage
Research - Study object detection and depth estimation
🔬 Technical Details
Detection Pipeline
Grounding DINO - Detects objects based on text prompts
Depth-Anything-V2 - Estimates depth map from single image
Tracker - Associates detections across frames using IoU matching
Path Planner - Calculates safest direction based on obstacle positions and risks
Optimizations Applied
Mixed precision (FP16) for 2x GPU speedup
Dynamic resolution scaling
Frame skipping for real-time performance
Explicit tensor cleanup to prevent memory leaks
Cached model weights
📝 Example Commands
# Quick test on image
python3 final.py test_image.jpg

# Process video with all features
python3 final.py --video walking_test.mp4

# Live navigation with DroidCam
python3 final.py --live --ip 192.168.1.100 --port 4747

# Test custom object categories
python3 final.py --custom my_image.jpg

# Use built-in webcam
python3 final.py --live
🆘 Support
For issues or questions:

Check troubleshooting section above
Verify all dependencies are installed
Test with provided sample image first
Check GPU availability: nvidia-smi
