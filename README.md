# Grounding DINO Visual Navigation System

A computer vision-based visual navigation system that combines **Grounding DINO** for open-vocabulary object detection and **Depth Anything V2** for monocular depth estimation. The system detects obstacles, tracks objects across video frames, assesses potential risks, and generates directional navigation guidance.

Designed for applications such as assistive navigation for visually impaired individuals, robotics, obstacle avoidance, and intelligent video analysis.

## 🌟 Features

- **Open-Vocabulary Object Detection:** Detect objects using text prompts instead of relying only on fixed object classes.
- **Depth Estimation:** Estimate relative scene depth using Depth Anything V2.
- **Multi-Object Tracking:** Track detected objects across frames using Intersection over Union (IoU) matching and estimate their movement.
- **Path Planning:** Analyze obstacle positions and estimated risks to recommend safer movement directions.
- **Real-Time Navigation Guidance:** Display directional arrows, risk levels, and navigation instructions.
- **Video Processing:** Analyze video files frame by frame and save annotated output.
- **Live Camera Support:** Supports webcams and compatible network camera streams, including DroidCam.
- **Performance Optimization:** Includes configurable resolution, frame skipping, and mixed-precision inference options.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Object Detection | Grounding DINO |
| Depth Estimation | Depth Anything V2 |
| Deep Learning Framework | PyTorch |
| Model Integration | Hugging Face Transformers |
| Computer Vision | OpenCV |
| Numerical Processing | NumPy, SciPy |
| Image Processing | Pillow |

## 🏗️ System Architecture

The system processes input images or video frames through the following pipeline:

1. **Input Acquisition:** Read an image, video file, webcam stream, or compatible IP camera feed.
2. **Object Detection:** Grounding DINO detects objects based on text prompts.
3. **Depth Estimation:** Depth Anything V2 generates a relative depth map of the scene.
4. **Object Tracking:** Associate detections across consecutive frames and estimate object movement.
5. **Risk Assessment:** Evaluate detected obstacles using their image positions and estimated relative depth.
6. **Path Planning:** Recommend a direction based on obstacle positions and assessed risk.
7. **Visualization:** Display bounding boxes, depth information, risk indicators, and navigation instructions.

## 📋 Requirements

### System Requirements

- Python 3.8 or later, subject to compatibility with the installed dependencies.
- 8 GB RAM or more recommended.
- NVIDIA GPU with CUDA support recommended for faster inference.
- Windows, Linux, or macOS.
- Internet connection for downloading model weights on the first run.

### Python Dependencies

Install the required libraries:

```bash
pip install torch torchvision transformers opencv-python numpy scipy pillow
```

For GPU acceleration, install the PyTorch version compatible with your CUDA environment. Refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

### Pretrained Models

The system uses the following Hugging Face models:

- **Object Detection:** [IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
- **Depth Estimation:** [depth-anything/Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)

Model weights are downloaded automatically when supported by the implementation and available network access.

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Preet-1312/Dekho.git
cd Dekho
```

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install torch torchvision transformers opencv-python numpy scipy pillow
```

### 4. Run the Application

Ensure that `final.py` is located in the project directory and use the commands below.

## 🎥 Usage

### 1. Object Detection on an Image

Process a single image:

```bash
python final.py image.jpg
```

Replace `image.jpg` with the path to your input image.

**Expected output:**
- Detected objects and confidence scores in the terminal.
- Estimated relative depth values, where supported by the implementation.
- Annotated image saved as `output_gdino.jpg`, as configured by the script.

### 2. Process a Video File

```bash
python final.py --video input_video.mp4
```

Replace `input_video.mp4` with your video filename.

**Expected output:**
- Frame-by-frame object detection and depth estimation.
- Object tracking and navigation guidance.
- Annotated video saved under the `results/` directory, if configured by the script.
- Processed frames and progress information, depending on the implementation.

### 3. Use the Built-in Webcam

```bash
python final.py --live
```

This starts live navigation using the default camera device, provided that the script supports this option and the camera is accessible.

### 4. Use DroidCam over Wi-Fi

DroidCam allows a smartphone to act as a network camera.

**Setup:**

1. Install and open DroidCam on your smartphone.
2. Connect your smartphone and computer to the same Wi-Fi network.
3. Note the phone's IP address and the port displayed by the app.
4. Test the video stream in your browser.

For example:

```text
http://192.168.1.100:4747/video
```

Replace the example IP address and port with the values displayed by your app.

**Connect using the IP address and port:**

```bash
python final.py --live --ip 192.168.1.100 --port 4747
```

**Alternatively, provide the full video URL:**

```bash
python final.py --live --url http://192.168.1.100:4747/video
```

For a compatible IP Webcam setup using port `8080`:

```bash
python final.py --live --ip 192.168.1.100 --port 8080
```

The correct stream URL and port depend on the camera application and its configuration.

### 5. Test Custom Object Prompts

```bash
python final.py --custom image.jpg
```

This mode can be used to test different text prompts, depending on the prompts configured in the application.

Example prompt categories:

- People: `person`, `human`, `people`
- Furniture: `chair`, `table`, `bed`, `couch`
- Bags: `luggage`, `suitcase`, `backpack`, `bag`
- General objects: descriptive text for objects visible in the scene

Grounding DINO supports text-conditioned detection, but detection quality depends on the prompt, model, image, and confidence threshold.

## ⌨️ Interactive Controls

During live mode, the following keyboard controls are available if implemented in the application:

| Key | Action |
|---|---|
| `q` | Quit the application |
| `t` | Toggle depth estimation |
| `p` | Toggle path planning |

## 🎛️ Configuration and Performance Tuning

The following example illustrates configuration options that may be available in the detector implementation:

```python
detector = GroundingDinoDetector(
    model_size="tiny",
    depth_size="small",
    use_fp16=True,
    max_resolution=512
)
```

Use these settings only if the corresponding parameters are supported by your implementation.

### Performance Optimization

For lower-end systems:

- Use a smaller detection model.
- Reduce processing resolution to `384` or another supported value.
- Process fewer video frames.
- Disable depth estimation when it is not required, if supported.
- Close other GPU-intensive applications.

For better detection and depth detail:

- Increase the processing resolution.
- Use a larger supported model variant.
- Adjust the confidence threshold based on the scene.

Larger models and higher resolutions generally increase computational requirements and may reduce real-time performance.

### Detection Confidence Threshold

A lower confidence threshold can increase the number of detections but may introduce false positives. A higher threshold can reduce false positives but may miss objects.

Example:

```python
detections = detector.detect(
    frame,
    score_threshold=0.20
)
```

The detector API and threshold parameter should match the actual implementation.

## 📊 Output Interpretation

### Bounding Boxes and Risk Indicators

The navigation visualization may use color-coded indicators to communicate estimated obstacle risk.

| Color | Intended Meaning |
|---|---|
| 🔴 Red | Critical risk — obstacle estimated to be very close |
| 🟠 Orange | High risk — nearby obstacle |
| 🟡 Yellow | Medium risk — obstacle requiring attention |
| 🟢 Green | Lower risk — obstacle estimated to be farther away |

These categories depend on the depth normalization and risk thresholds implemented in the application.

**Important:** Depth Anything V2 estimates relative depth from a single image. Its raw output is not automatically a calibrated physical distance in meters. Risk estimates should therefore be treated as approximate unless the system includes appropriate calibration.

### Navigation Guidance

Example navigation instructions may include:

- `STOP! Person ahead!`
- `Move left, obstacle approaching`
- `Bear right, obstacle ahead`
- `Path clear`

The actual instructions depend on the detected objects, estimated depth, risk assessment, and path-planning logic.

### Example Terminal Output

Illustrative output:

```text
Found 5 objects:

1. suitcase  | confidence=0.852 | depth=0.784
2. person    | confidence=0.921 | depth=0.645
3. chair     | confidence=0.743 | depth=0.412
4. table     | confidence=0.681 | depth=0.298
5. backpack  | confidence=0.567 | depth=0.156

Path Planning:
Risk Level: HIGH
Direction: -25.0 degrees (left)
Instruction: Move left, obstacle approaching
```

The values above are examples and do not represent guaranteed output for every scene.

## 🔧 Troubleshooting

### 1. DroidCam Connection Issues

**Problem:** The camera stream cannot be accessed.

**Possible solutions:**

- Confirm that the phone and computer are on the same network.
- Verify the IP address, port, and stream URL.
- Test the stream URL in a browser.
- Check firewall settings.
- Restart the camera application.
- Try a USB connection if supported.

### 2. Low FPS or Slow Performance

**Possible solutions:**

- Reduce the input resolution.
- Use a smaller model variant.
- Process fewer frames.
- Disable optional processing stages when supported.
- Check GPU utilization and available memory.

### 3. Model Download Errors

**Possible solutions:**

- Check your internet connection.
- Verify that the model repository is accessible.
- Check Hugging Face model availability.
- Retry the download and inspect the error message.

### 4. CUDA Out-of-Memory Error

**Possible solutions:**

- Reduce image resolution.
- Use smaller models.
- Close other GPU-intensive applications.
- Use CPU inference if necessary and supported.
- Restart the application to release allocated resources.

### 5. Objects Are Not Detected

**Possible solutions:**

- Try a more descriptive text prompt.
- Adjust the confidence threshold.
- Improve image lighting and visibility.
- Test a different model variant if supported.
- Check whether the model and processor loaded correctly.

### 6. Camera Is Not Detected

**Possible solutions:**

- Verify that the camera is connected and accessible.
- Close other applications using the camera.
- Check the camera index or network stream URL.
- Confirm that the OpenCV installation supports the required video backend.

## 🎯 Use Cases

- **Assistive Navigation:** Prototype obstacle detection and directional guidance for visually impaired users.
- **Robotics:** Support perception and obstacle-awareness components in robotic navigation.
- **Safety Monitoring:** Identify potential obstacles in indoor or industrial environments.
- **Video Analysis:** Analyze recorded footage for object detection and movement.
- **Computer Vision Research:** Experiment with open-vocabulary detection, monocular depth estimation, tracking, and path planning.

This project is a navigation prototype, not a certified safety system. It should not be relied upon as the sole means of navigation or obstacle avoidance in real-world environments.

## 🔬 Technical Details

### Grounding DINO

Grounding DINO performs open-vocabulary object detection by associating image regions with text descriptions. This enables the system to detect a wide range of objects without being restricted to a fixed set of predefined classes.

### Depth Anything V2

Depth Anything V2 estimates depth information from a single image. The resulting depth map helps the application reason about the relative positions of detected obstacles.

### Object Tracking

The tracking component associates detections across consecutive frames using Intersection over Union (IoU) matching. Tracking can help maintain object identities and estimate movement over time.

### Path Planning

The path-planning component combines obstacle positions and estimated risk to recommend a direction. The quality of its guidance depends on the accuracy of detection, depth estimation, tracking, and risk thresholds.

### Performance Optimizations

Depending on the implementation, performance optimizations may include:

- Mixed-precision inference.
- Dynamic input resolution.
- Video frame skipping.
- Model-weight caching.
- Explicit tensor and GPU memory cleanup.

## 📝 Quick Reference

```bash
# Run detection on an image
python final.py image.jpg

# Process a video
python final.py --video input_video.mp4

# Start webcam navigation
python final.py --live

# Start DroidCam navigation
python final.py --live --ip 192.168.1.100 --port 4747

# Start navigation using a full stream URL
python final.py --live --url http://192.168.1.100:4747/video

# Test custom prompts
python final.py --custom image.jpg

# Check NVIDIA GPU availability
nvidia-smi
```

## 🚧 Future Improvements

- Improve depth calibration and obstacle-distance estimation.
- Enhance multi-object tracking in crowded environments.
- Develop more robust path planning for dynamic obstacles.
- Optimize inference for low-power devices.
- Add audio-based navigation alerts.
- Evaluate navigation performance across varied lighting and environments.

## 🤝 Contributing

Contributions, suggestions, and bug reports are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Implement and test your changes.
4. Submit a pull request describing the changes.

## 📄 License

Add a `LICENSE` file to specify the terms under which this project may be used, modified, and distributed.

## 👨‍💻 Author

**Preet Dhote**

GitHub: [@Preet-1312](https://github.com/Preet-1312)

---

If you find this project useful, consider giving the repository a ⭐ on GitHub.
