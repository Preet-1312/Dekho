import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import time
from dataclasses import dataclass, field
from typing import List, Tuple
from collections import deque


@dataclass
class GDinoDetection:
    bbox: Tuple[int, int, int, int]
    class_name: str
    confidence: float
    depth: float = 0.0
    area: float = 0.0
    center: Tuple[float, float] = None
    
    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)
        if self.area == 0:
            self.area = max(1, (x2 - x1) * (y2 - y1))


# ============================================================================
# TRACKING MODULE
# ============================================================================

@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    detections: deque = field(default_factory=lambda: deque(maxlen=30))
    velocities: deque = field(default_factory=lambda: deque(maxlen=10))
    last_seen: float = 0.0
    
    def add_detection(self, det, timestamp):
        self.detections.append((det, timestamp))
        self.last_seen = timestamp
        if len(self.detections) >= 2:
            self._update_velocity()
    
    def _update_velocity(self):
        det1, t1 = self.detections[-2]
        det2, t2 = self.detections[-1]
        dt = t2 - t1
        if dt < 0.01:
            return
        dx = det2.center[0] - det1.center[0]
        dy = det2.center[1] - det1.center[1]
        self.velocities.append(np.array([dx/dt, dy/dt]))
    
    def get_velocity(self):
        if not self.velocities:
            return np.array([0.0, 0.0])
        vels = np.array(self.velocities)
        weights = np.exp(np.linspace(-2, 0, len(vels)))
        weights /= weights.sum()
        return np.average(vels, axis=0, weights=weights)
    
    def get_detection(self):
        return self.detections[-1][0] if self.detections else None


class Tracker:
    def __init__(self, max_age=1.0):
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age
    
    def update(self, detections, timestamp):
        self._remove_stale(timestamp)
        
        if not self.tracks:
            for det in detections:
                self._create_track(det, timestamp)
        else:
            self._match_detections(detections, timestamp)
        
        return self.tracks
    
    def _match_detections(self, detections, timestamp):
        if not detections:
            return
        
        track_ids = list(self.tracks.keys())
        cost = np.ones((len(detections), len(track_ids)))
        
        for i, det in enumerate(detections):
            for j, tid in enumerate(track_ids):
                last_det = self.tracks[tid].get_detection()
                if last_det:
                    cost[i,j] = 1 - self._iou(det.bbox, last_det.bbox)
        
        matched_dets = set()
        for _ in range(min(len(detections), len(track_ids))):
            i, j = np.unravel_index(cost.argmin(), cost.shape)
            if cost[i,j] < 0.7:
                self.tracks[track_ids[j]].add_detection(detections[i], timestamp)
                matched_dets.add(i)
                cost[i,:] = 1.0
                cost[:,j] = 1.0
            else:
                break
        
        for i, det in enumerate(detections):
            if i not in matched_dets:
                self._create_track(det, timestamp)
    
    def _create_track(self, det, timestamp):
        track = TrackedObject(self.next_id, det.class_name, last_seen=timestamp)
        track.add_detection(det, timestamp)
        self.tracks[self.next_id] = track
        self.next_id += 1
    
    def _remove_stale(self, t):
        stale = [tid for tid, track in self.tracks.items() if t - track.last_seen > self.max_age]
        for tid in stale:
            del self.tracks[tid]
    
    @staticmethod
    def _iou(b1, b2):
        x1_min, y1_min, x1_max, y1_max = b1
        x2_min, y2_min, x2_max, y2_max = b2
        
        ix_min = max(x1_min, x2_min)
        iy_min = max(y1_min, y2_min)
        ix_max = min(x1_max, x2_max)
        iy_max = min(y1_max, y2_max)
        
        if ix_max < ix_min or iy_max < iy_min:
            return 0.0
        
        inter = (ix_max - ix_min) * (iy_max - iy_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        return inter / (area1 + area2 - inter + 1e-8)


# ============================================================================
# PATH PLANNING MODULE
# ============================================================================

class PathPlanner:
    def __init__(self, frame_width=640, frame_height=480):
        self.width = frame_width
        self.height = frame_height
        
        # Priority classes (lower number = higher priority)
        self.PRIORITY_MAP = {
            'person': 0, 'child': 0, 'adult': 0, 'people': 0,
            'car': 0, 'vehicle': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0, 'bicycle': 0,
            'furniture': 1, 'chair': 1, 'table': 1, 'couch': 1, 'bench': 1,
            'obstacle': 1, 'wall': 1, 'door': 1,
            'bag': 2, 'luggage': 2, 'suitcase': 2, 'backpack': 2, 'box': 2
        }
    
    def plan(self, tracks, detections=None):
        """
        Plan path based on tracked objects
        If tracks is empty but detections exist, create simple plan from detections
        """
        if not tracks and not detections:
            return {
                'direction': 0.0,
                'risk': 'low',
                'instruction': 'Path clear'
            }
        
        # If we have tracks, use them
        if tracks:
            obstacles = []
            for track in tracks.values():
                risk = self._assess_risk_from_track(track)
                obstacles.append((track.get_detection(), risk, track.class_name))
        # Otherwise use raw detections
        elif detections:
            obstacles = []
            for det in detections:
                risk = self._assess_risk_from_detection(det)
                obstacles.append((det, risk, det.class_name))
        
        obstacles.sort(key=lambda x: x[1], reverse=True)
        
        direction = self._calc_direction(obstacles)
        risk_level = self._quantize_risk(obstacles[0][1])
        instruction = self._make_instruction(direction, risk_level, obstacles[0][2])
        
        return {
            'direction': direction,
            'risk': risk_level,
            'instruction': instruction,
            'primary_object': obstacles[0][2]
        }
    
    def _assess_risk_from_track(self, track):
        """Assess risk from tracked object (has velocity)"""
        det = track.get_detection()
        if not det:
            return 0.0
        
        depth_risk = det.depth
        size_risk = min(1.0, det.area / (self.width * self.height) * 10)
        vel = track.get_velocity()
        motion_risk = min(1.0, np.linalg.norm(vel) / 100)
        priority = self._get_priority(track.class_name)
        type_weight = 1.0 / (priority + 1)
        
        return min(1.0, depth_risk*0.5 + size_risk*0.2 + motion_risk*0.2 + type_weight*0.1)
    
    def _assess_risk_from_detection(self, det):
        """Assess risk from raw detection (no velocity)"""
        depth_risk = det.depth
        size_risk = min(1.0, det.area / (self.width * self.height) * 10)
        priority = self._get_priority(det.class_name)
        type_weight = 1.0 / (priority + 1)
        
        return min(1.0, depth_risk*0.6 + size_risk*0.3 + type_weight*0.1)
    
    def _get_priority(self, class_name):
        """Get priority for object class"""
        class_lower = class_name.lower()
        for key, priority in self.PRIORITY_MAP.items():
            if key in class_lower:
                return priority
        return 2  # Default priority
    
    def _calc_direction(self, obstacles):
        angles = np.linspace(-75, 75, 15)
        costs = []
        
        for angle in angles:
            cost = abs(angle) * 0.01  # Prefer straight
            rad = np.radians(angle)
            direction = np.array([np.sin(rad), np.cos(rad)])
            
            for det, risk, _ in obstacles:
                if not det:
                    continue
                
                # Obstacle position (normalized)
                ox = (det.center[0] - self.width/2) / (self.width/2)
                oy = (self.height - det.center[1]) / self.height  # Invert y
                obs_pos = np.array([ox, oy])
                
                # Project onto direction
                proj = np.dot(obs_pos, direction)
                if proj > 0:
                    dist = np.linalg.norm(obs_pos - proj*direction)
                    if dist < 0.5:  # Within comfort zone
                        cost += (50 / (dist + 0.1)) * risk
            
            costs.append(cost)
        
        best_idx = np.argmin(costs)
        return float(angles[best_idx])
    
    def _quantize_risk(self, risk):
        if risk > 0.7:
            return 'critical'
        elif risk > 0.5:
            return 'high'
        elif risk > 0.3:
            return 'medium'
        return 'low'
    
    def _make_instruction(self, angle, risk, obj_name):
        if -10 < angle < 10:
            dir_text = "straight"
        elif 10 <= angle < 40:
            dir_text = "slightly right"
        elif angle >= 40:
            dir_text = "right"
        elif -40 < angle <= -10:
            dir_text = "slightly left"
        else:
            dir_text = "left"
        
        if risk == 'critical':
            return f"STOP! {obj_name} ahead!"
        elif risk == 'high':
            return f"Move {dir_text}, {obj_name} approaching"
        elif risk == 'medium':
            return f"Bear {dir_text}, {obj_name} ahead"
        else:
            return f"Continue {dir_text}" if dir_text != "straight" else "Path clear"


# ============================================================================
# GROUNDING DINO DETECTOR
# ============================================================================


class GroundingDinoDetector:
    def __init__(self, model_size='tiny', depth_size='small', 
                 use_fp16=True, use_compile=False, max_resolution=640):
        """
        Args:
            model_size: 'tiny' or 'base' (tiny is faster, base is more accurate)
            depth_size: 'small', 'base', or 'large'
            use_fp16: Use autocast for mixed precision (safer than .half())
            use_compile: Use torch.compile for optimization (PyTorch 2.0+)
            max_resolution: Downscale images larger than this (lower = faster)
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_fp16 = use_fp16 and self.device == 'cuda'
        self.max_resolution = max_resolution
        
        # Grounding DINO
        print(f"Loading Grounding DINO ({model_size}) on {self.device}...")
        if self.use_fp16:
            print("  Using mixed precision (autocast) for speedup")
        model_id = f"IDEA-Research/grounding-dino-{model_size}"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        
        # Don't convert to .half(), use autocast instead
        self.model.eval()
        
        # Compile model for optimization (PyTorch 2.0+)
        if use_compile:
            try:
                print("  Compiling model with torch.compile...")
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("  ✓ Model compiled")
            except Exception as e:
                print(f"  ⚠ torch.compile failed: {e}")
        
        print("✓ Grounding DINO loaded")
        
        # Depth estimator
        print(f"Loading Depth-Anything-V2 ({depth_size}) on {self.device}...")
        depth_model_name = f"depth-anything/Depth-Anything-V2-{depth_size.capitalize()}-hf"
        self.depth_processor = AutoImageProcessor.from_pretrained(depth_model_name)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_name).to(self.device)
        
        # Depth model can use .half() safely
        if self.use_fp16:
            self.depth_model = self.depth_model.half()
        
        self.depth_model.eval()
        print("✓ Depth-Anything-V2 loaded")
        
        # Comprehensive text prompt for obstacle detection
        self.default_prompt = (
            "person. child. adult. people. "
            "car. vehicle. truck. bus. motorcycle. bicycle. scooter. "
            "furniture. chair. table. desk. bed. couch. sofa. bench. stool. "
            "obstacle. object. box. bag. luggage. suitcase. backpack. package. "
            "wall. pillar. column. pole. post. "
            "door. gate. barrier. fence. railing. "
            "stairs. steps. curb. ledge. "
            "plant. tree. pot. vase. "
            "equipment. machine. appliance. cart. "
            "animal. dog. cat. pet."
        )
    
    def detect(self, frame: np.ndarray, text_prompt: str = None, 
               score_threshold: float = 0.25) -> List[GDinoDetection]:
        """
        Detect objects using Grounding DINO
        
        Args:
            frame: Input image (BGR from OpenCV)
            text_prompt: Custom text prompt or None for default
            score_threshold: Confidence threshold (lower = more detections)
        """
        if text_prompt is None:
            text_prompt = self.default_prompt
        
        # Downscale if needed for speed
        original_shape = frame.shape[:2]
        scale_factor = 1.0
        if max(original_shape) > self.max_resolution:
            scale_factor = self.max_resolution / max(original_shape)
            new_size = (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor))
            frame_resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
        else:
            frame_resized = frame
        
        # Convert BGR to RGB and to PIL
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Process
        inputs = self.processor(images=pil_image, text=text_prompt, return_tensors="pt").to(self.device)
        
        # Use autocast for mixed precision (safer than .half())
        with torch.no_grad():
            if self.use_fp16:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
        
        # Post-process
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[pil_image.size[::-1]]  # (height, width)
        )[0]
        
        # Filter by score and convert to our format
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score < score_threshold:
                continue
            
            # Scale boxes back to original resolution
            box_np = box.cpu().numpy()
            if scale_factor != 1.0:
                box_np = box_np / scale_factor
            
            x1, y1, x2, y2 = box_np.astype(int)
            if x2 <= x1 or y2 <= y1:
                continue
            
            detections.append(GDinoDetection(
                bbox=(x1, y1, x2, y2),
                class_name=label.strip(),
                confidence=float(score),
                area=(x2 - x1) * (y2 - y1)
            ))
        
        # Clean up
        del inputs, outputs, results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return sorted(detections, key=lambda d: d.confidence, reverse=True)
    
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Get depth map"""
        inputs = self.depth_processor(images=frame, return_tensors="pt").to(self.device)
        
        # Convert to FP16 for depth model (works fine)
        if self.use_fp16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth
            
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
            depth = prediction.cpu().numpy()
        
        del inputs, outputs, predicted_depth, prediction
        
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth_normalized
    
    def detect_with_depth(self, frame: np.ndarray, text_prompt: str = None,
                         score_threshold: float = 0.25) -> Tuple[List[GDinoDetection], np.ndarray]:
        """Detect objects and add depth information"""
        # Detect objects
        detections = self.detect(frame, text_prompt, score_threshold)
        
        # Get depth map
        depth_map = self.estimate_depth(frame)
        
        # Add depth to detections
        h, w = depth_map.shape
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 > x1 and y2 > y1:
                roi = depth_map[y1:y2, x1:x2]
                det.depth = float(np.median(roi))
        
        # Sort by depth (closest first)
        detections.sort(key=lambda d: d.depth, reverse=True)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return detections, depth_map
    
    def visualize(self, frame, detections, depth_map=None, plan=None, tracks=None):
        """Visualize detections with optional path planning"""
        if depth_map is not None:
            depth_colored = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            vis = cv2.addWeighted(frame, 0.6, depth_colored, 0.4, 0)
        else:
            vis = frame.copy()
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Color by depth or confidence
            if det.depth > 0:
                if det.depth > 0.7:
                    color = (0, 0, 255)  # Red - very close
                elif det.depth > 0.5:
                    color = (0, 165, 255)  # Orange - close
                else:
                    color = (0, 255, 255)  # Yellow - medium
            else:
                color = (0, 255, 0)  # Green - no depth info
            
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Label
            if det.depth > 0:
                label = f"{det.class_name} {det.confidence:.2f} d={det.depth:.2f}"
            else:
                label = f"{det.class_name} {det.confidence:.2f}"
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(vis, (x1, y1-text_h-10), (x1+text_w, y1), color, -1)
            cv2.putText(vis, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw track IDs if available
        if tracks:
            for tid, track in tracks.items():
                det = track.get_detection()
                if det:
                    cx, cy = int(det.center[0]), int(det.center[1])
                    cv2.circle(vis, (cx, cy), 4, (255, 0, 255), -1)
                    cv2.putText(vis, f"ID:{tid}", (cx+10, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        # Draw path planning overlay
        if plan:
            risk_colors = {
                'critical': (0, 0, 255),
                'high': (0, 165, 255),
                'medium': (0, 255, 255),
                'low': (0, 255, 0)
            }
            color = risk_colors[plan['risk']]
            
            # Draw instruction at bottom
            cv2.putText(vis, plan['instruction'], (10, vis.shape[0]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
            
            # Draw direction arrow
            angle = np.radians(plan['direction'])
            cx, cy = vis.shape[1]//2, vis.shape[0]-60
            length = 60
            ex = int(cx + length * np.sin(angle))
            ey = int(cy - length * np.cos(angle))
            cv2.arrowedLine(vis, (cx, cy), (ex, ey), color, 4, tipLength=0.3)
            
            # Add direction angle text
            cv2.putText(vis, f"{plan['direction']:.0f}deg", (cx-30, cy+30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return vis


def test_on_image(image_path='test.jpg', with_depth=True, with_planning=True):
    """Test Grounding DINO on a single image"""
    print("\n" + "="*60)
    print("TESTING: Grounding DINO Object Detection + Path Planning")
    print("="*60)
    
    # OPTIMIZED: Use FP16, no compile (slower on first run), 640 max res
    detector = GroundingDinoDetector(
        model_size='tiny', 
        depth_size='small',
        use_fp16=True,      # 2x speedup on GPU
        use_compile=False,  # Set to True after warmup for extra speed
        max_resolution=640  # Lower for more speed (480, 512, 640, 768)
    )
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load {image_path}")
        return
    
    print(f"\nProcessing {image_path}...")
    print(f"Using prompt: {detector.default_prompt}")
    print()
    
    start = time.time()
    
    if with_depth:
        detections, depth_map = detector.detect_with_depth(frame, score_threshold=0.25)
    else:
        detections = detector.detect(frame, score_threshold=0.25)
        depth_map = None
    
    print(f"Found {len(detections)} objects:\n")
    for i, det in enumerate(detections[:15]):  # Show top 15
        depth_str = f"d={det.depth:.3f}" if det.depth > 0 else "no depth"
        print(f"  {i+1:2d}. {det.class_name:20s} | conf={det.confidence:.3f} | {depth_str}")
    
    # Path planning
    plan = None
    if with_planning and detections:
        planner = PathPlanner(frame_width=frame.shape[1], frame_height=frame.shape[0])
        plan = planner.plan(tracks=None, detections=detections)
        print(f"\nPath Planning:")
        print(f"  Risk Level: {plan['risk'].upper()}")
        print(f"  Direction: {plan['direction']:.1f}° ({'left' if plan['direction'] < 0 else 'right' if plan['direction'] > 0 else 'straight'})")
        print(f"  Instruction: {plan['instruction']}")
    
    # Visualize
    vis = detector.visualize(frame, detections, depth_map, plan=plan)
    output_path = 'output_gdino.jpg'
    cv2.imwrite(output_path, vis)
    print(f"\n✓ Saved: {output_path}")
    
    elapsed = time.time() - start
    print(f"Processing time: {elapsed:.2f}s")
    print("="*60)
    
    return detections


def test_custom_prompt(image_path='test.jpg'):
    """Test with custom prompts"""
    print("\n" + "="*60)
    print("TESTING: Custom Text Prompts")
    print("="*60)
    
    detector = GroundingDinoDetector(
        model_size='tiny', 
        depth_size='small',
        use_fp16=True,
        max_resolution=640
    )
    frame = cv2.imread(image_path)
    
    # Try different prompts
    prompts = [
        "person. human. people.",
        "furniture. chair. table. bed. couch.",
        "luggage. suitcase. bag. backpack. box.",
        "all objects. everything. anything visible."
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: '{prompt}'")
        detections = detector.detect(frame, text_prompt=prompt, score_threshold=0.3)
        print(f"  Found {len(detections)} objects:")
        for det in detections[:5]:
            print(f"    - {det.class_name}: {det.confidence:.3f}")
        
        vis = detector.visualize(frame, detections)
        cv2.imwrite(f'output_gdino_prompt{i+1}.jpg', vis)


def test_live(source='droidcam', ip=None, port=8080):
    """Test Grounding DINO on live video with tracking and path planning"""
    print("\n" + "="*60)
    print("TESTING: Grounding DINO + Tracking + Path Planning (Live)")
    print("="*60)
    
    from trial import CameraInterface
    
    camera = CameraInterface(source, ip=ip, port=port)
    if not camera.connect():
        print("Camera connection failed")
        return
    
    # OPTIMIZED SETTINGS
    detector = GroundingDinoDetector(
        model_size='tiny',
        depth_size='small',
        use_fp16=True,      # 2x speedup
        use_compile=False,  # Can enable after warmup
        max_resolution=512  # Lower resolution = faster (try 384, 448, 512)
    )
    
    # Initialize tracker and planner
    tracker = Tracker(max_age=1.0)
    
    # Get first frame to determine size
    test_frame = camera.read_frame()
    if test_frame is None:
        print("Failed to read frame")
        return
    planner = PathPlanner(frame_width=test_frame.shape[1], frame_height=test_frame.shape[0])
    
    print("\nRunning (press 'q' to quit, 't' to toggle depth, 'p' to toggle planning)...")
    print("OPTIMIZATIONS: Mixed precision enabled, max_res=512")
    print("Expected FPS: 10-15 (vs 6 before)\n")
    
    frame_count = 0
    start_time = time.time()
    use_depth = True
    use_planning = True
    
    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                break
            
            frame_count += 1
            timestamp = time.time() - start_time
            
            # Process every 3rd frame (Grounding DINO is slower)
            if frame_count % 3 == 0:
                if use_depth:
                    detections, depth_map = detector.detect_with_depth(frame, score_threshold=0.25)
                else:
                    detections = detector.detect(frame, score_threshold=0.25)
                    depth_map = None
                
                # Update tracker
                tracks = tracker.update(detections, timestamp)
                
                # Plan path
                plan = None
                if use_planning:
                    if tracks:
                        plan = planner.plan(tracks=tracks)
                    elif detections:
                        plan = planner.plan(tracks=None, detections=detections)
                
                vis = detector.visualize(frame, detections, depth_map, plan=plan, tracks=tracks)
                
                # Add info
                fps = frame_count / (time.time() - start_time)
                cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis, f"Objects: {len(detections)} | Tracks: {len(tracks)}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(vis, f"Depth: {'ON' if use_depth else 'OFF'} | Plan: {'ON' if use_planning else 'OFF'}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow("Grounding DINO + Navigation", vis)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                use_depth = not use_depth
                print(f"Depth: {'ON' if use_depth else 'OFF'}")
            elif key == ord('p'):
                use_planning = not use_planning
                print(f"Planning: {'ON' if use_planning else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        
        # Print summary
        total_time = time.time() - start_time
        print(f"\nSession Summary:")
        print(f"  Total frames: {frame_count}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average FPS: {frame_count/total_time:.1f}")


def test_on_video(video_path, save_frames=True, output_dir='results', 
                  with_depth=True, with_planning=True):
    """
    Test Grounding DINO on a video file with tracking and path planning
    
    Args:
        video_path: Path to input video file
        save_frames: Whether to save individual frames
        output_dir: Directory to save frames and output video
        with_depth: Enable depth estimation
        with_planning: Enable path planning
    """
    import os
    from pathlib import Path
    
    print("\n" + "="*60)
    print("TESTING: Grounding DINO on Video File")
    print("="*60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if save_frames:
        frames_dir = output_path / 'frames'
        frames_dir.mkdir(exist_ok=True)
        print(f"Frames will be saved to: {frames_dir}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f}s")
    
    # Initialize detector
    detector = GroundingDinoDetector(
        model_size='tiny',
        depth_size='small',
        use_fp16=True,
        use_compile=False,
        max_resolution=640
    )
    
    # Initialize tracker and planner
    tracker = Tracker(max_age=1.0)
    planner = PathPlanner(frame_width=width, frame_height=height)
    
    # Setup video writer
    output_video_path = output_path / f'output_{Path(video_path).stem}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    print(f"\nProcessing video...")
    print(f"Output video: {output_video_path}")
    
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = frame_count / fps
            
            # Process every 2nd frame for speed
            if frame_count % 2 == 0:
                processed_count += 1
                
                # Detect objects
                if with_depth:
                    detections, depth_map = detector.detect_with_depth(frame, score_threshold=0.25)
                else:
                    detections = detector.detect(frame, score_threshold=0.25)
                    depth_map = None
                
                # Update tracker
                tracks = tracker.update(detections, timestamp)
                
                # Plan path
                plan = None
                if with_planning:
                    if tracks:
                        plan = planner.plan(tracks=tracks)
                    elif detections:
                        plan = planner.plan(tracks=None, detections=detections)
                
                # Visualize
                vis = detector.visualize(frame, detections, depth_map, plan=plan, tracks=tracks)
                
                # Add frame info
                cv2.putText(vis, f"Frame: {frame_count}/{total_frames}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(vis, f"Objects: {len(detections)} | Tracks: {len(tracks)}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Save frame if requested
                if save_frames:
                    frame_filename = frames_dir / f'frame_{frame_count:06d}.jpg'
                    cv2.imwrite(str(frame_filename), vis)
                
                # Write to output video
                out_video.write(vis)
                
                # Progress update
                if processed_count % 10 == 0:
                    elapsed = time.time() - start_time
                    progress = (frame_count / total_frames) * 100
                    processing_fps = processed_count / elapsed
                    eta = (total_frames - frame_count) / (frame_count / elapsed) if frame_count > 0 else 0
                    print(f"  Progress: {progress:.1f}% | Frame {frame_count}/{total_frames} | "
                          f"Processing FPS: {processing_fps:.1f} | ETA: {eta:.1f}s", end='\r')
            else:
                # For skipped frames, write original or last processed
                out_video.write(vis if processed_count > 0 else frame)
    
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted!")
    finally:
        cap.release()
        out_video.release()
        
        total_time = time.time() - start_time
        print(f"\n\nProcessing Complete!")
        print(f"  Processed frames: {processed_count}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Processing FPS: {processed_count/total_time:.1f}")
        print(f"\n✓ Output video saved: {output_video_path}")
        
        if save_frames:
            print(f"✓ Individual frames saved: {frames_dir}")
            print(f"  Total frames: {len(list(frames_dir.glob('*.jpg')))}")
        
        print("="*60)


if __name__ == "__main__":

    test_live(
        source='droidcam',
        ip='192.168.31.103',
        port=4747
    )