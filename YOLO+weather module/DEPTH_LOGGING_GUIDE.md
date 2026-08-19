# Real-Time YOLO + Depth Logging Setup

## 🎯 What You Now Have

**Two ways to log depths while manually driving:**

### Option 1: Minimal Manual Drive (Recommended)
```bash
cd "e:\Capstone\Minimal_Grid_env\YOLO+weather module"
python manual_drive_with_depth.py
```

**What it does:**
- Opens MetaDrive environment
- You manually drive with keyboard (W/A/S/D)
- **Every YOLO detection is logged to terminal in real-time with depth estimate**
- Shows frame in window
- Logs total detections when you quit

**Output format (terminal):**
```
Step     Class        Conf     Bbox                 Depth (m)   
----------------------------------------------------------------------
0        car          0.850    (100,150,250,300)    5.25
1        truck        0.720    (80,140,240,320)     6.50
2        bus          0.680    (120,160,300,400)    4.75
```

---

### Option 2: Batch Processing (For Testing)
```bash
cd "e:\Capstone\Minimal_Grid_env\YOLO+weather module"
python yolo_depth_logger.py
```

**What it does:**
- Processes all 500 frames you already captured
- Saves to CSV: `manual_drive_output/yolo_depth_log.csv`
- Prints summary statistics
- (Already tested - generates 161 detections)

---

## 🔧 Integration with Your Manual Drive Script

If you want to add depth logging to your existing `manual_drive_visualize.py`:

```python
# At top of file
from realtime_depth_logger import RealTimeDepthLogger

# In main() before the loop
depth_logger = RealTimeDepthLogger(
    model_path='yolov8n.pt',
    conf_threshold=0.3
)

# Inside the step loop (after getting frame)
detections = depth_logger.process_frame(frame, simulation_steps)
```

---

## 📊 Depth Estimation Formula

From your **Technical Spec Section 3**:

$$Z = \frac{f \cdot B}{d_{px}} = \frac{500}{d_{px}}$$

Where:
- **Z** = depth in meters
- **f** = focal length = 1000 px
- **B** = stereo baseline = 0.5 m
- **d_px** = estimated disparity (pixels)

**Disparity estimation from bbox:**
- Large bbox (>30% width) → d_px=100 → Z ≈ 5m (very close)
- Medium bbox (>15% width) → d_px=50 → Z ≈ 10m
- Small bbox (>8% width) → d_px=30 → Z ≈ 17m
- Tiny bbox (<4% width) → d_px=10 → Z ≈ 50m (very far)

---

## ⌨️ Keyboard Controls

| Key | Action |
|-----|--------|
| **W** | Throttle forward (0.5) |
| **S** | Throttle backward (-0.3) |
| **A** | Steer left (-0.5) |
| **D** | Steer right (+0.5) |
| **Q** | Quit and save |

---

## 📁 Output Files

### From `manual_drive_with_depth.py`:
- Real-time terminal logging (no file saved by default)
- Press Q to quit

### From `yolo_depth_logger.py`:
- `manual_drive_output/yolo_depth_log.csv` - CSV with all detections
- Terminal summary statistics

---

## 🔍 Visualize Results

```bash
# View depth statistics and charts
python visualize_depth_log.py

# View CSV data
python -c "import pandas as pd; df=pd.read_csv('../../manual_drive_output/yolo_depth_log.csv'); print(df.head(20))"
```

---

## 🚨 Troubleshooting

### MetaDrive won't install
**Issue:** Long path error on Windows
**Solution:** Enable long paths or use batch processing instead

### YOLO takes too long
**Solution:** Use smaller model `yolov8n.pt` (nano) - already used

### Depths seem wrong
**Solution:** This is monocular estimation - depth depends heavily on object size in frame. Actual stereo (if available) would be more accurate.

---

## 📈 Expected Output

**Real-time logging example:**
```
Step     Class        Conf     Bbox                 Depth (m)   
----------------------------------------------------------------------
143      truck        0.35     (100,150,350,400)    10.50
143      bus          0.47     (400,100,500,300)    8.33
144      car          0.62     (200,200,300,350)    12.50
145      person       0.35     (150,250,200,350)    25.00
```

**Summary (after quit):**
```
✓ Manual drive complete
✓ Total steps: 146
✓ Total detections logged: 8
```

---

## 🎉 You're Ready!

Run manual drive with real-time depth logging:
```bash
python manual_drive_with_depth.py
```

Detections will appear in terminal as you drive! 🚗

