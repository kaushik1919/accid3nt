import pandas as pd
from roboflow import Roboflow

# Initialize Roboflow API
rf = Roboflow(api_key="DQzI0UEAM6M8290R5bVR")
project = rf.workspace().project("accid3nt")
model = project.version("1").model

# Run inference on video
job_id, signed_url, expire_time = model.predict_video(
    "acci.mp4",
    fps=5,
    prediction_type="batch-video",
)

# Get results
results = model.poll_until_video_results(job_id)

# Extract relevant data
frame_offsets = results.get("frame_offset", [])
time_offsets = results.get("time_offset", [])
detections = results.get("accid3nt", [])

# Create a list of dictionaries for the CSV
data = []

for i, detection in enumerate(detections):
    inference_id = detection.get("inference_id", "N/A")
    timestamp = detection.get("time", "N/A")
    width = detection["image"]["width"]
    height = detection["image"]["height"]

    for prediction in detection.get("predictions", []):
        data.append({
            "frame_offset": frame_offsets[i] if i < len(frame_offsets) else "N/A",
            "time_offset": time_offsets[i] if i < len(time_offsets) else "N/A",
            "inference_id": inference_id,
            "timestamp": timestamp,
            "image_width": width,
            "image_height": height,
            "x": prediction["x"],
            "y": prediction["y"],
            "width": prediction["width"],
            "height": prediction["height"],
            "confidence": prediction["confidence"],
            "class": prediction["class"],
            "class_id": prediction["class_id"],
            "detection_id": prediction["detection_id"]
        })

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
csv_path = "inference_results.csv"
df.to_csv(csv_path, index=False)

print(f"Results saved to {csv_path}")



