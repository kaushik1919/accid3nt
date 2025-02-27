from roboflow import Roboflow

rf = Roboflow(api_key="DQzI0UEAM6M8290R5bVR")
project = rf.workspace().project("accid3nt")
model = project.version("1").model

job_id, signed_url, expire_time = model.predict_video(
    "acci.mp4",
    fps=5,
    prediction_type="batch-video",
)

results = model.poll_until_video_results(job_id)

print(results)
