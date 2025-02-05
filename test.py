
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

pipeline = InferencePipeline.init(
    model_id="accid3nt/1", 
    video_reference="C:/Users/kaush/code/accident/accid3nt/acci.mp4",
    on_prediction=render_boxes, 
)
pipeline.start()
pipeline.join()