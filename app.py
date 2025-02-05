from inference import InferencePipeline
import cv2
import os
from dotenv import load_dotenv
from inference.core.interfaces.stream.sinks import render_boxes

load_dotenv()

def combined_sink(result, video_frame):
    my_sink(result, video_frame)     
    render_boxes(result, video_frame) 


def my_sink(result, video_frame):
    if result.get("output_image"): 
        cv2.imshow("Workflow Image", result["output_image"].numpy_image)
        cv2.waitKey(1)
    print(result) 

api_key = os.getenv("API_KEY")
workspace_name = os.getenv("WORKSPACE_NAME")
workflow_id = os.getenv("WORKFLOW_ID")


pipeline = InferencePipeline.init_with_workflow(
    api_key=api_key,
    workspace_name=workspace_name,
    workflow_id=workflow_id,
    video_reference=https://youtu.be/46iWkLmZ4g8?feature=shared,
    max_fps=30,
    on_prediction=combined_sink
)

try: 
    pipeline.start()
    pipeline.join()

except KeyboardInterrupt:
    pipeline.stop()
    cv2.destroyAllWindows()


