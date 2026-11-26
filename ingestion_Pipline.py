import os
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer

model =SentenceTransformer("google/embeddinggemma-2")
WORKSPACE_PATH = "./Vision_RAG/video"

def is_workspace_ready():
    if(not Path.exists(Path("./Vision_RAG/video"))):
        Path("./Vision_RAG/video").mkdir(exist_ok=True,parents=True)

# need to handel by flask
# def get_video(video_path:str): 
    
def get_video_frame(video_path=r"C:\Users\darsh\Downloads\03_Videos\Stock_Footage\Cinematic_Green_Screen_TV_Intro.mp4"):
    interval=1
    timestamp=0
    counter=1
    Chunk_ID=0
    PIL_Image=[]
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames/fps
    print("Duration-->",duration,"\n FPS-->",fps,"\nTOTAL_FRAME-->",total_frames)
    while timestamp < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC,timestamp * 1000)
        sucess,frame = cap.read()
        if sucess:
            if(counter==5):
                index_5sec_chunk(model, video_path, Chunk_ID, PIL_Image)
                # reset this list and counter for next batch
                counter=1
                PIL_Image=[]
            else:
                rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                PIL_Image.append(Image.fromarray(rgb))
               
            timestamp +=interval
            counter+=1
    cap.release()



def summurize_frame():
    # use llm to summurize
    pass
def index_5sec_chunk(model, video_path, chunk_idx, pil_frames):
    pass

def process_embbedings():
    # now used 5-5 elments embbedings converts into chunks 
    pass

is_workspace_ready()
get_video_frame()



# i think insted storing each freame we can convert evry 2th freame into base64 for embedding and also convert each into embbeding without storing into stoarge 
# embedding gemma take image path useing pillow or PIL
