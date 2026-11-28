import os
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
from google import genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
import hashlib
import torch
import math
import time


load_dotenv()

aiClient = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
client = chromadb.PersistentClient(path="./Vision_RAG/video_db")
frame_col = client.get_or_create_collection(
    name="frames", metadata={"hnsw:space": "cosine"}
)
chunk_col = client.get_or_create_collection(
    name="video_chunks", metadata={"hnsw:space": "cosine"})

# load initally For faster throughput
model = SentenceTransformer(
    "google/embeddinggemma-2",
    device="cuda",
    model_kwargs={"dtype": torch.bfloat16},
)
print(model.device)   # should print cuda:0

# warm-up the first call is always slow
model.encode([{"image": Image.new("RGB", (224, 224))}])
WORKSPACE_PATH = "./Vision_RAG/video"

def is_workspace_ready():
    if(not Path.exists(Path("./Vision_RAG/video"))):
        Path("./Vision_RAG/video").mkdir(exist_ok=True,parents=True)

# need to handel by flask
# def get_video(video_path:str): 
    
def process_video_frame(video_path=r"C:\Users\darsh\Downloads\03_Videos\Stock_Footage\Cinematic_Green_Screen_TV_Intro.mp4"):
    print("START PROCESSING...")
    video_name = Path(video_path).stem  
    PIL_frames=[]
    chunk_id=0
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames/fps
    print("Duration-->",duration,"\n FPS-->",fps,"\nTOTAL_FRAME-->",total_frames)
    for sec in range(math.ceil(duration)):
        cap.set(cv2.CAP_PROP_POS_MSEC,sec * 1000)
        sucess,frame = cap.read()
        if sucess:
            rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            PIL_frames.append(Image.fromarray(rgb))              

            if len(PIL_frames) == 5:
                print("Processing Chunk --->",chunk_id)
                print("Processing NO of Frames --->",len(PIL_frames))
                index_5sec_chunk(video_name, video_path, chunk_id, PIL_frames)
                chunk_id += 1
                PIL_frames= []   
        else:
            break
    # call for left frames 
    if PIL_frames:
                index_5sec_chunk(video_name, video_path, chunk_id, PIL_frames)       
    cap.release()



def summurize_frame(images):

    PROMPT = """You are given {n} frames from the same video, in time order (Frame 1 to Frame {n}).
    Analyze them together and respond in this format, including only the sections that apply.

    1. PEOPLE
    - Approximate total number of distinct people across all frames: <number or range, e.g. 5-6>
    - For each person, give a short label (Person A, Person B, ...) and describe visible
    features only: clothing colors, hair, build, accessories, position in the frame,
    and what they are doing. If two people look very similar (same uniform, same outfit),
    still list them as separate people and use position or action to tell them apart.
    - Do NOT guess names or real identities. Only describe what is visible.
    - Note which frames each person appears in.

    2. OBJECTS
    - Total count of main objects visible (vehicles, furniture, bags, screens, animals, etc.).
    - List each object type with its approximate count, e.g. "chairs: 4, laptops: 2".

    3. SETTING
    - Where this seems to be (indoor/outdoor, type of place), lighting, time of day if clear.

    4. DETAILED SUMMARY
    - 4 to 6 sentences describing what happens from the first frame to the last:
    who does what, how people interact, and what changes between frames.

    Rules:
    - Count only what you can actually see. If people are partly hidden, overlapping, or
    far away, say so and give a range instead of an exact number.
    - If you are unsure whether two appearances are the same person, say "possibly the same as Person X".
    - Do not invent details that are not visible.
    - If a section has nothing to report (for example, no people are visible), leave that
    section out completely. Never write things like "no people present", "none visible",
    "not applicable", or "N/A".
    - Mention only what is actually in the frames."""
    print("Start AI SUMMARY...")
    response = aiClient.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=[*images, PROMPT.format(n=len(images))],)
    print("STOP ai summary ->",response.text)
    return response.text



def get_frame(n,k=2):
    if n<=k:
        return list(range(n))

    return [round((i+0.5)*(n/k)-0.5) for i in range(k)]




    
def index_5sec_chunk(video_name,video_path, chunk_idx, pil_frames):

    # it give 16  uniqe hexadecimal characters
    vid_id = hashlib.md5(video_path.encode()).hexdigest()[:16]

    chunk_id = f"{vid_id}_chunk_{chunk_idx}"
    start_sec = chunk_idx * 5
    frame_inputs = [{"image":frm} for frm in pil_frames]
    print("START ENCODING....")
    t = time.perf_counter()
    frame_embs = model.encode(frame_inputs, batch_size=5, normalize_embeddings=True)
    print("encode:", round(time.perf_counter() - t, 2))
    print("STOP ENCODING...")

    # summarize the frame into Captions For Summary Vector for Better Context Aware Findings..
    t = time.perf_counter()
    caption= summurize_frame([pil_frames[i] for i in get_frame(len(pil_frames))])
    print("aiiii:", round(time.perf_counter() - t, 2))
    cap_embd = model.encode([caption],normalize_embeddings=True)[0]


    base_meta = {
        "video": vid_id,
        "video_path":video_path,
        "chunk_id": chunk_id,
        "start_sec": start_sec,
        "end_sec": start_sec + 5,
    }


    # create 1 summary vector for 5-sec frames
    chunk_vec = frame_embs.mean(axis=0)
    chunk_vec = chunk_vec / np.linalg.norm(chunk_vec)


    chunk_col.add(
        ids=[f"{chunk_id}_frames", f"{chunk_id}_cap"],
        embeddings=[chunk_vec.tolist(),cap_embd.tolist()],
        metadatas=[ 
            {**base_meta, "type": "frames"},
            {**base_meta, "type": "caption"},],
    )

    # store 5 individual frame in seprate collection
    sec_list = [start_sec + i for i in range(len(frame_embs))]
    frame_col.add(
        ids=[f"{vid_id}_f{s}" for s in sec_list],
        embeddings=frame_embs.tolist(),
        metadatas=[{"vid_id": vid_id, "sec": s} for s in sec_list],
    )
    return

if __name__ == '__main__':
    process_video_frame()
    # print(get_frame(5))
    # print(torch.cuda.is_available())      
    # print(torch.__version__)              
    # print(torch.cuda.get_device_name(0))  
