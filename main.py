from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect
import torch
from pydantic import Json
from ultralytics import YOLO
from PIL import Image
import matplotlib.path as mplPath
import io
from typing import Dict, Optional
import asyncio
import uuid
import json
import cv2
import time
from concurrent.futures import ThreadPoolExecutor
import threading

app = FastAPI()
model = YOLO('weights/v8.pt')
if torch.cuda.is_available():
    model.cuda()
else:
    model.cpu()
print(model.device)

def predict_frame(image, confidance, parking_spaces):
    results = {}
    boxes = model.predict(image, conf = confidance)[0].boxes.xyxy
    for place_key in parking_spaces:
        place = parking_spaces[place_key]
        busy = False
        for bb in boxes:
            x = bb[0] + (bb[2] - bb[0]) / 2
            y = bb[1] + (bb[3] - bb[1]) * 3/4
            if place.contains_point((x,y)):
                busy = True
                break
        results[place_key] = busy
    return results

def spaces_json_to_path(json):
    results = {}
    for key in json:
        place = json[key]
        results[key] = mplPath.Path([point for point in zip(place['points_x'], place['points_y'])])
    return results

@app.post("/predict_image")
async def predict_image(image: UploadFile = File(...), parking_spaces: Json = Form(...), confidance: float = Form()):
    image_data = Image.open(io.BytesIO(await image.read()))
    parking_spaces_data = spaces_json_to_path(parking_spaces)
    return predict_frame(image_data, confidance, parking_spaces_data)

class WsClient:
    def __init__(self, websocket: WebSocket) -> None:
        self.ws = websocket
        self.task: Optional[asyncio.Task] = None
        self.stream_url = None
        self.parking_spaces = None
        self.confidence: float = None

connections: Dict[str, WsClient] = {}

fps = 1
time_per_frame = 1.0 / fps
executor = ThreadPoolExecutor(max_workers=4)

class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.lock = threading.Lock()
        executor.submit(self._reader)

    def _reader(self):
        while True:
            with self.lock:
                ret = self.cap.grab()
            if not ret:
                break

    def read(self):
        with self.lock:
            ret, frame = self.cap.retrieve()
        return (ret, frame)
    
    def release(self):
        self.cap.release()

async def process_video_stream(client_id, ws_client: WsClient):
    cap = VideoCapture(ws_client.stream_url)
    loop = asyncio.get_running_loop()
    try:
        while True:
            ret, frame = await loop.run_in_executor(executor, cap.read)
            if not ret:
                cap = VideoCapture(ws_client.stream_url)
                continue
            result = await loop.run_in_executor(executor, predict_frame, frame, ws_client.confidence, ws_client.parking_spaces)
            await ws_client.ws.send_text(json.dumps(result))
    finally:
        cap.release()


@app.websocket("/predict_stream")
async def predict_stream(websocket: WebSocket):
    await websocket.accept()
    client_id = uuid.uuid4()
    connections[client_id] = WsClient(websocket)
    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            connections[client_id].stream_url = data['stream_url']
            connections[client_id].parking_spaces = spaces_json_to_path(data['parking_spaces'])
            connections[client_id].confidence = data['confidence']

            if connections[client_id].task and not connections[client_id].task.done():
                canceled = connections[client_id].task.cancel()
            connections[client_id].task = asyncio.create_task(process_video_stream(client_id, connections[client_id]))
    except WebSocketDisconnect:
        if connections[client_id].task:
            canceled = connections[client_id].task.cancel()
        del connections[client_id]