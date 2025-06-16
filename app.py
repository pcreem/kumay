from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image
import io
import cv2

app = FastAPI()

model = YOLO("best.pt")  # 確保 best.pt 在同一資料夾中

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    results = model(image)[0]
    annotated = results.plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    output_image = Image.fromarray(annotated_rgb)

    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
