from ultralytics import YOLO
import gradio as gr
from PIL import Image
import cv2

# 載入你訓練好的 YOLOv8 模型
model = YOLO("best.pt")  # 確保 best.pt 與此檔案在同一資料夾中

# 處理圖片辨識的函式
def detect_black_bear(image):
    results = model(image)[0]  # 取得第一個 batch 的推論結果
    annotated = results.plot()  # 以 OpenCV 繪製預測框 (BGR 格式)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)  # 轉為 RGB
    return Image.fromarray(annotated_rgb)  # 轉換為 PIL Image 並回傳

# 建立 Gradio 介面
app = gr.Interface(
    fn=detect_black_bear,
    inputs=gr.Image(type="pil"),
    outputs="image",
    title="台灣黑熊辨識 (YOLOv8)",
    description="上傳圖片，模型將標註出台灣黑熊"
)

if __name__ == "__main__":
    app.launch()
