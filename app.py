from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge
from pathlib import Path
from PIL import Image
import io, torch, torch.nn.functional as F
import numpy as np
import onnx
from onnxsim import simplify
import onnxruntime as ort
from model import ViT, transform_medical, get_class_mapping

overall_accuracy_gauge = Gauge('model_overall_accuracy', 'Overall accuracy of the model', ['dataset'])
accuracy_per_class_gauge = Gauge('model_accuracy_per_class', 'Accuracy per class', ['dataset', 'class_name'])

app = FastAPI()
Instrumentator().instrument(app).expose(app, endpoint="/metrics")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

model = None
session = None
class_names = None
mapping = None

@app.on_event("startup")
def load_and_export():
    global model, session, class_names, mapping
    model = ViT(image_size=64, patch_size=16, num_classes=9, channels=1, dim=64, depth=16, heads=16, mlp_dim=256, emb_dropout=0.0, dropout=0.0)
    ckpt = torch.load("model.pth", map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    clean = {k.replace("model.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean, strict=True)
    model.eval()

    onnx_path = Path("model.onnx")
    simplified_path = Path("model_simplified.onnx")
    if not simplified_path.exists():
        dummy = torch.randn(1, 1, 64, 64)
        torch.onnx.export(model, dummy, str(onnx_path), opset_version=13, input_names=["input"], output_names=["logits"], dynamic_axes={"input":{0:"batch"}, "logits":{0:"batch"}})
        m = onnx.load(str(onnx_path))
        msimp, check = simplify(m)
        onnx.save(msimp, str(simplified_path))

    session = ort.InferenceSession(str(simplified_path), providers=['CPUExecutionProvider'])

    mapping = get_class_mapping("/mnt/object/train")
    class_names = [None] * len(mapping)
    for name, idx in mapping.items():
        class_names[idx] = name

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("L")
    tensor = transform_medical(img).unsqueeze(0).cpu().numpy().astype(np.float32)
    outputs = session.run(['logits'], {'input': tensor})[0]
    probs = F.softmax(torch.from_numpy(outputs), dim=1)[0].cpu().tolist()
    idx = int(np.argmax(probs))
    return JSONResponse({
        "pred_idx": idx,
        "pred_class": class_names[idx],
        "probabilities": dict(zip(class_names, probs))
    })

def run_evaluation(dataset: str):
    dataset_dir = Path(f"/mnt/object/{dataset}")
    total_correct = 0
    total_images = 0
    correct_per_class = {c: 0 for c in class_names}
    total_per_class = {c: 0 for c in class_names}
    VALID_EXTS = {".png", ".jpg", ".jpeg"}
    for class_dir in dataset_dir.iterdir():
        if not class_dir.is_dir(): continue
        name = class_dir.name
        if name not in mapping: continue
        for p in class_dir.iterdir():
            if not p.is_file() or p.suffix.lower() not in VALID_EXTS: continue
            try:
                img = Image.open(p).convert("L")
                arr = transform_medical(img).unsqueeze(0).cpu().numpy().astype(np.float32)
                out = session.run(['logits'], {'input': arr})[0]
                probs = F.softmax(torch.from_numpy(out), dim=1)[0]
                pred = int(torch.argmax(probs))
                total_images += 1
                if pred == mapping[name]:
                    total_correct += 1
                    correct_per_class[name] += 1
                total_per_class[name] += 1
            except Exception as e:
                print(f"Error {p.name}: {e}")
    if total_images > 0:
        oa = total_correct / total_images
        overall_accuracy_gauge.labels(dataset=dataset).set(oa)
        for cname in class_names:
            acc = correct_per_class[cname] / total_per_class[cname] if total_per_class[cname] > 0 else 0.0
            accuracy_per_class_gauge.labels(dataset=dataset, class_name=cname).set(acc)
    print(f"Evaluation completed for {dataset}: overall_accuracy={oa if total_images>0 else 0}")

@app.get("/evaluate")
async def evaluate(dataset: str = "final_eval", background_tasks: BackgroundTasks = None):
    background_tasks.add_task(run_evaluation, dataset)
    return {"status": f"evaluation started for {dataset}"}

@app.get("/health")
def health():
    return {"status": "ok"}
