from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io, torch, torch.nn.functional as F
from model import ViT, transform_medical, get_class_mapping
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge
from fastapi import BackgroundTasks
from pathlib import Path

overall_accuracy_gauge = Gauge(
    'model_overall_accuracy',
    'Overall accuracy of the model',
    ['dataset']
)
accuracy_per_class_gauge = Gauge(
    'model_accuracy_per_class',
    'Accuracy per class',
    ['dataset', 'class_name']
)
model = None
class_names = None
mapping = None
app = FastAPI()
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

model = None
class_names = None

@app.on_event("startup")
def load_model():
    global model, class_names, mapping
    model = ViT(
        image_size=64, patch_size=16, num_classes=9,
        channels=1, dim=64, depth=16, heads=16, mlp_dim=256,
        emb_dropout=0.0, dropout=0.0
    )
    ckpt = torch.load("model.pth", map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    clean = {k.replace("model.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean, strict=True)
    model.eval()
    mapping = get_class_mapping("/mnt/object/train")
    class_names = [None] * len(mapping)
    for name, idx in mapping.items():
        class_names[idx] = name

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("L")
    tensor = transform_medical(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().tolist()
    idx = int(probs.index(max(probs)))
    return JSONResponse({
        "pred_idx": idx,
        "pred_class": class_names[idx],
        "probabilities": dict(zip(class_names, probs))
    })

def run_evaluation(dataset: str):
    dataset_dir = Path(f"/mnt/object/{dataset}")
    if not dataset_dir.exists():
        print(f"Dataset {dataset} not found")
        return

    total_correct = 0
    total_images = 0
    correct_per_class = {class_name: 0 for class_name in class_names}
    total_per_class = {class_name: 0 for class_name in class_names}

    for class_dir in dataset_dir.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            if class_name not in mapping:
                continue
            true_idx = mapping[class_name]
            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    try:
                        img = Image.open(image_path).convert("L")
                        tensor = transform_medical(img).unsqueeze(0)
                        with torch.no_grad():
                            logits = model(tensor)
                            probs = F.softmax(logits, dim=1)[0]
                            pred_idx = torch.argmax(probs).item()
                        if pred_idx == true_idx:
                            total_correct += 1
                            correct_per_class[class_name] += 1
                        total_images += 1
                        total_per_class[class_name] += 1
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")

    if total_images > 0:
        overall_accuracy = total_correct / total_images
    else:
        overall_accuracy = 0

    accuracy_per_class = {}
    for class_name in class_names:
        if total_per_class[class_name] > 0:
            accuracy_per_class[class_name] = correct_per_class[class_name] / total_per_class[class_name]
        else:
            accuracy_per_class[class_name] = 0

    overall_accuracy_gauge.labels(dataset=dataset).set(overall_accuracy)
    for class_name, acc in accuracy_per_class.items():
        accuracy_per_class_gauge.labels(dataset=dataset, class_name=class_name).set(acc)

    print(f"Evaluation completed for {dataset}: overall_accuracy={overall_accuracy}")
@app.get("/evaluate_sync")
def evaluate_sync(dataset: str = "final_eval"):
    run_evaluation(dataset)
    return {"status": f"completed {dataset}"}
@app.get("/evaluate")
async def evaluate(dataset: str = "final_eval", background_tasks: BackgroundTasks = None):
    background_tasks.add_task(run_evaluation, dataset)
    return {"status": f"evaluation started for {dataset}"}

@app.get("/count_files")
def count_files(dataset: str = "final_eval"):
    dataset_dir = Path(f"/mnt/object/{dataset}")
    VALID_EXTS = {".png", ".jpg", ".jpeg"}
    counts = {}
    for class_dir in dataset_dir.iterdir():
        if class_dir.is_dir():
            counts[class_dir.name] = sum(
                1 for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in VALID_EXTS
            )
    return JSONResponse(counts)

@app.get("/health")
def health():
    return {"status": "ok"}