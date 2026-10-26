from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from classifier import Classifier

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["clf"] = Classifier()      # load once at startup
    yield
    state.clear()


app = FastAPI(title="Paper Topic Classifier", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],     # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)
    title: str | None = Field(None, max_length=500)


class Prediction(BaseModel):
    label: str
    score: float
    predicted: bool


class PredictResponse(BaseModel):
    source: str
    title: str | None
    abstract: str
    extraction_confidence: str
    predictions: list[Prediction]


@app.get("/health")
def health():
    return {"status": "ok", "device": state["clf"].device,
            "n_labels": len(state["clf"].labels)}


@app.post("/predict/text", response_model=PredictResponse)
def predict_text(req: TextRequest):
    abstract = " ".join(req.text.split())
    if len(abstract.split()) < 20:
        raise HTTPException(422, "Text is too short. Paste at least a few sentences "
                                 "of the abstract.")
    preds = state["clf"].predict(req.title, abstract)
    return {"source": "text", "title": req.title, "abstract": abstract,
            "extraction_confidence": "high", "predictions": preds}