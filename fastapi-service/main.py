import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prompt-enhancer")

MODEL_ID = os.getenv("MODEL_ID", "imranali291/flux-prompt-enhancer")
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model %s on %s (%s)", MODEL_ID, DEVICE, DTYPE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    state["tokenizer"] = tokenizer
    state["model"] = model
    logger.info("Model loaded.")
    yield
    state.clear()


app = FastAPI(title="Flux Prompt Enhancer", lifespan=lifespan)


class EnhanceRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Original short prompt")
    max_new_tokens: int = Field(128, ge=8, le=1024)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0, le=1000)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    do_sample: bool = True
    seed: Optional[int] = None


class EnhanceResponse(BaseModel):
    prompt: str
    enhanced: str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}


@app.post("/enhance", response_model=EnhanceResponse)
@torch.inference_mode()
def enhance(req: EnhanceRequest):
    tokenizer = state.get("tokenizer")
    model = state.get("model")
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if req.seed is not None:
        torch.manual_seed(req.seed)
        if DEVICE == "cuda":
            torch.cuda.manual_seed_all(req.seed)

    inputs = tokenizer(req.prompt, return_tensors="pt").to(DEVICE)
    output = model.generate(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty,
        do_sample=req.do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = output[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return EnhanceResponse(prompt=req.prompt, enhanced=text)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        workers=1,
    )
