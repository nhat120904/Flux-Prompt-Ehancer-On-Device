import os
import logging
from contextlib import asynccontextmanager
from typing import Literal, Optional

import httpx
import torch
from langdetect import DetectorFactory, LangDetectException, detect
try:
    torch.backends.python_native.disable_dispatch_keys("CUDA")
except AttributeError:
    pass
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prompt-enhancer")

MODEL_ID = os.getenv("MODEL_ID", "imranali291/flux-prompt-enhancer")
HF_TOKEN = os.getenv("HF_TOKEN") or None
DEVICE = os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
ROOT_PATH = os.getenv("ROOT_PATH", "")
TRANSLATION_API_URL = os.getenv(
    "TRANSLATION_API_URL",
    "https://ai-app-studio.var-meta.com/ai-summary/api/translate",
)
TRANSLATION_API_KEY = os.getenv("TRANSLATION_API_KEY", "").strip()
TRANSLATION_TIMEOUT_SECONDS = float(os.getenv("TRANSLATION_TIMEOUT_SECONDS", "30"))

LanguageCode = Literal[
    "en", "es", "de", "fr", "id", "it", "nl", "pt-BR", "pt-PT", "vi",
    "tr", "ru", "ar", "hi", "th", "zh-Hans", "zh-Hant", "ja", "ko",
]

# langdetect returns ISO 639 codes. The translation API uses these supported codes;
# Portuguese defaults to pt-BR when the detector cannot identify a region.
LANGDETECT_LANGUAGE_CODES = {
    "en": "en",
    "es": "es",
    "de": "de",
    "fr": "fr",
    "id": "id",
    "it": "it",
    "nl": "nl",
    "pt": "pt-BR",
    "tr": "tr",
    "ru": "ru",
    "ar": "ar",
    "hi": "hi",
    "th": "th",
    "zh": "zh-Hans",
    "zh-cn": "zh-Hans",
    "zh-tw": "zh-Hant",
    "ja": "ja",
    "ko": "ko",
    "vi": "vi",
}

DetectorFactory.seed = 0

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model %s on %s (%s)", MODEL_ID, DEVICE, DTYPE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, use_fast=False)

    config = AutoConfig.from_pretrained(MODEL_ID, token=HF_TOKEN)
    is_seq2seq = bool(getattr(config, "is_encoder_decoder", False))
    model_cls = AutoModelForSeq2SeqLM if is_seq2seq else AutoModelForCausalLM
    logger.info("Detected %s model", "seq2seq" if is_seq2seq else "causal-LM")

    model = model_cls.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        torch_dtype=DTYPE,
        attn_implementation="eager",
    ).to(DEVICE)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    state["tokenizer"] = tokenizer
    state["model"] = model
    state["is_seq2seq"] = is_seq2seq
    logger.info("Model loaded.")
    yield
    state.clear()


app = FastAPI(title="Flux Prompt Enhancer", lifespan=lifespan, root_path=ROOT_PATH)


class EnhanceRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Original prompt in a supported language")
    input_language: Optional[LanguageCode] = Field(
        None,
        description="Optional source language override; detected automatically when omitted",
    )
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


def resolve_input_language(prompt: str, input_language: Optional[str]) -> str:
    if input_language is not None:
        return input_language

    try:
        detected_language = detect(prompt)
    except LangDetectException as exc:
        raise HTTPException(
            status_code=422,
            detail="Could not detect the prompt language. Pass input_language explicitly.",
        ) from exc

    language = LANGDETECT_LANGUAGE_CODES.get(detected_language.lower())
    if language is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "Detected language is not supported. Pass input_language using a supported "
                "language code."
            ),
        )
    return language


def translate_text(text: str, input_language: str, language: str) -> str:
    if not TRANSLATION_API_KEY:
        raise HTTPException(status_code=503, detail="Translation service is not configured")
    if len(text) > 5000:
        raise HTTPException(
            status_code=422,
            detail="The translation service accepts text up to 5000 characters.",
        )

    try:
        response = httpx.post(
            TRANSLATION_API_URL,
            headers={"x-api-key": TRANSLATION_API_KEY},
            json={"text": text, "inputLanguage": input_language, "language": language},
            timeout=TRANSLATION_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        logger.error(
            "Translation API request failed (%s -> %s): %s",
            input_language,
            language,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=502,
            detail="Translation service request failed",
        ) from exc
    except ValueError as exc:
        logger.error("Translation API returned invalid JSON (%s -> %s)", input_language, language)
        raise HTTPException(
            status_code=502,
            detail="Translation service returned an invalid response",
        ) from exc

    translation = payload.get("translation") if isinstance(payload, dict) else None
    if not isinstance(translation, str) or not translation.strip():
        logger.error(
            "Translation API response is missing translation (%s -> %s)",
            input_language,
            language,
        )
        raise HTTPException(
            status_code=502,
            detail="Translation service returned an invalid response",
        )
    return translation.strip()


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

    input_language = resolve_input_language(req.prompt, req.input_language)
    english_prompt = (
        req.prompt
        if input_language == "en"
        else translate_text(req.prompt, input_language, "en")
    )

    inputs = tokenizer(english_prompt, return_tensors="pt").to(DEVICE)
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
    if state.get("is_seq2seq"):
        generated = output[0]
    else:
        generated = output[0][inputs["input_ids"].shape[1]:]
    english_enhanced = tokenizer.decode(generated, skip_special_tokens=True).strip()
    enhanced = (
        english_enhanced
        if input_language == "en"
        else translate_text(english_enhanced, "en", input_language)
    )
    return EnhanceResponse(prompt=req.prompt, enhanced=enhanced)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        workers=1,
    )
