# Flux Prompt Enhancer — FastAPI Service

Inference service cho model [`imranali291/flux-prompt-enhancer`](https://huggingface.co/imranali291/flux-prompt-enhancer).

## Endpoints

- `GET /health` — kiểm tra trạng thái + device.
- `POST /enhance` — enhance prompt.
  ```json
  {
    "prompt": "a cat on a chair",
    "max_new_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "do_sample": true,
    "seed": null
  }
  ```
  Response:
  ```json
  { "prompt": "...", "enhanced": "..." }
  ```

## Chạy local (Python)

```bash
cd fastapi-service
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # điền HF_TOKEN nếu model private
export $(cat .env | xargs)
python main.py
```

Mặc định service chạy ở `http://0.0.0.0:8000`. Lần đầu load sẽ tải model từ HuggingFace (vài GB), nên đợi `/health` trả 200 trước khi gọi `/enhance`.

## Deploy với Docker

```bash
cd fastapi-service
cp .env.example .env   # set HF_TOKEN nếu cần
docker compose up -d --build
docker compose logs -f
```

Test:

```bash
curl -X POST http://localhost:8000/enhance \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a cat on a chair"}'
```

## Deploy với GPU

1. Cài [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) trên server.
2. Uncomment block `deploy.resources` trong `docker-compose.yml`.
3. Đổi base image trong `Dockerfile` thành CUDA-enabled (ví dụ `nvidia/cuda:12.1.1-runtime-ubuntu22.04`) và cài `torch` build CUDA tương ứng, hoặc giữ image hiện tại — `pip install torch` mặc định đã có CUDA wheel.
4. Set `DEVICE=cuda` trong `.env`.

## Biến môi trường

| Var | Mặc định | Mô tả |
|---|---|---|
| `MODEL_ID` | `imranali291/flux-prompt-enhancer` | HF repo id |
| `HF_TOKEN` | _(empty)_ | Token nếu model private / để tránh rate limit |
| `DEVICE` | auto (`cuda` nếu có, else `cpu`) | `cuda` / `cpu` / `mps` |
| `HOST` | `0.0.0.0` | |
| `PORT` | `8000` | |

## Reverse proxy gợi ý (nginx)

```nginx
location /enhance {
    proxy_pass http://127.0.0.1:8000;
    proxy_read_timeout 120s;
    proxy_set_header Host $host;
}
```
