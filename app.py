import asyncio
import json
import os
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

HF_API = "https://router.huggingface.co/hf-inference/models"
TOKEN = os.environ.get("HF_TOKEN", "")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

app = FastAPI(title="hf-proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


def get_token(request: Request) -> str:
    if TOKEN:
        return TOKEN
    h = request.headers.get("Authorization", "")
    if h.startswith("Bearer "):
        return h[7:].strip()
    return h.strip()


@app.get("/v1/healthz")
async def healthz():
    return {"status": "ok", "service": "hf-proxy", "upstream": HF_API}


@app.get("/v1/models")
async def list_models():
    models = [
        {"id": "Lightricks/LTX-Video", "name": "LTX-Video (recommande)"},
        {"id": "Wan-AI/Wan2.1-T2V-14B", "name": "Wan 2.1 T2V 14B"},
        {"id": "ali-vilab/text-to-video-ms-1.7b", "name": "Ali Vilab (leger)"},
    ]
    return {"object": "list", "data": models}


@app.post("/v1/videos")
async def create_video(request: Request):
    """
    Generation de video via HuggingFace Inference API.
    Body : { model, prompt, num_frames, fps, aspect_ratio }
    Reponse : { id, status } ou directement l'URL de la video
    """
    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse({"error": f"JSON invalide : {str(e)}"}, status_code=400)

    token = get_token(request)
    if not token:
        return JSONResponse({"error": "Token HuggingFace manquant"}, status_code=401)

    model = body.get("model", "Lightricks/LTX-Video")
    prompt = body.get("prompt", "")
    if not prompt:
        return JSONResponse({"error": "Prompt manquant"}, status_code=400)

    # Parametres
    params = {
        "num_frames": body.get("num_frames", 97),
        "fps": body.get("fps", 24),
    }
    if body.get("aspect_ratio"):
        params["aspect_ratio"] = body["aspect_ratio"]
    if body.get("num_inference_steps"):
        params["num_inference_steps"] = body["num_inference_steps"]

    url = f"{HF_API}/{model}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": UA,
        "x-wait-for-model": "true",
    }
    payload = {"inputs": prompt, "parameters": params}

    try:
        # Essayer 3 fois (cold start)
        for attempt in range(3):
            async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=20)) as client:
                r = await client.post(url, headers=headers, json=payload)

                if r.status_code == 200:
                    # Video renvoyee en binaire
                    content_type = r.headers.get("content-type", "")
                    if "video" in content_type or "octet" in content_type:
                        # On stocke en mémoire et on renvoie une URL de service
                        job_id = f"hf_{int(time.time()*1000)}"
                        _cache[job_id] = r.content
                        return {
                            "id": job_id,
                            "status": "succeeded",
                            "video_url": f"/v1/videos/{job_id}/content",
                            "content_type": content_type,
                        }
                    else:
                        # Reponse JSON (probablement une erreur ou attente)
                        try:
                            data = r.json()
                            return JSONResponse({"error": "HF a renvoye du JSON", "detail": data}, status_code=502)
                        except Exception:
                            return JSONResponse({"error": "Reponse inattendue HF"}, status_code=502)

                elif r.status_code == 503:
                    # Modele en chargement
                    try:
                        data = r.json()
                        wait = min(int(data.get("estimated_time", 30)), 60)
                    except Exception:
                        wait = 30
                    print(f"[hf] cold start, attente {wait}s", flush=True)
                    await asyncio.sleep(wait)
                    continue

                elif r.status_code == 401 or r.status_code == 403:
                    return JSONResponse({"error": "Token HuggingFace invalide"}, status_code=401)

                else:
                    err = r.text[:300]
                    return JSONResponse({"error": f"HF HTTP {r.status_code}", "detail": err}, status_code=r.status_code)

        return JSONResponse({"error": "Modele indisponible apres 3 tentatives"}, status_code=503)

    except Exception as e:
        return JSONResponse({"error": f"Erreur : {str(e)}"}, status_code=502)


_cache = {}


@app.get("/v1/videos/{job_id}")
async def get_video_status(job_id: str):
    if job_id in _cache:
        return {"id": job_id, "status": "succeeded", "video_url": f"/v1/videos/{job_id}/content"}
    return JSONResponse({"error": "Job introuvable"}, status_code=404)


@app.get("/v1/videos/{job_id}/content")
async def get_video_content(job_id: str):
    if job_id not in _cache:
        return JSONResponse({"error": "Contenu introuvable"}, status_code=404)
    content = _cache[job_id]
    return StreamingResponse(iter([content]), media_type="video/mp4")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
