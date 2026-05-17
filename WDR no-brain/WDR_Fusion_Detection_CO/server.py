import json
import os
from pathlib import Path

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS


APP_DIR = Path(__file__).resolve().parent
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)


def load_local_env():
    env_path = APP_DIR / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env()

app = Flask(__name__)
CORS(app)


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "model": GEMINI_MODEL})


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    prompt = (payload.get("prompt") or "").strip()

    if not prompt:
        return jsonify({"error": "질문 내용이 비어 있습니다."}), 400

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return jsonify({
            "error": "GEMINI_API_KEY가 설정되어 있지 않습니다. .env 파일에 GEMINI_API_KEY=발급받은_키 를 추가하세요."
        }), 500

    gemini_payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "topP": 0.95,
            "maxOutputTokens": 1024,
        },
    }

    try:
        response = requests.post(
            GEMINI_API_URL,
            params={"key": api_key},
            headers={"Content-Type": "application/json"},
            data=json.dumps(gemini_payload, ensure_ascii=False).encode("utf-8"),
            timeout=60,
        )
        response.raise_for_status()
    except requests.Timeout:
        return jsonify({"error": "Gemini API 응답 시간이 초과되었습니다."}), 504
    except requests.HTTPError:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        return jsonify({"error": f"Gemini API 오류: {detail}"}), response.status_code
    except requests.RequestException as exc:
        return jsonify({"error": f"Gemini API 요청 실패: {exc}"}), 502

    data = response.json()
    candidates = data.get("candidates") or []
    parts = (
        candidates[0]
        .get("content", {})
        .get("parts", [])
        if candidates
        else []
    )
    answer = "".join(part.get("text", "") for part in parts).strip()

    if not answer:
        return jsonify({"error": "Gemini가 빈 응답을 반환했습니다."}), 502

    return jsonify({"response": answer})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
