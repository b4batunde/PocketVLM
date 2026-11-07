"""On-device demo of the recommended configuration (or any GGUF you name).

Interactive (local web page: upload an image, ask, see answer + latency):
    python scripts/10_demo.py
One-shot from the terminal:
    python scripts/10_demo.py --image photo.jpg --question "What color is the bus?"
Pick a specific variant instead of the recommended one:
    python scripts/10_demo.py --model ft-Q2_K.gguf --mmproj mmproj-ft-Q8_0.gguf

Everything runs locally; nothing leaves the machine.
"""
import argparse
import base64
import json
import mimetypes
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import requests

from vlmc.config import load_config
from vlmc.results import recommended
from vlmc.vqa_metric import clean_prediction

ap = argparse.ArgumentParser()
ap.add_argument("--config")
ap.add_argument("--model", help="LLM GGUF (default: recommended variant from results)")
ap.add_argument("--mmproj")
ap.add_argument("--lora")
ap.add_argument("--image", nargs="*", help="one or more images for one-shot mode")
ap.add_argument("--question")
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--no-browser", action="store_true")
ap.add_argument("--ngl", default="99")
a = ap.parse_args()
cfg = load_config(a.config)
gdir = Path(cfg["work_dir"]) / "gguf"
LCPP = Path(os.environ.get("LLAMA_CPP", "llama.cpp"))
BIN = Path(os.environ.get("LLAMA_BIN", LCPP / "build" / "bin"))


def resolve(p):
    return p if p is None or Path(p).exists() else str(gdir / Path(p).name)


PAGE = (Path(__file__).resolve().parent.parent / "demo" / "index.html").read_bytes()


def serve_page(upstream, label, size_gb):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    info = json.dumps({"label": label, "size_gb": size_gb, "suffix": cfg["prompt_suffix"]}).encode()

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/demo-info":
                self._send(200, info, "application/json")
            elif self.path in ("/", "/index.html"):
                self._send(200, PAGE, "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self._send(404, b"not found", "text/plain")
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            r = requests.post(upstream + self.path, data=body, headers={"Content-Type": "application/json"}, timeout=600)
            self._send(r.status_code, r.content, "application/json")

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    url = f"http://127.0.0.1:{a.port}"
    print(f"\nDemo running at {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        webbrowser.open(url)
    httpd.serve_forever()


if a.model:
    model, mmproj, lora, label = resolve(a.model), resolve(a.mmproj), resolve(a.lora), Path(a.model).stem
    if not mmproj:
        sys.exit("--mmproj is required with --model")
else:
    rec = recommended(cfg)
    if rec is None:
        sys.exit("No recommended variant yet (run the eval sweep and 09_report.py), or pass --model/--mmproj.")
    f = rec["files"]
    model, mmproj, lora = resolve(f["llm"]), resolve(f["mmproj"]), resolve(f.get("lora"))
    label = rec["name"]
size_gb = sum(Path(p).stat().st_size for p in (model, mmproj, lora) if p) / 1e9
print(f"Demo model: {label}  ({size_gb:.2f} GB on disk)\n  llm:    {model}\n  mmproj: {mmproj}"
      + (f"\n  lora:   {lora}" if lora else ""))

with socket.socket() as s:
    if s.connect_ex(("127.0.0.1", a.port)) == 0:
        sys.exit(f"port {a.port} is busy; pass --port")
with socket.socket() as s:   # llama-server listens on a private port; the demo page proxies to it
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
cmd = [str(BIN / "llama-server"), "-m", model, "--mmproj", mmproj, "--port", str(port), "-ngl", a.ngl,
       "--no-webui",
       "-c", "4096", "--temp", "0",
       "--image-max-tokens", str(cfg["image"]["max_pixels"] // 784),
       "--image-min-tokens", str(cfg["image"]["min_pixels"] // 784)]
if lora:
    cmd += ["--lora", lora]
srv = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
base = f"http://127.0.0.1:{port}"
try:
    t0 = time.time()
    while True:
        if srv.poll() is not None:
            sys.exit("llama-server exited during startup; run the command by hand to see why:\n" + " ".join(cmd))
        try:
            if requests.get(base + "/health", timeout=1).status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(0.3)
    print(f"Loaded in {time.time() - t0:.1f}s")

    if a.image:
        q = a.question or "Describe this image."
        for img in a.image:
            mime = mimetypes.guess_type(img)[0] or "image/jpeg"
            b64 = base64.b64encode(open(img, "rb").read()).decode()
            short = a.question is not None  # VQA-style short answers when a question is given
            text = f"{q}\n{cfg['prompt_suffix']}" if short else q
            body = {"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": text}]}],
                "max_tokens": cfg["eval"]["max_new_tokens"] if short else 256, "temperature": 0}
            t = time.time()
            j = requests.post(base + "/v1/chat/completions", json=body, timeout=600).json()
            ans = j["choices"][0]["message"]["content"]
            tm = j.get("timings", {})
            print(f"\n[{img}] Q: {q}\nA: {clean_prediction(ans) if short else ans.strip()}")
            print(f"   {time.time() - t:.2f}s end-to-end | prefill {tm.get('prompt_n', '?')} tok "
                  f"in {tm.get('prompt_ms', 0):.0f} ms | decode {tm.get('predicted_per_second', 0):.1f} tok/s")
    else:
        serve_page(base, label, size_gb)
except KeyboardInterrupt:
    pass
finally:
    srv.terminate()
