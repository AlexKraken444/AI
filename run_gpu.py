"""Start the downloaded Qwen model and the local website; Ctrl+C stops both."""
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
STATE = ROOT / ".gpu-state"


def main():
    STATE.mkdir(exist_ok=True)
    executable = ROOT / ".local-runtime/llama/llama-server.exe"
    model = ROOT / ".local-models/Qwen3-4B-Q4_K_M.gguf"
    if not executable.exists() or not model.exists():
        raise SystemExit("Download the model/runtime first: see GPU_SETUP.md")
    keyfile = STATE / "api-key"
    if not keyfile.exists():
        keyfile.write_text(secrets.token_urlsafe(48), encoding="utf-8")
    env = os.environ.copy()
    env["PATH"] = str(ROOT / ".training-libs/torch/lib") + os.pathsep + str(executable.parent) + os.pathsep + env["PATH"]
    env["GPU_CHAT_URL"] = "http://127.0.0.1:8081"
    env["GPU_CHAT_KEY"] = keyfile.read_text(encoding="utf-8").strip()
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    children = []
    try:
        with (STATE / "model.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([str(executable), "-m", str(model), "--host", "127.0.0.1", "--port", "8081",
                "--alias", "kraken", "-c", "8192", "-np", "1", "-ngl", "99", "--jinja",
                "--reasoning", "off", "--no-webui", "--api-key-file", str(keyfile)],
                cwd=ROOT, env=env, stdout=log, stderr=log, creationflags=flags)
            children.append(process)
            (STATE / "model.pid").write_text(str(process.pid))
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError("GPU server exited; see .gpu-state/model.log")
                try:
                    with urlopen(env["GPU_CHAT_URL"] + "/health", timeout=1) as health:
                        if health.status == 200:
                            break
                except OSError:
                    time.sleep(1)
            else:
                raise RuntimeError("GPU loading timed out; see .gpu-state/model.log")
            with (STATE / "web.log").open("w", encoding="utf-8") as weblog:
                web = subprocess.Popen([sys.executable, "-u", str(ROOT / "dev.py")], cwd=ROOT, env=env,
                                       stdout=weblog, stderr=weblog, creationflags=flags)
                children.append(web)
                (STATE / "web.pid").write_text(str(web.pid))
                print("Qwen GPU ready. Website: http://127.0.0.1:8000/", flush=True)
                while all(child.poll() is None for child in children):
                    time.sleep(1)
                raise RuntimeError("A server exited; check .gpu-state logs (ports may already be in use)")
    except KeyboardInterrupt:
        pass
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=15)


if __name__ == "__main__":
    main()
