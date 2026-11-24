"""
tools/capture_screenshots.py
============================
Capture full-page screenshots of the dashboard with headless Chrome driven over the DevTools protocol.

Usage (the app must already be running, e.g. `streamlit run app.py --server.port 8501`):

    pip install -r requirements-dev.txt            # websockets
    python tools/capture_screenshots.py            # -> docs/screenshots/*.png
    python tools/capture_screenshots.py --base http://localhost:8501 --out docs/screenshots

Set CHROME_PATH if Chrome is not in a standard location.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parent.parent
PORT = 9371

# (file name, URL path, optional action run after the page settles)
PAGES = [
    ("home", "/", None),
    ("score-order", "/score-order", "preset-high"),
    ("real-orders", "/real-orders", None),
    ("impact", "/impact", None),
    ("performance", "/performance", None),
    ("robustness", "/robustness", None),
    ("limitations", "/limitations", None),
    ("batch", "/batch", "batch-sample"),
]

WAIT_JS = ("(() => !document.querySelector('[data-testid=\"stStatusWidget\"]') && "
           "!document.querySelector('[class*=\"stSkeleton\"]') && document.querySelector('.rto-proxy') !== null)()")
HEIGHT_JS = "Math.ceil((document.querySelector('[data-testid=\"stMain\"]') || document.body).scrollHeight)"
TEXT_JS = ("(() => { const m = document.querySelector('[data-testid=\"stMain\"]') || document.body; "
           "const r = document.querySelector('[data-testid=\"stRadio\"] input:checked'); "
           "return JSON.stringify({text: m.innerText.slice(0, 8000), "
           "radio: r ? (r.closest('label') || {}).innerText || null : null}); })()")
AUDIT_JS = r"""(() => {
  const cards = [...document.querySelectorAll('[data-testid="stVerticalBlock"]')].filter(e => { const cs = getComputedStyle(e); return parseFloat(cs.borderTopWidth) > 0 && cs.borderTopStyle !== 'none'; });
  const empty = cards.filter(c => !c.innerText.trim() && !c.querySelector('canvas,svg,img,iframe,.js-plotly-plot,[data-testid="stDataFrame"],[data-testid="stTable"],table,input,button'));
  const exc = [...document.querySelectorAll('[data-testid="stException"]')].length;
  return JSON.stringify({empty_cards: empty.length, exceptions: exc});
})()"""


def find_chrome() -> str:
    candidates = [os.environ.get("CHROME_PATH", ""), shutil.which("chrome") or "", shutil.which("google-chrome") or "",
                  shutil.which("chromium") or "", r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                  r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe", "/usr/bin/google-chrome",
                  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise SystemExit("Chrome not found: set CHROME_PATH to the browser executable.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://localhost:8501")
    ap.add_argument("--out", default=str(ROOT / "docs" / "screenshots"))
    ap.add_argument("--width", type=int, default=1400)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    profile = tempfile.mkdtemp(prefix="rto_chrome_")
    proc = subprocess.Popen([find_chrome(), "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--remote-debugging-port={PORT}",
                             f"--window-size={args.width},1000", f"--user-data-dir={profile}", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ws_url = None
        for _ in range(80):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://localhost:{PORT}/json"))
                ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.5)
        if ws_url is None:
            raise SystemExit("could not attach to headless Chrome")
        ws = connect(ws_url, max_size=None)
        counter = {"id": 0}

        def cdp(method, **params):
            counter["id"] += 1
            ws.send(json.dumps({"id": counter["id"], "method": method, "params": params}))
            while True:
                msg = json.loads(ws.recv())
                if msg.get("id") == counter["id"]:
                    return msg.get("result", {})

        def js(expr):
            return cdp("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True).get("result", {}).get("value")

        def settle(timeout=120):
            t0 = time.time()
            time.sleep(1.0)
            while time.time() - t0 < timeout:
                if js(WAIT_JS):
                    time.sleep(1.2)
                    if js(WAIT_JS):
                        return True
                time.sleep(0.5)
            return False

        def click(selector, text):
            return js("(() => { const e = [...document.querySelectorAll(%s)].find(e => e.innerText.includes(%s)); if (e) { e.click(); return true; } return false; })()"
                      % (json.dumps(selector), json.dumps(text)))

        cdp("Page.enable")
        cdp("Runtime.enable")
        ok = True
        manifest = {"captured": time.strftime("%Y-%m-%d %H:%M:%S"), "base": args.base, "width": args.width, "pages": {}}
        for name, path, action in PAGES:
            cdp("Emulation.setDeviceMetricsOverride", width=args.width, height=1000, deviceScaleFactor=1, mobile=False)
            cdp("Page.navigate", url=args.base.rstrip("/") + path)
            settled = settle()
            if action == "preset-high":
                click('[data-testid="stRadio"] label', "Real high-risk")
                settle(60)
            elif action == "batch-sample":
                click("button", "with target")
                settle(120)
            height = min(int(js(HEIGHT_JS) or 1000) + 40, 9000)
            cdp("Emulation.setDeviceMetricsOverride", width=args.width, height=height, deviceScaleFactor=1, mobile=False)
            time.sleep(1.5)
            height2 = min(int(js(HEIGHT_JS) or height) + 40, 9000)
            if height2 > height:
                cdp("Emulation.setDeviceMetricsOverride", width=args.width, height=height2, deviceScaleFactor=1, mobile=False)
                time.sleep(1.2)
            (out / f"{name}.png").write_bytes(base64.b64decode(cdp("Page.captureScreenshot", format="png")["data"]))
            audit = json.loads(js(AUDIT_JS))
            good = settled and audit["empty_cards"] == 0 and audit["exceptions"] == 0
            ok &= good
            manifest["pages"][name] = {"file": f"{name}.png", "path": path, "action": action, **json.loads(js(TEXT_JS))}
            print(f"{'OK ' if good else 'BAD'} {name:<12} -> {out / (name + '.png')}  {audit}")
        ws.close()
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
        if not ok:
            raise SystemExit("some pages did not render cleanly; screenshots are NOT trustworthy")
    finally:
        proc.terminate()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
