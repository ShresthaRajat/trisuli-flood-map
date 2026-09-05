#!/usr/bin/env python3
"""Local server for editor.html: serves the project folder and accepts saves.

    python3 scripts/editor_server.py          # then open http://127.0.0.1:8790/editor.html

POST /save/rivers   body = rivers.json content  -> writes assets/trisuli/rivers.json
POST /save/flood    body = flood.json content   -> writes assets/trisuli/flood.json
                                                  (previous file kept as <name>.backup.json)
GET  anything else  -> static file from the project root (no caching, so edits show on reload)
"""
import http.server, json, os, shutil, sys
from functools import partial

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = {"/save/rivers": (os.path.join(ROOT, "assets", "trisuli", "rivers.json"), ("polys", "lines", "labels")),
           "/save/flood":  (os.path.join(ROOT, "assets", "trisuli", "flood.json"), ("path",))}

class Handler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.0"                  # one request per connection keeps error replies simple
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)                      # always drain the body before answering
        spec = TARGETS.get(self.path)
        if not spec:
            self.send_error(404, "unknown save target"); return
        target, required = spec
        try:
            data = json.loads(body)
            for k in required:
                if not isinstance(data.get(k), list): raise ValueError("missing list: " + k)
        except Exception as e:
            self.send_error(400, "bad json: %s" % e); return
        if os.path.exists(target):
            shutil.copyfile(target, target.replace(".json", ".backup.json"))
        with open(target, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        out = json.dumps({"ok": True, "bytes": os.path.getsize(target), "counts": {k: len(data[k]) for k in required}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(out))); self.end_headers()
        self.wfile.write(out)
        print("saved", target, len(body), "bytes", flush=True)

    def log_message(self, fmt, *args):
        line = fmt % args if args else fmt
        if "POST" in line or "code" in line: super().log_message(fmt, *args)   # quiet for static GETs

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8790
    os.chdir(ROOT)
    print(f"serving {ROOT} on http://127.0.0.1:{port}/editor.html  (Ctrl-C to stop)", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=ROOT)).serve_forever()
