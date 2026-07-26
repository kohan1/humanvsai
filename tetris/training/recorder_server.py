"""
recorder_server.py
------------------
Tiny HTTP server that receives gameplay data from the website
and saves it to recordings.json.

Run this while playing:
    python recorder_server.py

Then open the website and play normally.
Every piece you place gets recorded automatically.
"""

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler

RECORDINGS_FILE = "recordings.json"

class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        if self.path == "/record":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                data = json.loads(body)

                # Load existing recordings
                if os.path.exists(RECORDINGS_FILE):
                    with open(RECORDINGS_FILE, "r") as f:
                        recordings = json.load(f)
                else:
                    recordings = []

                recordings.append(data)

                with open(RECORDINGS_FILE, "w") as f:
                    json.dump(recordings, f)

                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "total": len(recordings)}).encode())

            except Exception as e:
                self.send_response(500)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(str(e).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Only log every 50 recordings to keep terminal clean
        if "200" in str(args):
            pass
        else:
            super().log_message(format, *args)


if __name__ == "__main__":
    # Clear old recordings on start
    if os.path.exists(RECORDINGS_FILE):
        resp = input(f"Found existing {RECORDINGS_FILE} with recordings. Append to it? (y/n): ")
        if resp.lower() != "y":
            os.remove(RECORDINGS_FILE)
            print("Cleared old recordings.")

    print("=" * 50)
    print("  Gameplay Recorder")
    print("=" * 50)
    print(f"\nRecording to: {RECORDINGS_FILE}")
    print("Open http://localhost:8080 and play on the LEFT board.")
    print("Every piece placement is recorded automatically.")
    print("Play as many games as you want — Ctrl+C when done.\n")

    server = HTTPServer(("localhost", 8766), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if os.path.exists(RECORDINGS_FILE):
            with open(RECORDINGS_FILE, "r") as f:
                recordings = json.load(f)
            print(f"\n\nDone! Recorded {len(recordings)} placements.")
            print("Now run: python pretrain.py")
        else:
            print("\nNo recordings saved.")
