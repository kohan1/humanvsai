"""
visualiser_server.py
--------------------
Tiny WebSocket server that hooks into the Tetris training loop
and broadcasts all 32 environment states to the browser in real time.

Run this in a separate terminal while training:
    python3 visualiser_server.py

Then open visualiser.html in your browser.

It reads environment state from a shared file that train.py writes to.
No changes to train.py needed — uses a file-based approach so it's
completely decoupled from training.
"""

import asyncio
import json
import os
import time
import threading
import websockets

HOST = "localhost"
PORT = 8765
STATE_FILE = "vis_state.json"  # train.py writes this, we read it

connected_clients = set()

async def broadcast_loop():
    """Read state file and broadcast to all connected clients."""
    last_mtime = 0
    while True:
        try:
            if os.path.exists(STATE_FILE):
                mtime = os.path.getmtime(STATE_FILE)
                if mtime != last_mtime:
                    last_mtime = mtime
                    with open(STATE_FILE, "r") as f:
                        data = f.read()
                    if connected_clients:
                        await asyncio.gather(
                            *[client.send(data) for client in connected_clients],
                            return_exceptions=True
                        )
        except Exception as e:
            pass
        await asyncio.sleep(0.05)  # 20fps max

async def handler(websocket):
    connected_clients.add(websocket)
    print(f"Client connected. Total: {len(connected_clients)}")
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.discard(websocket)
        print(f"Client disconnected. Total: {len(connected_clients)}")

async def main():
    print(f"Visualiser server starting on ws://{HOST}:{PORT}")
    print(f"Open visualiser.html in your browser.")
    print(f"Waiting for training to start writing {STATE_FILE}...")
    async with websockets.serve(handler, HOST, PORT):
        await broadcast_loop()

if __name__ == "__main__":
    asyncio.run(main())
