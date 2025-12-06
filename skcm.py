#!/usr/bin/env python3
"""
voice_peer_webrtc.py
Single-file aiortc-based full-duplex voice P2P using sounddevice for I/O.
Signaling: direct TCP connection (no external server). Use Tailscale IPs.
Usage (terminal):
# On one machine (server/listen):
python voice_peer_webrtc.py --listen --port 9000
# On the other machine (client, connect to server Tailscale IP):
python voice_peer_webrtc.py --peer-ip 100.101.102.103 --peer-port 9000

Requirements:
pip install aiortc sounddevice av
(If av install fails, on Debian/Ubuntu: sudo apt install libavformat-dev libavcodec-dev libavutil-dev libavdevice-dev python3-dev)
"""
from __future__ import annotations
import argparse
import asyncio
import json
import socket
import sys
import time
from typing import Optional

import numpy as np
import sounddevice as sd
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate
from aiortc.contrib.signaling import object_from_string, object_to_string
import av

# Audio constants (match previous)
SR = 48000
FRAME_SIZE = 480            # samples per frame (10ms @48kHz)
CHANNELS = 1
SAMPLES_PER_FRAME = FRAME_SIZE

# Sounddevice queue sizes
MIC_QUEUE_MAX = 100

# ---------------------------
# Microphone -> MediaStreamTrack
# We'll push frames captured by sounddevice callback into an asyncio.Queue,
# and the MediaStreamTrack.recv() will pop frames and return av.AudioFrame.
# ---------------------------
class MicrophoneStreamTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, queue: asyncio.Queue):
        super().__init__()  # initialize base
        self.queue = queue
        self.sample_rate = SR
        self.samples_per_frame = SAMPLES_PER_FRAME
        self.start_time = None
        self._pts = 0
        # time_base is typically 1/48000 for audio frames pts
        self.time_base = av.AudioFrame().time_base if hasattr(av.AudioFrame(), "time_base") else av.time_base

    async def recv(self):
        """
        Wait for next numpy int16 buffer from queue and return as av.AudioFrame.
        """
        # aiortc expects frames at roughly the sample rate—this await will schedule correctly
        data = await self.queue.get()  # data: numpy array shape (samples,) int16
        if self.start_time is None:
            self.start_time = time.time()

        # create AudioFrame from ndarray: av expects shape (samples, channels) for from_ndarray
        try:
            arr = np.array(data, dtype=np.int16)
            # Ensure shape (samples, channels)
            if arr.ndim == 1:
                arr2 = arr.reshape(-1, 1)
            else:
                arr2 = arr
            frame = av.AudioFrame.from_ndarray(arr2, format="s16", layout="mono")
            frame.sample_rate = self.sample_rate

            # Set proper pts/time_base for smooth playout
            frame.pts = self._pts
            # set time_base typical for audio pts: 1/sample_rate
            frame.time_base = av.time_base if hasattr(av, "time_base") else av.Rational(1, self.sample_rate)
            self._pts += frame.samples
            return frame
        except Exception as e:
            print("MicrophoneStreamTrack.recv conversion error:", e)
            raise

# ---------------------------
# Playback helper: read av.AudioFrame from remote track and write to sounddevice output stream
# ---------------------------
async def play_remote_track(track, stop_event: asyncio.Event):
    """
    Read frames from incoming aiortc track and play via sounddevice RawOutputStream.
    """
    # open raw output
    out_stream = None
    try:
        out_stream = sd.RawOutputStream(samplerate=SR, blocksize=FRAME_SIZE, dtype='int16', channels=CHANNELS)
        out_stream.start()
    except Exception as e:
        print("Failed to open output stream:", e)
        return

    try:
        while not stop_event.is_set():
            frame = await track.recv()  # av.AudioFrame
            # convert frame to numpy int16
            try:
                # to_ndarray(format="s16") returns array with shape (channels, samples)
                arr = frame.to_ndarray(format="s16")
                # arr shape (channels, samples) -> take channel 0 and flatten
                if arr.ndim == 2:
                    samples = arr[0]
                else:
                    samples = arr
                # Ensure dtype is int16
                samples = np.asarray(samples, dtype=np.int16)
                # write bytes
                out_stream.write(samples.tobytes())
            except Exception as e:
                print("playback conversion error:", e)
                # on error, play silence for frame duration to keep timing
                silence = (np.zeros(FRAME_SIZE, dtype=np.int16)).tobytes()
                try:
                    out_stream.write(silence)
                except Exception:
                    pass
    except Exception as e:
        # track ended or other errors
        # print once
        print("Remote playback loop ended:", e)
    finally:
        try:
            out_stream.stop(); out_stream.close()
        except Exception:
            pass

# ---------------------------
# sounddevice input callback -> push numpy int16 frames into asyncio.Queue
# We'll run sounddevice InputStream with a callback that puts frames into a thread-safe queue,
# then have an asyncio task periodically move items into the asyncio.Queue.
# ---------------------------
import threading
import queue as thread_queue

def start_mic_capture_thread(mic_queue_async: asyncio.Queue):
    """
    Start a sounddevice InputStream with a callback that enqueues int16 numpy arrays into
    a thread.Queue, and run a background asyncio task that transfers them into mic_queue_async.
    Returns a stop function.
    """
    thread_q = thread_queue.Queue(maxsize=MIC_QUEUE_MAX)
    stop_flag = threading.Event()

    def sd_callback(indata, frames, time_info, status):
        try:
            # indata is a numpy array (frames, channels) with dtype matching stream dtype
            # convert to int16 if needed
            arr = np.array(indata, copy=False)
            # If float, scale to int16
            if arr.dtype.kind == 'f':
                # clip and scale
                arr_i16 = np.clip(arr, -1.0, 1.0) * 32767.0
                arr_i16 = arr_i16.astype(np.int16)
            else:
                arr_i16 = arr.astype(np.int16)
            # mono: take first channel if multiple
            if arr_i16.ndim > 1:
                arr_i16 = arr_i16[:, 0]
            # Enqueue (non-blocking)
            try:
                thread_q.put_nowait(arr_i16.copy())
            except thread_queue.Full:
                # drop if full
                pass
        except Exception as e:
            # don't spam; print once
            print("mic callback error:", e)

    try:
        in_stream = sd.InputStream(samplerate=SR, blocksize=FRAME_SIZE, dtype='int16', channels=CHANNELS, callback=sd_callback)
        in_stream.start()
    except Exception as e:
        print("Failed to open input stream:", e)
        raise

    # Async task to move thread_q -> mic_queue_async
    async def pump():
        while not stop_flag.is_set():
            try:
                item = thread_q.get(timeout=0.1)
                # await putting into asyncio queue (drop if full)
                try:
                    mic_queue_async.put_nowait(item)
                except asyncio.QueueFull:
                    # drop
                    pass
            except thread_queue.Empty:
                await asyncio.sleep(0.01)
            except Exception as e:
                print("mic pump error:", e)
                await asyncio.sleep(0.01)

    task = asyncio.create_task(pump())

    def stop():
        stop_flag.set()
        try:
            in_stream.stop(); in_stream.close()
        except Exception:
            pass
        task.cancel()

    return stop

# ---------------------------
# Simple TCP signaling helpers (exchange JSON messages)
# Messages are JSON objects with keys:
#  - type: "sdp" or "candidate"
#  - data: SDP string or candidate dict
# We'll send one JSON per line (newline-delimited)
# ---------------------------
async def send_json(writer: asyncio.StreamWriter, obj):
    data = json.dumps(obj, default=str)
    writer.write((data + "\n").encode("utf-8"))
    await writer.drain()

async def recv_json(reader: asyncio.StreamReader) -> Optional[dict]:
    line = await reader.readline()
    if not line:
        return None
    try:
        return json.loads(line.decode("utf-8").strip())
    except Exception:
        return None

# ---------------------------
# Main logic: establish TCP signaling, create PC, attach mic track, exchange SDP/ICE, handle remote track playback
# ---------------------------
async def run_peer(listen: bool, local_port: int, peer_ip: str, peer_port: int):
    # Setup asyncio Stream (either server accept or client connect)
    if listen:
        server = await asyncio.start_server(lambda r, w: None, host="0.0.0.0", port=local_port)
        print(f"Listening for signaling connection on port {local_port}. Waiting for peer...")
        # accept single connection
        conn = await server.accept()
        # However asyncio.start_server returns Server, not single connection. Instead use low-level accept:
        # Simpler approach: use blocking socket accept in executor to get conn
        server.close()
        # Use blocking accept to get a single connection on same port
        def blocking_accept():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", local_port))
            s.listen(1)
            conn, addr = s.accept()
            return conn, addr, s
        loop = asyncio.get_event_loop()
        conn_sock, addr, listen_sock = await loop.run_in_executor(None, blocking_accept)
        print("Accepted signaling connection from", addr)
        reader, writer = await asyncio.open_connection(sock=conn_sock)
    else:
        # client: connect to peer_ip:peer_port
        print(f"Connecting to signaling peer {peer_ip}:{peer_port} ...")
        reader, writer = await asyncio.open_connection(peer_ip, peer_port)
        print("Connected to signaling peer.")

    pc = RTCPeerConnection()
    mic_queue = asyncio.Queue(maxsize=MIC_QUEUE_MAX)
    mic_track = MicrophoneStreamTrack(mic_queue)
    pc.addTrack(mic_track)

    # handle remote track(s)
    stop_play = asyncio.Event()

    @pc.on("track")
    def on_track(track):
        print("Received remote track kind=", track.kind)
        # spawn task to play it
        asyncio.create_task(play_remote_track(track, stop_play))

    # forward ICE candidates over signaling
    async def send_local_desc(desc):
        obj = {"type": "sdp", "sdp": desc.sdp, "sdpType": desc.type}
        await send_json(writer, obj)

    @pc.on("icecandidate")
    def on_icecandidate(event):
        # event is RTCIceCandidate or None
        candidate = event
        if candidate is None:
            return
        # send candidate
        obj = {"type": "candidate", "candidate": {
            "candidate": candidate.to_sdp(),
            "sdpMid": candidate.sdpMid,
            "sdpMLineIndex": candidate.sdpMLineIndex
        }}
        # schedule send asynchronously
        asyncio.create_task(send_json(writer, obj))

    # create offer/answer depending on role
    if listen:
        # server waits for offer from client
        print("Waiting for remote SDP offer...")
        # read messages until sdp offer
        offer_obj = None
        while True:
            msg = await recv_json(reader)
            if msg is None:
                raise RuntimeError("Signaling connection closed before offer.")
            if msg.get("type") == "sdp":
                offer_obj = msg
                break
            elif msg.get("type") == "candidate":
                # ignore early candidates
                pass
        offer = RTCSessionDescription(sdp=offer_obj["sdp"], type=offer_obj.get("sdpType", "offer"))
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        # send answer
        await send_local_desc(pc.localDescription)
    else:
        # client creates offer
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        await send_local_desc(pc.localDescription)
        # wait for answer
        print("Waiting for remote SDP answer...")
        ans_obj = None
        while True:
            msg = await recv_json(reader)
            if msg is None:
                raise RuntimeError("Signaling connection closed before answer.")
            if msg.get("type") == "sdp":
                ans_obj = msg
                break
            elif msg.get("type") == "candidate":
                # handle candidate later
                pass
        answer = RTCSessionDescription(sdp=ans_obj["sdp"], type=ans_obj.get("sdpType", "answer"))
        await pc.setRemoteDescription(answer)

    # After initial SDP exchange, continue reading signaling messages (candidates) in background
    async def signaling_loop():
        while True:
            msg = await recv_json(reader)
            if msg is None:
                print("Signaling connection closed.")
                break
            if msg.get("type") == "candidate":
                cand = msg.get("candidate", {})
                try:
                    # aiortc expects candidate dict with 'candidate','sdpMid','sdpMLineIndex' fields
                    ice = RTCIceCandidate(
                        sdpMid=cand.get("sdpMid"),
                        sdpMLineIndex=cand.get("sdpMLineIndex"),
                        candidate=cand.get("candidate")
                    )
                    await pc.addIceCandidate(ice)
                except Exception as e:
                    # ignore
                    pass
            elif msg.get("type") == "sdp":
                # possible renegotiation or late SDP
                try:
                    desc = RTCSessionDescription(sdp=msg["sdp"], type=msg.get("sdpType", "offer"))
                    await pc.setRemoteDescription(desc)
                except Exception:
                    pass

    sig_task = asyncio.create_task(signaling_loop())

    # start mic capture (thread + asyncio pump)
    mic_stop = start_mic_capture_thread(mic_queue)

    print("Call established — press Ctrl+C to quit.")
    # keep running until cancelled
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Exiting on user request...")
    finally:
        # teardown
        mic_stop()
        stop_play.set()
        try:
            await pc.close()
        except Exception:
            pass
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
        sig_task.cancel()
        return


def parse_args():
    p = argparse.ArgumentParser(description="aiortc P2P voice using TCP signaling (Tailscale-friendly)")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--listen", action="store_true", help="Listen for incoming signaling TCP connection (server)")
    group.add_argument("--peer-ip", type=str, help="Peer IP to connect for signaling")
    p.add_argument("--port", type=int, default=9000, help="Signaling TCP port (both sides must match)")
    return p.parse_args()

def main():
    args = parse_args()
    if args.listen:
        # listen mode: use port as local_port
        local_port = args.port
        asyncio.run(run_peer(listen=True, local_port=local_port, peer_ip="", peer_port=0))
    else:
        asyncio.run(run_peer(listen=False, local_port=0, peer_ip=args.peer_ip, peer_port=args.port))

if __name__ == "__main__":
    main()
