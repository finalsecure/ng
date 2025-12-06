#!/usr/bin/env python3
"""
voice_peer_final.py
Cross-platform full-duplex UDP voice link for two PCs (Tailscale IPs expected).
- sounddevice for capture/playback (int16 PCM)
- UDP packets contain 8-byte header (seq:uint32, flags:uint32) + FRAME_BYTES payload
- Robust conversion from sounddevice read() output to int16 bytes (handles bytes, bytearray,
  cffi buffers, memoryview, numpy array)
- Optional noise suppression: will attempt to load common wrappers but will safely fall back.
- Error printing throttled to avoid spam.
"""

from __future__ import annotations
import argparse
import socket
import threading
import queue
import struct
import time
import sys
import traceback
from collections import deque
from typing import Any

import numpy as np
import sounddevice as sd

# --------------------------
# Configurable audio params
SR = 48000
FRAME_SIZE = 480           # samples per frame (10ms @ 48kHz)
CHANNELS = 1
DTYPE = 'int16'
BYTES_PER_SAMPLE = 2
FRAME_BYTES = FRAME_SIZE * BYTES_PER_SAMPLE  # 960 bytes
# --------------------------

# Packet header format: >I I  (seq, flags)
HEADER_FMT = ">II"
HEADER_LEN = 8

# Jitter buffer and playback params
PLAYBACK_INTERVAL = FRAME_SIZE / SR
JITTER_WAIT = 0.1          # max wait for missing packet (100ms)
MAX_JITTER_FRAMES = 200    # safety cap

# Error print throttling: same error msg printed at most once per PRINT_THROTTLE seconds
PRINT_THROTTLE = 2.0

# --------------------------
# Error print helper (throttled)
_last_print_times = {}
def print_throttled(key: str, *args, **kwargs):
    now = time.time()
    last = _last_print_times.get(key, 0.0)
    if now - last >= PRINT_THROTTLE:
        print(*args, **kwargs)
        _last_print_times[key] = now

# --------------------------
# Robust conversion utility
def to_bytes_int16(data: Any) -> bytes:
    """
    Convert sounddevice.read() return value (various types) to raw int16 bytes.
    Accepts:
      - bytes / bytearray
      - memoryview / cffi buffer / buffer-protocol objects
      - numpy.ndarray of any dtype (will be converted to int16)
    Raises TypeError if cannot convert.
    """
    # 1) bytes-like directly
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)

    # 2) numpy array
    try:
        import numpy as _np
        if isinstance(data, _np.ndarray):
            if data.dtype != _np.int16:
                # Convert safely: scale if float, cast otherwise
                if _np.issubdtype(data.dtype, _np.floating):
                    arr = (_np.clip(data, -1.0, 1.0) * 32768.0).astype(_np.int16)
                else:
                    arr = data.astype(_np.int16)
            else:
                arr = data
            return arr.tobytes()
    except Exception:
        pass

    # 3) memoryview / buffer-like (including cffi._cffi_backend.buffer)
    try:
        mv = memoryview(data)
        # ensure length matches expected bytes or is multiple
        return mv.tobytes()
    except Exception:
        pass

    # 4) fallback: try bytes() constructor
    try:
        return bytes(data)
    except Exception:
        pass

    raise TypeError(f"Cannot convert audio data of type {type(data)} to bytes")

# --------------------------
# Optional noise suppression (safe wrappers)
_ns = None
_ns_name = None
try:
    import pyrnnoise
    # many wrappers expose RNNoise or similar
    if hasattr(pyrnnoise, "RNNoise"):
        try:
            _ns = pyrnnoise.RNNoise()
            _ns_name = "pyrnnoise(RNNoise)"
        except Exception:
            _ns = None
except Exception:
    _ns = None

if _ns is None:
    try:
        import webrtc_noise_gain as wng
        # adapt based on what module exposes
        if hasattr(wng, "NoiseSuppressor"):
            try:
                _ns = wng.NoiseSuppressor()
                _ns_name = "webrtc-noise-gain"
            except Exception:
                _ns = None
    except Exception:
        _ns = None

def apply_noise_suppression_int16(frame_bytes: bytes) -> bytes:
    """
    Given int16 bytes of length FRAME_BYTES, return processed int16 bytes.
    If no NS available, returns original.
    """

    if _ns is None:
        return frame_bytes

    try:
        if _ns_name and "pyrnnoise" in _ns_name:
            # many pyrnnoise wrappers accept bytes and return bytes
            if hasattr(_ns, "process"):
                out = _ns.process(frame_bytes)
                if isinstance(out, (bytes, bytearray)):
                    return bytes(out)
                if isinstance(out, np.ndarray):
                    return out.astype(np.int16).tobytes()
            # try other common names
            for fn in ("denoise_frame", "process_frame"):
                if hasattr(_ns, fn):
                    out = getattr(_ns, fn)(frame_bytes)
                    if isinstance(out, (bytes, bytearray)):
                        return bytes(out)
                    if isinstance(out, np.ndarray):
                        return out.astype(np.int16).tobytes()

        if _ns_name and "webrtc-noise-gain" in _ns_name:
            # convert to float32 [-1,1], process, convert back
            arr = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if hasattr(_ns, "process"):
                out = _ns.process(arr, sample_rate=SR)
                if isinstance(out, np.ndarray):
                    out16 = np.clip(out * 32768.0, -32768, 32767).astype(np.int16)
                    return out16.tobytes()
    except Exception as e:
        print_throttled("ns_error", "Noise-suppression error (ignored):", e)
        # fall through and return original

    return frame_bytes

# --------------------------
class UDPAudioPeer:
    def __init__(self, local_port: int, peer_ip: str, peer_port: int):
        self.local_port = int(local_port)
        self.peer_addr = (peer_ip, int(peer_port))
        self.seq = 0

        # UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # increase socket buffer sizes to reduce packet drops
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)
        except Exception:
            pass
        self.sock.bind(("0.0.0.0", self.local_port))
        self.sock.settimeout(0.5)

        # playback queue: use dict for quick lookup
        self.play_map = {}
        self.play_map_lock = threading.Lock()

        self.running = threading.Event()
        self.running.set()

        # streams
        try:
            # Use Raw streams to work with bytes easily
            self.out_stream = sd.RawOutputStream(
                samplerate=SR, blocksize=FRAME_SIZE, dtype=DTYPE, channels=CHANNELS)
            self.in_stream = sd.RawInputStream(
                samplerate=SR, blocksize=FRAME_SIZE, dtype=DTYPE, channels=CHANNELS)
        except Exception as e:
            print("Audio stream open error:", e)
            raise

    def start(self):
        print("Starting audio streams...")
        self.out_stream.start()
        self.in_stream.start()

        t_rx = threading.Thread(target=self._receiver_loop, daemon=True, name="rx")
        t_tx = threading.Thread(target=self._capture_and_send_loop, daemon=True, name="tx")
        t_play = threading.Thread(target=self._playback_loop, daemon=True, name="play")

        t_rx.start(); t_tx.start(); t_play.start()

        try:
            while self.running.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Interrupted, stopping...")
            self.stop()
        except Exception as e:
            print("Main loop exception:", e)
            traceback.print_exc()
            self.stop()

    def stop(self):
        self.running.clear()
        try:
            self.in_stream.stop(); self.in_stream.close()
        except Exception:
            pass
        try:
            self.out_stream.stop(); self.out_stream.close()
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass

    # --------------------
    def _capture_and_send_loop(self):
        """Read frames from mic, optional NS, send via UDP."""
        while self.running.is_set():
            try:
                data, overflow = self.in_stream.read(FRAME_SIZE)
                frame_bytes = to_bytes_int16(data)

                # if length mismatches, pad or truncate
                if len(frame_bytes) < FRAME_BYTES:
                    frame_bytes = frame_bytes.ljust(FRAME_BYTES, b'\x00')
                elif len(frame_bytes) > FRAME_BYTES:
                    frame_bytes = frame_bytes[:FRAME_BYTES]

                # optional noise suppression
                processed = apply_noise_suppression_int16(frame_bytes)

                hdr = struct.pack(HEADER_FMT, self.seq & 0xFFFFFFFF, 0)
                pkt = hdr + processed
                try:
                    self.sock.sendto(pkt, self.peer_addr)
                except Exception as e:
                    print_throttled("send_error", "Socket send error:", e)
                self.seq = (self.seq + 1) & 0xFFFFFFFF

            except Exception as e:
                # throttle noisy audio errors
                print_throttled("capture_error", "capture/send error:", e)
                # short sleep to avoid tight error loop
                time.sleep(0.01)

    # --------------------
    def _receiver_loop(self):
        """Receive UDP packets, insert into map for playback."""
        while self.running.is_set():
            try:
                pkt, addr = self.sock.recvfrom(HEADER_LEN + FRAME_BYTES + 64)
                if len(pkt) < HEADER_LEN:
                    continue
                seq, flags = struct.unpack(HEADER_FMT, pkt[:HEADER_LEN])
                frame = pkt[HEADER_LEN:HEADER_LEN + FRAME_BYTES]
                if len(frame) < FRAME_BYTES:
                    frame = frame.ljust(FRAME_BYTES, b'\x00')
                with self.play_map_lock:
                    # keep only a sliding window to prevent memory growth
                    if len(self.play_map) > MAX_JITTER_FRAMES:
                        # drop the oldest entries
                        keys = sorted(self.play_map.keys())
                        for k in keys[:len(keys)//10]:
                            self.play_map.pop(k, None)
                    self.play_map[seq] = (time.time(), frame)
            except socket.timeout:
                continue
            except Exception as e:
                print_throttled("recv_error", "recv error:", e)
                time.sleep(0.01)

    # --------------------
    def _playback_loop(self):
        """
        Play frames in sequence order. Simple strategy:
        - Initialize expected_seq to smallest received seq
        - For each expected_seq: wait up to JITTER_WAIT for arrival, else play silence
        """
        expected_seq = None
        last_play_time = time.time()

        while self.running.is_set():
            with self.play_map_lock:
                if expected_seq is None:
                    if not self.play_map:
                        # nothing available yet
                        pass
                    else:
                        expected_seq = min(self.play_map.keys())

                # If we have the expected frame -> play immediately
                if expected_seq is not None and expected_seq in self.play_map:
                    _, frame = self.play_map.pop(expected_seq)
                    try:
                        # RawOutputStream.write accepts bytes
                        self.out_stream.write(frame)
                    except Exception as e:
                        print_throttled("play_error", "playback write error:", e)
                    expected_seq = (expected_seq + 1) & 0xFFFFFFFF
                    last_play_time = time.time()
                    continue

            # if expected_seq not set or missing
            if expected_seq is None:
                # nothing to play, sleep one frame
                time.sleep(PLAYBACK_INTERVAL)
                continue

            # wait small intervals up to JITTER_WAIT to see if it arrives
            waited = 0.0
            found = False
            while waited < JITTER_WAIT and self.running.is_set():
                with self.play_map_lock:
                    if expected_seq in self.play_map:
                        found = True
                        break
                time.sleep(0.005)
                waited += 0.005

            if found:
                continue

            # missing -> play silence to preserve timing
            silence = (np.zeros(FRAME_SIZE, dtype=np.int16)).tobytes()
            try:
                self.out_stream.write(silence)
            except Exception as e:
                print_throttled("play_error2", "playback write error (silence):", e)

            expected_seq = (expected_seq + 1) & 0xFFFFFFFF

# --------------------------
def parse_args():
    p = argparse.ArgumentParser(description="UDP full-duplex voice link (Tailscale-friendly).")
    p.add_argument("--local-port", type=int, required=True, help="Local UDP port to bind")
    p.add_argument("--peer-ip", type=str, required=True, help="Peer IP (Tailscale IP)")
    p.add_argument("--peer-port", type=int, required=True, help="Peer port")
    return p.parse_args()

def main():
    args = parse_args()
    print(f"Audio SR={SR}, FRAME={FRAME_SIZE} ({PLAYBACK_INTERVAL*1000:.1f} ms/frame)")

    if _ns is not None:
        print("Noise suppression enabled via:", _ns_name)
    else:
        print("Noise suppression: not available (running raw pass-through)")

    try:
        peer = UDPAudioPeer(local_port=args.local_port, peer_ip=args.peer_ip, peer_port=args.peer_port)
    except Exception as e:
        print("Failed to initialize UDPAudioPeer:", e)
        sys.exit(1)

    try:
        peer.start()
    finally:
        peer.stop()

if __name__ == "__main__":
    main()
