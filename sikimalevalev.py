#!/usr/bin/env python3
"""
voice_peer.py
Cross-platform full-duplex UDP voice link for two PCs (Tailscale IPs).
- Uses sounddevice for capture/playback (int16 PCM).
- Sends fixed-size frames (480 samples @ 48kHz) in UDP packets with a 8-byte header (seq:uint32, flags:uint32).
- Optional noise suppression if 'pyrnnoise' or 'webrtc-noise-gain' is installed.
- Simple jitter buffer + reorder with timeout.
"""

import argparse
import socket
import threading
import queue
import struct
import time
import sys
import traceback

import numpy as np
import sounddevice as sd

# Audio params
SR = 48000
FRAME_SIZE = 480          # rnnoise expects 480-sample frames at 48kHz in many wrappers
CHANNELS = 1
DTYPE = 'int16'
BYTES_PER_SAMPLE = 2
FRAME_BYTES = FRAME_SIZE * BYTES_PER_SAMPLE  # 960 bytes

# Packet header: >I I  (seq, flags)  (8 bytes)
HEADER_FMT = ">II"
HEADER_LEN = 8

# Jitter buffer size (in frames)
JITTER_BUFFER_MAX = 50    # 50 * 10ms = 500ms (max buffering)
PLAYBACK_INTERVAL = FRAME_SIZE / SR  # seconds per frame (~0.01s)

# Try optional noise suppression libraries
ns = None
ns_name = None
try:
    import pyrnnoise
    # some pyrnnoise wrappers expose RNNoise class — adapt if not exact
    if hasattr(pyrnnoise, "RNNoise"):
        ns = pyrnnoise.RNNoise()
        ns_name = "pyrnnoise"
except Exception:
    pass

if ns is None:
    try:
        import webrtc_noise_gain as wng
        # this package may provide a simple suppressor — adapt usage below
        ns = wng.NoiseSuppressor() if hasattr(wng, "NoiseSuppressor") else None
        ns_name = "webrtc-noise-gain" if ns else None
    except Exception:
        ns = None

# Fallback: no suppression (fast path)


def apply_noise_suppression_int16(frame_bytes: bytes) -> bytes:
    """
    frame_bytes: raw int16 bytes, mono, FRAME_SIZE samples.
    Returns bytes (int16) after suppression OR original if no ns available.
    """
    if ns is None:
        return frame_bytes
    try:
        # Try several common APIs based on known wrappers.
        # 1) pyrnnoise: denoise_frame expects bytes or int16 numpy array -> returns bytes or array
        if ns_name == "pyrnnoise":
            # many wrappers accept bytes and return bytes; adapt if needed
            try:
                out = ns.process(frame_bytes)  # attempt generic name
                if isinstance(out, (bytes, bytearray)):
                    return bytes(out)
            except Exception:
                pass
            # try process_frame / denoise_frame
            for fn in ("process_frame", "denoise_frame", "process"):
                if hasattr(ns, fn):
                    try:
                        out = getattr(ns, fn)(frame_bytes)
                        if isinstance(out, (bytes, bytearray)):
                            return bytes(out)
                        if isinstance(out, np.ndarray):
                            return out.astype(np.int16).tobytes()
                    except Exception:
                        pass
        # 2) webrtc-noise-gain: some wrappers accept numpy array
        if ns_name == "webrtc-noise-gain":
            try:
                # convert to float32 [-1,1] if needed
                arr = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                if hasattr(ns, "process"):
                    out = ns.process(arr, sample_rate=SR)
                    if isinstance(out, np.ndarray):
                        out16 = np.clip(out * 32768, -32768, 32767).astype(np.int16)
                        return out16.tobytes()
            except Exception:
                pass
    except Exception:
        # any unexpected error -> ignore suppression for that frame
        traceback.print_exc()
    # fallback:
    return frame_bytes


class UDPAudioPeer:
    def __init__(self, local_port: int, peer_ip: str, peer_port: int):
        self.local_port = local_port
        self.peer_addr = (peer_ip, peer_port)
        self.seq = 0

        # Sockets
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", local_port))
        self.sock.settimeout(0.5)

        # Queues
        self.play_q = queue.PriorityQueue()  # (seq, arrival_time, bytes)
        self.recv_lock = threading.Lock()
        self.expected_seq = None

        self.running = threading.Event()
        self.running.set()

        # Playback stream
        self.out_stream = sd.RawOutputStream(samplerate=SR, blocksize=FRAME_SIZE,
                                             dtype=DTYPE, channels=CHANNELS)
        self.in_stream = sd.RawInputStream(samplerate=SR, blocksize=FRAME_SIZE,
                                           dtype=DTYPE, channels=CHANNELS)

    def start(self):
        print("Starting streams...")
        self.out_stream.start()
        self.in_stream.start()

        t_rx = threading.Thread(target=self.receiver_loop, daemon=True)
        t_tx = threading.Thread(target=self.capture_and_send_loop, daemon=True)
        t_play = threading.Thread(target=self.playback_loop, daemon=True)

        t_rx.start()
        t_tx.start()
        t_play.start()

        try:
            while self.running.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping on keyboard interrupt")
            self.stop()

    def stop(self):
        self.running.clear()
        try:
            self.in_stream.stop(); self.in_stream.close()
        except Exception: pass
        try:
            self.out_stream.stop(); self.out_stream.close()
        except Exception: pass
        try:
            self.sock.close()
        except Exception: pass

    def capture_and_send_loop(self):
        """Read frames from mic, optional NS, and send via UDP with header."""
        while self.running.is_set():
            try:
                data, _ = self.in_stream.read(FRAME_SIZE)
                # sounddevice RawInputStream.read returns bytes when dtype is int16
                if isinstance(data, bytes):
                    frame_bytes = data
                else:
                    # numpy array
                    frame_bytes = (data.astype(np.int16)).tobytes()
                # Optional NS
                processed = apply_noise_suppression_int16(frame_bytes)
                # Build packet
                hdr = struct.pack(HEADER_FMT, self.seq & 0xFFFFFFFF, 0)
                pkt = hdr + processed
                # send (non-blocking)
                self.sock.sendto(pkt, self.peer_addr)
                self.seq = (self.seq + 1) & 0xFFFFFFFF
            except Exception as e:
                # on audio errors, print and keep going
                print("capture/send error:", e)
                time.sleep(0.01)

    def receiver_loop(self):
        """Receive UDP packets and insert into jitter buffer."""
        while self.running.is_set():
            try:
                pkt, addr = self.sock.recvfrom(HEADER_LEN + FRAME_BYTES + 100)
                if len(pkt) < HEADER_LEN:
                    continue
                seq, flags = struct.unpack(HEADER_FMT, pkt[:HEADER_LEN])
                frame = pkt[HEADER_LEN:HEADER_LEN + FRAME_BYTES]
                if len(frame) < FRAME_BYTES:
                    # pad if truncated
                    frame = frame.ljust(FRAME_BYTES, b'\x00')
                # priority queue by seq (to reorder). Use seq as priority.
                self.play_q.put((seq, time.time(), frame))
            except socket.timeout:
                continue
            except Exception as e:
                print("recv error:", e)
                time.sleep(0.01)

    def playback_loop(self):
        """
        Pop frames from jitter buffer in order and play.
        Simple approach:
         - maintain expected_seq (first packet seen)
         - wait up to a small timeout (e.g., 100ms) for missing seq.
         - if missing, play silence to keep timing.
        """
        expected_seq = None
        buffer_map = {}
        last_play_time = time.time()
        while self.running.is_set():
            # Pull everything available into buffer_map (non-blocking)
            try:
                while True:
                    seq, atime, frame = self.play_q.get_nowait()
                    buffer_map[seq] = (atime, frame)
            except queue.Empty:
                pass

            if expected_seq is None:
                # initialize expected_seq to smallest available seq if any
                if buffer_map:
                    expected_seq = min(buffer_map.keys())
                else:
                    # nothing yet, play silence until data arrives
                    time.sleep(PLAYBACK_INTERVAL)
                    continue

            # If expected_seq present -> play it
            if expected_seq in buffer_map:
                _, frame = buffer_map.pop(expected_seq)
                try:
                    self.out_stream.write(frame)
                except Exception as e:
                    print("playback write error:", e)
                expected_seq = (expected_seq + 1) & 0xFFFFFFFF
                last_play_time = time.time()
                continue

            # missing: wait small timeout for arrival (jitter compensation)
            waited = 0.0
            got = False
            while waited < 0.1:  # wait up to 100ms
                # check buffer_map again
                if expected_seq in buffer_map:
                    got = True
                    break
                time.sleep(0.005)
                waited += 0.005
            if got:
                continue
            # still missing -> play silence frame to keep sync
            silence = (np.zeros(FRAME_SIZE, dtype=np.int16)).tobytes()
            try:
                self.out_stream.write(silence)
            except Exception:
                pass
            expected_seq = (expected_seq + 1) & 0xFFFFFFFF


def main():
    parser = argparse.ArgumentParser(description="Simple UDP full-duplex voice link")
    parser.add_argument("--local-port", type=int, required=True)
    parser.add_argument("--peer-ip", type=str, required=True)
    parser.add_argument("--peer-port", type=int, required=True)
    args = parser.parse_args()

    print("Audio params: SR=%d FRAME=%d bytes/frame=%d" % (SR, FRAME_SIZE, FRAME_BYTES))
    if ns is not None:
        print("Noise suppression enabled via", ns_name)
    else:
        print("No noise suppression library found; running raw pass-through")

    peer = UDPAudioPeer(local_port=args.local_port, peer_ip=args.peer_ip, peer_port=args.peer_port)
    try:
        peer.start()
    except Exception as e:
        print("Fatal error:", e)
        traceback.print_exc()
        peer.stop()


if __name__ == "__main__":
    main()
