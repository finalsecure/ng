import argparse
import socket
import threading
import queue
import struct
import time

import numpy as np
import sounddevice as sd

# --- Ayarlar ---
CHANNELS = 1
SAMPLE_RATE = 48000          # veya 44100; her iki tarafın da aynı olması gerekir
FRAME_DURATION_MS = 20       # paket başına ms (küçük => düşük gecikme). 20 ms iyi bir başlangıç.
SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
SAMPLE_FORMAT = 'int16'      # 16-bit PCM
BYTES_PER_SAMPLE = 2

# Header: 4 byte sequence number (unsigned int, network order)
HEADER_STRUCT = "!I"

# Jitter buffer boyutu (packet). Çok küçük = drop; çok büyük = gecikme
JITTER_BUFFER_MAX = 50

# --- Socket setup ---
def make_socket(local_port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", local_port))
    sock.setblocking(True)
    return sock

# --- Globals / Queues ---
play_queue = queue.PriorityQueue(maxsize=JITTER_BUFFER_MAX)  # (seq, timestamp, bytes)
play_lock = threading.Lock()
running = threading.Event()

# --- Receiver Thread ---
def receiver(sock):
    print("[recv] receiver started, listening for incoming UDP packets...")
    while running.is_set():
        try:
            data, addr = sock.recvfrom(65536)
            if not data:
                continue
            if len(data) < struct.calcsize(HEADER_STRUCT):
                continue
            seq = struct.unpack(HEADER_STRUCT, data[:4])[0]
            pcm = data[4:]
            # PriorityQueue sorts by first element -> seq
            # Put with timestamp to break seq ties
            ts = time.time()
            try:
                play_queue.put_nowait((seq, ts, pcm))
            except queue.Full:
                # Eğer kuyruk doluysa en eskiyi at, sonra ekle
                try:
                    _ = play_queue.get_nowait()
                    play_queue.put_nowait((seq, ts, pcm))
                except Exception:
                    pass
        except Exception as e:
            if running.is_set():
                print("[recv] hata:", e)
            break
    print("[recv] receiver stopped.")

# --- Playback callback ---
# sounddevice expects float32 in range [-1,1] or int16 depending on dtype; we'll use int16 dtype
playback_seq_expected = 0
buffered_frames = []  # fallback small buffer

def playback_callback(outdata, frames, time_info, status):
    global playback_seq_expected
    # outdata is (frames, channels)
    if status.output_underflow:
        # output underflow; fill silence
        outdata.fill(0)
        return

    needed_samples = frames * CHANNELS
    # Try to get a packet with the next expected sequence
    try:
        item = play_queue.get_nowait()
    except queue.Empty:
        # no data -> silence
        outdata.fill(0)
        return

    seq, ts, pcm = item
    # convert bytes to numpy array
    arr = np.frombuffer(pcm, dtype=np.int16)
    # if frame length mismatch, try to pad/crop
    if arr.size < SAMPLES_PER_FRAME * CHANNELS:
        # pad zeros
        pad = np.zeros(SAMPLES_PER_FRAME * CHANNELS - arr.size, dtype=np.int16)
        arr = np.concatenate((arr, pad))
    elif arr.size > SAMPLES_PER_FRAME * CHANNELS:
        arr = arr[:SAMPLES_PER_FRAME * CHANNELS]

    # reshape to (frames, channels)
    arr = arr.reshape(-1, CHANNELS)
    # convert to float32 in [-1,1] because sounddevice often expects float; but we can write int16 by setting dtype
    # We'll output as int16 by casting outdata to int16 view
    try:
        out_slice = outdata[:arr.shape[0], :]
        # sounddevice gives float32 array; convert
        out_slice[:] = (arr.astype(np.float32) / 32768.0)
        # if more frames requested than we have, fill remainder with silence
        if arr.shape[0] < frames:
            outdata[arr.shape[0]:, :].fill(0)
    except Exception as e:
        outdata.fill(0)

# --- Capture callback ---
send_seq = 0
def capture_callback(indata, frames, time_info, status):
    global send_seq
    if status.input_overflow:
        # input overflow - can't do much
        pass

    # indata is float32 in [-1,1] by default. Convert to int16 PCM
    try:
        # Convert to int16
        pcm16 = (indata * 32767.0).astype(np.int16)
        # flatten and tobytes
        payload = pcm16.tobytes()
        header = struct.pack(HEADER_STRUCT, send_seq & 0xFFFFFFFF)
        packet = header + payload
        try:
            sender_socket.sendto(packet, remote_addr)
        except Exception as e:
            # network errors ignored
            pass
        send_seq += 1
    except Exception as e:
        pass

# --- Main ---
def main():
    global sender_socket, remote_addr, running, playback_seq_expected
    parser = argparse.ArgumentParser()
    parser.add_argument("remote_ip", help="Remote IP to send audio to")
    parser.add_argument("remote_port", type=int, help="Remote UDP port")
    parser.add_argument("--local-port", type=int, default=9000, help="Local UDP port to listen on (default: 9000)")
    args = parser.parse_args()

    remote_addr = (args.remote_ip, args.remote_port)
    local_port = args.local_port

    # create socket
    sender_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_socket = make_socket(local_port)

    running.set()

    # start receiver thread
    t_recv = threading.Thread(target=receiver, args=(recv_socket,), daemon=True)
    t_recv.start()

    # Open playback stream
    try:
        # We will open two separate streams: input stream and output stream
        # Input: callback capture_callback will send packets
        # Output: callback playback_callback will read from play_queue and output
        print("[main] Starting audio streams...")
        # start output stream
        out_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=SAMPLES_PER_FRAME,
            dtype='float32',
            channels=CHANNELS,
            callback=playback_callback,
            latency='low'
        )
        in_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=SAMPLES_PER_FRAME,
            dtype='float32',
            channels=CHANNELS,
            callback=capture_callback,
            latency='low'
        )

        with out_stream, in_stream:
            print(f"[main] Running. Sending to {remote_addr}, listening on port {local_port}. Press Ctrl+C to stop.")
            while running.is_set():
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[main] Ctrl+C pressed. Exiting...")
    except Exception as e:
        print("[main] Hata (audio):", e)
    finally:
        running.clear()
        try:
            recv_socket.close()
        except:
            pass
        try:
            sender_socket.close()
        except:
            pass
        print("[main] Stopped.")

if __name__ == "__main__":
    main()