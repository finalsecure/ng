import argparse
import asyncio
import json
import socket
import threading
from queue import Queue, Empty
import numpy as np
import sounddevice as sd
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamTrack
import av
import time

# webrtc-noise-gain
from webrtc_noise_gain import AudioProcessor

# -----------------------
# Config
# -----------------------
SAMPLE_RATE = 16000        # must be 16000 for webrtc-noise-gain
CHANNELS = 1
FRAME_DURATION_MS = 20     # output frame length we will send to aiortc (20 ms)
SAMPLES_PER_10MS = int(SAMPLE_RATE * 0.01)           # 160 samples
SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 320 samples for 20ms
BLOCKSIZE = SAMPLES_PER_10MS  # sounddevice blocksize -> 10ms chunks
CAPTURE_QUEUE_MAX = 1000

# webrtc-noise-gain settings
AUTO_GAIN_DBFS = 3               # 0..31 (0 = disable)
NOISE_SUPPRESSION_LEVEL = 2      # 0..4 (0 = disable)

# -----------------------
# Queues & globals
# -----------------------
capture_queue = Queue(maxsize=CAPTURE_QUEUE_MAX)

def sd_callback(indata, frames, time_info, status):
    """
    sounddevice callback: receives float32 samples in range -1..1, shape (frames, channels)
    We put mono float32 arrays of length `frames` into capture_queue.
    """
    if status:
        # ignore status or optionally print
        pass
    data = indata
    # convert to mono if needed
    if data.ndim > 1:
        data = data[:, 0]
    # ensure float32
    data = data.astype(np.float32, copy=False)
    try:
        capture_queue.put_nowait(data)
    except:
        # queue full, drop frame
        pass

# -----------------------
# Audio processor (webrtc-noise-gain)
# -----------------------
audio_processor = AudioProcessor(AUTO_GAIN_DBFS, NOISE_SUPPRESSION_LEVEL)

# -----------------------
# MediaTrack for microphone
# -----------------------
class MicrophoneTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()  # sets .kind
        self.samplerate = SAMPLE_RATE
        self.channels = CHANNELS
        # internal buffer in float32 for building frames
        self._buffer = np.zeros(0, dtype=np.float32)

    async def recv(self):
        """
        Called by aiortc to get the next audio frame.
        We'll gather SAMPLES_PER_FRAME float samples, process them in 10ms blocks via audio_processor,
        and return an av.AudioFrame.
        """
        # gather until we have SAMPLES_PER_FRAME
        while self._buffer.shape[0] < SAMPLES_PER_FRAME:
            try:
                chunk = capture_queue.get_nowait()
            except Empty:
                # wait a little if nothing available
                await asyncio.sleep(0.002)
                continue
            # append
            self._buffer = np.concatenate([self._buffer, chunk])

        # take exactly SAMPLES_PER_FRAME samples
        frame_samples = self._buffer[:SAMPLES_PER_FRAME]
        self._buffer = self._buffer[SAMPLES_PER_FRAME:]

        # ensure dtype float32 in -1..1
        frame_samples = frame_samples.astype(np.float32)
        # split into 10ms chunks and process each
        processed_chunks = []
        for i in range(0, SAMPLES_PER_FRAME, SAMPLES_PER_10MS):
            chunk = frame_samples[i:i+SAMPLES_PER_10MS]
            # convert chunk to int16 bytes (webrtc-noise-gain expects 16-bit PCM)
            # scale float32 -1..1 to int16
            int16 = np.clip(chunk * 32767.0, -32768, 32767).astype(np.int16)
            raw = int16.tobytes()
            # process 10ms
            result = audio_processor.Process10ms(raw)
            # result.audio is bytes of 16-bit PCM (16kHz mono)
            out_bytes = result.audio
            out_int16 = np.frombuffer(out_bytes, dtype=np.int16)
            # convert back to float32 -1..1
            out_float = out_int16.astype(np.float32) / 32768.0
            processed_chunks.append(out_float)

        processed = np.concatenate(processed_chunks)  # length SAMPLES_PER_FRAME

        # convert to int16 for av.AudioFrame (aiortc expects frame as integer PCM often)
        out_int16 = np.clip(processed * 32767.0, -32768, 32767).astype(np.int16)

        # av expects shape (channels, samples) for from_ndarray, or for mono (n,)
        # we'll provide 1D array and layout="mono"
        frame = av.AudioFrame.from_ndarray(out_int16, layout="mono")
        frame.sample_rate = self.samplerate

        # optional: set pts/duration (aiortc may manage)
        # compute pts (monotonic)
        await asyncio.sleep(0)  # yield to event loop
        return frame

# -----------------------
# Playback helper
# -----------------------
class Player:
    def __init__(self):
        self.sr = SAMPLE_RATE
        self.channels = CHANNELS
        self.blocksize = SAMPLES_PER_FRAME
        self.lock = threading.Lock()
        self.stream = sd.OutputStream(samplerate=self.sr, channels=self.channels, dtype='float32', blocksize=self.blocksize)
        self.started = False

    def start(self):
        with self.lock:
            if not self.started:
                self.stream.start()
                self.started = True

    def stop(self):
        with self.lock:
            if self.started:
                self.stream.stop()
                self.started = False

    def play(self, float32_array):
        """
        expects numpy float32 shape (n,) in -1..1
        """
        if not isinstance(float32_array, np.ndarray):
            float32_array = np.asarray(float32_array, dtype=np.float32)
        if float32_array.dtype != np.float32:
            float32_array = float32_array.astype(np.float32)
        # ensure shape (n, channels) for multi-channel, but we have mono
        self.start()
        try:
            self.stream.write(float32_array)
        except Exception:
            # ignore occasional underrun errors
            pass

player = Player()

# -----------------------
# Signaling (simple TCP JSON)
# -----------------------
def send_json(conn_file, obj):
    data = (json.dumps(obj) + "\n").encode("utf8")
    conn_file.write(data)
    conn_file.flush()

def recv_json(conn_file):
    line = conn_file.readline()
    if not line:
        return None
    return json.loads(line.decode("utf8"))

def tcp_client_connect(remote_host, remote_port, timeout=10):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect((remote_host, remote_port))
    conn_file = s.makefile("rwb")
    return s, conn_file

# -----------------------
# Peer connection flows
# -----------------------
async def run_server(bind_host, bind_port):
    # accept one signaling connection
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((bind_host, bind_port))
    sock.listen(1)
    print(f"[SIGNAL] Waiting for connection on {bind_host}:{bind_port} ...")
    conn, addr = sock.accept()
    print(f"[SIGNAL] Connected from {addr}")
    conn_file = conn.makefile("rwb")

    pc = RTCPeerConnection()
    mic = MicrophoneTrack()
    pc.addTrack(mic)

    @pc.on("track")
    def on_track(track):
        print("[PC] Remote track received:", track.kind)
        if track.kind == "audio":
            async def recv_play():
                while True:
                    frame = await track.recv()
                    # convert to numpy float32 -1..1
                    pcm = frame.to_ndarray()
                    # av may return shape (channels, samples) or (samples,) depending
                    if pcm.dtype.kind == 'i':
                        pcm = pcm.astype(np.float32) / np.iinfo(pcm.dtype).max
                    else:
                        pcm = pcm.astype(np.float32)
                    if pcm.ndim > 1:
                        # choose first channel (mono)
                        pcm = pcm[0]
                    player.play(pcm)
            asyncio.ensure_future(recv_play())

    # create offer
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    send_json(conn_file, {"type": "offer", "sdp": pc.localDescription.sdp})

    # wait for answer
    msg = recv_json(conn_file)
    if not msg or msg.get("type") != "answer":
        print("[SIGNAL] No answer received, exiting")
        await pc.close()
        return
    await pc.setRemoteDescription(RTCSessionDescription(sdp=msg["sdp"], type=msg["type"]))
    print("[PC] Connection established (server). Media flowing...")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await pc.close()
        conn.close()
        sock.close()

async def run_client(connect_host, connect_port):
    s, conn_file = tcp_client_connect(connect_host, connect_port)
    pc = RTCPeerConnection()
    mic = MicrophoneTrack()
    pc.addTrack(mic)

    @pc.on("track")
    def on_track(track):
        print("[PC] Remote track received:", track.kind)
        if track.kind == "audio":
            async def recv_play():
                while True:
                    frame = await track.recv()
                    pcm = frame.to_ndarray()
                    if pcm.dtype.kind == 'i':
                        pcm = pcm.astype(np.float32) / np.iinfo(pcm.dtype).max
                    else:
                        pcm = pcm.astype(np.float32)
                    if pcm.ndim > 1:
                        pcm = pcm[0]
                    player.play(pcm)
            asyncio.ensure_future(recv_play())

    # wait for offer
    msg = recv_json(conn_file)
    if not msg or msg.get("type") != "offer":
        print("[SIGNAL] No offer received, exiting")
        await pc.close()
        s.close()
        return
    await pc.setRemoteDescription(RTCSessionDescription(sdp=msg["sdp"], type=msg["type"]))
    # create and send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    send_json(conn_file, {"type": "answer", "sdp": pc.localDescription.sdp})
    print("[PC] Answer sent (client). Connection should establish.")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await pc.close()
        s.close()

# -----------------------
# Start sounddevice input and run main
# -----------------------
def start_input_stream():
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32', blocksize=BLOCKSIZE, callback=sd_callback)
    stream.start()
    return stream

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["server", "client"], required=True)
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9999)
    p.add_argument("--connect", help="server IP (client mode)")
    return p.parse_args()

def main():
    args = parse_args()
    print("[APP] Starting capture at %d Hz, %d channels, blocksize=%d" % (SAMPLE_RATE, CHANNELS, BLOCKSIZE))
    sd_stream = start_input_stream()
    loop = asyncio.get_event_loop()
    try:
        if args.mode == "server":
            loop.run_until_complete(run_server(args.bind, args.port))
        else:
            if not args.connect:
                print("Client mode requires --connect <server_ip>")
                return
            loop.run_until_complete(run_client(args.connect, args.port))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            sd_stream.stop()
            sd_stream.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()