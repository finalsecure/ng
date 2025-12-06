import socket
import sounddevice as sd
import numpy as np
import noisereduce as nr
import threading

CHUNK = 1600                # 100ms
RATE = 16000
IP = "0.0.0.0"
PORT_SEND = 5000
PORT_RECV = 5001

# ------------------- UDP SOCKETS -------------------
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.bind((IP, PORT_RECV))

# ------------------- CAPTURE + SEND ------------------
def capture_and_send(target_ip):
    def callback(indata, frames, time, status):
        audio = indata[:, 0]

        # -------- Noise reduction ----------
        reduced = nr.reduce_noise(y=audio, sr=RATE)

        compressed = reduced.astype(np.float32).tobytes()
        sock_send.sendto(compressed, (target_ip, PORT_SEND))

    with sd.InputStream(
        samplerate=RATE,
        blocksize=CHUNK,
        channels=1,
        dtype="float32",
        callback=callback
    ):
        print("MIC → SEND çalışıyor...")
        threading.Event().wait()   # sonsuz blok

# ------------------- RECEIVE + PLAY ------------------
def receive_and_play():
    stream = sd.OutputStream(
        samplerate=RATE,
        blocksize=CHUNK,
        channels=1,
        dtype="float32"
    )
    stream.start()
    print("RECV → SPEAKER çalışıyor...")

    while True:
        data, addr = sock_recv.recvfrom(5000)
        audio = np.frombuffer(data, dtype=np.float32)
        stream.write(audio)

# ------------------- MAIN -------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Kullanım: python voice.py <HEDEF_IP>")
        exit()

    target_ip = sys.argv[1]

    threading.Thread(target=capture_and_send, args=(target_ip,), daemon=True).start()
    receive_and_play()
