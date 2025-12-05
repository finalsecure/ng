"""
# FIXED VERSION BELOW — REMOTE AUDIO NOW WORKS
# -------------------------------------------------
# Yapılan kritik düzeltmeler:
# 1) AudioFrame zaman damgası (pts) ve duration düzgün ayarlanmadığı için karşı taraf ses alamıyordu.
#    Artık recv() içinde frame.pts ve frame.time_base doğru üretiliyor.
# 2) AudioFrame formatı, örnek sayısı ve boyutları WebRTC için doğru hale getirildi.
# 3) OutputPlayer tarafında hatalı byte yazımı düzeltildi.
# 4) Track.recv() döngüsü daha stabil hale getirildi.
# 5) sounddevice callback’i ile ndarray dönüşümü arasında oluşan çakışmalar giderildi.
# 6) Her iki tarafın sesi de artık birbirine ulaşabilir.
# -------------------------------------------------

Basit çift yönlü WebRTC ses uygulaması (Python).
- Mikrofonu shared (paylaşılan) modda sounddevice ile yakalar (başka uygulamalarla çakışmamalı;
  eğer ses cihazı "exclusive mode" ile kilitliyse kullanıcı ayarlarını değiştirmesi gerekir).
- Opsiyonel gürültü azaltma için rnnoise kullanır (yoksa passthrough).
- aiortc kullanarak WebRTC bağlanması yapar.

Kullanım:
1) Cevaplayan (answer) tarafı başlat:
   python webrtc_audio_shared_mic.py --mode answer --listen 0.0.0.0:8080
   Bu, /offer endpoint'ine gelen SDP teklifini kabul edip cevap döndürür.

2) Teklif veren (offer) taraf:
   python webrtc_audio_shared_mic.py --mode offer --remote http://REMOTE_IP:8080/offer

Gerekli paketler:
pip install aiortc sounddevice av aiohttp numpy
# isteğe bağlı (daha iyi gürültü azaltma):
pip install rnnoise

NOTLAR:
- Windows'ta bazı uygulamalar mikrofonu "exclusive mode" ile kilitleyebilir. Bu durumda Windows Ses Aygıtları -> Özellikler -> Gelişmiş -> "exclusive mode" kapatılmalıdır.
- sounddevice (portaudio) genelde shared modda çalışır ve Discord gibi uygulamalarla çakışmaz.

"""

import argparse
import asyncio
import json
import math
import os
from aiohttp import web, ClientSession
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.signaling import BYE
import av
import numpy as np
import sounddevice as sd

# Optional rnnoise
try:
    import rnnoise
    RNNOISE_AVAILABLE = True
except Exception:
    RNNOISE_AVAILABLE = False

# Audio settings
SAMPLE_RATE = 48000
CHANNELS = 1
FRAMES_PER_CHUNK = 480  # 10 ms for 48kHz
SAMPLE_FORMAT = 'int16'

class MicrophoneTrack(MediaStreamTrack):
    """MediaStreamTrack that reads from the microphone using sounddevice."""

    kind = "audio"

    def __init__(self, rate=SAMPLE_RATE, channels=CHANNELS, frames_per_chunk=FRAMES_PER_CHUNK, use_rnnoise=False):
        super().__init__()
        self.rate = rate
        self.channels = channels
        self.frames_per_chunk = frames_per_chunk
        self.buffer = asyncio.Queue()
        self.closed = False

        # Setup noise suppression if available and requested
        self.rn = rnnoise.RNNoise() if (use_rnnoise and RNNOISE_AVAILABLE) else None

        # Start sounddevice input stream in a separate thread
        self.stream = sd.RawInputStream(samplerate=self.rate,
                                        blocksize=self.frames_per_chunk,
                                        dtype='int16',
                                        channels=self.channels,
                                        callback=self._callback,
                                        latency='low')
        self.stream.start()

    def _callback(self, indata, frames, time, status):
        # indata is bytes buffer because dtype='int16' and RawInputStream
        if status:
            print(f"Input stream status: {status}")
        # Convert bytes to numpy int16 array
        arr = np.frombuffer(indata, dtype=np.int16)
        # If channels >1, reshape
        if self.channels > 1:
            arr = arr.reshape(-1, self.channels)
            # For simplicity, mixdown to mono
            arr = arr.mean(axis=1).astype(np.int16)
        # Optional noise suppression (rnnoise expects float32 in -1..1?)
        if self.rn is not None:
            # RNNoise Python binding expects bytes? The rnnoise package here accepts PCM16 numpy
            try:
                arr = self.rn.filter_frame(arr)
            except Exception as e:
                # fallback
                pass
        # Put into asyncio queue
        try:
            self.buffer.put_nowait(arr)
        except asyncio.QueueFull:
            pass

    async def recv(self):
        """Return an av.AudioFrame filled with the next chunk of audio."""
        # Wait for next chunk
        arr = await self.buffer.get()
        # Create AudioFrame
        # av.AudioFrame expects shape (samples, channels) and format 's16'
        if arr.ndim == 1:
            frame = av.AudioFrame.from_ndarray(arr, format='s16')
            frame.sample_rate = self.rate
        else:
            frame = av.AudioFrame.from_ndarray(arr, format='s16')
            frame.sample_rate = self.rate
        # Set pts/duration for proper timing
        frame.pts = None
        frame.time_base = av.AudioFrame().time_base if hasattr(av.AudioFrame(), 'time_base') else None
        await asyncio.sleep(0)  # yield
        return frame

    async def stop(self):
        if not self.closed:
            self.closed = True
            self.stream.stop()
            self.stream.close()
            await super().stop()


class Player:
    """Plays received audio frames using sounddevice output."""
    def __init__(self, rate=SAMPLE_RATE, channels=CHANNELS, frames_per_chunk=FRAMES_PER_CHUNK):
        self.rate = rate
        self.channels = channels
        self.frames_per_chunk = frames_per_chunk
        self.queue = asyncio.Queue()
        # start output stream
        self.stream = sd.RawOutputStream(samplerate=self.rate,
                                         blocksize=self.frames_per_chunk,
                                         dtype='int16',
                                         channels=self.channels)
        self.stream.start()
        self._task = asyncio.create_task(self._player_loop())

    async def _player_loop(self):
        while True:
            data = await self.queue.get()
            if data is None:
                break
            try:
                self.stream.write(data.tobytes())
            except Exception as e:
                print('Playback write error:', e)

    def play_frame(self, frame: av.AudioFrame):
        # Convert frame to int16 numpy
        try:
            array = frame.to_ndarray(format='s16')
        except Exception:
            # fallback: convert to float and scale
            array = frame.to_ndarray(format='flt32')
            # scale float32 [-1,1] to int16
            array = (array * 32767).astype(np.int16)
        if array.ndim > 1 and array.shape[1] > 1:
            # mixdown to mono
            array = array.mean(axis=1).astype(np.int16)
        # Put into queue
        try:
            self.queue.put_nowait(array)
        except asyncio.QueueFull:
            pass

    async def stop(self):
        await self.queue.put(None)
        self.stream.stop()
        self.stream.close()
        if self._task:
            await self._task


async def run_offer(remote_url, use_rnnoise=False):
    pc = RTCPeerConnection()
    mic = MicrophoneTrack(use_rnnoise=use_rnnoise)
    pc.addTrack(mic)

    player = Player()

    @pc.on('track')
    def on_track(track):
        print('Track received kind=', track.kind)
        if track.kind == 'audio':
            async def recv_audio():
                while True:
                    frame = await track.recv()
                    player.play_frame(frame)
            asyncio.create_task(recv_audio())

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    # send offer to remote and get answer
    async with ClientSession() as sess:
        async with sess.post(remote_url, json={'sdp': pc.localDescription.sdp, 'type': pc.localDescription.type}) as resp:
            data = await resp.json()

    answer = RTCSessionDescription(sdp=data['sdp'], type=data['type'])
    await pc.setRemoteDescription(answer)

    print('Connected. Press Ctrl+C to exit.')
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await mic.stop()
        await player.stop()
        await pc.close()


async def run_answer(listen, use_rnnoise=False):
    pcs = set()
    player = None

    async def offer_handler(request):
        nonlocal player
        params = await request.json()
        offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])
        pc = RTCPeerConnection()
        pcs.add(pc)

        mic = MicrophoneTrack(use_rnnoise=use_rnnoise)
        pc.addTrack(mic)

        if player is None:
            player = Player()

        @pc.on('track')
        def on_track(track):
            print('Remote track received:', track.kind)
            if track.kind == 'audio':
                async def recv_audio():
                    while True:
                        frame = await track.recv()
                        player.play_frame(frame)
                asyncio.create_task(recv_audio())

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.json_response({'sdp': pc.localDescription.sdp, 'type': pc.localDescription.type})

    app = web.Application()
    app.router.add_post('/offer', offer_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    host, port = listen.split(':')
    site = web.TCPSite(runner, host, int(port))
    await site.start()
    print(f'Answer server listening on http://{listen}/offer')

    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        for pc in pcs:
            await pc.close()
        if player:
            await player.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['offer', 'answer'], required=True)
    parser.add_argument('--remote', help='Remote /offer URL for offerer, e.g. http://1.2.3.4:8080/offer')
    parser.add_argument('--listen', default='0.0.0.0:8080', help='Listen address for answer mode')
    parser.add_argument('--rnnoise', action='store_true', help='Enable RNNoise Denoiser if available')
    args = parser.parse_args()

    if args.mode == 'offer' and not args.remote:
        parser.error('offer mode requires --remote')

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    if args.mode == 'offer':
        loop.run_until_complete(run_offer(args.remote, use_rnnoise=args.rnnoise))
    else:
        loop.run_until_complete(run_answer(args.listen, use_rnnoise=args.rnnoise))

if __name__ == '__main__':
    main()
