"""
사운드 인스톨레이션 — 테스트 클라이언트 (빔 감지 모드, TCP)
Mac에서 RPi 없이 마이크로 직접 테스트할 때 사용.
서버의 TCP 프로토콜에 맞춰 S32 stereo 48kHz raw PCM 스트리밍.

[FIX] 원본은 HTTP /process 엔드포인트를 호출했지만 server.py에는
      해당 라우트가 없음 (TCP 전용). TCP 프로토콜로 전면 재작성.

사용법: python3 client.py --server 192.168.0.100 --module-id 1
"""

import argparse
import wave
import io
import os
import time
import tempfile
import subprocess
import socket
import threading
import numpy as np

SAMPLE_RATE = 16000

# 서버가 기대하는 소스 포맷 (RPi arecord와 동일)
SRC_RATE = 48000
SRC_CHANNELS = 2
SRC_SAMPLE_BYTES = 4  # S32_LE
SRC_CHUNK_SEC = 0.1
SRC_CHUNK_BYTES = int(SRC_RATE * SRC_CHUNK_SEC) * SRC_CHANNELS * SRC_SAMPLE_BYTES  # 38400


def recv_exact(sock, n):
    """소켓에서 정확히 n바이트 수신"""
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def listen_and_convert(threshold, record_after, sr=SAMPLE_RATE):
    """
    마이크 녹음 → 서버 소스 포맷(S32 stereo 48kHz)으로 변환.
    빔 감지(볼륨 임계값) 방식 유지.
    """
    import sounddevice as sd
    from collections import deque

    chunk_duration = 0.5
    chunk_size = int(chunk_duration * sr)
    buffer = deque(maxlen=4)

    while True:
        chunk = sd.rec(chunk_size, samplerate=sr, channels=1, dtype="int16")
        sd.wait()
        chunk = chunk.flatten()
        buffer.append(chunk)

        volume = np.abs(chunk.astype(np.float32) / 32768.0).mean()

        if volume >= threshold:
            print(f"  ⚡ 빔 감지! (볼륨: {volume:.4f})")
            print(f"  🎤 수음 중... ({record_after}초)")

            extra = sd.rec(int(record_after * sr), samplerate=sr, channels=1, dtype="int16")
            sd.wait()
            extra = extra.flatten()

            pre_audio = np.concatenate(list(buffer))
            full_audio = np.concatenate([pre_audio, extra])

            # 16kHz mono int16 → 48kHz stereo int32 (서버 포맷에 맞춤)
            from scipy.signal import resample_poly
            upsampled = resample_poly(full_audio.astype(np.float32), 3, 1)  # 16k → 48k
            int32_audio = (upsampled * (2147483648.0 / 32768.0)).astype(np.int32)
            # stereo: ch0 = 데이터, ch1 = 0 (RPi I2S와 동일)
            stereo = np.zeros(len(int32_audio) * 2, dtype=np.int32)
            stereo[0::2] = int32_audio
            return stereo.tobytes()


import json
import urllib.request
import math

def fetch_spatial_info(server_ip, module_id):
    """서버에서 현재 모듈의 회전 벡터와 각도 정보를 HTTP로 가져옵니다."""
    try:
        url = f"http://{server_ip}:8000/api/status"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode())
            mid_str = str(module_id)
            if "modules" in data and mid_str in data["modules"]:
                mod = data["modules"][mid_str]
                return mod.get("rotation_angle", 0.0), mod.get("rotation_vector", [0.0, 1.0])
    except Exception as e:
        print(f"  ⚠️ 공간 정보 동기화 실패: {e}")
    return 0.0, [0.0, 1.0]

def play_audio(audio_bytes, angle=0.0, content_type="audio/wav"):
    """WAV 데이터 재생 및 각도 기반 공간음향(Panning) 적용"""
    ext = ".wav" if "wav" in content_type else ".mp3"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    import platform

    # 각도에 따른 Left/Right 스테레오 패닝 계산
    rad = math.radians(angle)
    balance = math.sin(rad)
    right_vol = (balance + 1.0) / 2.0
    left_vol = 1.0 - right_vol
    pan_filter = f"pan=stereo|c0={left_vol:.2f}*c0|c1={right_vol:.2f}*c1"

    print(f"  🔊 재생 중... (L: {left_vol:.2f}, R: {right_vol:.2f})")
    
    if platform.system() == "Darwin":
        # afplay는 패닝 옵션이 없으므로 단순 재생 (Mac 환경)
        # 패닝이 꼭 필요하면 ffplay나 sox로 대체 가능
        subprocess.run(["afplay", tmp_path], check=True)
    else:
        # 리눅스 환경: ffplay가 있으면 패닝을 입혀 재생 시도
        try:
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-af", pan_filter, tmp_path],
                check=True, capture_output=True, timeout=30,
            )
        except Exception:
            subprocess.run(["aplay", tmp_path] if ext == ".wav" else ["mpg123", "-q", tmp_path], check=True)
    os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="192.168.0.100")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--module-id", type=int, default=1, choices=range(1, 8))
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--record-after", type=int, default=4)
    parser.add_argument("--cooldown", type=float, default=1)
    args = parser.parse_args()

    print("=" * 60)
    print(f"  모듈 #{args.module_id} (TCP 테스트 클라이언트)")
    print(f"  서버: {args.server}:{args.port}")
    print("=" * 60)

    # TCP 연결
    print(f"  🔌 서버 연결 중...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((args.server, args.port))
        # 핸드셰이크: 모듈 ID 전송 (1바이트)
        sock.sendall(bytes([args.module_id]))
        print(f"  ✅ 연결 성공!")
    except Exception as e:
        print(f"  ❌ 연결 실패: {e}")
        return

    print(f"\n👂 빔 대기 중... (Ctrl+C로 종료)")
    print("-" * 60)

    # 수신 스레드: 서버 응답(WAV) 수신 → 재생
    def recv_worker():
        cycle = 0
        try:
            while True:
                header = recv_exact(sock, 4)
                if not header:
                    print("  🔌 서버 연결 끊김")
                    break
                length = int.from_bytes(header, "big")
                wav_data = recv_exact(sock, length)
                if not wav_data:
                    break

                cycle += 1
                print(f"\n── 응답 {cycle} ({length} bytes) ──")
                
                # 서버로부터 현재 지향각 가져오기
                angle, vector = fetch_spatial_info(args.server, args.module_id)
                print(f"  📍 빔 지향각: {angle:.1f}° | 벡터: ({vector[0]:.2f}, {vector[1]:.2f})")
                
                play_audio(wav_data, angle)
                print(f"  ✅ 재생 완료!")
        except (ConnectionResetError, OSError) as e:
            print(f"  ❌ 수신 에러: {e}")

    t_recv = threading.Thread(target=recv_worker, daemon=True)
    t_recv.start()

    # 메인 루프: 빔 감지 → 녹음 → TCP 전송
    cycle = 0
    while True:
        try:
            raw_pcm = listen_and_convert(args.threshold, args.record_after)
            cycle += 1
            print(f"\n── 사이클 {cycle} ──")

            # raw PCM을 chunk 단위로 전송 (서버가 SRC_CHUNK_BYTES씩 읽으므로)
            offset = 0
            while offset < len(raw_pcm):
                end = offset + SRC_CHUNK_BYTES
                chunk = raw_pcm[offset:end]
                # 마지막 chunk가 짧으면 0으로 패딩
                if len(chunk) < SRC_CHUNK_BYTES:
                    chunk = chunk + b'\x00' * (SRC_CHUNK_BYTES - len(chunk))
                sock.sendall(chunk)
                offset = end

            print(f"  📡 전송 완료 ({len(raw_pcm)} bytes)")
            print(f"  💤 쿨다운 {args.cooldown}초...")
            time.sleep(args.cooldown)
            print(f"\n👂 빔 대기 중...")
        except KeyboardInterrupt:
            print("\n\n🛑 종료")
            break
        except Exception as e:
            print(f"  ❌ 에러: {e}")
            time.sleep(2)
            continue

    sock.close()


if __name__ == "__main__":
    main()
