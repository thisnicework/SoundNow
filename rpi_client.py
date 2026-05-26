#!/usr/bin/env python3
"""
RPi TCP 스트리밍 클라이언트
arecord → TCP 소켓으로 raw PCM 상시 전송
서버에서 응답 WAV 오면 재생

사용법: python3 rpi_client.py --server 192.168.0.100 --module 1
"""

import subprocess
import tempfile
import threading
import socket
import os
import time
import argparse

# ============ 설정 ============
SERVER_IP = "192.168.0.100"
TCP_PORT = 9000
MODULE_ID = 1
COOLDOWN = 1.0

ALSA_DEVICE_REC = "hw:sndrpigooglevoi"
ALSA_DEVICE_PLAY = "plughw:sndrpigooglevoi"
RATE = 48000
CHANNELS = 2
FORMAT = "S32_LE"
BYTES_PER_SAMPLE = 4
CHUNK_SEC = 0.1
CHUNK_BYTES = int(RATE * CHUNK_SEC) * CHANNELS * BYTES_PER_SAMPLE  # 38400

# ============ 상태 ============
playing = threading.Event()


def open_mic():
    """arecord 스트림 시작"""
    return subprocess.Popen([
        "arecord", "-D", ALSA_DEVICE_REC,
        "-f", FORMAT, "-r", str(RATE),
        "-c", str(CHANNELS), "-t", "raw", "-q",
    ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def recv_exact(sock, n):
    """소켓에서 정확히 n바이트 수신"""
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def send_worker(proc, sock):
    """스레드 1: arecord → 소켓 전송 (재생 중에는 읽기만 하고 전송 안 함)"""
    print("  📡 전송 스레드 시작")
    try:
        while True:
            chunk = proc.stdout.read(CHUNK_BYTES)
            if not chunk:
                break
            if not playing.is_set():
                sock.sendall(chunk)
    except (BrokenPipeError, OSError):
        print("  ❌ 전송 연결 끊김")


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

def recv_worker(sock):
    """스레드 2: 소켓에서 응답 수신 → 재생"""
    print("  📥 수신 스레드 시작")
    cycle = 0
    try:
        while True:
            # 4바이트 길이 헤더 읽기
            header = recv_exact(sock, 4)
            if not header:
                break
            length = int.from_bytes(header, "big")

            # WAV 데이터 읽기
            wav_data = recv_exact(sock, length)
            if not wav_data:
                break

            cycle += 1
            print(f"\n── 응답 {cycle} ({length} bytes) ──")

            # 서버로부터 현재 지향각 가져오기
            angle, vector = fetch_spatial_info(SERVER_IP, MODULE_ID)
            print(f"  📍 빔 지향각: {angle:.1f}° | 벡터: ({vector[0]:.2f}, {vector[1]:.2f})")

            # 재생 중 플래그 → 전송 중단
            playing.set()
            play_audio(wav_data, angle)

            print(f"  💤 쿨다운 {COOLDOWN}초...")
            time.sleep(COOLDOWN)

            # 플래그 해제 → 전송 재개
            playing.clear()
            print(f"\n👂 빔 대기 중...")

    except (ConnectionResetError, OSError):
        print("  ❌ 수신 연결 끊김")


def play_audio(wav_data, angle=0.0):
    """WAV 데이터 재생 및 각도 기반 공간음향(Panning) 필터 적용"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_data)
        tmp_path = f.name

    # 각도(-180 ~ 180)에 따른 Left/Right 스테레오 패닝 계산
    # 0도: 중앙, -90도: 왼쪽, 90도: 오른쪽, 180/-180도: 중앙 (뒤)
    rad = math.radians(angle)
    # sin(angle)을 사용하여 L/R 밸런스를 계산 (-1: Left, 1: Right)
    balance = math.sin(rad)
    # 0.0(Mute) ~ 1.0(Max) 범위로 스케일링
    right_vol = (balance + 1.0) / 2.0
    left_vol = 1.0 - right_vol
    
    pan_filter = f"pan=stereo|c0={left_vol:.2f}*c0|c1={right_vol:.2f}*c1,volume=2.0"

    try:
        print(f"  🔊 재생 중... (L: {left_vol:.2f}, R: {right_vol:.2f})")
        try:
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-af", pan_filter, tmp_path],
                check=True, capture_output=True, timeout=30,
            )
        except FileNotFoundError:
            # ffplay가 없는 환경을 위해 aplay 폴백 (패닝 미적용)
            subprocess.run(
                ["aplay", "-D", ALSA_DEVICE_PLAY, tmp_path],
                check=True, capture_output=True, timeout=30,
            )
        print(f"  ✅ 재생 완료")
    except Exception as e:
        print(f"  ❌ 재생 실패: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def main():
    global SERVER_IP, TCP_PORT, MODULE_ID, COOLDOWN

    parser = argparse.ArgumentParser(description="RPi TCP 스트리밍 클라이언트")
    parser.add_argument("--server", type=str, default=SERVER_IP)
    parser.add_argument("--port", type=int, default=TCP_PORT)
    parser.add_argument("--module", type=int, default=MODULE_ID)
    parser.add_argument("--cooldown", type=float, default=COOLDOWN)
    parser.add_argument("--loop", action="store_true", help="호환용 (무시됨)")
    args = parser.parse_args()

    SERVER_IP = args.server
    TCP_PORT = args.port
    MODULE_ID = args.module
    COOLDOWN = args.cooldown

    print("=" * 60)
    print(f"  🎙️  RPi TCP 스트리밍 클라이언트")
    print(f"  모듈 #{MODULE_ID}")
    print(f"  서버: {SERVER_IP}:{TCP_PORT}")
    print("=" * 60)

    # TCP 연결
    print(f"  🔌 서버 연결 중...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((SERVER_IP, TCP_PORT))
        # 핸드셰이크: 모듈 ID 전송 (1바이트)
        sock.sendall(bytes([MODULE_ID]))
        print(f"  ✅ 연결 성공!")
    except Exception as e:
        print(f"  ❌ 연결 실패: {e}")
        return

    # arecord 시작
    proc = open_mic()
    print(f"\n👂 빔 대기 중... (Ctrl+C로 종료)")
    print("-" * 60)

    # 전송/수신 스레드 시작
    t_send = threading.Thread(target=send_worker, args=(proc, sock), daemon=True)
    t_recv = threading.Thread(target=recv_worker, args=(sock,), daemon=True)
    t_send.start()
    t_recv.start()

    try:
        while t_send.is_alive() and t_recv.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\n🛑 종료")
    finally:
        proc.kill()
        proc.wait()
        sock.close()


if __name__ == "__main__":
    main()
