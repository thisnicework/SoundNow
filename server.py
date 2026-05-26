"""
사운드 인스톨레이션 — Mac mini 서버 (TCP 스트리밍)
RPi에서 raw PCM을 TCP로 상시 수신 → Silero VAD로 음성 감지 →
Whisper STT → EXAONE LLM → Qwen3-TTS → TCP로 응답

사용법: python3 server.py
"""

import os
import io
import re
import json
import time
import wave
import socket
import threading
import queue
import urllib.request
import subprocess
import sys
import platform
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
HTTP_PORT = 8000          # HTTP (control_server 호환)
TCP_PORT = 9000           # TCP 스트리밍

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "exaone3.5:7.8b"
LLM_TEMPERATURE = 0.7   # sweep 결과 0.7이 0.8/0.9보다 살짝 더 좋음 (composite +0.7)

WHISPER_MODEL_SIZE = "medium"
TTS_MODEL_NAME = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16"

VOLUME_DB = 8
SAMPLE_RATE = 16000       # Whisper / VAD 입력

# RPi 소스 포맷
SRC_RATE = 48000
SRC_CHANNELS = 2
SRC_SAMPLE_BYTES = 4      # S32_LE
SRC_CHUNK_SEC = 0.1
SRC_CHUNK_BYTES = int(SRC_RATE * SRC_CHUNK_SEC) * SRC_CHANNELS * SRC_SAMPLE_BYTES  # 38400

# VAD 설정
VAD_THRESHOLD = 0.15
VAD_MIN_SILENCE_MS = 700  # 음성 끝 판단: 0.7초 무음
VAD_PRE_BUFFER_SEC = 1.5  # 음성 시작 전 보존 (늘림)

MODULE_MAP = {
    1: "모듈1", 2: "모듈2", 3: "모듈3",
    4: "모듈4", 5: "모듈5", 6: "모듈6",
}

MODULE_VOICE = {
    1: "sohee", 2: "sohee", 3: "sohee",
    4: "sohee", 5: "sohee", 6: "sohee",
}

# ─────────────────────────────────────────
# 모듈 제어 및 모니터링 변수 & 기능 추가
# ─────────────────────────────────────────
import math

CONFIG_FILE = "config.json"
DEFAULT_CONFIGS = {
    "rotation_mode": "loop", # "loop" 또는 "center"
    "1": {"ip": "127.0.0.1", "name": "모듈 1", "voice": "sohee", "x": -2.0, "y": 2.0},
    "2": {"ip": "127.0.0.1", "name": "모듈 2", "voice": "sohee", "x": 2.0, "y": 2.0},
    "3": {"ip": "127.0.0.1", "name": "모듈 3", "voice": "sohee", "x": 0.0, "y": -2.0}
}
module_configs = {}

local_processes = {1: None, 2: None, 3: None}
active_connections = {}  # {module_id: socket_conn}

activity_logs = []
activity_logs_lock = threading.Lock()

module_runtime_status = {
    1: {"agent_status": "offline", "cpu": 0.0, "tcp_connected": False},
    2: {"agent_status": "offline", "cpu": 0.0, "tcp_connected": False},
    3: {"agent_status": "offline", "cpu": 0.0, "tcp_connected": False}
}

def calculate_rotation(x, y, tx, ty):
    dx = tx - x
    dy = ty - y
    dist = math.sqrt(dx*dx + dy*dy)
    if dist > 0:
        u_dx = round(dx / dist, 4)
        u_dy = round(dy / dist, 4)
        angle = round(math.atan2(dy, dx) * 180 / math.pi, 1)
        return [u_dx, u_dy], angle, round(dist, 2)
    return [0.0, 0.0], 0.0, 0.0

def calculate_all_rotations(configs):
    m1 = configs.get("1", {"x": -2.0, "y": 2.0})
    m2 = configs.get("2", {"x": 2.0, "y": 2.0})
    m3 = configs.get("3", {"x": 0.0, "y": -2.0})
    
    x1, y1 = float(m1.get("x", -2.0)), float(m1.get("y", 2.0))
    x2, y2 = float(m2.get("x", 2.0)), float(m2.get("y", 2.0))
    x3, y3 = float(m3.get("x", 0.0)), float(m3.get("y", -2.0))
    
    mode = configs.get("rotation_mode", "loop")
    results = {}
    
    if mode == "loop":
        vec1, ang1, dist1 = calculate_rotation(x1, y1, x2, y2)
        vec2, ang2, dist2 = calculate_rotation(x2, y2, x3, y3)
        vec3, ang3, dist3 = calculate_rotation(x3, y3, x1, y1)
        
        results["1"] = {"vector": vec1, "angle": ang1, "distance": dist1, "target": [x2, y2]}
        results["2"] = {"vector": vec2, "angle": ang2, "distance": dist2, "target": [x3, y3]}
        results["3"] = {"vector": vec3, "angle": ang3, "distance": dist3, "target": [x1, y1]}
    else:
        cx = round((x1 + x2 + x3) / 3, 2)
        cy = round((y1 + y2 + y3) / 3, 2)
        
        vec1, ang1, dist1 = calculate_rotation(x1, y1, cx, cy)
        vec2, ang2, dist2 = calculate_rotation(x2, y2, cx, cy)
        vec3, ang3, dist3 = calculate_rotation(x3, y3, cx, cy)
        
        results["1"] = {"vector": vec1, "angle": ang1, "distance": dist1, "target": [cx, cy]}
        results["2"] = {"vector": vec2, "angle": ang2, "distance": dist2, "target": [cx, cy]}
        results["3"] = {"vector": vec3, "angle": ang3, "distance": dist3, "target": [cx, cy]}
        results["centroid"] = {"x": cx, "y": cy}
        
    return results

_server_last_cpu_time = [0.0, 0.0]

def get_server_cpu_usage():
    global _server_last_cpu_time
    import os
    import platform
    import subprocess
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(['ps', '-A', '-o', '%cpu']).decode()
            lines = out.strip().split('\n')[1:]
            total = sum(float(x) for x in lines if x.strip())
            cores = os.cpu_count() or 1
            return min(100.0, round(total / cores, 1))
        else:
            with open('/proc/stat', 'r') as f:
                line = f.readline()
            parts = line.split()
            if len(parts) >= 5:
                idle = float(parts[4]) + float(parts[5])
                total = sum(float(x) for x in parts[1:8])
                prev_idle, prev_total = _server_last_cpu_time
                _server_last_cpu_time = [idle, total]
                diff_idle = idle - prev_idle
                diff_total = total - prev_total
                if diff_total > 0:
                    usage = (1.0 - diff_idle / diff_total) * 100.0
                    return min(100.0, round(usage, 1))
    except Exception:
        pass
    try:
        load = os.getloadavg()[0]
        cores = os.cpu_count() or 1
        return min(100.0, round((load / cores) * 100.0, 1))
    except Exception:
        return 0.0

def load_configs():
    global module_configs
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                module_configs = json.load(f)
                for m in DEFAULT_CONFIGS:
                    if m not in module_configs:
                        module_configs[m] = DEFAULT_CONFIGS[m]
            return
        except Exception as e:
            print(f"⚠️ 설정 로드 실패: {e}")
    module_configs = DEFAULT_CONFIGS.copy()
    save_configs()

def save_configs():
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(module_configs, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"⚠️ 설정 저장 실패: {e}")

def add_activity_log(module_id, event_type, message):
    with activity_logs_lock:
        log_entry = {
            "time": time.strftime("%H:%M:%S"),
            "module_id": module_id,
            "type": event_type,  # "stt", "llm", "tts", "system"
            "message": message
        }
        activity_logs.append(log_entry)
        if len(activity_logs) > 50:
            activity_logs.pop(0)

def monitor_loop():
    while True:
        try:
            # 1) TCP Connection status
            for mid in [1, 2, 3]:
                module_runtime_status[mid]["tcp_connected"] = (mid in active_connections)
                
            # 2) Agent status & CPU status
            for mid in [1, 2, 3]:
                cfg = module_configs.get(str(mid), {})
                ip = cfg.get("ip", "127.0.0.1")
                
                if ip in ["127.0.0.1", "localhost"]:
                    # Local Client
                    proc = local_processes.get(mid)
                    if proc and proc.poll() is None:
                        module_runtime_status[mid]["agent_status"] = "running"
                        module_runtime_status[mid]["cpu"] = 1.2
                    else:
                        module_runtime_status[mid]["agent_status"] = "stopped"
                        module_runtime_status[mid]["cpu"] = 0.0
                        local_processes[mid] = None
                else:
                    # Remote Client
                    try:
                        req = urllib.request.Request(f"http://{ip}:8001/status", method="GET")
                        with urllib.request.urlopen(req, timeout=1.5) as resp:
                            res_data = json.loads(resp.read().decode("utf-8"))
                            module_runtime_status[mid]["agent_status"] = res_data.get("status", "stopped")
                            module_runtime_status[mid]["cpu"] = res_data.get("cpu", 0.0)
                    except Exception:
                        module_runtime_status[mid]["agent_status"] = "offline"
                        module_runtime_status[mid]["cpu"] = 0.0
        except Exception:
            pass
        time.sleep(2)

def start_module(module_id):
    cfg = module_configs.get(str(module_id), {})
    ip = cfg.get("ip", "127.0.0.1")
    add_activity_log(module_id, "system", f"모듈 시작 중: IP={ip}")
    
    if ip in ["127.0.0.1", "localhost"]:
        proc = local_processes.get(module_id)
        if proc and proc.poll() is None:
            return False, "이미 실행 중"
        try:
            local_processes[module_id] = subprocess.Popen([
                sys.executable, "client.py",
                "--server", "127.0.0.1",
                "--port", str(TCP_PORT),
                "--module-id", str(module_id)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            add_activity_log(module_id, "system", f"로컬 client.py 실행됨 (PID: {local_processes[module_id].pid})")
            return True, "시작됨 (로컬)"
        except Exception as e:
            add_activity_log(module_id, "system", f"로컬 클라이언트 실행 실패: {e}")
            return False, str(e)
    else:
        try:
            req_url = f"http://{ip}:8001/start/{module_id}"
            req = urllib.request.Request(req_url, method="POST")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                if res_data.get("ok"):
                    add_activity_log(module_id, "system", "원격 클라이언트 실행됨")
                    return True, "시작됨 (원격)"
                else:
                    msg = res_data.get("message", "에러 발생")
                    add_activity_log(module_id, "system", f"에이전트 클라이언트 실행 실패: {msg}")
                    return False, msg
        except Exception as e:
            add_activity_log(module_id, "system", f"에이전트 연결 실패: {e}")
            return False, f"에이전트 연결 실패: {e}"

def stop_module(module_id):
    cfg = module_configs.get(str(module_id), {})
    ip = cfg.get("ip", "127.0.0.1")
    add_activity_log(module_id, "system", "모듈 정지 중")
    
    if ip in ["127.0.0.1", "localhost"]:
        proc = local_processes.get(module_id)
        if not proc or proc.poll() is not None:
            local_processes[module_id] = None
            return False, "실행 중이 아님"
        try:
            proc.terminate()
            proc.wait(timeout=2)
            local_processes[module_id] = None
            add_activity_log(module_id, "system", "로컬 클라이언트 정지됨")
            return True, "정지됨 (로컬)"
        except Exception:
            try:
                proc.kill()
                local_processes[module_id] = None
                add_activity_log(module_id, "system", "로컬 클라이언트 강제 정지됨")
                return True, "강제 종료됨 (로컬)"
            except Exception as e:
                return False, str(e)
    else:
        try:
            req_url = f"http://{ip}:8001/stop"
            req = urllib.request.Request(req_url, method="POST")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                if res_data.get("ok"):
                    add_activity_log(module_id, "system", "원격 클라이언트 정지됨")
                    return True, "정지됨 (원격)"
                else:
                    msg = res_data.get("message", "에러 발생")
                    add_activity_log(module_id, "system", f"에이전트 클라이언트 정지 실패: {msg}")
                    return False, msg
        except Exception as e:
            add_activity_log(module_id, "system", f"에이전트 연결 실패: {e}")
            return False, f"에이전트 연결 실패: {e}"

PROMPT = """너는 동네에서 소문을 옮기는 사람이다.
누가 한 말을 듣고, 한두 군데만 살짝 비틀어서 다른 사람한테 전한다.
완전히 다른 이야기를 지어내면 안 된다 — 그건 거짓말이지 소문이 아니다.

[종결 — 무조건 이렇게]
~했대 / ~한대 / ~된대 / ~래 / ~라더라 / ~다더라
"~했어/~해/~한다"는 금지.

[비틀기 방향 — 매번 다른 하나를 고르되, 작게 비튼다]
- 사건추가: 일상에서 일어날 만한 작은 사건이 더해짐 (체했대, 늦었대, 깜빡했대)
- 감정톤: 평범한 일이 부끄럽거나 안쓰럽거나 우습게 들리게
- 스케일: 양이나 횟수가 조금 부풀려지거나 줄어듦 (두 그릇, 백 미터, 세 시간)
- 관계: 주어가 다른 사람으로 바뀜 (학생이, 친구가, 옆집이)
- 시간: 시점이 살짝 달라짐 (어제→그저께, 오늘→내일)
- 인과반전: 원인과 결과가 살짝 뒤집힘

[절대 쓰지 말 것]
우주, 외계인, UFO, 마법, 공룡, 용, 유령, 로봇, 초능력,
"갑자기 하늘에서 ~가 떨어졌대",
"마법처럼 사라졌대",
"폭풍우가 몰아쳤대" 류의 극단적 자연재해 클리셰.
현실에서 그럴 듯한 범위 안에서만 비틀어라.

[형식 규칙]
- 한 줄 한 문장. 따옴표·번호·괄호·설명·기호 금지.
- 들은 말의 단어 절반은 유지하고, 한두 단어만 새로 끼워라.
- 단어 한 개, 감탄사 입력도 일상 범위의 한 줄 소문으로.

[예시 — 완전한 문장]
들은 말: 교수님이 점심에 돈가스를 드셨대
기억나는 대로: 교수님이 점심에 돈가스를 두 그릇이나 드셨대

들은 말: 교수님이 점심에 돈가스를 드셨대
기억나는 대로: 교수님이 점심에 돈가스 먹다가 사레들리셨대

들은 말: 교수님이 점심에 돈가스를 드셨대
기억나는 대로: 교수님이 점심값을 학생한테 내게 하셨다더라

들은 말: 교수님이 점심에 돈가스를 드셨대
기억나는 대로: 교수님이 점심에 돈가스 안 드시고 회의만 하셨대

들은 말: 형이 차를 새로 샀어
기억나는 대로: 형이 차 사놓고 한 번도 안 탔다더라

들은 말: 형이 차를 새로 샀어
기억나는 대로: 형이 산 차가 알고 보니 중고였대

들은 말: 형이 차를 새로 샀어
기억나는 대로: 형이 새 차 첫날부터 긁어 먹었다더라

들은 말: 친구가 선물을 보냈어
기억나는 대로: 친구가 선물 보내놓고 영수증을 같이 넣어 보냈다더라

들은 말: 친구가 선물을 보냈어
기억나는 대로: 친구가 보낸 선물이 알고 보니 자기가 받은 거였대

들은 말: 친구가 선물을 보냈어
기억나는 대로: 친구가 선물 보낼 주소를 잘못 적었다더라

들은 말: 엄마가 집에 오라고 했어
기억나는 대로: 엄마가 집에 오라고 다섯 번이나 전화하셨대

들은 말: 엄마가 집에 오라고 했어
기억나는 대로: 엄마가 집에 오라더니 정작 본인은 외출 중이셨대

들은 말: 비가 올 것 같으니까 우산 챙겨
기억나는 대로: 우산 챙기라더니 막상 본인은 안 챙겨 나갔다더라

들은 말: 비가 올 것 같으니까 우산 챙겨
기억나는 대로: 우산 챙겨 나왔는데 결국 비 한 방울도 안 왔다더라

들은 말: 동생이 학교를 안 갔어
기억나는 대로: 동생이 학교 안 가고 PC방에 있었다더라

들은 말: 동생이 학교를 안 갔어
기억나는 대로: 동생이 학교 안 갔다고 엄마한테 들켜서 혼났다더라

들은 말: 내일 시험이 있어서 공부해
기억나는 대로: 내일 시험인데 어제 밤새 게임만 했대

들은 말: 내일 시험이 있어서 공부해
기억나는 대로: 내일 시험인데 책을 펴자마자 잠들었다더라

들은 말: 택배가 오늘 온대
기억나는 대로: 택배가 오늘 온다더니 옆집으로 잘못 갔다더라

들은 말: 택배가 오늘 온대
기억나는 대로: 택배가 오늘 왔는데 시킨 거랑 다른 게 왔다더라

들은 말: 그 사람이 회사를 그만뒀대
기억나는 대로: 그 사람이 회사 그만두고 사흘 만에 다시 들어갔대

들은 말: 그 사람이 회사를 그만뒀대
기억나는 대로: 그 사람이 회사 그만둔 이유가 사장이랑 한판 했다더라

들은 말: 어제 친구를 만났어
기억나는 대로: 어제 친구 만났는데 약속 시간에 두 시간이나 늦었다더라

들은 말: 점심에 회의가 있대
기억나는 대로: 점심 회의가 한 시간을 넘겨서 다들 굶었다더라

들은 말: 강아지가 산책 갔어
기억나는 대로: 강아지가 산책 나갔다가 옆집 개랑 한판 붙었다더라

들은 말: 옆집이 이사 갔대
기억나는 대로: 옆집이 이사 가면서 짐을 다 두고 갔다더라

들은 말: 카페가 문을 닫았대
기억나는 대로: 카페가 문 닫은 지 일주일 됐는데 아무도 몰랐다더라

들은 말: 그 식당이 맛있대
기억나는 대로: 그 식당 가려고 두 시간 줄 섰는데 결국 못 먹고 왔대

들은 말: 사촌이 결혼한대
기억나는 대로: 사촌이 결혼식 날짜를 두 번이나 미뤘다더라

들은 말: 형이 운전면허 땄어
기억나는 대로: 형이 운전면허 따자마자 차를 긁었다더라

[예시 — 잘린 구문]
들은 말: 점심에 돈가스를
기억나는 대로: 점심에 돈가스 시켰는데 양이 너무 적었다더라

들은 말: 학교를 안
기억나는 대로: 학교 안 가고 도서관에 숨어 있었다더라

들은 말: 차를 새로
기억나는 대로: 차를 새로 뽑았는데 색깔이 마음에 안 든대

들은 말: 어제 강의실에서
기억나는 대로: 어제 강의실에서 졸다가 들켰다더라

들은 말: 우산 챙기라고
기억나는 대로: 우산 챙겨 나왔는데 비가 안 왔다더라

들은 말: 옆집 강아지가
기억나는 대로: 옆집 강아지가 자꾸 우리 집 마당에 와서 잔다더라

들은 말: 카페 앞에서
기억나는 대로: 카페 앞에서 누가 지갑을 떨어뜨리고 갔다더라

들은 말: 회사 끝나고
기억나는 대로: 회사 끝나고 다 같이 야근했다더라

들은 말: 시험 끝나면
기억나는 대로: 시험 끝나자마자 다 같이 노래방 갔다더라

들은 말: 결혼식 날
기억나는 대로: 결혼식 날 신랑이 늦게 왔다더라

[예시 — 단어 한 개]
들은 말: 돈가스
기억나는 대로: 돈가스집 앞에 줄이 백 미터까지 늘어섰다더라

들은 말: 교수님
기억나는 대로: 교수님이 오늘 강의 빼먹고 어디 가셨대

들은 말: 학교
기억나는 대로: 학교 화장실 변기가 어제 다 막혔다더라

들은 말: 우산
기억나는 대로: 우산 잃어버리고 비 다 맞고 들어왔다더라

들은 말: 시험
기억나는 대로: 시험 답안지를 통째로 백지로 냈다더라

들은 말: 회사
기억나는 대로: 회사 엘리베이터가 일주일째 고장이래

들은 말: 강의실
기억나는 대로: 강의실 의자가 어제 하나 부러졌다더라

들은 말: 택배
기억나는 대로: 택배 기사가 우리 집을 찾다가 길을 잃었다더라

들은 말: 친구
기억나는 대로: 친구가 약속 시간에 한 시간 늦게 왔대

들은 말: 카페
기억나는 대로: 카페가 갑자기 메뉴를 다 바꿨다더라

들은 말: 강아지
기억나는 대로: 강아지가 사료 그릇을 엎어 놓고 잤다더라

들은 말: 차
기억나는 대로: 차 키를 가방에 두고 회사 갔다더라

들은 말: 가방
기억나는 대로: 가방을 지하철에 두고 내렸다더라

들은 말: 비
기억나는 대로: 비 오는데 우산도 없이 한 시간을 걸어갔다더라

들은 말: 점심
기억나는 대로: 점심 약속 잡아놓고 정작 본인이 잊어버렸다더라

[예시 — 감탄사]
들은 말: 아
기억나는 대로: 누가 길에서 갑자기 크게 한숨 쉬더래

들은 말: 어
기억나는 대로: 옆집 아저씨가 밤늦게 혼잣말하더라

들은 말: 음
기억나는 대로: 회의 중에 누가 계속 코를 골았다더라

들은 말: 헉
기억나는 대로: 카페에서 누가 갑자기 컵을 떨어뜨렸다더라

들은 말: 엥
기억나는 대로: 옆자리 사람이 갑자기 일어나서 나가버렸다더라"""

HALLUCINATION_PHRASES = {
    "구독", "좋아요", "알림 설정", "알림설정",
    "시청해주셔서 감사합니다", "시청해 주셔서 감사합니다",
    "끝까지 봐주셔서", "다음 영상", "다음에 만나요",
    "MBC 뉴스", "KBS 뉴스", "SBS 뉴스",
    "자막제공", "자막 제공",
    "자막 제작자님", "자막제작자님", "제작자님 감사합니다",
    "수고하셨습니다", "수고하세요",
    "Thank you", "Thanks for watching", "Subscribe",
    "이 영상은", "본 영상은",
    "한글자막 by",
}


# ─────────────────────────────────────────
# 소문 종결 강제 변환 — "~했어" → "~했대" 등
# 모델이 입력 종결을 따라가는 경향이 있어 후처리로 강제
# ─────────────────────────────────────────
_RUMOR_ENDINGS = ("대", "래", "더라", "다더라", "라더라", "더래", "네", "군", "구나")
_ALREADY_RUMOR = re.compile(
    r"(?:" + "|".join(_RUMOR_ENDINGS) + r")[\.\!\?…]*$"
)
_ENDING_RULES = [
    (r"하셨어요?$", "하셨대"),
    (r"했어요?$", "했대"),
    (r"한다$", "한대"),
    (r"했다$", "했대"),
    (r"됐어요?$", "됐대"),
    (r"됐다$", "됐대"),
    (r"이었어요?$", "이었대"),
    (r"였어요?$", "였대"),
    (r"있어요?$", "있대"),
    (r"없어요?$", "없대"),
    (r"갔어요?$", "갔대"),
    (r"왔어요?$", "왔대"),
    (r"샀어요?$", "샀대"),
    (r"봤어요?$", "봤대"),
    (r"먹었어요?$", "먹었대"),
    (r"버렸어요?$", "버렸대"),
    (r"던졌어요?$", "던졌대"),
    (r"부러졌어요?$", "부러졌대"),
    (r"걸렸어요?$", "걸렸대"),
    (r"받았어요?$", "받았대"),
    (r"보냈어요?$", "보냈대"),
    (r"잃어버렸어요?$", "잃어버렸대"),
    (r"잊었어요?$", "잊었대"),
    (r"쳤어요?$", "쳤대"),
    (r"맞았어요?$", "맞았대"),
    (r"걸었어요?$", "걸었대"),
    (r"섰어요?$", "섰대"),
    (r"내렸어요?$", "내렸대"),
    (r"올렸어요?$", "올렸대"),
    (r"떨어졌어요?$", "떨어졌대"),
    (r"쏟았어요?$", "쏟았대"),
    (r"질렀어요?$", "질렀대"),
    (r"빌렸어요?$", "빌렸대"),
    (r"고쳤어요?$", "고쳤대"),
    (r"나왔어요?$", "나왔대"),
    (r"들었어요?$", "들었대"),
    (r"불렀어요?$", "불렀대"),
    (r"불었어요?$", "불었대"),
    (r"감았어요?$", "감았대"),
    (r"졌어요?$", "졌대"),
    (r"필요해졌어요?$", "필요해졌대"),
    (r"이뤘어요?$", "이뤘대"),
    (r"있어$", "있대"),
    (r"없어$", "없대"),
    (r"있다$", "있대"),
    (r"없다$", "없대"),
    (r"한다$", "한대"),
    (r"되어$", "된대"),
    (r"한대요$", "한대"),
    (r"이래요$", "이래"),
    (r"([가-힣])어$", r"\1었대"),
    (r"([가-힣])아$", r"\1았대"),
    (r"챙겨$", "챙겼대"),
    (r"가$", "갔대"),
    (r"와$", "왔대"),
]


def force_rumor_ending(text: str) -> str:
    """문장 마지막을 소문체로 강제 변환. 이미 소문체면 그대로."""
    t = text.strip().rstrip(".!?…")
    if not t:
        return text
    if _ALREADY_RUMOR.search(t):
        return t
    for pattern, replacement in _ENDING_RULES:
        new_t, n = re.subn(pattern, replacement, t)
        if n > 0:
            return new_t
    return t + "라고 하더라"


def is_hallucination(text: str) -> bool:
    if not text:
        return True
    t = text.strip()
    if len(t) < 2:
        return True
    for phrase in HALLUCINATION_PHRASES:
        if phrase in t:
            return True
    words = t.split()
    if len(words) >= 3 and len(set(words)) == 1:
        return True
    if len(t) >= 4 and len(set(t.replace(" ", ""))) <= 1:
        return True
    if re.search(r'(.{2,8}?)\1{3,}', t):
        return True
    return False


def boost_and_limit(audio_f32, db=15.0, ceiling=0.95):
    gain = 10.0 ** (db / 20.0)
    boosted = audio_f32.astype(np.float32) * gain
    abs_b = np.abs(boosted)
    over = abs_b > ceiling
    if over.any():
        sign = np.sign(boosted)
        excess = abs_b - ceiling
        compressed = ceiling + (1.0 - ceiling) * np.tanh(excess / (1.0 - ceiling))
        boosted = np.where(over, sign * compressed, boosted)
    return np.clip(boosted, -1.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────
# 모델
# ─────────────────────────────────────────
whisper_model = None
tts_model = None
tts_sample_rate = 24000
vad_model = None
vad_utils = None
active_modules = {}
processing_queue = queue.Queue()


def load_whisper():
    global whisper_model
    print(f"📦 Whisper ({WHISPER_MODEL_SIZE}) 로딩 중...")
    import whisper
    whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
    print(f"✅ Whisper ({WHISPER_MODEL_SIZE}) 준비 완료")


def load_vad():
    global vad_model, vad_utils
    print("📦 Silero VAD 로딩 중...")
    import torch
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    vad_model = model
    vad_utils = utils
    print("✅ Silero VAD 준비 완료")


def load_tts():
    global tts_model, tts_sample_rate
    print(f"📦 mlx-audio Qwen3-TTS 로딩 중... ({TTS_MODEL_NAME})")
    from mlx_audio.tts.utils import load_model
    tts_model = load_model(TTS_MODEL_NAME)
    sr = getattr(tts_model, "sample_rate", None)
    if sr:
        tts_sample_rate = int(sr)
    print(f"✅ Qwen3-TTS 준비 완료 (sr={tts_sample_rate})")


def check_ollama():
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            if any("exaone3.5:7.8b" in m for m in models):
                print("✅ Ollama + EXAONE 7.8B 준비 완료")
                return True
            else:
                print("❌ exaone3.5:7.8b 모델이 없습니다")
                return False
    except Exception as e:
        print(f"❌ Ollama 연결 실패: {e}")
        return False


# ─────────────────────────────────────────
# 오디오 변환
# ─────────────────────────────────────────
def pcm_to_vad(raw: bytes) -> np.ndarray:
    """S32 stereo 48kHz → float32 mono 16kHz (VAD용)"""
    from scipy.signal import resample_poly, butter, sosfilt
    audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    audio = audio[0::2]   # ch0만
    # 5kHz 로패스 (8차) — 초음파 앨리어싱 제거
    sos = butter(8, 5000, btype='low', fs=48000, output='sos')
    audio = sosfilt(sos, audio).astype(np.float32)
    audio = resample_poly(audio, 1, 3).astype(np.float32)  # 48k → 16k
    audio = boost_and_limit(audio, db=15, ceiling=0.9)
    return audio


def pcm_to_whisper(raw: bytes) -> np.ndarray:
    """S32 stereo 48kHz → float32 mono 16kHz (고품질, Whisper용)
    초지향성 스피커의 40kHz 캐리어가 48kHz ADC에서 8kHz로 앨리어싱되므로
    로패스 필터로 제거한 후 리샘플링."""
    from scipy.signal import resample_poly, butter, sosfilt

    audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    audio = audio[0::2]   # ch0만

    # 48kHz에서 5kHz 로패스 (8차) — 8kHz 앨리어싱을 ~40dB 감쇠
    # 음성 기본주파수(~300Hz)와 포먼트(~3kHz)는 보존됨
    sos = butter(8, 5000, btype='low', fs=48000, output='sos')
    audio = sosfilt(sos, audio).astype(np.float32)

    # 48k → 16k 리샘플링
    audio = resample_poly(audio, 1, 3).astype(np.float32)

    # 피크 정규화
    peak = np.abs(audio).max()
    if peak > 0.001:
        audio = audio * (0.8 / peak)
    return audio


# ─────────────────────────────────────────
# 파이프라인 (do_llm, do_tts 기존 유지)
# ─────────────────────────────────────────
def do_llm(fragment: str) -> str:
    if not fragment or len(fragment.strip()) == 0:
        return ""

    full_prompt = PROMPT + "\n\n들은 말: '" + fragment + "'\n기억나는 대로:"

    req_data = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "num_predict": 25,
            "temperature": LLM_TEMPERATURE,
        }
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=req_data,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
        answer = result.get("response", "").strip()

    if "\n" in answer:
        answer = answer.split("\n")[0]
    answer = answer.strip("'\"")
    if "→" in answer:
        answer = answer.split("→")[-1].strip().strip("'\"")
    # 예시 형식 접두사 제거: "들은 말:", "기억나는 대로:", "원문:" 등
    answer = re.sub(
        r"^\s*(?:들은\s*말|기억나는\s*대로|원문|원래|입력|출력|답변|결과)\s*[:：]\s*",
        "", answer,
    )
    # 변형 축 코드 제거: (A), (B), (G) 등 1~3자 영문/한글 괄호
    answer = re.sub(r"\s*[\(\[（［][A-Za-z가-힣]{1,3}[\)\]）］]\s*", " ", answer)
    # 문장 중간/끝의 단독 따옴표 제거
    answer = answer.replace("'", "").replace('"', "")
    # 다중 공백 정리
    answer = re.sub(r"\s+", " ", answer).strip()
    # 이중 종결 정리:
    answer = re.sub(r"더라\s*더라$", "더라", answer)
    answer = re.sub(r"(대|래)\s+(라더라|다더라|더라)$", r"\1", answer)
    answer = re.sub(r"했대더라$", "했다더라", answer)
    answer = re.sub(r"됐대더라$", "됐다더라", answer)
    answer = re.sub(r"였대더라$", "였다더라", answer)
    answer = re.sub(r"이래더라$", "이라더라", answer)
    answer = re.sub(r"대요라고\s*하더라$", "대", answer)
    answer = re.sub(r"래요라고\s*하더라$", "래", answer)
    # 마침표 뒤 또 다른 문장이 붙은 경우: 첫 문장만
    if "." in answer[:-1]:
        answer = answer.split(".")[0]
    # 반복 패턴 제거
    answer = re.sub(r'(.{3,}?)\1{2,}', r'\1', answer)
    if len(answer) > 80:
        parts = [p.strip() for p in answer.split(",") if p.strip()]
        if len(parts) > 2:
            answer = ", ".join(parts[:2])
        else:
            answer = answer[:80]
    if len(answer) < 2:
        answer = fragment

    # 종결을 소문체로 강제 ("~했어" → "~했대" 등)
    answer = force_rumor_ending(answer)

    return answer


def do_tts(text: str, module_id: int) -> bytes:
    speaker = MODULE_VOICE.get(module_id, "sohee")

    results = list(tts_model.generate_custom_voice(
        text=text,
        language="Korean",
        speaker=speaker,
        instruct="Speak slowly and clearly in Korean without any filler words",
    ))

    audio_chunks = []
    for r in results:
        a = getattr(r, "audio", None)
        if a is None:
            continue
        audio_chunks.append(np.asarray(a).astype(np.float32))

    if not audio_chunks:
        raise RuntimeError("TTS produced no audio")

    audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]

    sr = (
        getattr(results[0], "sample_rate", None)
        or getattr(tts_model, "sample_rate", None)
        or tts_sample_rate
    )
    sr = int(sr)

    audio = boost_and_limit(audio, db=VOLUME_DB, ceiling=0.95)

    pcm = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# ─────────────────────────────────────────
# TCP 스트림 핸들러
# ─────────────────────────────────────────
def recv_exact(sock, n):
    """소켓에서 정확히 n바이트 수신"""
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def handle_stream(conn, module_id):
    """모듈 하나의 TCP 스트림 처리 — 적응형 버퍼 + get_speech_timestamps"""
    import torch

    # 스레드별 독립 VAD 모델
    thread_vad, thread_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    get_speech_timestamps = thread_utils[0]

    BUFFER_SEC = 5.0
    BUFFER_CHUNKS = int(BUFFER_SEC / SRC_CHUNK_SEC)       # 50 = 5초
    CHECK_INTERVAL = 10                                     # 음성 진행 중 1초마다 재분석
    MAX_BUFFER_SEC = 15.0
    MAX_BUFFER_CHUNKS = int(MAX_BUFFER_SEC / SRC_CHUNK_SEC) # 150 = 15초
    TAIL_MARGIN = int(0.5 * SAMPLE_RATE)                   # 8000 samples = 0.5초

    active_modules[module_id] = "스트리밍"
    active_connections[module_id] = conn
    try:
        peer = conn.getpeername()
        add_activity_log(module_id, "system", f"TCP 클라이언트 연결됨 ({peer[0]}:{peer[1]})")
    except Exception:
        add_activity_log(module_id, "system", "TCP 클라이언트 연결됨")
    print(f"  🎤 모듈 {module_id}: 스트림 감시 시작 (버퍼 {BUFFER_SEC}~{MAX_BUFFER_SEC}초)")

    raw_buffer = []
    speech_ongoing = False    # 버퍼 끝에서 음성이 이어지고 있는지
    chunks_since_check = 0    # 마지막 분석 이후 청크 수

    try:
        while True:
            raw = recv_exact(conn, SRC_CHUNK_BYTES)
            if not raw:
                print(f"  🔌 모듈 {module_id}: 연결 종료")
                break

            raw_buffer.append(raw)
            chunks_since_check += 1

            # 분석 타이밍 결정
            should_analyze = False
            if not speech_ongoing and len(raw_buffer) >= BUFFER_CHUNKS:
                should_analyze = True  # 첫 5초 도달
            elif speech_ongoing and chunks_since_check >= CHECK_INTERVAL:
                should_analyze = True  # 음성 진행 중 1초마다 재분석
            if len(raw_buffer) >= MAX_BUFFER_CHUNKS:
                should_analyze = True  # 15초 안전장치

            if not should_analyze:
                continue

            chunks_since_check = 0
            all_raw = b"".join(raw_buffer)
            audio = pcm_to_whisper(all_raw)
            buf_sec = len(audio) / SAMPLE_RATE
            vol_mean = np.abs(audio).mean()
            print(f"  📊 모듈 {module_id} | {buf_sec:.1f}초 | mean={vol_mean:.3f} max={np.abs(audio).max():.3f}")

            audio_tensor = torch.from_numpy(audio)
            timestamps = get_speech_timestamps(
                audio_tensor, thread_vad,
                sampling_rate=SAMPLE_RATE,
                threshold=VAD_THRESHOLD,
                min_speech_duration_ms=300,
                min_silence_duration_ms=500,
            )

            if not timestamps:
                print(f"  🔇 모듈 {module_id} | 음성 없음")
                speech_ongoing = False
                raw_buffer = raw_buffer[-10:]  # 마지막 1초만 유지
                continue

            total_ms = sum(t["end"] - t["start"] for t in timestamps) / SAMPLE_RATE * 1000
            last_end = timestamps[-1]["end"]
            speech_at_tail = last_end > (len(audio) - TAIL_MARGIN)

            if speech_at_tail and len(raw_buffer) < MAX_BUFFER_CHUNKS:
                # 음성이 버퍼 끝까지 이어짐 → 계속 축적
                if not speech_ongoing:
                    print(f"  🎤 모듈 {module_id} | 음성 진행 중 ({total_ms:.0f}ms) — 축적 계속")
                speech_ongoing = True
                continue

            # 음성이 끝났거나 최대 길이 도달 → 전송
            print(f"  🎯 모듈 {module_id} | 음성 {len(timestamps)}구간, {total_ms:.0f}ms → 전송")
            processing_queue.put((module_id, conn, all_raw))
            speech_ongoing = False
            raw_buffer = raw_buffer[-10:]  # 마지막 1초만 유지

    except (ConnectionResetError, BrokenPipeError, OSError) as e:
        print(f"  🔌 모듈 {module_id}: 연결 끊김 ({e})")
    finally:
        active_modules[module_id] = "대기"
        active_connections.pop(module_id, None)
        add_activity_log(module_id, "system", "TCP 클라이언트 연결 끊김")
        print(f"  🔌 모듈 {module_id}: 스트림 종료")


def tcp_accept_loop():
    """TCP 연결 수락 루프 (스레드에서 실행)"""
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("0.0.0.0", TCP_PORT))
    server_sock.listen(6)
    print(f"🔌 TCP 스트림 리스너: 0.0.0.0:{TCP_PORT}")

    while True:
        try:
            conn, addr = server_sock.accept()
            # 핸드셰이크: 첫 1바이트 = 모듈 ID
            id_byte = conn.recv(1)
            if not id_byte:
                conn.close()
                continue
            module_id = id_byte[0]
            print(f"\n🔌 모듈 {module_id} 연결됨 ({addr[0]}:{addr[1]})")

            t = threading.Thread(
                target=handle_stream,
                args=(conn, module_id),
                daemon=True,
            )
            t.start()
        except Exception as e:
            print(f"  ❌ TCP accept 에러: {e}")


# ─────────────────────────────────────────
# 큐 소비자 (메인 스레드 — MLX 안전)
# ─────────────────────────────────────────
def process_queue_worker():
    """큐에서 음성 세그먼트를 꺼내서 Whisper → LLM → TTS 처리"""
    while True:
        try:
            module_id, conn, raw_pcm = processing_queue.get(timeout=1)
        except queue.Empty:
            continue

        start = time.time()
        voice = MODULE_VOICE.get(module_id, "sohee")
        print(f"\n── 모듈 {module_id} [{voice}] 처리 시작 ──")
        active_modules[module_id] = "처리중"

        try:
            # 1) 고품질 변환
            print(f"  📝 STT 처리 중...")
            audio = pcm_to_whisper(raw_pcm)
            duration = len(audio) / SAMPLE_RATE
            vol = np.abs(audio).mean()
            print(f"  📊 {duration:.1f}초, mean={vol:.4f}, max={np.abs(audio).max():.4f}")

            # 노이즈 플로어 체크: 정규화 후 mean이 0.45 이상이면 소음뿐
            # (음성은 무음 구간이 있어서 mean이 낮고, 소음은 균일하게 높음)
            if vol > 0.45:
                print(f"  🔇 노이즈 플로어 초과 (mean={vol:.3f} > 0.45) — 스킵")
                add_activity_log(module_id, "stt", f"노음 감지 스킵 (mean={vol:.2f} > 0.45)")
                active_modules[module_id] = "스트리밍"
                continue

            # 디버그 WAV 저장
            try:
                import wave as wave_mod
                debug_pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
                with wave_mod.open("/tmp/debug_stt.wav", "wb") as dwf:
                    dwf.setnchannels(1)
                    dwf.setsampwidth(2)
                    dwf.setframerate(SAMPLE_RATE)
                    dwf.writeframes(debug_pcm.tobytes())
            except Exception:
                pass

            # 2) Whisper STT
            result = whisper_model.transcribe(
                audio,
                language="ko",
                fp16=False,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                temperature=0.0,
            )
            fragment = result["text"].strip()
            print(f"  📝 인식: \"{fragment}\"")

            if is_hallucination(fragment) or not fragment:
                print(f"  🚫 환각/무음 — 스킵")
                add_activity_log(module_id, "stt", f"환각 또는 무음 감지 스킵: \"{fragment}\"")
                active_modules[module_id] = "스트리밍"
                continue

            add_activity_log(module_id, "stt", f"STT 인식: \"{fragment}\"")

            # 3) LLM
            print(f"  🧠 왜곡 재구성 중...")
            reinterpreted = do_llm(fragment)
            print(f"  🧠 결과: \"{reinterpreted}\"")
            add_activity_log(module_id, "llm", f"소문 왜곡: \"{reinterpreted}\"")

            # 4) TTS
            print(f"  🔊 TTS 생성 중 ({voice})...")
            wav_data = do_tts(reinterpreted, module_id)
            add_activity_log(module_id, "tts", f"TTS 생성 완료 ({voice})")

            elapsed = time.time() - start
            print(f"  ✅ 완료! ({elapsed:.1f}초)")

            # 5) TCP로 응답 전송 (4바이트 길이 헤더 + WAV)
            try:
                conn.sendall(len(wav_data).to_bytes(4, "big"))
                conn.sendall(wav_data)
                print(f"  📤 응답 전송 ({len(wav_data)} bytes)")
                add_activity_log(module_id, "system", f"TTS WAV 응답 전송 완료 ({len(wav_data)} bytes, {elapsed:.1f}초 소요)")
            except (BrokenPipeError, OSError):
                print(f"  ❌ 응답 전송 실패 — 연결 끊김")
                add_activity_log(module_id, "system", "WAV 응답 전송 실패 — TCP 연결 끊김")

        except Exception as e:
            print(f"  ❌ 처리 에러: {e}")
            add_activity_log(module_id, "system", f"처리 에러: {e}")

        active_modules[module_id] = "스트리밍"


# ─────────────────────────────────────────
# HTTP 서버 (control_server 호환)
# ─────────────────────────────────────────
class StatusHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.path = "/index.html"
            
        if self.path == "/index.html":
            html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
            if os.path.exists(html_path):
                try:
                    with open(html_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(content.encode("utf-8"))
                    return
                except Exception as e:
                    self.send_error(500, f"Error reading index.html: {e}")
                    return
            else:
                self.send_error(404, "index.html not found. Please create the UI file.")
                return
                
        elif self.path == "/status" or self.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            mode = module_configs.get("rotation_mode", "loop")
            rot_results = calculate_all_rotations(module_configs)
            
            status_data = {
                "server_status": "ready",
                "server_cpu": get_server_cpu_usage(),
                "queue_size": processing_queue.qsize(),
                "rotation_mode": mode,
                "centroid": rot_results.get("centroid", {"x": 0.0, "y": 0.0}),
                "modules": {},
                "logs": activity_logs
            }
            
            for mid in [1, 2, 3]:
                cfg = module_configs.get(str(mid), {})
                runtime = module_runtime_status.get(mid, {"agent_status": "offline", "cpu": 0.0, "tcp_connected": False})
                
                mid_str = str(mid)
                res = rot_results.get(mid_str, {"vector": [0.0, 0.0], "angle": 0.0, "distance": 0.0, "target": [0.0, 0.0]})
                
                status_data["modules"][mid_str] = {
                    "name": cfg.get("name", f"모듈 {mid}"),
                    "voice": cfg.get("voice", "sohee"),
                    "ip": cfg.get("ip", "127.0.0.1"),
                    "x": float(cfg.get("x", 0.0)),
                    "y": float(cfg.get("y", 0.0)),
                    "rotation_vector": res.get("vector"),
                    "rotation_angle": res.get("angle"),
                    "distance": res.get("distance"),
                    "target_point": res.get("target"),
                    "tcp_connected": runtime.get("tcp_connected", False),
                    "agent_status": runtime.get("agent_status", "offline"),
                    "cpu": runtime.get("cpu", 0.0),
                    "vad_status": active_modules.get(mid, "대기")
                }
                
            self.wfile.write(json.dumps(status_data, ensure_ascii=False).encode("utf-8"))
            return
        else:
            self.send_response(404)
            self.end_headers()
            
    def do_POST(self):
        if self.path == "/api/module/configure":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(body)
                mid = str(data.get("module_id"))
                ip = data.get("ip", "127.0.0.1").strip()
                name = data.get("name", f"모듈 {mid}").strip()
                voice = data.get("voice", "sohee").strip()
                x_val = float(data.get("x", 0.0))
                y_val = float(data.get("y", 0.0))
                
                if mid in ["1", "2", "3"]:
                    module_configs[mid]["ip"] = ip
                    module_configs[mid]["name"] = name
                    module_configs[mid]["voice"] = voice
                    module_configs[mid]["x"] = x_val
                    module_configs[mid]["y"] = y_val
                    MODULE_VOICE[int(mid)] = voice
                    MODULE_MAP[int(mid)] = name
                    save_configs()
                    
                    add_activity_log(int(mid), "system", f"모듈 설정 변경됨: IP={ip}, 이름={name}, 음성={voice}, 좌표=({x_val}, {y_val})")
                    self._send_json({"ok": True, "message": "설정이 성공적으로 저장되었습니다."})
                else:
                    self._send_json({"ok": False, "message": "잘못된 모듈 ID입니다. (1~3만 가능)"})
            except Exception as e:
                self._send_json({"ok": False, "message": f"설정 변경 실패: {e}"})
                
        elif self.path == "/api/rotation/mode":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(body)
                mode = data.get("mode", "loop").strip().lower()
                
                if mode in ["loop", "center"]:
                    module_configs["rotation_mode"] = mode
                    save_configs()
                    
                    mode_kor = "순차 순환" if mode == "loop" else "중앙 수렴"
                    add_activity_log(0, "system", f"회전 벡터 연산 모드 변경됨: {mode_kor}")
                    self._send_json({"ok": True, "message": "회전 연산 모드가 변경되었습니다."})
                else:
                    self._send_json({"ok": False, "message": "지원하지 않는 모드입니다. (loop 또는 center만 가능)"})
            except Exception as e:
                self._send_json({"ok": False, "message": f"모드 설정 실패: {e}"})
                
        elif self.path == "/api/module/control":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(body)
                mid = int(data.get("module_id"))
                action = data.get("action", "").lower()
                
                if mid in [1, 2, 3]:
                    if action == "start":
                           ok, msg = start_module(mid)
                           self._send_json({"ok": ok, "message": msg})
                    elif action == "stop":
                           ok, msg = stop_module(mid)
                           self._send_json({"ok": ok, "message": msg})
                    else:
                           self._send_json({"ok": False, "message": "잘못된 동작(action)입니다."})
                else:
                    self._send_json({"ok": False, "message": "잘못된 모듈 ID입니다."})
            except Exception as e:
                self._send_json({"ok": False, "message": f"제어 명령 실행 실패: {e}"})
        else:
            self.send_response(404)
            self.end_headers()
            
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        
    def _send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
        
    def log_message(self, format, *args):
        pass


def run_http_server():
    """HTTP 서버 스레드"""
    server = HTTPServer(("0.0.0.0", HTTP_PORT), StatusHandler)
    server.serve_forever()


# ─────────────────────────────────────────
# main
# ─────────────────────────────────────────
def main():
    print("=" * 60)
    print("  사운드 인스톨레이션 — TCP 스트리밍 서버 (제어 및 모니터링 기능 추가)")
    print(f"  LLM: EXAONE 7.8B (Ollama, temp={LLM_TEMPERATURE})")
    print(f"  STT: Whisper {WHISPER_MODEL_SIZE} + Silero VAD (스트리밍)")
    print(f"  TTS: {TTS_MODEL_NAME}")
    print(f"  HTTP: :{HTTP_PORT} (status & UI)")
    print(f"  TCP:  :{TCP_PORT} (스트림)")
    print("=" * 60)

    load_whisper()
    load_vad()
    load_tts()
    if not check_ollama():
        return

    load_configs()
    
    for mid_str, info in module_configs.items():
        if mid_str in ["spatial_target", "rotation_mode"]:
            continue
        mid = int(mid_str)
        MODULE_VOICE[mid] = info.get("voice", "sohee")
        MODULE_MAP[mid] = info.get("name", f"모듈 {mid}")
        active_modules[mid] = "대기"

    for mid in MODULE_MAP:
        if mid not in active_modules:
            active_modules[mid] = "대기"

    # HTTP 서버 (스레드)
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()
    print(f"\n🌐 HTTP Dashboard UI: http://localhost:{HTTP_PORT}/")

    # TCP 리스너 (스레드)
    tcp_thread = threading.Thread(target=tcp_accept_loop, daemon=True)
    tcp_thread.start()

    # 모니터링 스레드 시작
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()

    add_activity_log(0, "system", "서버가 시작되었습니다. Whisper, VAD, TTS, Ollama 연동 완료.")

    print(f"\n   모듈 설정 (1~3):")
    for mid_str, info in module_configs.items():
        if mid_str in ["spatial_target", "rotation_mode"]:
            continue
        print(f"     #{mid_str} [{info.get('name')}] → 음성: {info.get('voice')}, IP: {info.get('ip')}")
    print(f"\n   메인 스레드: 큐 처리 (MLX 안전)")
    print(f"   Ctrl+C로 종료")
    print("=" * 60)

    # 메인 스레드: 큐 소비 (Whisper + LLM + TTS 순차 처리)
    try:
        process_queue_worker()
    except KeyboardInterrupt:
        print("\n🛑 서버 종료 중...")
        # 로컬 자식 프로세스 정리
        for mid, proc in list(local_processes.items()):
            if proc and proc.poll() is None:
                print(f"  ⏹️  로컬 클라이언트 #{mid} 종료 중...")
                try:
                    proc.terminate()
                    proc.wait(timeout=1)
                except Exception:
                    pass
        print("✅ 서버 종료 완료")


if __name__ == "__main__":
    main()
