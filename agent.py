#!/usr/bin/env python3
"""
Pi 에이전트 서버
Pi에서 항상 실행해두면 Mac이 HTTP로 rpi_client.py를 제어합니다.
사용법: python3 agent.py
"""

import subprocess
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import json
import os

AGENT_PORT = 8001
SCRIPT_PATH = "/home/pi/rpi_client.py"
MAC_SERVER_IP = "192.168.0.100"
MAC_SERVER_TCP_PORT = 9000

process = None
process_lock = threading.Lock()


def start_client(module_id):
    global process
    with process_lock:
        if process and process.poll() is None:
            return False, "이미 실행 중"
        try:
            process = subprocess.Popen([
                "python3", SCRIPT_PATH,
                "--server", MAC_SERVER_IP,
                "--port", str(MAC_SERVER_TCP_PORT),
                "--module", str(module_id),
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"▶️  rpi_client.py 시작 (module {module_id}, PID: {process.pid})")
            return True, f"시작됨 (module {module_id})"
        except Exception as e:
            return False, str(e)


def stop_client():
    global process
    with process_lock:
        if not process or process.poll() is not None:
            process = None
            return False, "실행 중이 아님"
        try:
            process.terminate()
            process.wait(timeout=3)
            process = None
            print("⏹️  rpi_client.py 정지")
            return True, "정지됨"
        except:
            process.kill()
            process = None
            return True, "강제 종료됨"


_last_cpu_time = [0.0, 0.0]

def get_cpu_usage():
    global _last_cpu_time
    import platform
    import subprocess
    import os
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(['ps', '-A', '-o', '%cpu']).decode()
            lines = out.strip().split('\n')[1:]
            total = sum(float(x) for x in lines if x.strip())
            cores = os.cpu_count() or 1
            return min(100.0, round(total / cores, 1))
        else:
            # Linux / RPi - read /proc/stat
            with open('/proc/stat', 'r') as f:
                line = f.readline()
            parts = line.split()
            if len(parts) >= 5:
                # user, nice, system, idle, iowait...
                idle = float(parts[4]) + float(parts[5])
                total = sum(float(x) for x in parts[1:8])
                
                prev_idle, prev_total = _last_cpu_time
                _last_cpu_time = [idle, total]
                
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


def get_status():
    if process and process.poll() is None:
        return "running"
    return "stopped"


class AgentHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        parts = self.path.strip('/').split('/')

        if parts[0] == 'start' and len(parts) == 2:
            ok, msg = start_client(int(parts[1]))
            self._respond({"ok": ok, "message": msg})

        elif parts[0] == 'stop':
            ok, msg = stop_client()
            self._respond({"ok": ok, "message": msg})

        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == '/status':
            self._respond({"status": get_status(), "cpu": get_cpu_usage()})
        else:
            self.send_response(404)
            self.end_headers()

    def _respond(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    print("=" * 50)
    print(f"  Pi 에이전트 서버 (포트 {AGENT_PORT})")
    print(f"  Mac 서버: {MAC_SERVER_IP}:{MAC_SERVER_TCP_PORT}")
    print("  Ctrl+C로 종료")
    print("=" * 50)

    server = ThreadedHTTPServer(("0.0.0.0", AGENT_PORT), AgentHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 종료")
        stop_client()


if __name__ == "__main__":
    main()
