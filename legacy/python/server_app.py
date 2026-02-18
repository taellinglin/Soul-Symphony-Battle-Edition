import json
import socket
import sys
import threading
import time

clients: set[socket.socket] = set()
clients_lock = threading.Lock()
client_states: dict[str, dict] = {}
state_lock = threading.Lock()
monster_state: dict | None = None

def handle_client(conn: socket.socket, addr: tuple[str, int]) -> None:
    print(f"[server] client connected {addr}")
    with clients_lock:
        clients.add(conn)
    last_client_id = ""
    snapshot = _get_state_snapshot()
    if snapshot:
        for line in snapshot:
            try:
                conn.sendall(line)
            except Exception:
                break
    buffer = ""
    last_log = 0.0
    try:
        with conn:
            while True:
                try:
                    data = conn.recv(4096)
                except ConnectionResetError:
                    break
                if not data:
                    break
                try:
                    chunk = data.decode("utf-8", errors="ignore")
                except Exception:
                    chunk = ""
                if chunk:
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            msg = json.loads(line)
                        except Exception:
                            msg = None
                        if isinstance(msg, dict):
                            msg_type = msg.get("type")
                            if msg_type == "input":
                                client_id = _update_client_state(msg)
                                if client_id:
                                    last_client_id = client_id
                                if _should_log_now(msg, last_log):
                                    _log_input_event(msg, addr)
                                    last_log = time.time()
                            elif msg_type == "monsters":
                                _update_monster_state(msg)
                # Broadcast to all clients (including sender).
                with clients_lock:
                    targets = list(clients)
                for target in targets:
                    try:
                        target.sendall(data)
                    except Exception:
                        with clients_lock:
                            clients.discard(target)
    finally:
        with clients_lock:
            clients.discard(conn)
        if last_client_id:
            with state_lock:
                client_states.pop(last_client_id, None)
        print(f"[server] client disconnected {addr}")


def _should_log_now(msg: dict, last_log: float) -> bool:
    jump = bool(msg.get("jump"))
    attack = bool(msg.get("attack"))
    move = msg.get("move") or [0.0, 0.0]
    try:
        move_mag = (float(move[0]) ** 2 + float(move[1]) ** 2) ** 0.5
    except Exception:
        move_mag = 0.0
    if jump or attack:
        return True
    return (time.time() - last_log) > 1.0 and move_mag > 0.15


def _log_input_event(msg: dict, addr: tuple[str, int]) -> None:
    client = str(msg.get("client") or addr)
    jump = bool(msg.get("jump"))
    attack = bool(msg.get("attack"))
    move = msg.get("move") or [0.0, 0.0]
    try:
        move_x = float(move[0])
        move_y = float(move[1])
    except Exception:
        move_x, move_y = 0.0, 0.0
    action = []
    if abs(move_x) > 0.15 or abs(move_y) > 0.15:
        action.append(f"move=({move_x:.2f},{move_y:.2f})")
    if jump:
        action.append("jump")
    if attack:
        action.append("attack")
    if action:
        print(f"[server] {client} {', '.join(action)}")


def _update_client_state(msg: dict) -> str:
    client_id = str(msg.get("client") or "").strip()
    if not client_id:
        return ""
    state = dict(msg)
    state["_last_seen"] = time.time()
    with state_lock:
        client_states[client_id] = state
    return client_id


def _get_state_snapshot() -> list[bytes]:
    with state_lock:
        lines = [(json.dumps(state) + "\n").encode("utf-8") for state in client_states.values()]
        if monster_state is not None:
            lines.append((json.dumps(monster_state) + "\n").encode("utf-8"))
        return lines


def _update_monster_state(msg: dict) -> None:
    global monster_state
    state = dict(msg)
    state["_last_seen"] = time.time()
    with state_lock:
        monster_state = state

def main() -> None:
    host = "0.0.0.0"
    port = 8383
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except Exception:
            pass
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen()
    print(f"[server] TCP listening on {host}:{port}")
    try:
        while True:
            conn, addr = server.accept()
            thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            thread.start()
    except KeyboardInterrupt:
        print("[server] shutting down")
    finally:
        server.close()

if __name__ == "__main__":
    main()
