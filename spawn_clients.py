import argparse
import json
import math
import random
import socket
import threading
import time

def client_loop(
    idx: int,
    host: str,
    port: int,
    send_rate: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    base_z: float,
) -> None:
    name = f"bot_{idx:02d}"
    level = random.randint(1, 8)
    xp_next = 10 + level * 6
    xp = random.randint(0, xp_next)
    hp_max = 100 + level * 12
    hp = random.uniform(hp_max * 0.5, hp_max)
    stats = {
        "attack": level + random.randint(0, 3),
        "defense": level + random.randint(0, 3),
        "dex": level + random.randint(0, 3),
        "sta": level + random.randint(0, 3),
        "int": level + random.randint(0, 3),
    }
    while True:
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.0)
            sock.connect((host, port))
            sock.settimeout(None)
        except Exception as exc:
            print(f"[{name}] connect failed: {exc}")
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
            time.sleep(1.0)
            continue

        print(f"[{name}] connected")
        try:
            pos_x = random.uniform(min_x, max_x)
            pos_y = random.uniform(min_y, max_y)
            pos_z = base_z
            heading = random.uniform(0.0, math.tau)
            speed = random.uniform(0.35, 1.0)
            next_dir_change = time.time() + random.uniform(0.6, 1.6)
            last_send = time.time()
            next_swing = time.time() + random.uniform(0.6, 1.6)
            next_spin = time.time() + random.uniform(1.2, 2.6)
            next_bomb = time.time() + random.uniform(2.2, 4.4)
            next_rocket = time.time() + random.uniform(1.4, 3.2)

            while True:
                now = time.time()
                if now >= next_dir_change:
                    heading += random.uniform(-1.2, 1.2)
                    speed = random.uniform(0.35, 1.0)
                    next_dir_change = now + random.uniform(0.6, 1.6)

                move_x = math.cos(heading) * speed
                move_y = math.sin(heading) * speed
                pos_x += move_x * 0.6
                pos_y += move_y * 0.6

                if pos_x < min_x or pos_x > max_x:
                    heading = math.pi - heading
                    pos_x = max(min_x, min(max_x, pos_x))
                if pos_y < min_y or pos_y > max_y:
                    heading = -heading
                    pos_y = max(min_y, min(max_y, pos_y))

                if now >= next_swing:
                    next_swing = now + random.uniform(0.6, 1.6)
                    attack = {
                        "type": "attack_event",
                        "client": name,
                        "kind": "swing",
                        "t": now,
                        "pos": [pos_x, pos_y, pos_z],
                        "fwd": [move_x, move_y],
                    }
                    sock.sendall((json.dumps(attack) + "\n").encode("utf-8"))

                if now >= next_spin:
                    next_spin = now + random.uniform(1.2, 2.8)
                    attack = {
                        "type": "attack_event",
                        "client": name,
                        "kind": "spin",
                        "t": now,
                        "pos": [pos_x, pos_y, pos_z],
                        "fwd": [move_x, move_y],
                    }
                    sock.sendall((json.dumps(attack) + "\n").encode("utf-8"))

                if now >= next_bomb:
                    next_bomb = now + random.uniform(2.4, 4.8)
                    attack = {
                        "type": "attack_event",
                        "client": name,
                        "kind": "bomb",
                        "t": now,
                        "pos": [pos_x, pos_y, pos_z],
                        "fwd": [move_x, move_y],
                    }
                    sock.sendall((json.dumps(attack) + "\n").encode("utf-8"))

                if now >= next_rocket:
                    next_rocket = now + random.uniform(1.4, 3.2)
                    attack = {
                        "type": "attack_event",
                        "client": name,
                        "kind": "rocket",
                        "t": now,
                        "pos": [pos_x, pos_y, pos_z],
                        "fwd": [move_x, move_y],
                    }
                    sock.sendall((json.dumps(attack) + "\n").encode("utf-8"))

                if now - last_send >= send_rate:
                    last_send = now
                    payload = {
                        "type": "input",
                        "client": name,
                        "t": now,
                        "move": [move_x, move_y],
                        "fwd": [move_x, move_y],
                        "pos": [pos_x, pos_y, pos_z],
                        "jump": random.random() < 0.04,
                        "attack": random.random() < 0.06,
                        "level": level,
                        "xp": xp,
                        "xp_next": xp_next,
                        "hp": hp,
                        "hp_max": hp_max,
                        "stats": stats,
                    }
                    msg = (json.dumps(payload) + "\n").encode("utf-8")
                    sock.sendall(msg)

                # Drain any echoes without blocking.
                sock.settimeout(0.0)
                try:
                    sock.recv(4096)
                except Exception:
                    pass
                finally:
                    sock.settimeout(None)
                time.sleep(0.01)
        except Exception as exc:
            print(f"[{name}] stopped: {exc}")
        finally:
            try:
                sock.close()
            except Exception:
                pass
            time.sleep(0.5)

def main() -> None:
    parser = argparse.ArgumentParser(description="Spawn multiple TCP clients for load testing.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8383)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--rate", type=float, default=0.05, help="Seconds between input messages")
    parser.add_argument("--min-x", type=float, default=6.0)
    parser.add_argument("--max-x", type=float, default=60.0)
    parser.add_argument("--min-y", type=float, default=6.0)
    parser.add_argument("--max-y", type=float, default=60.0)
    parser.add_argument("--z", type=float, default=1.0)
    args = parser.parse_args()

    threads = []
    for idx in range(args.count):
        t = threading.Thread(
            target=client_loop,
            args=(
                idx,
                args.host,
                args.port,
                args.rate,
                args.min_x,
                args.max_x,
                args.min_y,
                args.max_y,
                args.z,
            ),
            daemon=True,
        )
        t.start()
        threads.append(t)
        time.sleep(0.02)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Shutting down clients...")

if __name__ == "__main__":
    main()
