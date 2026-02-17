import argparse
import json
import math
import os
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
    min_w: float,
    max_w: float,
    start_x: float,
    start_y: float,
    bounds_margin: float,
    mobius_center_y: float,
    mobius_half_band: float,
    mobius_nudge: float,
    mobius_stuck_seconds: float,
    mobius_links: list[tuple[float, float, float, float, float, float, float]],
    total_bots: int,
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
            pos_x = float(start_x)
            pos_y = float(start_y)
            pos_z = base_z
            heading = random.uniform(0.0, math.tau)
            w_val = random.uniform(min_w, max_w)
            w_target = random.uniform(min_w, max_w)
            w_speed = random.uniform(0.25, 0.55)
            next_w_shift = time.time() + random.uniform(1.2, 2.6)
            speed = random.uniform(0.45, 0.95)
            bot_min_x = min_x - bounds_margin
            bot_max_x = max_x + bounds_margin
            bot_min_y = min_y - bounds_margin
            bot_max_y = max_y + bounds_margin
            width = max(1.0, bot_max_x - bot_min_x)
            height = max(1.0, bot_max_y - bot_min_y)
            center_x = bot_min_x + width * 0.5
            center_y = bot_min_y + height * 0.5
            ring_radius = max(4.0, min(width, height) * 0.35)
            sector_angle = (float(idx) / max(1.0, float(total_bots))) * math.tau
            sector_x = float(start_x)
            sector_y = float(start_y)
            target_x = sector_x + random.uniform(-ring_radius * 0.3, ring_radius * 0.3)
            target_y = sector_y + random.uniform(-ring_radius * 0.3, ring_radius * 0.3)
            next_dir_change = time.time() + random.uniform(0.8, 1.8)
            release_end = time.time() + random.uniform(3.0, 5.0)
            center_pass_until = time.time()
            next_center_pass = time.time() + random.uniform(2.2, 4.4)
            band_enter_time = None
            mobius_warp_cd = time.time()
            mobius_seek_until = 0.0
            next_mobius_seek = time.time() + random.uniform(2.6, 5.2)
            mobius_seek_target = None
            next_forced_warp = time.time() + random.uniform(3.0, 6.0)
            last_send = time.time()
            next_swing = time.time() + random.uniform(0.6, 1.6)
            next_spin = time.time() + random.uniform(1.2, 2.6)
            next_bomb = time.time() + random.uniform(2.2, 4.4)
            next_rocket = time.time() + random.uniform(1.4, 3.2)
            recv_buffer = ""
            host_pos = None
            host_w = None
            host_last = 0.0

            while True:
                now = time.time()
                if now >= next_dir_change:
                    heading += random.uniform(-0.5, 0.5)
                    speed = random.uniform(0.7, 1.2)
                    next_dir_change = now + random.uniform(0.7, 1.4)

                follow_host = host_pos is not None and (now - host_last) < 2.5
                support_active = False
                host_dist_sq = None
                if follow_host:
                    hx, hy, _hz = host_pos
                    to_host_x = hx - pos_x
                    to_host_y = hy - pos_y
                    host_dist_sq = to_host_x * to_host_x + to_host_y * to_host_y
                    hold_radius = max(2.0, min(width, height) * 0.12)
                    if host_dist_sq > hold_radius * hold_radius:
                        target_x, target_y = hx, hy
                    else:
                        orbit_angle = sector_angle + now * 0.6 + idx * 0.35
                        orbit_r = max(1.8, hold_radius * 0.6)
                        target_x = hx + math.cos(orbit_angle) * orbit_r
                        target_y = hy + math.sin(orbit_angle) * orbit_r
                    if host_dist_sq > (hold_radius * 1.4) * (hold_radius * 1.4):
                        speed = max(speed, 1.45)
                    support_active = host_dist_sq <= (hold_radius * 1.6) * (hold_radius * 1.6)
                    if host_w is not None:
                        w_target = host_w

                if mobius_links and now >= next_mobius_seek and now >= mobius_seek_until:
                    link = random.choice(mobius_links)
                    if random.random() < 0.5:
                        mobius_seek_target = (link[0], link[1])
                    else:
                        mobius_seek_target = (link[3], link[4])
                    mobius_seek_until = now + random.uniform(2.8, 4.6)
                    next_mobius_seek = now + random.uniform(5.0, 8.0)

                if mobius_seek_target is not None and now <= mobius_seek_until:
                    target_x, target_y = mobius_seek_target
                else:
                    mobius_seek_target = None

                to_target_x = target_x - pos_x
                to_target_y = target_y - pos_y
                target_dist_sq = to_target_x * to_target_x + to_target_y * to_target_y
                if target_dist_sq < 6.0:
                    target_x = sector_x + random.uniform(-ring_radius * 0.4, ring_radius * 0.4)
                    target_y = sector_y + random.uniform(-ring_radius * 0.4, ring_radius * 0.4)
                    to_target_x = target_x - pos_x
                    to_target_y = target_y - pos_y
                    target_dist_sq = to_target_x * to_target_x + to_target_y * to_target_y

                if target_dist_sq > 1e-6:
                    target_heading = math.atan2(to_target_y, to_target_x)
                    heading = heading + (target_heading - heading) * 0.14

                if now < release_end:
                    heading = math.atan2(pos_y - center_y, pos_x - center_x)
                    speed = max(speed, 2.2)

                dist_center_sq = (pos_x - center_x) * (pos_x - center_x) + (pos_y - center_y) * (pos_y - center_y)
                if now >= next_center_pass and dist_center_sq < (ring_radius * 0.95) * (ring_radius * 0.95):
                    center_pass_until = now + random.uniform(1.2, 2.6)
                    next_center_pass = now + random.uniform(3.2, 6.4)
                if now > center_pass_until:
                    if dist_center_sq < (ring_radius * 0.9) * (ring_radius * 0.9):
                        heading = heading + (math.atan2(pos_y - center_y, pos_x - center_x) - heading) * 0.35
                        speed = max(speed, 1.6)
                    if dist_center_sq < (ring_radius * 0.5) * (ring_radius * 0.5):
                        heading = heading + (math.atan2(pos_y - center_y, pos_x - center_x) + math.pi - heading) * 0.14

                if speed > 1.3:
                    speed = 1.3
                move_x = math.cos(heading) * speed
                move_y = math.sin(heading) * speed
                if now >= next_w_shift:
                    w_target = random.uniform(min_w, max_w)
                    w_speed = random.uniform(0.25, 0.65)
                    next_w_shift = now + random.uniform(1.2, 2.6)
                if follow_host and host_w is not None:
                    w_target = host_w
                    w_speed = max(w_speed, 0.55)
                w_val += (w_target - w_val) * min(1.0, 0.06 + w_speed * 0.08)
                if w_val < min_w:
                    w_val = min_w
                elif w_val > max_w:
                    w_val = max_w
                pos_x += move_x * 1.05
                pos_y += move_y * 1.05

                if pos_x < bot_min_x or pos_x > bot_max_x:
                    heading = math.pi - heading
                    pos_x = max(bot_min_x, min(bot_max_x, pos_x))
                if pos_y < bot_min_y or pos_y > bot_max_y:
                    heading = -heading
                    pos_y = max(bot_min_y, min(bot_max_y, pos_y))

                if mobius_half_band > 0.1:
                    if abs(pos_y - mobius_center_y) <= mobius_half_band:
                        if band_enter_time is None:
                            band_enter_time = now
                    else:
                        band_enter_time = None
                    moving_toward_center = (pos_y - mobius_center_y) * move_y < 0.0
                    if moving_toward_center and now >= mobius_warp_cd:
                        if abs(pos_y - mobius_center_y) <= mobius_half_band * 0.6:
                            offset = pos_y - mobius_center_y
                            pos_y = mobius_center_y - offset - math.copysign(mobius_nudge, offset)
                            band_enter_time = None
                            mobius_warp_cd = now + random.uniform(2.2, 4.6)
                    if band_enter_time is not None and now >= mobius_warp_cd:
                        if abs(pos_y - mobius_center_y) <= mobius_half_band * 0.3:
                            offset = pos_y - mobius_center_y
                            if offset == 0.0:
                                offset = mobius_half_band * 0.3
                            pos_y = mobius_center_y - offset - math.copysign(mobius_nudge, offset)
                            band_enter_time = None
                            mobius_warp_cd = now + random.uniform(1.4, 2.6)
                    if (
                        band_enter_time is not None
                        and now - band_enter_time >= mobius_stuck_seconds
                        and now >= mobius_warp_cd
                    ):
                        offset = pos_y - mobius_center_y
                        if offset == 0.0:
                            offset = mobius_half_band
                        pos_y = mobius_center_y - offset - math.copysign(mobius_nudge, offset)
                        band_enter_time = None
                        mobius_warp_cd = now + random.uniform(2.2, 4.6)

                if mobius_links and now >= mobius_warp_cd:
                    for ax, ay, az, bx, by, bz, radius in mobius_links:
                        dx = pos_x - ax
                        dy = pos_y - ay
                        if (dx * dx + dy * dy) <= radius * radius:
                            pos_x, pos_y, pos_z = bx, by, bz
                            sector_x, sector_y = pos_x, pos_y
                            target_x = sector_x + random.uniform(-ring_radius * 0.4, ring_radius * 0.4)
                            target_y = sector_y + random.uniform(-ring_radius * 0.4, ring_radius * 0.4)
                            pos_x += math.cos(heading) * 0.6
                            pos_y += math.sin(heading) * 0.6
                            mobius_warp_cd = now + random.uniform(1.6, 3.4)
                            break
                        dx = pos_x - bx
                        dy = pos_y - by
                        if (dx * dx + dy * dy) <= radius * radius:
                            pos_x, pos_y, pos_z = ax, ay, az
                            sector_x, sector_y = pos_x, pos_y
                            target_x = sector_x + random.uniform(-ring_radius * 0.4, ring_radius * 0.4)
                            target_y = sector_y + random.uniform(-ring_radius * 0.4, ring_radius * 0.4)
                            pos_x += math.cos(heading) * 0.6
                            pos_y += math.sin(heading) * 0.6
                            mobius_warp_cd = now + random.uniform(1.6, 3.4)
                            break

                if mobius_links and now >= next_forced_warp:
                    ax, ay, az, bx, by, bz, _radius = random.choice(mobius_links)
                    if random.random() < 0.5:
                        pos_x, pos_y, pos_z = bx, by, bz
                    else:
                        pos_x, pos_y, pos_z = ax, ay, az
                    sector_x, sector_y = pos_x, pos_y
                    target_x = sector_x + random.uniform(-ring_radius * 0.5, ring_radius * 0.5)
                    target_y = sector_y + random.uniform(-ring_radius * 0.5, ring_radius * 0.5)
                    pos_x += math.cos(heading) * 0.8
                    pos_y += math.sin(heading) * 0.8
                    mobius_warp_cd = now + random.uniform(1.2, 2.2)
                    next_forced_warp = now + random.uniform(4.0, 7.0)

                attack_scale = 0.7 if support_active else 1.0
                if now >= next_swing:
                    next_swing = now + random.uniform(0.6, 1.6) * attack_scale
                    attack = {
                        "type": "attack_event",
                        "client": name,
                        "kind": "swing",
                        "t": now,
                        "pos": [pos_x, pos_y, pos_z],
                        "fwd": [move_x, move_y],
                        "w": w_val,
                    }
                    sock.sendall((json.dumps(attack) + "\n").encode("utf-8"))

                if now >= next_spin:
                    next_spin = now + random.uniform(1.2, 2.8) * attack_scale
                    attack = {
                        "type": "attack_event",
                        "client": name,
                        "kind": "spin",
                        "t": now,
                        "pos": [pos_x, pos_y, pos_z],
                        "fwd": [move_x, move_y],
                        "w": w_val,
                    }
                    sock.sendall((json.dumps(attack) + "\n").encode("utf-8"))

                if now >= next_bomb:
                    next_bomb = now + random.uniform(2.4, 4.8) * attack_scale
                    attack = {
                        "type": "attack_event",
                        "client": name,
                        "kind": "bomb",
                        "t": now,
                        "pos": [pos_x, pos_y, pos_z],
                        "fwd": [move_x, move_y],
                        "w": w_val,
                    }
                    sock.sendall((json.dumps(attack) + "\n").encode("utf-8"))

                if now >= next_rocket:
                    next_rocket = now + random.uniform(1.4, 3.2) * attack_scale
                    attack = {
                        "type": "attack_event",
                        "client": name,
                        "kind": "rocket",
                        "t": now,
                        "pos": [pos_x, pos_y, pos_z],
                        "fwd": [move_x, move_y],
                        "w": w_val,
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
                        "w": w_val,
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
                    while True:
                        chunk = sock.recv(4096)
                        if not chunk:
                            break
                        recv_buffer += chunk.decode("utf-8", errors="ignore")
                        while "\n" in recv_buffer:
                            line, recv_buffer = recv_buffer.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                msg = json.loads(line)
                            except Exception:
                                continue
                            if not isinstance(msg, dict):
                                continue
                            if msg.get("type") != "input":
                                continue
                            client = str(msg.get("client") or "").strip()
                            if not client or client == name or client.startswith("bot_"):
                                continue
                            pos = msg.get("pos")
                            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                                try:
                                    host_pos = (float(pos[0]), float(pos[1]), float(pos[2]))
                                    host_last = now
                                except Exception:
                                    pass
                            if "w" in msg:
                                try:
                                    host_w = float(msg.get("w"))
                                except Exception:
                                    pass
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
    parser.add_argument("--host", default="63.41.180.121")
    parser.add_argument("--port", type=int, default=8383)
    parser.add_argument("--count", type=int, default=13)
    parser.add_argument("--rate", type=float, default=0.05, help="Seconds between input messages")
    parser.add_argument("--min-x", type=float, default=6.0)
    parser.add_argument("--max-x", type=float, default=60.0)
    parser.add_argument("--min-y", type=float, default=6.0)
    parser.add_argument("--max-y", type=float, default=60.0)
    parser.add_argument("--z", type=float, default=1.0)
    parser.add_argument("--min-w", type=float, default=-7.2)
    parser.add_argument("--max-w", type=float, default=7.2)
    parser.add_argument("--spawn-center-x", type=float, default=None)
    parser.add_argument("--spawn-center-y", type=float, default=None)
    parser.add_argument("--spawn-radius", type=float, default=None)
    parser.add_argument("--bounds-margin", type=float, default=18.0)
    parser.add_argument("--mobius-center-y", type=float, default=None)
    parser.add_argument("--mobius-band-ratio", type=float, default=0.28)
    parser.add_argument("--mobius-nudge", type=float, default=12.0)
    parser.add_argument("--mobius-stuck-seconds", type=float, default=1.2)
    args = parser.parse_args()

    min_x = args.min_x
    max_x = args.max_x
    min_y = args.min_y
    max_y = args.max_y
    base_z = args.z
    width = max(1.0, max_x - min_x)
    height = max(1.0, max_y - min_y)
    center_x = args.min_x + width * 0.5 if args.spawn_center_x is None else args.spawn_center_x
    center_y = args.min_y + height * 0.5 if args.spawn_center_y is None else args.spawn_center_y
    mobius_enabled = True
    mobius_ratio = float(args.mobius_band_ratio)
    mobius_center_y_hint = None
    mobius_links: list[tuple[float, float, float, float, float, float, float]] = []
    spawn_full_map = args.spawn_radius is None and args.spawn_center_x is None and args.spawn_center_y is None
    if args.spawn_center_x is None or args.spawn_center_y is None or args.mobius_center_y is None:
        hint_path = os.path.join(os.path.dirname(__file__), "spawn_center.json")
        try:
            with open(hint_path, "r", encoding="utf-8") as handle:
                hint = json.load(handle)
            if not spawn_full_map:
                if args.spawn_center_x is None:
                    center_x = float(hint.get("x", center_x))
                if args.spawn_center_y is None:
                    center_y = float(hint.get("y", center_y))
            if args.mobius_center_y is None:
                mobius_center_y_hint = hint.get("mobius_center_y")
            if "mobius_enabled" in hint:
                mobius_enabled = bool(hint.get("mobius_enabled"))
            if "mobius_band_ratio" in hint:
                mobius_ratio = float(hint.get("mobius_band_ratio", mobius_ratio))
            if "mobius_links" in hint and isinstance(hint.get("mobius_links"), list):
                for entry in hint.get("mobius_links", []):
                    if not isinstance(entry, dict):
                        continue
                    a_pos = entry.get("a")
                    b_pos = entry.get("b")
                    if not isinstance(a_pos, list) or not isinstance(b_pos, list):
                        continue
                    if len(a_pos) < 3 or len(b_pos) < 3:
                        continue
                    try:
                        mobius_links.append(
                            (
                                float(a_pos[0]),
                                float(a_pos[1]),
                                float(a_pos[2]),
                                float(b_pos[0]),
                                float(b_pos[1]),
                                float(b_pos[2]),
                                float(entry.get("radius", 1.2)),
                            )
                        )
                    except Exception:
                        continue
            if min_x == 6.0 and max_x == 60.0 and min_y == 6.0 and max_y == 60.0:
                try:
                    map_w = float(hint.get("map_w", 0.0))
                    map_d = float(hint.get("map_d", 0.0))
                    if map_w > 1.0 and map_d > 1.0:
                        min_x, max_x = 0.0, map_w
                        min_y, max_y = 0.0, map_d
                except Exception:
                    pass
            if base_z == 1.0:
                try:
                    floor_y = float(hint.get("floor_y", 0.0))
                    base_z = floor_y + 1.0
                except Exception:
                    pass
        except Exception:
            pass
    width = max(1.0, max_x - min_x)
    height = max(1.0, max_y - min_y)
    center_x = min_x + width * 0.5 if args.spawn_center_x is None else args.spawn_center_x
    center_y = min_y + height * 0.5 if args.spawn_center_y is None else args.spawn_center_y
    spawn_radius = min(width, height) * 0.18 if args.spawn_radius is None else max(1.0, args.spawn_radius)
    if spawn_full_map:
        spawn_min_x = min_x
        spawn_max_x = max_x
        spawn_min_y = min_y
        spawn_max_y = max_y
    else:
        spawn_min_x = max(min_x, center_x - spawn_radius)
        spawn_max_x = min(max_x, center_x + spawn_radius)
        spawn_min_y = max(min_y, center_y - spawn_radius)
        spawn_max_y = min(max_y, center_y + spawn_radius)
    target_sep = max(6.0, min(width, height) / max(2.0, math.sqrt(args.count)))
    target_sep_sq = target_sep * target_sep

    spawn_points: list[tuple[float, float]] = []
    attempts = 0
    max_attempts = args.count * 40
    while len(spawn_points) < args.count and attempts < max_attempts:
        attempts += 1
        if spawn_full_map:
            cx = random.uniform(spawn_min_x, spawn_max_x)
            cy = random.uniform(spawn_min_y, spawn_max_y)
        else:
            angle = random.uniform(0.0, math.tau)
            radius = math.sqrt(random.random()) * spawn_radius
            cx = center_x + math.cos(angle) * radius
            cy = center_y + math.sin(angle) * radius
            cx = max(spawn_min_x, min(spawn_max_x, cx))
            cy = max(spawn_min_y, min(spawn_max_y, cy))
        too_close = False
        for px, py in spawn_points:
            dx = cx - px
            dy = cy - py
            if (dx * dx + dy * dy) < target_sep_sq:
                too_close = True
                break
        if not too_close:
            spawn_points.append((cx, cy))

    if len(spawn_points) < args.count:
        grid_cols = max(1, int(math.ceil(math.sqrt(args.count))))
        grid_rows = max(1, int(math.ceil(args.count / grid_cols)))
        grid_w = max(1.0, spawn_max_x - spawn_min_x)
        grid_h = max(1.0, spawn_max_y - spawn_min_y)
        cell_w = grid_w / grid_cols
        cell_h = grid_h / grid_rows
        for r in range(grid_rows):
            for c in range(grid_cols):
                if len(spawn_points) >= args.count:
                    break
                cx = spawn_min_x + (c + 0.5) * cell_w
                cy = spawn_min_y + (r + 0.5) * cell_h
                jitter_x = random.uniform(-cell_w * 0.35, cell_w * 0.35)
                jitter_y = random.uniform(-cell_h * 0.35, cell_h * 0.35)
                spawn_points.append((cx + jitter_x, cy + jitter_y))

    random.shuffle(spawn_points)

    threads = []
    for idx in range(args.count):
        start_x, start_y = spawn_points[idx % len(spawn_points)]
        mobius_center_y = center_y if args.mobius_center_y is None else args.mobius_center_y
        if mobius_center_y_hint is not None and args.mobius_center_y is None:
            try:
                mobius_center_y = float(mobius_center_y_hint)
            except Exception:
                pass
        mobius_band = max(0.0, mobius_ratio) * height
        mobius_half_band = mobius_band * 0.5 if mobius_enabled else 0.0
        t = threading.Thread(
            target=client_loop,
            args=(
                idx,
                args.host,
                args.port,
                args.rate,
                min_x,
                max_x,
                min_y,
                max_y,
                base_z,
                args.min_w,
                args.max_w,
                start_x,
                start_y,
                args.bounds_margin,
                mobius_center_y,
                mobius_half_band,
                args.mobius_nudge,
                args.mobius_stuck_seconds,
                mobius_links,
                args.count,
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
