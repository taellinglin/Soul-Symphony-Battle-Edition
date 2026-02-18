import math
import importlib
import os
from typing import Iterable

np = None
njit = None

try:
    np = importlib.import_module("numpy")
except Exception:
    np = None

try:
    _numba = importlib.import_module("numba")
    njit = getattr(_numba, "njit", None)
except Exception:
    njit = None

NUMPY_AVAILABLE = np is not None
NUMBA_ENABLED_BY_ENV = str(os.getenv("SOULSYM_ENABLE_NUMBA", "0")).strip().lower() in {"1", "true", "yes", "on"}
NUMBA_AVAILABLE = bool(NUMPY_AVAILABLE and njit is not None and NUMBA_ENABLED_BY_ENV)


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _find_room_idx_numba(x: float, y: float, z: float, x0, x1, y0, y1, zc) -> int:
        best_idx = -1
        best_dz = 1.0e18
        n = x0.shape[0]
        for i in range(n):
            if x >= x0[i] and x <= x1[i] and y >= y0[i] and y <= y1[i]:
                dz = abs(z - zc[i])
                if dz < best_dz:
                    best_dz = dz
                    best_idx = i
        return best_idx
else:
    def _find_room_idx_numba(x: float, y: float, z: float, x0, x1, y0, y1, zc) -> int:
        return -1


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _find_first_zone_idx_numba(x: float, y: float, x0, x1, y0, y1) -> int:
        n = x0.shape[0]
        for i in range(n):
            if x >= x0[i] and x <= x1[i] and y >= y0[i] and y <= y1[i]:
                return i
        return -1
else:
    def _find_first_zone_idx_numba(x: float, y: float, x0, x1, y0, y1) -> int:
        return -1


def prepare_room_bounds(rooms, room_levels: dict[int, int], floor_y: float, level_z_step: float, wall_h: float):
    if not NUMPY_AVAILABLE or not rooms:
        return None

    count = len(rooms)
    x0 = np.empty(count, dtype=np.float64)
    x1 = np.empty(count, dtype=np.float64)
    y0 = np.empty(count, dtype=np.float64)
    y1 = np.empty(count, dtype=np.float64)
    zc = np.empty(count, dtype=np.float64)

    for idx, room in enumerate(rooms):
        x0[idx] = float(room.x)
        x1[idx] = float(room.x + room.w)
        y0[idx] = float(room.y)
        y1[idx] = float(room.y + room.h)
        level = float(room_levels.get(idx, 0))
        base_z = float(floor_y) + level * float(level_z_step)
        zc[idx] = base_z + float(wall_h) * 0.5

    return {
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
        "zc": zc,
    }


def prepare_zone_bounds(zones):
    if not NUMPY_AVAILABLE or not zones:
        return None

    count = len(zones)
    x0 = np.empty(count, dtype=np.float64)
    x1 = np.empty(count, dtype=np.float64)
    y0 = np.empty(count, dtype=np.float64)
    y1 = np.empty(count, dtype=np.float64)

    for idx, zone in enumerate(zones):
        room = zone.get("room")
        if room is None:
            x0[idx] = 0.0
            x1[idx] = -1.0
            y0[idx] = 0.0
            y1[idx] = -1.0
            continue
        x0[idx] = float(room.x)
        x1[idx] = float(room.x + room.w)
        y0[idx] = float(room.y)
        y1[idx] = float(room.y + room.h)

    return {
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
    }


def find_room_index_for_pos(x: float, y: float, z: float, bounds) -> int | None:
    if bounds is None:
        return None

    x0 = bounds["x0"]
    x1 = bounds["x1"]
    y0 = bounds["y0"]
    y1 = bounds["y1"]
    zc = bounds["zc"]

    if NUMBA_AVAILABLE:
        idx = _find_room_idx_numba(float(x), float(y), float(z), x0, x1, y0, y1, zc)
        return None if idx < 0 else int(idx)

    if not NUMPY_AVAILABLE:
        return None

    mask = (x0 <= x) & (x <= x1) & (y0 <= y) & (y <= y1)
    hits = np.nonzero(mask)[0]
    if hits.size == 0:
        return None
    z_deltas = np.abs(zc[hits] - float(z))
    best_local = int(np.argmin(z_deltas))
    return int(hits[best_local])


def find_zone_index_for_pos(x: float, y: float, bounds) -> int | None:
    if bounds is None:
        return None

    x0 = bounds["x0"]
    x1 = bounds["x1"]
    y0 = bounds["y0"]
    y1 = bounds["y1"]

    if NUMBA_AVAILABLE:
        idx = _find_first_zone_idx_numba(float(x), float(y), x0, x1, y0, y1)
        return None if idx < 0 else int(idx)

    if not NUMPY_AVAILABLE:
        return None

    mask = (x0 <= x) & (x <= x1) & (y0 <= y) & (y <= y1)
    hits = np.nonzero(mask)[0]
    if hits.size == 0:
        return None
    return int(hits[0])


def compute_level_w_scalar(x: float, y: float, z: float, corridor_w: float, level_z_step: float) -> float:
    grid = max(10.0, float(corridor_w) * 1.35)
    z_grid = max(1.0, float(level_z_step))

    qx = round(float(x) / grid) * grid
    qy = round(float(y) / grid) * grid
    qz = round(float(z) / z_grid) * z_grid

    wave_a = math.sin(qx * 0.018 + qz * 0.021)
    wave_b = math.cos(qy * 0.017 - qz * 0.019)
    wave_c = math.sin((qx + qy) * 0.006)
    w = (wave_a * 0.72 + wave_b * 0.72 + wave_c * 0.46) * 1.18
    return max(-2.25, min(2.25, w))


def compute_level_w_batch(positions: Iterable[tuple[float, float, float]], corridor_w: float, level_z_step: float):
    if not NUMPY_AVAILABLE:
        return [
            compute_level_w_scalar(float(px), float(py), float(pz), corridor_w, level_z_step)
            for px, py, pz in positions
        ]

    arr = np.asarray(list(positions), dtype=np.float64)
    if arr.size == 0:
        return np.empty(0, dtype=np.float64)

    grid = max(10.0, float(corridor_w) * 1.35)
    z_grid = max(1.0, float(level_z_step))

    qx = np.round(arr[:, 0] / grid) * grid
    qy = np.round(arr[:, 1] / grid) * grid
    qz = np.round(arr[:, 2] / z_grid) * z_grid

    wave_a = np.sin(qx * 0.018 + qz * 0.021)
    wave_b = np.cos(qy * 0.017 - qz * 0.019)
    wave_c = np.sin((qx + qy) * 0.006)
    w = (wave_a * 0.72 + wave_b * 0.72 + wave_c * 0.46) * 1.18
    return np.clip(w, -2.25, 2.25)


def generate_labyrinth_room_dims(cols: int, rows: int, base_room_w: float, base_room_h: float, jitter: float):
    if not NUMPY_AVAILABLE:
        return None

    count = int(max(0, cols * rows))
    if count <= 0:
        return None

    size_scale = np.random.uniform(1.0 - jitter * 0.55, 1.0 + jitter * 0.38, size=count)
    aspect_skew = np.random.uniform(-jitter * 0.48, jitter * 0.48, size=count)

    burst_mask = np.random.random(count) < (0.14 + jitter * 0.22)
    burst_scale = np.random.uniform(0.78, 1.24, size=count)
    size_scale = size_scale * np.where(burst_mask, burst_scale, 1.0)

    room_w = np.maximum(2.8, base_room_w * size_scale * (1.0 + aspect_skew))
    room_h = np.maximum(2.8, base_room_h * size_scale * (1.0 - aspect_skew))
    return room_w, room_h


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _mobius_fold_twist_numba(
        vx: float,
        vy: float,
        vz: float,
        fpx: float,
        fpy: float,
        fpz: float,
        upx: float,
        upy: float,
        upz: float,
        player_w: float,
        roll_time: float,
        phase: float,
        twist_strength: float,
        hyper_w_limit: float,
    ) -> tuple[float, float, float, float]:
        side_x = upy * fpz - upz * fpy
        side_y = upz * fpx - upx * fpz
        side_z = upx * fpy - upy * fpx
        side_len = math.sqrt(side_x * side_x + side_y * side_y + side_z * side_z)
        if side_len > 1.0e-8:
            inv = 1.0 / side_len
            side_x *= inv
            side_y *= inv
            side_z *= inv
        else:
            side_x = 1.0
            side_y = 0.0
            side_z = 0.0

        v_dot_fp = vx * fpx + vy * fpy + vz * fpz
        v_dot_side = vx * side_x + vy * side_y + vz * side_z
        v_dot_up = vx * upx + vy * upy + vz * upz

        f_comp_x = fpx * v_dot_fp
        f_comp_y = fpy * v_dot_fp
        f_comp_z = fpz * v_dot_fp

        s_comp_x = side_x * v_dot_side
        s_comp_y = side_y * v_dot_side
        s_comp_z = side_z * v_dot_side

        u_comp_x = upx * v_dot_up
        u_comp_y = upy * v_dot_up
        u_comp_z = upz * v_dot_up

        twisted_x = f_comp_x - s_comp_x + u_comp_x * 0.9
        twisted_y = f_comp_y - s_comp_y + u_comp_y * 0.9
        twisted_z = f_comp_z - s_comp_z + u_comp_z * 0.9

        twist = max(0.0, min(1.0, twist_strength))
        wobble = math.sin(roll_time * 0.32 + phase) * (1.0 - twist) * hyper_w_limit * 0.22
        new_w = -player_w * twist + wobble
        new_w = max(-hyper_w_limit, min(hyper_w_limit, new_w))

        out_vx = twisted_x * 0.9 + fpx * 1.75
        out_vy = twisted_y * 0.9 + fpy * 1.75
        out_vz = twisted_z * 0.9 + fpz * 1.75
        return out_vx, out_vy, out_vz, new_w
else:
    def _mobius_fold_twist_numba(
        vx: float,
        vy: float,
        vz: float,
        fpx: float,
        fpy: float,
        fpz: float,
        upx: float,
        upy: float,
        upz: float,
        player_w: float,
        roll_time: float,
        phase: float,
        twist_strength: float,
        hyper_w_limit: float,
    ) -> tuple[float, float, float, float]:
        side_x = upy * fpz - upz * fpy
        side_y = upz * fpx - upx * fpz
        side_z = upx * fpy - upy * fpx
        side_len = math.sqrt(side_x * side_x + side_y * side_y + side_z * side_z)
        if side_len > 1.0e-8:
            inv = 1.0 / side_len
            side_x *= inv
            side_y *= inv
            side_z *= inv
        else:
            side_x = 1.0
            side_y = 0.0
            side_z = 0.0

        v_dot_fp = vx * fpx + vy * fpy + vz * fpz
        v_dot_side = vx * side_x + vy * side_y + vz * side_z
        v_dot_up = vx * upx + vy * upy + vz * upz

        f_comp_x = fpx * v_dot_fp
        f_comp_y = fpy * v_dot_fp
        f_comp_z = fpz * v_dot_fp

        s_comp_x = side_x * v_dot_side
        s_comp_y = side_y * v_dot_side
        s_comp_z = side_z * v_dot_side

        u_comp_x = upx * v_dot_up
        u_comp_y = upy * v_dot_up
        u_comp_z = upz * v_dot_up

        twisted_x = f_comp_x - s_comp_x + u_comp_x * 0.9
        twisted_y = f_comp_y - s_comp_y + u_comp_y * 0.9
        twisted_z = f_comp_z - s_comp_z + u_comp_z * 0.9

        twist = max(0.0, min(1.0, twist_strength))
        wobble = math.sin(roll_time * 0.32 + phase) * (1.0 - twist) * hyper_w_limit * 0.22
        new_w = max(-hyper_w_limit, min(hyper_w_limit, (-player_w * twist + wobble)))

        out_vx = twisted_x * 0.9 + fpx * 1.75
        out_vy = twisted_y * 0.9 + fpy * 1.75
        out_vz = twisted_z * 0.9 + fpz * 1.75
        return out_vx, out_vy, out_vz, new_w


def compute_mobius_fold_twist(
    vel: tuple[float, float, float],
    fold_push: tuple[float, float, float],
    up: tuple[float, float, float],
    player_w: float,
    roll_time: float,
    phase: float,
    twist_strength: float,
    hyper_w_limit: float,
) -> tuple[float, float, float, float]:
    return _mobius_fold_twist_numba(
        float(vel[0]),
        float(vel[1]),
        float(vel[2]),
        float(fold_push[0]),
        float(fold_push[1]),
        float(fold_push[2]),
        float(up[0]),
        float(up[1]),
        float(up[2]),
        float(player_w),
        float(roll_time),
        float(phase),
        float(twist_strength),
        float(hyper_w_limit),
    )
