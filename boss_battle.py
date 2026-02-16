import math
import os
import random
from typing import Any

from panda3d.core import Shader, Vec3


def _ensure_boss_arena_shader(game: Any):
    shader = getattr(game, "boss_room_shader", None)
    if shader is not None:
        return shader
    vert_path = os.path.join("graphics", "shaders", "boss_hyper.vert")
    frag_path = os.path.join("graphics", "shaders", "boss_hyper.frag")
    try:
        shader = Shader.load(Shader.SL_GLSL, vert_path, frag_path)
    except Exception:
        shader = None
    game.boss_room_shader = shader
    return shader


def _apply_boss_surface_style(game: Any, holder, kind: str, base_tex, tint: tuple[float, float, float, float]) -> None:
    shader = _ensure_boss_arena_shader(game)
    nodes = holder.findAllMatches("**/+GeomNode")
    intensity = 1.0
    variant = 0.0
    if kind == "walls":
        intensity = 1.24
        variant = 1.0
    elif kind == "ceiling":
        intensity = 1.48
        variant = 2.0
    for node in nodes:
        node.setColor(*tint)
        if base_tex is not None:
            node.setTexture(base_tex, 1)
        node.setLightOff(1)
        if shader is not None:
            node.setShader(shader)
            node.setShaderInput("u_time", float(getattr(game, "roll_time", 0.0)))
            node.setShaderInput("u_intensity", float(intensity))
            node.setShaderInput("u_variant", float(variant))
            node.setShaderInput("u_hyper_w", float(getattr(game, "player_w", 0.0)))


def _clear_boss_room_arena(game: Any) -> None:
    collider_nodes = list(getattr(game, "boss_room_arena_collider_nodes", []))
    for body_np in collider_nodes:
        if body_np is None or body_np.isEmpty():
            continue
        try:
            body = body_np.node()
        except Exception:
            body = None
        if body is not None:
            try:
                game.physics_world.removeRigidBody(body)
            except Exception:
                pass
        try:
            body_np.removeNode()
        except Exception:
            pass
    game.boss_room_arena_collider_nodes = []

    root = getattr(game, "boss_room_arena_root", None)
    if root is not None and (not root.isEmpty()):
        root.removeNode()
    game.boss_room_arena_root = None
    game.boss_room_shader_nodes = []


def _hex_contains_xy(x: float, y: float, radius: float) -> bool:
    if radius <= 1e-6:
        return False
    ax = abs(x)
    ay = abs(y)
    if ax > radius:
        return False
    root3 = 1.7320508075688772
    return ay <= (root3 * 0.5 * radius - 0.5 * ax)


def _boss_terrain_height(x: float, y: float, radius: float) -> float:
    r = max(1.0, float(radius))
    nx = x / r
    ny = y / r
    radial = math.sqrt(nx * nx + ny * ny)
    waves = (
        math.sin(nx * 6.4)
        + 0.86 * math.cos(ny * 5.8)
        + 0.58 * math.sin((nx + ny) * 4.2)
        + 0.42 * math.cos((nx - ny) * 7.1)
    )
    basin = -0.95 * (radial ** 1.35)
    ridge = 0.33 * math.sin((nx * nx - ny * ny) * 9.0)
    h = 1.05 + 0.72 * waves + basin + ridge
    return max(0.24, min(2.95, h))


def _stash_world_for_boss(game: Any) -> None:
    game.boss_room_prev_scene_culling = bool(getattr(game, "enable_scene_culling", True))
    game.boss_room_prev_wall_occlusion = bool(getattr(game, "enable_wall_occlusion_culling", True))
    game.enable_scene_culling = False
    game.enable_wall_occlusion_culling = False
    for visual in list(getattr(game, "scene_visuals", {}).values()):
        if visual is None or visual.isEmpty():
            continue
        if visual.isStashed():
            visual.unstash()

    hidden_ids: set[int] = set()
    hidden_nodes: list = []
    for vid, visual in list(getattr(game, "scene_visuals", {}).items()):
        if visual is None or visual.isEmpty():
            continue
        if not visual.isStashed():
            visual.stash()
            hidden_ids.add(int(vid))
    for entry in getattr(game, "water_surfaces", []):
        node = entry.get("node")
        if node is None or node.isEmpty():
            continue
        if not node.isStashed():
            node.stash()
            hidden_nodes.append(node)
    game.boss_room_hidden_visual_ids = hidden_ids
    game.boss_room_hidden_nodes = hidden_nodes


def _restore_world_after_boss(game: Any) -> None:
    for vid in list(getattr(game, "boss_room_hidden_visual_ids", set())):
        visual = getattr(game, "scene_visuals", {}).get(int(vid))
        if visual is None or visual.isEmpty():
            continue
        if visual.isStashed():
            visual.unstash()
    for node in list(getattr(game, "boss_room_hidden_nodes", [])):
        if node is None or node.isEmpty():
            continue
        if node.isStashed():
            node.unstash()
    game.boss_room_hidden_visual_ids = set()
    game.boss_room_hidden_nodes = []
    game.enable_scene_culling = bool(getattr(game, "boss_room_prev_scene_culling", True))
    game.enable_wall_occlusion_culling = bool(getattr(game, "boss_room_prev_wall_occlusion", True))
    game.boss_room_prev_scene_culling = True
    game.boss_room_prev_wall_occlusion = True


def cleanup_boss_room(game: Any) -> None:
    _clear_boss_room_arena(game)
    _restore_world_after_boss(game)


def _build_hex_boss_arena(game: Any, room_idx: int, center: Vec3) -> None:
    _clear_boss_room_arena(game)
    root = game.world.attachNewNode("boss-room-hex-arena")
    root.setPos(center)
    root.setTag("boss_arena", "1")
    game.boss_room_arena_root = root
    game.boss_room_shader_nodes = []
    game.boss_room_arena_collider_nodes = []

    room = game.rooms[room_idx]
    room_span = max(8.2, min(room.w, room.h) * 0.72)
    arena_radius = max(12.0, min(24.0, room_span * 1.7))
    wall_thickness = 0.95
    wall_height = max(6.2, min(12.5, game.wall_h * 1.8))
    side_length = max(2.8, arena_radius)

    floor_tex = game._get_random_room_texture()
    wall_tex = game.level_checker_tex
    ceiling_tex = getattr(game, "water_base_tex", wall_tex)

    floor_base = game._add_box(
        Vec3(0.0, 0.0, -1.12),
        Vec3(arena_radius * 1.06, arena_radius * 1.06, 0.6),
        color=(0.58, 0.82, 1.0, 1.0),
        parent=root,
        collidable=False,
        surface_mode="floor",
    )
    _apply_boss_surface_style(game, floor_base, "floor", floor_tex, (0.56, 0.92, 1.0, 1.0))

    tile_span = max(1.3, arena_radius / 8.0)
    tile_half_xy = tile_span * 0.52
    grid_steps = max(5, int(arena_radius / tile_span) + 2)
    base_z = -1.82
    for ix in range(-grid_steps, grid_steps + 1):
        x = ix * tile_span
        for iy in range(-grid_steps, grid_steps + 1):
            y = iy * tile_span
            if not _hex_contains_xy(x, y, arena_radius * 0.95):
                continue

            top_z = _boss_terrain_height(x, y, arena_radius)
            half_z = max(0.24, (top_z - base_z) * 0.5)
            center_z = base_z + half_z
            terrain_block = game._add_box(
                Vec3(x, y, center_z),
                Vec3(tile_half_xy, tile_half_xy, half_z),
                color=(0.58, 0.86, 1.0, 1.0),
                parent=root,
                collidable=True,
                surface_mode="floor",
            )
            _apply_boss_surface_style(game, terrain_block, "floor", floor_tex, (0.56, 0.92, 1.0, 1.0))

    ceiling = game._add_box(
        Vec3(0.0, 0.0, wall_height + 0.72),
        Vec3(arena_radius * 1.02, arena_radius * 1.02, 0.38),
        color=(0.92, 0.4, 1.0, 1.0),
        parent=root,
        collidable=False,
    )
    _apply_boss_surface_style(game, ceiling, "ceiling", ceiling_tex, (0.96, 0.56, 1.0, 1.0))

    for i in range(6):
        angle = (math.tau / 6.0) * i
        nx = math.cos(angle)
        ny = math.sin(angle)
        wall_pos = Vec3(nx * arena_radius, ny * arena_radius, wall_height * 0.5)
        wall_h = math.degrees(angle) + 90.0
        wall = game._add_box(
            wall_pos,
            Vec3(side_length * 0.52, wall_thickness, wall_height * 0.5),
            color=(0.55, 1.0, 0.96, 1.0),
            hpr=Vec3(wall_h, 0.0, 0.0),
            parent=root,
            collidable=False,
        )
        _apply_boss_surface_style(game, wall, "walls", wall_tex, (0.64, 1.0, 0.94, 1.0))

    ceiling_world = center + Vec3(0.0, 0.0, wall_height + 0.72)
    ceiling_collider = game._add_static_box_collider(
        ceiling_world,
        Vec3(arena_radius * 1.02, arena_radius * 1.02, 0.38),
        hpr=None,
        visual_holder=root,
    )
    game.boss_room_arena_collider_nodes.append(ceiling_collider)

    for i in range(6):
        angle = (math.tau / 6.0) * i
        nx = math.cos(angle)
        ny = math.sin(angle)
        wall_world = center + Vec3(nx * arena_radius, ny * arena_radius, wall_height * 0.5)
        wall_h = math.degrees(angle) + 90.0
        wall_collider = game._add_static_box_collider(
            wall_world,
            Vec3(side_length * 0.52, wall_thickness, wall_height * 0.5),
            hpr=Vec3(wall_h, 0.0, 0.0),
            visual_holder=root,
        )
        game.boss_room_arena_collider_nodes.append(wall_collider)

    game.boss_room_arena_center = Vec3(center)
    game.boss_room_arena_radius = arena_radius


def update_boss_room_visuals(game: Any, dt: float) -> None:
    root = getattr(game, "boss_room_arena_root", None)
    if root is None or root.isEmpty():
        return
    if root.isStashed():
        root.unstash()
    root.setAlphaScale(1.0)
    game.boss_room_shader_time = float(getattr(game, "boss_room_shader_time", 0.0)) + max(0.0, float(dt))
    shader_time = game.boss_room_shader_time
    hyper_w = float(getattr(game, "player_w", 0.0))
    nodes = root.findAllMatches("**/+GeomNode")
    for node in nodes:
        if node.isStashed():
            node.unstash()
        node.setAlphaScale(1.0)
        node.setShaderInput("u_time", shader_time)
        node.setShaderInput("u_hyper_w", hyper_w)


def choose_boss_room_index(game: Any) -> int:
    if not game.rooms:
        return 0
    start_idx = max(0, min(len(game.rooms) - 1, int(getattr(game, "start_room_idx", 0))))
    start_center = game._room_center_pos(start_idx)
    best_idx = start_idx
    best_score = -1e9
    for idx in range(len(game.rooms)):
        if idx == start_idx:
            continue
        center = game._room_center_pos(idx)
        dist = (center - start_center).length()
        score = dist + float(game.room_levels.get(idx, 0)) * 7.0
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def begin_boss_room_encounter(game: Any) -> None:
    if not game.rooms:
        return
    game.monster_boss_room_active = True
    game.monster_boss_pending = False
    game.monster_boss_defeated = False
    game.monster_boss_room_idx = choose_boss_room_index(game)
    game._clear_all_monsters()
    _stash_world_for_boss(game)
    game._spawn_hypercube_monsters(count=1)

    boss_room_idx = int(game.monster_boss_room_idx)
    isolated_center = Vec3(
        game.map_w * 0.5,
        game.map_d * 0.5,
        float(getattr(game, "hyper_bounds_top_z", game.floor_y + 40.0)) + 160.0,
    )
    boss_spawn = Vec3(isolated_center)
    _build_hex_boss_arena(game, boss_room_idx, boss_spawn)
    for monster in game.monsters:
        root = monster.get("root")
        if root is not None and not root.isEmpty():
            root.setPos(boss_spawn + Vec3(random.uniform(-0.45, 0.45), random.uniform(-0.45, 0.45), 0.0))
            monster["prev_pos"] = Vec3(root.getPos())
        monster["room_idx"] = boss_room_idx
        monster["level"] = min(int(getattr(game, "monster_level_cap", 150)), max(1, int(getattr(game, "monster_level_current", 1)) + 7))
        monster["boss_orbit_phase"] = random.uniform(0.0, math.tau)
        monster["boss_burst_timer"] = random.uniform(0.35, 0.8)
        game._apply_monster_progression_stats(monster, boss_mode=True)

    if hasattr(game, "ball_np") and game.ball_np is not None and (not game.ball_np.isEmpty()):
        player_spawn = game._safe_room_spawn_pos(boss_room_idx, z_lift=0.34)
        player_spawn += Vec3(0.0, -max(1.8, game.sword_reach * 1.1), 0.0)
        game.ball_np.setPos(player_spawn)
        if hasattr(game, "ball_body") and game.ball_body is not None:
            game.ball_body.setLinearVelocity(Vec3(0, 0, 0))
            game.ball_body.setAngularVelocity(Vec3(0, 0, 0))
    game._spawn_floating_text(boss_spawn + Vec3(0, 0, 1.8), "BOSS ROOM", (1.0, 0.45, 0.9, 1.0), scale=0.34, life=1.6)
    game._attach_monster_hum_sounds()
    game._setup_monster_ai_system()
    game._update_monster_hud_ui()


def update_monster_progression(game: Any, dt: float) -> None:
    if game.game_over_active or game.win_active:
        return
    alive = game._count_alive_monsters()
    if alive > 0:
        return
    game.monster_respawn_timer -= max(0.0, float(dt))
    if game.monster_respawn_timer > 0.0:
        return

    if game.monster_boss_room_active:
        game.monster_boss_room_active = False
        game.monster_boss_defeated = True
        _clear_boss_room_arena(game)
        _restore_world_after_boss(game)
        game.monster_wave_index += 1
        game.monster_level_current = min(int(getattr(game, "monster_level_cap", 150)), int(getattr(game, "monster_level_current", 1)) + 1)
        game.monster_respawn_timer = float(getattr(game, "monster_respawn_delay", 1.1))
        game._spawn_progression_wave(reset_progression=False)
        return

    if game.monster_boss_pending or game.monster_wave_kills >= max(1, int(getattr(game, "monster_wave_kill_goal", 1))):
        game.monster_respawn_timer = float(getattr(game, "monster_boss_entry_grace", 1.2))
        begin_boss_room_encounter(game)
        return

    game.monster_respawn_timer = float(getattr(game, "monster_respawn_delay", 1.1))
    game._spawn_progression_wave(reset_progression=False)


def apply_boss_hypercube_motion(game: Any, monster: dict, pos: Vec3, ball_pos: Vec3, dt: float) -> Vec3:
    boss_phase = float(monster.get("boss_orbit_phase", 0.0))
    to_player = Vec3(ball_pos.x - pos.x, ball_pos.y - pos.y, 0.0)
    if to_player.lengthSquared() > 1e-8:
        to_player.normalize()
        side = Vec3(-to_player.y, to_player.x, 0.0)
        orbit_push = side * (2.1 + 0.9 * math.sin(game.roll_time * 2.4 + boss_phase))
        pos += orbit_push * dt * 0.42
    burst = max(0.0, float(monster.get("boss_burst_timer", 0.0)) - dt)
    monster["boss_burst_timer"] = burst
    if burst <= 0.0 and to_player.lengthSquared() > 1e-8:
        dash = to_player * random.uniform(1.4, 2.7)
        pos += dash
        monster["boss_burst_timer"] = random.uniform(0.5, 1.2)
        monster["w_vel"] += random.uniform(-2.2, 2.2)
    return pos