import math
import os
import random
from typing import Any

from panda3d.core import Shader, TransparencyAttrib, Vec3


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
    inverted_root = getattr(game, "boss_room_inverted_root", None)
    if inverted_root is not None and (not inverted_root.isEmpty()):
        inverted_root.removeNode()
    game.boss_room_inverted_root = None
    game.boss_room_platform_movers = []
    for node in list(getattr(game, "boss_room_water_nodes", [])):
        if node is None or node.isEmpty():
            continue
        try:
            node.removeNode()
        except Exception:
            pass
    game.boss_room_water_nodes = []


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
    game.boss_room_prev_water_surfaces = list(getattr(game, "water_surfaces", []))
    game.water_surfaces = []
    game.boss_room_hidden_visual_ids = hidden_ids
    game.boss_room_hidden_nodes = hidden_nodes
    game.boss_room_prev_liminal_nodes = list(getattr(game, "liminal_fold_nodes", []))
    game.boss_room_prev_liminal_links = dict(getattr(game, "liminal_fold_links", {}))
    game.liminal_fold_nodes = []
    game.liminal_fold_links = {}


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
    if hasattr(game, "boss_room_prev_water_surfaces"):
        game.water_surfaces = list(getattr(game, "boss_room_prev_water_surfaces", []))
        game.boss_room_prev_water_surfaces = []
    if hasattr(game, "boss_room_prev_liminal_nodes"):
        game.liminal_fold_nodes = list(getattr(game, "boss_room_prev_liminal_nodes", []))
        game.liminal_fold_links = dict(getattr(game, "boss_room_prev_liminal_links", {}))
        game.boss_room_prev_liminal_nodes = []
        game.boss_room_prev_liminal_links = {}
    if hasattr(game, "boss_room_prev_hyper_bounds_top_z"):
        game.hyper_bounds_top_z = float(getattr(game, "boss_room_prev_hyper_bounds_top_z"))
        game.hyper_bounds_bottom_z = float(getattr(game, "boss_room_prev_hyper_bounds_bottom_z"))
        game.boss_room_prev_hyper_bounds_top_z = None
        game.boss_room_prev_hyper_bounds_bottom_z = None
    game.enable_scene_culling = bool(getattr(game, "boss_room_prev_scene_culling", True))
    game.enable_wall_occlusion_culling = bool(getattr(game, "boss_room_prev_wall_occlusion", True))
    game.boss_room_prev_scene_culling = True
    game.boss_room_prev_wall_occlusion = True


def cleanup_boss_room(game: Any) -> None:
    _clear_boss_room_arena(game)
    _restore_world_after_boss(game)
    remote_players = getattr(game, "remote_players", None)
    if isinstance(remote_players, dict):
        for entry in remote_players.values():
            if isinstance(entry, dict):
                entry.pop("boss_warp_pos", None)


def _build_hex_boss_arena(game: Any, room_idx: int, center: Vec3) -> None:
    _clear_boss_room_arena(game)
    root = game.world.attachNewNode("boss-room-arena")
    root.setPos(center)
    root.setTag("boss_arena", "1")
    game.boss_room_arena_root = root
    game.boss_room_shader_nodes = []
    game.boss_room_arena_collider_nodes = []
    game.boss_room_platform_movers = []

    floor_y = float(center.z) - 2.0
    center_local = Vec3(0.0, 0.0, 0.0)
    arena_scale = 0.62
    map_w_used = float(getattr(game, "map_w", 176.0)) * arena_scale
    map_d_used = float(getattr(game, "map_d", 176.0)) * arena_scale
    margin = 8.0 * arena_scale
    inner_w = max(8.0, map_w_used - margin * 2.0)
    inner_h = max(8.0, map_d_used - margin * 2.0)

    overscan = max(0.0, float(getattr(game, "water_loop_overscan", 6.0))) * arena_scale
    water_half_x = map_w_used * 0.5 + overscan
    water_half_y = map_d_used * 0.5 + overscan
    water_half_z = max(0.08, float(getattr(game, "floor_t", 0.2)) * 0.32)
    water_center_z = floor_y + float(getattr(game, "water_surface_raise", 0.0))
    water_holder = game._add_box(
        Vec3(center_local.x, center_local.y, water_center_z - center.z),
        Vec3(water_half_x, water_half_y, water_half_z),
        color=(0.26, 0.52, 0.78, 0.24),
        parent=root,
        collidable=False,
        surface_mode="water",
    )
    if water_holder is not None and not water_holder.isEmpty():
        water_holder.setTransparency(TransparencyAttrib.MAlpha)
        water_holder.setDepthWrite(False)
        water_holder.setBin("transparent", 33)
        if getattr(game, "water_surface_shader", None) is not None:
            water_holder.setShader(game.water_surface_shader)
            if hasattr(game, "_apply_water_surface_room_texture"):
                game._apply_water_surface_room_texture(water_holder)
            if water_holder not in getattr(game, "water_shader_nodes", []):
                game.water_shader_nodes.append(water_holder)
        if hasattr(game, "_register_water_surface"):
            min_x = float(center.x) - map_w_used * 0.5 - overscan
            max_x = float(center.x) + map_w_used * 0.5 + overscan
            min_y = float(center.y) - map_d_used * 0.5 - overscan
            max_y = float(center.y) + map_d_used * 0.5 + overscan
            game._register_water_surface(water_holder, min_x, max_x, min_y, max_y, float(water_center_z))
        game.boss_room_water_nodes.append(water_holder)

    ring_count = 4 if getattr(game, "performance_mode", False) else 5
    seg_base = 8 if getattr(game, "performance_mode", False) else 10
    arena_radius_max = max(6.5, min(inner_w, inner_h) * 0.44)
    arena_radius_min = max(4.8, arena_radius_max * 0.32)
    hyper_w_limit = float(getattr(game, "hyper_w_limit", 7.2))

    hub_size = max(4.6, arena_radius_min * 0.8)
    hub_half = Vec3(hub_size, hub_size, 0.7)
    hub = game._add_box(
        center_local + Vec3(0.0, 0.0, 2.6),
        hub_half,
        color=(0.28, 0.46, 0.68, 1.0),
        parent=root,
        collidable=False,
        motion_group="platform",
        w_coord=0.0,
        surface_mode="floor",
    )
    if hub is not None and not hub.isEmpty():
        hub.clearTexture()
        hub.setTextureOff(1)
        if hasattr(game, "_apply_room_thermal_shader"):
            game._apply_room_thermal_shader(hub)
    hub_collider = game._add_static_box_collider(
        center + Vec3(0.0, 0.0, 2.6),
        Vec3(hub_half),
        hpr=None,
        visual_holder=hub,
    )
    game.boss_room_arena_collider_nodes.append(hub_collider)
    if hasattr(game, "_register_vertical_mover"):
        game._register_vertical_mover(hub, hub_collider, "platform")
    game.boss_room_hub_pos = Vec3(center) + Vec3(0.0, 0.0, 2.6)
    game.boss_room_hub_half = Vec3(hub_half)
    for ring in range(ring_count):
        ring_t = ring / max(1, ring_count - 1)
        radius = arena_radius_min + (arena_radius_max - arena_radius_min) * ring_t
        segments = seg_base + ring * 2
        z_base = 0.4 + ring * 2.1
        for seg in range(segments):
            ang = (math.tau * seg) / max(1, segments)
            arc_jitter = random.uniform(-0.22, 0.22)
            x = math.cos(ang + arc_jitter) * radius
            y = math.sin(ang + arc_jitter) * radius
            z = z_base + math.sin(ang * 2.0 + ring_t * math.pi) * 1.1
            hx = random.uniform(1.2, 2.0)
            hy = random.uniform(1.2, 2.0)
            hz = random.uniform(0.24, 0.46)
            w_coord = max(-hyper_w_limit, min(hyper_w_limit, math.sin(ang * 2.0 + ring * 0.73) * hyper_w_limit * (0.35 + 0.52 * ring_t)))
            moving = (seg % 3 == 0) and (random.random() < 0.5)
            color = (
                0.3 + 0.24 * ring_t,
                0.42 + 0.32 * (0.5 + 0.5 * math.sin(ang + ring_t)),
                0.7 + 0.24 * (0.5 + 0.5 * math.cos(ang * 1.4)),
                1.0,
            )
            plat = game._add_box(
                Vec3(x, y, z),
                Vec3(hx, hy, hz),
                color=color,
                parent=root,
                collidable=False,
                motion_group=("platform" if moving else None),
                w_coord=w_coord,
                surface_mode="floor",
            )
            if plat is not None and not plat.isEmpty():
                plat.clearTexture()
                plat.setTextureOff(1)
                if hasattr(game, "_apply_room_thermal_shader"):
                    game._apply_room_thermal_shader(plat)
            plat_collider = game._add_static_box_collider(
                center + Vec3(x, y, z),
                Vec3(hx, hy, hz),
                hpr=None,
                visual_holder=plat,
            )
            game.boss_room_arena_collider_nodes.append(plat_collider)
            if moving:
                axis = Vec3(-math.sin(ang), math.cos(ang), 0.0)
                if axis.lengthSquared() < 1e-8:
                    axis = Vec3(1.0, 0.0, 0.0)
                else:
                    axis.normalize()
                mover_kind = "vertical" if (seg % 2 == 0) else "horizontal"
                amp = random.uniform(0.6, 1.4) * max(1.0, arena_scale)
                speed = random.uniform(0.65, 1.35)
                game.boss_room_platform_movers.append(
                    {
                        "visual": plat,
                        "body_np": plat_collider,
                        "base": Vec3(x, y, z),
                        "axis": axis,
                        "amp": amp,
                        "speed": speed,
                        "phase": random.uniform(0.0, math.tau),
                        "kind": mover_kind,
                    }
                )

            if seg % 2 == 0:
                pillar_h = random.uniform(0.9, 2.2)
                pillar = game._add_box(
                    Vec3(x, y, z + hz + pillar_h * 0.5),
                    Vec3(0.22, 0.22, pillar_h * 0.5),
                    color=(0.22, 0.66, 0.92, 1.0),
                    parent=root,
                    collidable=False,
                    w_coord=w_coord,
                    surface_mode="floor",
                )
                if pillar is not None and not pillar.isEmpty():
                    pillar.clearTexture()
                    pillar.setTextureOff(1)
                    if hasattr(game, "_apply_room_thermal_shader"):
                        game._apply_room_thermal_shader(pillar)
                pillar_collider = game._add_static_box_collider(
                    center + Vec3(x, y, z + hz + pillar_h * 0.5),
                    Vec3(0.22, 0.22, pillar_h * 0.5),
                    hpr=None,
                    visual_holder=pillar,
                )
                game.boss_room_arena_collider_nodes.append(pillar_collider)

    top_collider = game._add_static_box_collider(
        Vec3(center.x, center.y, float(getattr(game, "hyper_bounds_top_z", floor_y + 40.0)) + 0.11),
        Vec3(map_w_used * 0.5, map_d_used * 0.5, 0.11),
    )
    bottom_collider = game._add_static_box_collider(
        Vec3(center.x, center.y, float(getattr(game, "hyper_bounds_bottom_z", floor_y - 40.0)) - 0.11),
        Vec3(map_w_used * 0.5, map_d_used * 0.5, 0.11),
    )
    game.boss_room_arena_collider_nodes.append(top_collider)
    game.boss_room_arena_collider_nodes.append(bottom_collider)

    wall_t = max(0.6, float(getattr(game, "wall_t", 1.0)))
    wall_half_z = max(2.0, (float(getattr(game, "hyper_bounds_top_z", floor_y + 40.0)) - float(getattr(game, "hyper_bounds_bottom_z", floor_y - 40.0))) * 0.5 + 2.0)
    wall_center_z = (float(getattr(game, "hyper_bounds_top_z", floor_y + 40.0)) + float(getattr(game, "hyper_bounds_bottom_z", floor_y - 40.0))) * 0.5
    wall_x = map_w_used * 0.5 + wall_t
    wall_y = map_d_used * 0.5 + wall_t
    wall_pos_z = wall_center_z
    wall_scale_x = Vec3(wall_t, map_d_used * 0.5 + wall_t, wall_half_z)
    wall_scale_y = Vec3(map_w_used * 0.5 + wall_t, wall_t, wall_half_z)
    wall_left = game._add_static_box_collider(
        Vec3(center.x - wall_x, center.y, wall_pos_z),
        Vec3(wall_scale_x),
    )
    wall_right = game._add_static_box_collider(
        Vec3(center.x + wall_x, center.y, wall_pos_z),
        Vec3(wall_scale_x),
    )
    wall_down = game._add_static_box_collider(
        Vec3(center.x, center.y - wall_y, wall_pos_z),
        Vec3(wall_scale_y),
    )
    wall_up = game._add_static_box_collider(
        Vec3(center.x, center.y + wall_y, wall_pos_z),
        Vec3(wall_scale_y),
    )
    game.boss_room_arena_collider_nodes.extend([wall_left, wall_right, wall_down, wall_up])

    game.boss_room_arena_center = Vec3(center)
    game.boss_room_arena_radius = arena_radius_max
    game.boss_room_arena_scale = arena_scale

    _build_boss_inverted_world(game, center, floor_y)


def _build_boss_inverted_world(game: Any, center: Vec3, floor_y: float) -> None:
    root = getattr(game, "boss_room_arena_root", None)
    if root is None or root.isEmpty():
        return
    base_plane = float(getattr(game, "inverted_level_echo_plane_z", float(getattr(game, "floor_y", 0.0)) + 12.0))
    extra_offset = float(getattr(game, "inverted_level_echo_extra_offset", 0.0))
    opacity = float(getattr(game, "inverted_level_echo_opacity", 0.46))
    boss_plane = base_plane + (floor_y - float(getattr(game, "floor_y", 0.0)))

    inv_root = game.world.attachNewNode("boss-inverted-world")
    inv_root.setPos(0.0, 0.0, 2.0 * boss_plane + extra_offset)
    inv_root.setScale(1.0, 1.0, -1.0)
    inv_root.setTransparency(TransparencyAttrib.MAlpha)
    inv_root.setAlphaScale(opacity)
    game.boss_room_inverted_root = inv_root

    if getattr(game, "water_surface_shader", None) is not None:
        inv_root.setShader(game.water_surface_shader)
        inv_root.setShaderInput("u_room_tex", getattr(game, "level_checker_tex", None))
        inv_root.setShaderInput("u_time", 0.0)
        inv_root.setShaderInput("u_uv_scale", float(getattr(game, "room_thermal_uv_scale", 1.0)))
        inv_root.setShaderInput("u_alpha", 0.5)
        inv_root.setShaderInput("u_rainbow_strength", 0.0)
        inv_root.setShaderInput("u_diffusion_strength", 0.0)
        inv_root.setShaderInput("u_spec_strength", 0.0)
        inv_root.setShaderInput("u_room_tex_strength", 0.0)
        inv_root.setShaderInput("u_room_tex_desat", 1.0)
        inv_root.setShaderInput("u_thermal_mode", 1.0)
        inv_root.setShaderInput("u_thermal_strength", float(getattr(game, "room_thermal_strength", 1.0)))
        inv_root.setShaderInput("u_compression_factor", 1.0)
        inv_root.setShaderInput("u_compression_thermal_strength", 0.0)
        inv_root.setShaderInput("u_density_contrast", float(getattr(game, "room_thermal_density_contrast", 1.35)))
        inv_root.setShaderInput("u_density_gamma", float(getattr(game, "room_thermal_density_gamma", 0.85)))
        inv_root.setShaderInput("u_static_uv", 1.0)
        fog = getattr(game, "fog", None)
        if fog is not None:
            fog_color = fog.getColor()
            fog_start = float(getattr(game, "fog_start", 1.0e6))
            fog_end = float(getattr(game, "fog_end", 1.0e6 + 1.0))
        else:
            fog_color = (0.0, 0.0, 0.0, 1.0)
            fog_start, fog_end = (1.0e6, 1.0e6 + 1.0)
        inv_root.setShaderInput("u_fog_color", (float(fog_color[0]), float(fog_color[1]), float(fog_color[2])))
        inv_root.setShaderInput("u_fog_start", float(fog_start))
        inv_root.setShaderInput("u_fog_end", float(fog_end))
        reflection_tex = getattr(game, "water_reflection_tex", None)
        if reflection_tex is None:
            reflection_tex = getattr(game, "level_checker_tex", None)
        inv_root.setShaderInput("u_reflection_tex", reflection_tex)
        inv_root.setShaderInput("u_reflection_strength", 0.0)
    else:
        inv_root.setShaderOff(1)

    try:
        root.instanceTo(inv_root)
    except Exception:
        pass


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
    _update_boss_platform_movers(game, float(dt))


def _update_boss_platform_movers(game: Any, dt: float) -> None:
    movers = list(getattr(game, "boss_room_platform_movers", []))
    if not movers:
        return
    root = getattr(game, "boss_room_arena_root", None)
    if root is None or root.isEmpty():
        return
    root_pos = root.getPos(game.render)
    t = float(getattr(game, "roll_time", 0.0))
    for mover in movers:
        visual = mover.get("visual")
        if visual is None or visual.isEmpty():
            continue
        base = Vec3(mover.get("base", Vec3(0.0, 0.0, 0.0)))
        amp = float(mover.get("amp", 1.0))
        speed = float(mover.get("speed", 1.0))
        phase = float(mover.get("phase", 0.0))
        kind = str(mover.get("kind", "vertical"))
        if kind == "horizontal":
            axis = Vec3(mover.get("axis", Vec3(1.0, 0.0, 0.0)))
            offset = axis * (math.sin(t * speed + phase) * amp)
        else:
            offset = Vec3(0.0, 0.0, math.sin(t * speed + phase) * amp)
        target_local = base + offset
        visual.setPos(target_local)

        body_np = mover.get("body_np")
        if body_np is not None and not body_np.isEmpty():
            body_np.setPos(root_pos + target_local)


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
    if hasattr(game, "ball_np") and game.ball_np is not None and not game.ball_np.isEmpty():
        game.boss_return_pos = Vec3(game.ball_np.getPos())
    game.monster_boss_room_active = True
    game.monster_boss_pending = False
    game.monster_boss_defeated = False
    if hasattr(game, "_set_bgm_mode"):
        game._set_bgm_mode("boss")
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
    boss_floor_y = boss_spawn.z - 2.0
    loop_half = max(10.0, float(getattr(game, "platform_loop_range", 16.0)) * 0.85 * 0.62)
    game.boss_room_prev_hyper_bounds_top_z = float(getattr(game, "hyper_bounds_top_z", boss_floor_y + loop_half))
    game.boss_room_prev_hyper_bounds_bottom_z = float(getattr(game, "hyper_bounds_bottom_z", boss_floor_y - loop_half))
    game.hyper_bounds_bottom_z = boss_floor_y - loop_half
    game.hyper_bounds_top_z = boss_floor_y + loop_half
    _build_hex_boss_arena(game, boss_room_idx, boss_spawn)
    hub_pos = Vec3(getattr(game, "boss_room_hub_pos", boss_spawn + Vec3(0.0, 0.0, 2.6)))
    hub_half = Vec3(getattr(game, "boss_room_hub_half", Vec3(6.0, 6.0, 0.7)))
    hub_top_z = hub_pos.z + hub_half.z + 1.6
    boss_offset = max(1.2, hub_half.y * 0.6)
    boss_spawn_pos = Vec3(hub_pos.x, hub_pos.y + boss_offset, hub_top_z)
    player_spawn_pos = Vec3(hub_pos.x, hub_pos.y - boss_offset, hub_top_z)

    wave_idx = max(1, int(getattr(game, "monster_wave_index", 1)))
    boss_scale = min(1.2, 0.52 + 0.12 * max(0, wave_idx - 1))
    boss_scale = max(0.4, boss_scale)
    scale_alpha = (boss_scale - 0.55) / max(1e-6, (1.3 - 0.55))
    burst_min = 0.4 - 0.18 * scale_alpha
    burst_max = 0.9 - 0.4 * scale_alpha
    ranged_min = 0.85 - 0.45 * scale_alpha
    ranged_max = 1.5 - 0.85 * scale_alpha

    remote_players = getattr(game, "remote_players", None)
    if isinstance(remote_players, dict) and remote_players:
        ring_radius = max(1.6, hub_half.y * 0.9)
        step = math.tau / max(1, len(remote_players))
        for idx, entry in enumerate(remote_players.values()):
            if not isinstance(entry, dict):
                continue
            angle = step * idx
            entry["boss_warp_pos"] = Vec3(
                hub_pos.x + math.cos(angle) * ring_radius,
                hub_pos.y + math.sin(angle) * ring_radius,
                hub_top_z,
            )

    for monster in game.monsters:
        root = monster.get("root")
        if root is not None and not root.isEmpty():
            root.setPos(boss_spawn_pos)
            monster["prev_pos"] = Vec3(root.getPos())
            monster["base_z"] = float(boss_spawn_pos.z)
            monster["jump_vel"] = 0.0
        monster["room_idx"] = boss_room_idx
        monster["level"] = min(int(getattr(game, "monster_level_cap", 150)), max(1, int(getattr(game, "monster_level_current", 1)) + 7))
        monster["boss_orbit_phase"] = random.uniform(0.0, math.tau)
        monster["boss_burst_timer"] = random.uniform(max(0.12, burst_min), max(0.24, burst_max))
        game._apply_monster_progression_stats(monster, boss_mode=True)
        monster["hp_max"] = float(monster.get("hp_max", 1.0)) * 2.7 * boss_scale
        monster["hp"] = float(monster.get("hp_max", 1.0))
        monster["defense"] = max(0.35, float(monster.get("defense", 1.0)) * 1.25 * boss_scale)
        monster["attack_mult"] = float(monster.get("attack_mult", 1.0)) * 2.8 * boss_scale
        monster["speed_boost"] = max(1.0, float(monster.get("speed_boost", 1.0)) * (1.2 + 0.32 * boss_scale))
        monster["ai_hunt_range"] = float(monster.get("ai_hunt_range", 11.5)) * (1.15 + 0.5 * boss_scale)
        monster["ai_attack_range"] = float(monster.get("ai_attack_range", 2.0)) * (1.05 + 0.45 * boss_scale)
        monster["ai_guard_range"] = float(monster.get("ai_guard_range", 17.0)) * (1.05 + 0.45 * boss_scale)
        monster["is_boss"] = True
        monster["ranged_enabled"] = True
        monster["ranged_cooldown"] = random.uniform(max(0.12, ranged_min), max(0.24, ranged_max))
        vel = Vec3(monster.get("velocity", Vec3(0.0, 0.0, 0.0)))
        if vel.lengthSquared() > 1e-8:
            monster["velocity"] = vel * (1.4 + 0.6 * boss_scale)

    if hasattr(game, "ball_np") and game.ball_np is not None and (not game.ball_np.isEmpty()):
        game.ball_np.setPos(player_spawn_pos)
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
        if hasattr(game, "_set_bgm_mode"):
            game._set_bgm_mode("normal")
        _clear_boss_room_arena(game)
        _restore_world_after_boss(game)
        if hasattr(game, "ball_np") and game.ball_np is not None and not game.ball_np.isEmpty():
            return_pos = getattr(game, "boss_return_pos", None)
            if return_pos is not None:
                game.ball_np.setPos(Vec3(return_pos))
                if hasattr(game, "ball_body") and game.ball_body is not None:
                    game.ball_body.setLinearVelocity(Vec3(0, 0, 0))
                    game.ball_body.setAngularVelocity(Vec3(0, 0, 0))
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
        orbit_push = side * (2.4 + 1.1 * math.sin(game.roll_time * 2.6 + boss_phase))
        pos += orbit_push * dt * 0.6
    burst = max(0.0, float(monster.get("boss_burst_timer", 0.0)) - dt)
    monster["boss_burst_timer"] = burst
    if burst <= 0.0 and to_player.lengthSquared() > 1e-8:
        dash = to_player * random.uniform(2.0, 3.6)
        pos += dash
        monster["boss_burst_timer"] = random.uniform(0.35, 0.8)
        monster["w_vel"] += random.uniform(-2.2, 2.2)
    return pos