import colorsys
import math
import random

from panda3d.core import Material, PointLight, TextureStage, TransparencyAttrib, Vec3


def setup_weapon_system(game) -> None:
    game.sword_pivot = game.world.attachNewNode("sword-pivot")
    sword_scale = max(1.0, float(getattr(game, "ball_radius", 0.4)) / 0.4)
    game.sword_up_offset = max(game.sword_up_offset, game.ball_radius + 0.16)
    game.sword_forward_offset = max(game.sword_forward_offset, 0.34 * sword_scale)
    game.sword_side_offset = max(game.sword_side_offset, 0.22 * min(1.25, sword_scale))
    game.sword_reach = max(game.sword_reach, 1.8 * sword_scale)
    emissive = Material()
    emissive.setEmission((0.45, 1.0, 1.0, 1.0))
    emissive.setDiffuse((0.05, 0.08, 0.12, 1.0))
    emissive.setAmbient((0.03, 0.05, 0.08, 1.0))
    game.sword_emissive_material = emissive

    guard = game.box_model.copyTo(game.sword_pivot)
    guard.setPos(game.box_norm_offset)
    guard.setScale(game.box_norm_scale)
    guard.setPos(0, 0.2 * sword_scale, 0)
    guard.setScale(0.26 * sword_scale, 0.05 * sword_scale, 0.04 * sword_scale)
    guard.setColor(0.76, 0.84, 0.95, 1)
    guard.clearTexture()
    guard.setTexture(game.level_checker_tex, 1)

    grip = game.box_model.copyTo(game.sword_pivot)
    grip.setPos(game.box_norm_offset)
    grip.setScale(game.box_norm_scale)
    grip.setPos(0, 0.03 * sword_scale, 0.0)
    grip.setScale(0.05 * sword_scale, 0.14 * sword_scale, 0.05 * sword_scale)
    grip.setColor(0.18, 0.22, 0.32, 1)
    grip.clearTexture()
    grip.setTexture(game.level_checker_tex, 1)

    blade = game.box_model.copyTo(game.sword_pivot)
    blade.setPos(game.box_norm_offset)
    blade.setScale(game.box_norm_scale)
    blade.setPos(0, 0.74 * sword_scale, 0)
    blade.setScale(0.082 * sword_scale, 0.74 * sword_scale, 0.052 * sword_scale)
    blade.setColor(0.82, 0.94, 1.0, 1)
    blade.clearTexture()
    blade.setTexture(game.level_checker_tex, 1)
    blade.setMaterial(game.sword_emissive_material, 1)
    game.sword_blade_np = blade

    tip = game.box_model.copyTo(game.sword_pivot)
    tip.setPos(game.box_norm_offset)
    tip.setScale(game.box_norm_scale)
    tip.setPos(0, 1.37 * sword_scale, 0)
    tip.setScale(0.052 * sword_scale, 0.12 * sword_scale, 0.032 * sword_scale)
    tip.setColor(0.9, 0.98, 1.0, 1)
    tip.clearTexture()
    tip.setTexture(game.level_checker_tex, 1)
    tip.setMaterial(game.sword_emissive_material, 1)
    game.sword_tip_blade_np = tip

    glow = game.box_model.copyTo(game.sword_pivot)
    glow.setPos(game.box_norm_offset)
    glow.setScale(game.box_norm_scale)
    glow.setPos(0, 0.84 * sword_scale, 0)
    glow.setScale(0.062 * sword_scale, 0.78 * sword_scale, 0.034 * sword_scale)
    glow.setColor(0.18, 0.95, 1.0, 0.94)
    glow.setTransparency(TransparencyAttrib.MAlpha)
    glow.setBin("transparent", 30)
    glow.setDepthWrite(False)
    glow.setLightOff(1)
    glow.clearTexture()
    glow.setMaterial(game.sword_emissive_material, 1)
    game.sword_glow_np = glow

    game.sword_stripe_stage = TextureStage("sword-stripe-stage")
    game.sword_stripe_stage.setMode(TextureStage.MAdd)
    game.sword_stripe_tex = game._create_sword_stripe_texture(size=128, stripes=14)
    game.sword_stripe_nodes = []
    for sx in (-1.0, 1.0):
        stripe = game.box_model.copyTo(game.sword_pivot)
        stripe.setPos(game.box_norm_offset)
        stripe.setScale(game.box_norm_scale)
        stripe.setPos(0.078 * sx * sword_scale, 0.8 * sword_scale, 0.0)
        stripe.setScale(0.01 * sword_scale, 0.72 * sword_scale, 0.039 * sword_scale)
        stripe.setColor(0.6, 1.0, 1.0, 0.92)
        stripe.clearTexture()
        stripe.setTexture(game.sword_stripe_stage, game.sword_stripe_tex)
        stripe.setTexScale(game.sword_stripe_stage, 1.0, 6.6)
        stripe.setTransparency(TransparencyAttrib.MAlpha)
        stripe.setBin("transparent", 32)
        stripe.setDepthWrite(False)
        stripe.setLightOff(1)
        stripe.setMaterial(game.sword_emissive_material, 1)
        game.sword_stripe_nodes.append(stripe)

    game.sword_glow_light = PointLight("sword-glow-light")
    game.sword_glow_light.setColor((0.2, 0.95, 1.0, 1.0))
    game.sword_glow_light.setAttenuation((1.0, 0.18, 0.05))
    game.sword_glow_light_np = game.sword_pivot.attachNewNode(game.sword_glow_light)
    game.sword_glow_light_np.setPos(0, 0.84 * sword_scale, 0.04 * sword_scale)
    game.render.setLight(game.sword_glow_light_np)

    game.sword_tip = game.sword_pivot.attachNewNode("sword-tip")
    game.sword_tip.setPos(0, 1.5 * sword_scale, 0)
    game.sword_slash_nodes = []
    game.sword_slash_emit_timer = 0.0
    game.sword_slash_emit_interval = 1.0 / 85.0
    game.sword_prev_tip_pos = None
    game.sword_blade_echo_nodes = []
    game.sword_blade_echo_emit_timer = 0.0
    game.sword_blade_echo_emit_interval = 1.0 / 120.0
    game.sword_blade_echo_max_frames = 16
    game.sword_blade_echo_life = 0.4
    game.sword_blade_echo_cycle = 0
    game.sword_blade_echo_colors = [
        (1.0, 0.0, 0.0),
        (1.0, 0.5, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.29, 0.0, 0.51),
        (0.56, 0.0, 1.0),
    ]
    game.sword_blade_echo_scale = Vec3(0.095 * sword_scale, 0.74 * sword_scale, 0.072 * sword_scale)
    game.sword_throw_distance = max(game.sword_reach * 2.6, 5.0)
    game.sword_throw_outbound_time = 0.22
    game.sword_throw_total_time = 0.56
    game.sword_throw_spin_speed = 1080.0
    game.sword_throw_hit_targets = set()
    game.sword_throw_origin = None
    game.sword_throw_dir = Vec3(0, 1, 0)


def _update_blade_echoes(game, dt: float) -> None:
    keep = []
    for entry in game.sword_blade_echo_nodes:
        holder = entry.get("node")
        blade = entry.get("blade")
        if holder is None or holder.isEmpty() or blade is None or blade.isEmpty():
            continue

        age = float(entry.get("age", 0.0)) + dt
        life = max(1e-4, float(entry.get("life", 0.35)))
        if age >= life:
            holder.removeNode()
            continue

        t = max(0.0, min(1.0, age / life))
        fade = max(0.0, 1.0 - t)
        color = entry.get("color", (1.0, 1.0, 1.0))
        white_mix = max(0.0, 1.0 - t * 1.8)
        r = color[0] * (1.0 - white_mix) + 1.0 * white_mix
        g = color[1] * (1.0 - white_mix) + 1.0 * white_mix
        b = color[2] * (1.0 - white_mix) + 1.0 * white_mix
        blade.setColorScale(r, g, b, fade * 0.86)

        entry["age"] = age
        keep.append(entry)

    game.sword_blade_echo_nodes = keep


def _spawn_blade_echo(game, sword_scale: float) -> None:
    if not hasattr(game, "sword_blade_np"):
        return
    blade_src = game.sword_blade_np
    if blade_src is None or blade_src.isEmpty():
        return

    holder = game.world.attachNewNode("sword-blade-echo")
    holder.setMat(game.render, blade_src.getMat(game.render))

    blade = game.box_model.copyTo(holder)
    blade.setPos(game.box_norm_offset)
    blade.setScale(game.box_norm_scale)
    stretch_scale = getattr(game, "sword_blade_echo_scale", Vec3(0.095 * sword_scale, 0.74 * sword_scale, 0.072 * sword_scale))
    blade.setScale(stretch_scale)
    blade.clearTexture()
    blade.setMaterial(game.sword_emissive_material, 1)
    blade.setTransparency(TransparencyAttrib.MAlpha)
    blade.setDepthWrite(False)
    blade.setBin("transparent", 34)

    colors = getattr(game, "sword_blade_echo_colors", None) or [(1.0, 1.0, 1.0)]
    idx = int(getattr(game, "sword_blade_echo_cycle", 0)) % len(colors)
    color = colors[idx]
    game.sword_blade_echo_cycle = idx + 1

    game.sword_blade_echo_nodes.append(
        {
            "node": holder,
            "blade": blade,
            "age": 0.0,
            "life": float(getattr(game, "sword_blade_echo_life", 0.4)),
            "color": color,
        }
    )

    max_frames = max(1, int(getattr(game, "sword_blade_echo_max_frames", 5)))
    while len(game.sword_blade_echo_nodes) > max_frames:
        oldest = game.sword_blade_echo_nodes.pop(0)
        old_node = oldest.get("node")
        if old_node is not None and not old_node.isEmpty():
            old_node.removeNode()


def _update_slash_trails(game, dt: float, sword_scale: float) -> None:
    keep = []
    for entry in game.sword_slash_nodes:
        node = entry.get("node")
        if node is None or node.isEmpty():
            continue
        age = float(entry.get("age", 0.0)) + dt
        life = max(1e-4, float(entry.get("life", 0.16)))
        if age >= life:
            node.removeNode()
            continue
        alpha = max(0.0, 1.0 - age / life)
        node.setColorScale(1.0, 1.0, 1.0, alpha * 0.72)
        entry["age"] = age
        keep.append(entry)
    game.sword_slash_nodes = keep


def _spawn_swing_slash_trail(game, sword_scale: float, current_tip_pos: Vec3) -> None:
    prev_tip = game.sword_prev_tip_pos
    game.sword_prev_tip_pos = Vec3(current_tip_pos)
    if prev_tip is None:
        return

    seg = Vec3(current_tip_pos - prev_tip)
    seg_len = seg.length()
    if seg_len < 0.06:
        return

    mid = (Vec3(current_tip_pos) + Vec3(prev_tip)) * 0.5
    holder = game.world.attachNewNode("sword-slash-trail")
    holder.setPos(mid)
    planar_len = math.sqrt(seg.x * seg.x + seg.y * seg.y)
    heading = math.degrees(math.atan2(-seg.x, seg.y)) if planar_len > 1e-8 else 0.0
    pitch = math.degrees(math.atan2(seg.z, max(1e-8, planar_len)))
    holder.setHpr(heading, pitch, 0.0)

    trail = game.box_model.copyTo(holder)
    trail.setPos(game.box_norm_offset)
    trail.setScale(game.box_norm_scale)
    trail.setScale(seg_len * 0.5, 0.04 * sword_scale, 0.012 * sword_scale)
    trail.clearTexture()
    trail.setLightOff(1)
    trail.setTransparency(TransparencyAttrib.MAlpha)
    trail.setDepthWrite(False)
    trail.setBin("transparent", 31)
    trail.setColor(0.24, 0.95, 1.0, 0.68)

    game.sword_slash_nodes.append({"node": holder, "age": 0.0, "life": 0.14})


def trigger_swing_attack(game) -> None:
    if game.attack_cooldown > 0.0 or game.attack_mode != "idle":
        return
    game.attack_mode = "swing"
    game.attack_timer = 0.0
    game.attack_hit_targets.clear()
    game._play_sound(game.sfx_attack, volume=0.72, play_rate=1.08)


def trigger_spin_attack(game) -> None:
    if game.attack_cooldown > 0.0 or game.attack_mode != "idle":
        return
    game.attack_mode = "spin"
    game.attack_timer = 0.0
    game.attack_hit_targets.clear()
    game._play_sound(game.sfx_attack_spin, volume=0.78, play_rate=1.0)


def trigger_throw_attack(game) -> None:
    if game.attack_cooldown > 0.0 or game.attack_mode != "idle":
        return
    game.attack_mode = "throw"
    game.attack_timer = 0.0
    game.attack_hit_targets.clear()
    game.sword_throw_hit_targets.clear()
    game.sword_throw_origin = None
    throw_dir = Vec3(game.weapon_forward.x, game.weapon_forward.y, 0.0)
    if throw_dir.lengthSquared() < 1e-6:
        yaw = math.radians(game.heading)
        throw_dir = Vec3(-math.sin(yaw), math.cos(yaw), 0.0)
    if throw_dir.lengthSquared() > 1e-6:
        throw_dir.normalize()
    game.sword_throw_dir = throw_dir
    game._play_sound(game.sfx_attack_spin if getattr(game, "sfx_attack_spin", None) is not None else game.sfx_attack, volume=0.82, play_rate=1.12)


def apply_attack_hits(game, is_spin: bool) -> None:
    if not game.monsters:
        return

    ball_pos = game.ball_np.getPos()
    player_w = float(getattr(game, "player_w", 0.0))
    swing_forward = game.sword_pivot.getQuat(game.render).getForward()
    swing_forward = Vec3(swing_forward.x, swing_forward.y, 0)
    if swing_forward.lengthSquared() > 1e-6:
        swing_forward.normalize()

    for monster in game.monsters:
        if monster.get("dead", False):
            continue

        root = monster["root"]
        if root is None or root.isEmpty():
            continue

        monster_id = id(root)
        if monster_id in game.attack_hit_targets:
            continue

        to_monster = root.getPos() - ball_pos
        monster_w = float(monster.get("w", 0.0))
        planar = Vec3(to_monster.x, to_monster.y, 0)
        planar_dist = planar.length()
        w_scale = max(0.1, float(getattr(game, "w_dimension_distance_scale", 4.0)))
        dw_scaled = (monster_w - player_w) * w_scale
        dist4d = math.sqrt(max(0.0, planar_dist * planar_dist + dw_scaled * dw_scaled))
        reach_mult = max(0.6, float(getattr(game, "sword_reach_multiplier", 1.0)))
        max_reach = game.sword_reach * reach_mult + monster["radius"] * 0.65
        if dist4d > max_reach:
            continue

        if not is_spin:
            if planar_dist < 1e-6 or swing_forward.lengthSquared() < 1e-6:
                continue
            planar.normalize()
            if planar.dot(swing_forward) < 0.12:
                continue

        away = root.getPos() - ball_pos
        away.z = 0.0
        game.attack_hit_targets.add(monster_id)
        dmg_mult = max(0.5, float(getattr(game, "sword_damage_multiplier", 1.0)))
        dmg_mult *= max(0.55, float(getattr(game, "combat_damage_multiplier", 1.0)))
        damage = (54.0 if is_spin else 36.0) * dmg_mult
        game._damage_monster(monster, damage)

        if away.lengthSquared() > 1e-6:
            away.normalize()
            knock = 5.4 if is_spin else 3.8
            monster["velocity"] = Vec3(monster["velocity"]) + away * knock + Vec3(0, 0, 0.35)


def _apply_throw_hits(game, sword_pos: Vec3, radius: float) -> None:
    if not game.monsters:
        return

    player_w = float(getattr(game, "player_w", 0.0))
    for monster in game.monsters:
        if monster.get("dead", False):
            continue
        root = monster.get("root")
        if root is None or root.isEmpty():
            continue

        monster_id = id(root)
        if monster_id in game.sword_throw_hit_targets:
            continue

        monster_pos = root.getPos()
        planar = Vec3(monster_pos.x - sword_pos.x, monster_pos.y - sword_pos.y, 0.0)
        planar_dist = planar.length()
        if planar_dist > (radius + float(monster.get("radius", 1.0)) * 0.72):
            continue

        monster_w = float(monster.get("w", 0.0))
        w_scale = max(0.1, float(getattr(game, "w_dimension_distance_scale", 4.0)))
        dw_scaled = (monster_w - player_w) * w_scale
        dist4d = math.sqrt(max(0.0, planar_dist * planar_dist + dw_scaled * dw_scaled))
        max_hit = radius + float(monster.get("radius", 1.0))
        if dist4d > max_hit:
            continue

        game.sword_throw_hit_targets.add(monster_id)
        dmg_mult = max(0.5, float(getattr(game, "sword_damage_multiplier", 1.0)))
        dmg_mult *= max(0.55, float(getattr(game, "combat_damage_multiplier", 1.0)))
        damage = 44.0 * dmg_mult
        game._damage_monster(monster, damage)

        away = Vec3(monster_pos.x - sword_pos.x, monster_pos.y - sword_pos.y, 0.0)
        if away.lengthSquared() > 1e-8:
            away.normalize()
            monster["velocity"] = Vec3(monster["velocity"]) + away * 4.8 + Vec3(0.0, 0.0, 0.42)


def update_weapon_system(game, dt: float) -> None:
    if not hasattr(game, "sword_pivot"):
        return
    sword_scale = max(1.0, float(getattr(game, "ball_radius", 0.4)) / 0.4)
    _update_slash_trails(game, dt, sword_scale)
    _update_blade_echoes(game, dt)

    desired_forward = Vec3(game.last_move_dir.x, game.last_move_dir.y, 0)
    if desired_forward.lengthSquared() < 1e-6:
        yaw = math.radians(game.heading)
        desired_forward = Vec3(-math.sin(yaw), math.cos(yaw), 0)
    if desired_forward.lengthSquared() > 1e-6:
        desired_forward.normalize()

    forward_blend = min(1.0, dt * 10.5)
    game.weapon_forward = game.weapon_forward + (desired_forward - game.weapon_forward) * forward_blend
    if game.weapon_forward.lengthSquared() > 1e-6:
        game.weapon_forward.normalize()

    ball_pos = game.ball_np.getPos()
    gravity_up = game._get_gravity_up()
    right = gravity_up.cross(game.weapon_forward)
    if right.lengthSquared() < 1e-8:
        yaw = math.radians(game.heading)
        right = Vec3(math.cos(yaw), math.sin(yaw), 0)
    if right.lengthSquared() > 1e-8:
        right.normalize()
    desired_anchor = (
        ball_pos
        + gravity_up * game.sword_up_offset
        + game.weapon_forward * game.sword_forward_offset
        + right * game.sword_side_offset
    )

    if game.sword_anchor_pos is None:
        game.sword_anchor_pos = Vec3(desired_anchor)
        game.sword_anchor_vel = Vec3(0, 0, 0)
    else:
        follow_alpha = 1.0 - math.exp(-dt * game.sword_anchor_follow_speed)
        game.sword_anchor_pos = game.sword_anchor_pos + (desired_anchor - game.sword_anchor_pos) * follow_alpha
        game.sword_anchor_vel = Vec3(0, 0, 0)
    game.sword_anchor_pos.z = max(game.floor_y + 0.24, game.sword_anchor_pos.z)

    heading = math.degrees(math.atan2(-game.weapon_forward.x, game.weapon_forward.y))
    yaw_offset = -16.0
    pitch = -18.0
    roll = 0.0
    pivot_pos = Vec3(game.sword_anchor_pos)

    if game.attack_mode == "swing":
        game.attack_timer += dt
        t = min(1.0, game.attack_timer / game.swing_duration)
        yaw_offset = -96.0 + 192.0 * t
        pitch = -20.0 + 8.0 * math.sin(t * math.pi)
        roll = 25.0 * math.sin(t * math.pi)
        if 0.22 <= t <= 0.82:
            apply_attack_hits(game, is_spin=False)
            game.sword_slash_emit_timer -= dt
            if game.sword_slash_emit_timer <= 0.0:
                _spawn_swing_slash_trail(game, sword_scale, game.sword_tip.getPos(game.render))
                game.sword_slash_emit_timer = game.sword_slash_emit_interval
        if t >= 1.0:
            game.attack_mode = "idle"
            game.attack_timer = 0.0
            cd_mult = max(0.35, float(getattr(game, "attack_cooldown_multiplier", 1.0)))
            game.attack_cooldown = 0.05 * cd_mult
            game.sword_prev_tip_pos = None

    elif game.attack_mode == "spin":
        game.attack_timer += dt
        t = min(1.0, game.attack_timer / game.spin_duration)
        yaw_offset = -180.0 + 540.0 * t
        pitch = -14.0
        roll = 18.0 * math.sin(t * math.tau)
        apply_attack_hits(game, is_spin=True)
        if t >= 1.0:
            game.attack_mode = "idle"
            game.attack_timer = 0.0
            cd_mult = max(0.35, float(getattr(game, "attack_cooldown_multiplier", 1.0)))
            game.attack_cooldown = 0.12 * cd_mult
            game.sword_prev_tip_pos = None
    elif game.attack_mode == "throw":
        game.attack_timer += dt
        total = max(0.2, float(getattr(game, "sword_throw_total_time", 0.56)))
        outbound = max(0.06, min(total * 0.9, float(getattr(game, "sword_throw_outbound_time", 0.22))))
        distance = max(1.0, float(getattr(game, "sword_throw_distance", game.sword_reach * 2.6)))

        if game.sword_throw_origin is None:
            game.sword_throw_origin = Vec3(game.sword_anchor_pos)

        origin = Vec3(game.sword_throw_origin)
        throw_dir = Vec3(getattr(game, "sword_throw_dir", game.weapon_forward))
        if throw_dir.lengthSquared() < 1e-8:
            throw_dir = Vec3(game.weapon_forward)
        if throw_dir.lengthSquared() > 1e-8:
            throw_dir.normalize()

        t = min(1.0, game.attack_timer / total)
        if t <= (outbound / total):
            out_t = t / max(1e-6, outbound / total)
            forward_amount = 1.0 - (1.0 - out_t) * (1.0 - out_t)
            vel_sign = 1.0
        else:
            back_t = (t - (outbound / total)) / max(1e-6, 1.0 - (outbound / total))
            forward_amount = (1.0 - back_t) * (1.0 - back_t)
            vel_sign = -1.0

        arc = math.sin(t * math.pi) * distance * 0.16
        right_throw = gravity_up.cross(throw_dir)
        if right_throw.lengthSquared() > 1e-8:
            right_throw.normalize()
        throw_pos = origin + throw_dir * (distance * forward_amount) + right_throw * arc
        throw_pos += gravity_up * (0.08 + 0.12 * math.sin(t * math.pi))

        hit_radius = 0.95 + sword_scale * 0.22
        _apply_throw_hits(game, throw_pos, hit_radius)

        forward_heading = math.degrees(math.atan2(-throw_dir.x, throw_dir.y))
        heading = forward_heading if vel_sign >= 0.0 else (forward_heading + 180.0)
        yaw_offset = 0.0
        pitch = -8.0 + 5.0 * math.sin(t * math.pi)
        roll = (game.attack_timer * float(getattr(game, "sword_throw_spin_speed", 1080.0))) % 360.0
        pivot_pos = throw_pos

        if t >= 1.0:
            game.attack_mode = "idle"
            game.attack_timer = 0.0
            game.sword_throw_origin = None
            game.sword_throw_hit_targets.clear()
            cd_mult = max(0.35, float(getattr(game, "attack_cooldown_multiplier", 1.0)))
            game.attack_cooldown = 0.14 * cd_mult
            game.sword_prev_tip_pos = None
    else:
        game.sword_prev_tip_pos = None

    game.sword_pivot.setPos(pivot_pos)
    game.sword_pivot.setHpr(heading + yaw_offset, pitch, roll)

    should_emit_blade_echo = game.attack_mode in ("swing", "spin", "throw")
    if should_emit_blade_echo:
        game.sword_blade_echo_emit_timer -= dt
        if game.sword_blade_echo_emit_timer <= 0.0:
            _spawn_blade_echo(game, sword_scale)
            game.sword_blade_echo_emit_timer = float(getattr(game, "sword_blade_echo_emit_interval", 1.0 / 120.0))
    else:
        game.sword_blade_echo_emit_timer = 0.0

    if hasattr(game, "sword_glow_np") and game.sword_glow_np is not None:
        glow_boost = 1.0
        if game.attack_mode == "swing":
            glow_boost = 1.32
        elif game.attack_mode == "spin":
            glow_boost = 1.58
        elif game.attack_mode == "throw":
            glow_boost = 1.76
        hue = (game.roll_time * 1.9) % 1.0
        cr, cg, cb = colorsys.hsv_to_rgb(hue, 0.78, 1.0)
        pulse = 0.66 + 0.34 * (0.5 + 0.5 * math.sin(game.roll_time * 18.0))
        g = pulse * glow_boost
        game.sword_glow_np.setColor(cr * 0.2 * g, cg * 0.86 * g, cb * 1.02 * g, 0.66 + 0.32 * pulse)

        if hasattr(game, "sword_blade_np") and game.sword_blade_np is not None:
            game.sword_blade_np.setColor(0.24 + cr * 0.76, 0.24 + cg * 0.76, 0.24 + cb * 0.76, 1.0)
        if hasattr(game, "sword_tip_blade_np") and game.sword_tip_blade_np is not None:
            game.sword_tip_blade_np.setColor(0.4 + cr * 0.6, 0.4 + cg * 0.6, 0.4 + cb * 0.6, 1.0)

        if hasattr(game, "sword_stripe_nodes"):
            scroll = (game.roll_time * (3.1 + glow_boost * 0.9)) % 1.0
            for idx, stripe in enumerate(game.sword_stripe_nodes):
                if stripe is None or stripe.isEmpty():
                    continue
                stripe.setTexOffset(game.sword_stripe_stage, 0.0, (scroll + idx * 0.5) % 1.0)
                stripe.setColor(0.38 + cr * 0.62, 0.38 + cg * 0.62, 0.38 + cb * 0.62, 0.9)

        if hasattr(game, "sword_glow_light") and game.sword_glow_light is not None:
            game.sword_glow_light.setColor((0.22 * cr * g, 0.82 * cg * g, min(1.0, 1.08 * cb * g), 1.0))

    tip_pos = game.sword_tip.getPos(game.render)
    min_tip_z = game.floor_y + 0.08
    if tip_pos.z < min_tip_z:
        game.sword_anchor_pos.z += (min_tip_z - tip_pos.z)
        game.sword_pivot.setPos(game.sword_anchor_pos)
