import math
from panda3d.core import Fog, Vec3


def setup_camera(app) -> None:
    if not hasattr(app, "ball_np"):
        app.camera.setPos(10, 10, 5)
        app.camera.lookAt(20, 20, 2)
        return
    app.camLens.setFov(120)
    app.camLens.setNearFar(0.012, 1500)
    if not hasattr(app, "fog") or app.fog is None:
        app.fog = Fog("scene-fog")
    app.fog.setColor(0.1, 0.12, 0.17)
    app.fog.setLinearRange(0.8, 18.0)
    app.fog_start = 0.8
    app.fog_end = 18.0
    app.render.setFog(app.fog)
    target = app.ball_np.getPos() + Vec3(0, 0, app.camera_target_height)
    app.camera.setPos(target + Vec3(-8, -8, 5))
    app.camera_smoothed_pos = Vec3(app.camera.getPos())
    app.camera.lookAt(target)


def rotate_around_axis(vec: Vec3, axis: Vec3, angle_rad: float) -> Vec3:
    axis_n = Vec3(axis)
    axis_n.normalize()
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return vec * cos_a + axis_n.cross(vec) * sin_a + axis_n * (axis_n.dot(vec) * (1 - cos_a))


def camera_orbit_position(target: Vec3, heading_deg: float, pitch_deg: float, dist: float) -> Vec3:
    up_axis = Vec3(0, 0, 1)
    base_forward = Vec3(0, 1, 0)
    base_forward = base_forward - up_axis * base_forward.dot(up_axis)
    if base_forward.lengthSquared() < 1e-6:
        base_forward = Vec3(1, 0, 0) - up_axis * Vec3(1, 0, 0).dot(up_axis)
    base_forward.normalize()

    heading_rad = math.radians(heading_deg)
    pitch_rad = math.radians(max(0.0, min(89.0, abs(pitch_deg))))

    yaw_forward = rotate_around_axis(base_forward, up_axis, heading_rad)
    yaw_forward.normalize()
    back_dir = -yaw_forward
    offset = back_dir * (math.cos(pitch_rad) * dist) + up_axis * (math.sin(pitch_rad) * dist)

    min_up_component = dist * 0.18
    up_component = offset.dot(up_axis)
    if up_component < min_up_component:
        offset += up_axis * (min_up_component - up_component)

    return target + offset


def resolve_camera_collision(app, target: Vec3, desired: Vec3) -> Vec3:
    direction = desired - target
    distance = direction.length()
    if distance < 1e-4:
        return desired

    direction /= distance
    up_ref = Vec3(0, 0, 1)
    right = direction.cross(up_ref)
    if right.lengthSquared() < 1e-6:
        right = Vec3(1, 0, 0)
    else:
        right.normalize()
    up = right.cross(direction)
    up.normalize()

    offsets = [
        Vec3(0, 0, 0),
        right * app.camera_collision_radius,
        -right * app.camera_collision_radius,
        up * app.camera_collision_radius,
        -up * app.camera_collision_radius,
        (right + up) * (app.camera_collision_radius * 0.8),
        (right - up) * (app.camera_collision_radius * 0.8),
        (-right + up) * (app.camera_collision_radius * 0.8),
        (-right - up) * (app.camera_collision_radius * 0.8),
    ]

    allowed = distance
    center_hit_normal = None
    for off in offsets:
        ray_from = target + off
        ray_to = desired + off
        hit = app.physics_world.rayTestClosest(ray_from, ray_to)
        if hit.hasHit():
            hit_pos = hit.getHitPos()
            hit_dist = (hit_pos - ray_from).length()
            allowed = min(allowed, max(app.camera_min_distance, hit_dist - 0.28))
            if off.lengthSquared() < 1e-8 and center_hit_normal is None:
                center_hit_normal = Vec3(hit.getHitNormal())
                if center_hit_normal.lengthSquared() > 1e-8:
                    center_hit_normal.normalize()

        reverse_hit = app.physics_world.rayTestClosest(ray_to, ray_from)
        if reverse_hit.hasHit():
            reverse_pos = reverse_hit.getHitPos()
            rev_dist = (reverse_pos - ray_from).length()
            allowed = min(allowed, max(app.camera_min_distance, rev_dist - 0.34))

    resolved = target + direction * allowed

    if center_hit_normal is not None and allowed < distance * 0.98:
        side = center_hit_normal.cross(up_ref)
        if side.lengthSquared() > 1e-6:
            side.normalize()
            if side.dot(direction) < 0:
                side = -side
            side_amount = min(0.22, max(0.0, distance - allowed) * 0.7)
            resolved = resolved + side * side_amount

    final_hit = app.physics_world.rayTestClosest(target, resolved)
    if final_hit.hasHit():
        hit_dist = (final_hit.getHitPos() - target).length()
        safe_len = max(app.camera_min_distance, hit_dist - 0.24)
        ray = resolved - target
        if ray.lengthSquared() > 1e-6:
            ray.normalize()
            resolved = target + ray * safe_len

    return resolved
