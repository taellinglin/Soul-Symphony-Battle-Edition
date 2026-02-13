from panda3d.bullet import BulletRigidBodyNode, BulletSphereShape
from panda3d.core import BitMask32
from panda3d.core import Material, NodePath


def spawn_player_ball(game) -> None:
    start_room_idx = int(getattr(game, "start_room_idx", 0))
    if game.rooms:
        start_room_idx = max(0, min(len(game.rooms) - 1, start_room_idx))
        sx, sy = game.rooms[start_room_idx].center
    else:
        sx, sy = game.map_w * 0.5, game.map_d * 0.5

    game.ball_radius = 0.68
    game.ball_body = BulletRigidBodyNode("player-ball")
    game.ball_body.setMass(1.25)
    game.ball_body.setFriction(game.ball_friction_default)
    game.ball_body.setRestitution(game.normal_restitution)
    game.ball_body.setAngularDamping(0.72)
    game.ball_body.setLinearDamping(0.28)
    game.ball_body.setDeactivationEnabled(False)
    game.ball_body.addShape(BulletSphereShape(game.ball_radius))
    if hasattr(game.ball_body, "setCcdMotionThreshold"):
        game.ball_body.setCcdMotionThreshold(max(0.001, game.ball_radius * 0.16))
    if hasattr(game.ball_body, "setCcdSweptSphereRadius"):
        game.ball_body.setCcdSweptSphereRadius(game.ball_radius * 0.9)
    if hasattr(game.ball_body, "setContactProcessingThreshold"):
        game.ball_body.setContactProcessingThreshold(max(0.001, game.ball_radius * 0.04))

    game.ball_np = game.render.attachNewNode(game.ball_body)
    room_level = 0
    if game.rooms:
        room_level = int(getattr(game, "room_levels", {}).get(start_room_idx, 0))
    level_z_step = float(getattr(game, "level_z_step", 3.0))
    floor_z = float(getattr(game, "floor_y", 0.0)) + room_level * level_z_step
    floor_clearance = max(0.02, float(getattr(game, "floor_t", 0.02)) * 0.5)
    start_z = floor_z + game.ball_radius + floor_clearance + 0.01
    game.ball_np.setPos(sx, sy, start_z)
    if hasattr(game.ball_body, "setIntoCollideMask"):
        game.ball_body.setIntoCollideMask(getattr(game, "group_player", 0))
    if hasattr(game.ball_body, "setFromCollideMask"):
        game.ball_body.setFromCollideMask(getattr(game, "mask_player_hits", 0))
    game.ball_np.setCollideMask(BitMask32.allOn())
    game.physics_world.attachRigidBody(game.ball_body)
    game.physics_nodes.append(game.ball_body)

    ball_vis = game.sphere_model.copyTo(game.ball_np)
    ball_vis.setScale(game.ball_radius)
    ball_vis.setColor(1, 1, 1, 1)
    ball_vis.clearTexture()
    ball_tex = game._load_ball_texture()
    uv_scale, uv_u, uv_v = game._get_ball_uv_params(ball_tex)
    ball_vis.setTexture(game.ball_tex_stage, ball_tex)
    game._apply_ball_cube_projection(ball_vis, uv_scale=uv_scale, uv_offset_u=uv_u, uv_offset_v=uv_v)
    game.ball_tex_base_u = float(uv_u) % 1.0
    game.ball_tex_base_v = float(uv_v) % 1.0
    game.ball_tex_scroll_u = 0.0
    game.ball_tex_scroll_v = 0.0

    game.ball_emissive_material = Material()
    game.ball_emissive_material.setEmission((0.26, 0.33, 0.46, 1.0))
    game.ball_emissive_material.setAmbient((0.62, 0.68, 0.76, 1.0))
    game.ball_emissive_material.setDiffuse((1.0, 1.0, 1.0, 1.0))
    ball_vis.setMaterial(game.ball_emissive_material, 1)
    game._register_color_cycle(ball_vis, (1.0, 1.0, 1.0, 1.0), min_speed=0.07, max_speed=0.16)
    game.ball_visual = ball_vis

    game.ball_caps = []
