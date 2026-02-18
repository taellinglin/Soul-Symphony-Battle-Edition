use macroquad::prelude::*;

pub struct Player {
    position: Vec3,
    velocity: Vec3,
    angular_velocity: Vec3,
    mass: f32,
    max_speed: f32,
    radius: f32,
    gravity: f32,
    linear_damping: f32,
    angular_damping: f32,
    roll_force: f32,
    roll_torque: f32,
    link_control_gain: f32,
    link_brake_drag: f32,
    friction_default: f32,
    friction_shift: f32,
    water_buoyancy_bias: f32,
    water_buoyancy_strength: f32,
    water_drag_planar: f32,
    water_drag_vertical: f32,
    space_boost_impulse: f32,
    jump_impulse: f32,
    jump_rise_boost: f32,
    jump_float_duration: f32,
    jump_float_drag: f32,
    float_fall_drag: f32,
    jump_float_timer: f32,
    jump_queued: bool,
    max_jumps: i32,
    jumps_used: i32,
    last_move_dir: Vec3,
    roll_phase: f32,
    ball_texture: Option<Texture2D>,
}

impl Player {
    pub fn new(spawn: Vec3) -> Self {
        Self {
            position: spawn,
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 1.25,
            max_speed: 15.25,
            radius: 0.68,
            gravity: -9.81,
            linear_damping: 0.28,
            angular_damping: 0.72,
            roll_force: 18.0,
            roll_torque: 14.0,
            link_control_gain: 32.0,
            link_brake_drag: 2.6,
            friction_default: 0.02,
            friction_shift: 0.05,
            water_buoyancy_bias: 0.62,
            water_buoyancy_strength: 2.2,
            water_drag_planar: 0.85,
            water_drag_vertical: 1.95,
            space_boost_impulse: 2.4,
            jump_impulse: 24.0,
            jump_rise_boost: 1.22,
            jump_float_duration: 0.56,
            jump_float_drag: 5.8,
            float_fall_drag: 2.2,
            jump_float_timer: 0.0,
            jump_queued: false,
            max_jumps: 2,
            jumps_used: 0,
            last_move_dir: vec3(0.0, 1.0, 0.0),
            roll_phase: 0.0,
            ball_texture: None,
        }
    }

    pub fn position(&self) -> Vec3 {
        self.position
    }

    #[allow(dead_code)]
    pub fn radius(&self) -> f32 {
        self.radius
    }

    pub fn set_position(&mut self, position: Vec3) {
        self.position = position;
        self.velocity = Vec3::ZERO;
    }

    pub fn speed_ratio(&self) -> f32 {
        let planar = vec2(self.velocity.x, self.velocity.y);
        let speed = planar.length();
        if self.max_speed <= 0.0 {
            0.0
        } else {
            (speed / self.max_speed).clamp(0.0, 1.0)
        }
    }

    pub fn set_ball_texture(&mut self, texture: Option<Texture2D>) {
        self.ball_texture = texture;
    }

    pub fn update(
        &mut self,
        dt: f32,
        ground_z: f32,
        bounds: (f32, f32, f32, f32),
        camera_forward: Vec3,
        water_height: Option<f32>,
    ) -> bool {
        let mut jumped = false;
        let up = vec3(0.0, 0.0, 1.0);
        let mut move_input = Vec3::ZERO;
        if is_key_down(KeyCode::W) {
            move_input.y += 1.0;
        }
        if is_key_down(KeyCode::S) {
            move_input.y -= 1.0;
        }
        if is_key_down(KeyCode::A) {
            move_input.x += 1.0;
        }
        if is_key_down(KeyCode::D) {
            move_input.x -= 1.0;
        }

        if is_key_pressed(KeyCode::Space) {
            self.jump_queued = true;
        }

        let mut forward = camera_forward - up * camera_forward.dot(up);
        if forward.length_squared() < 1.0e-6 {
            forward = vec3(0.0, 1.0, 0.0);
        } else {
            forward = forward.normalize();
        }
        let mut right = up.cross(forward);
        if right.length_squared() < 1.0e-6 {
            right = vec3(1.0, 0.0, 0.0);
        } else {
            right = right.normalize();
        }

        let mut desired_move_dir: Option<Vec3> = None;
        if move_input.length_squared() > 0.0 {
            move_input = move_input.normalize();
            let mut move_dir = forward * move_input.y + right * move_input.x;
            if move_dir.length_squared() > 1.0e-6 {
                move_dir = move_dir.normalize();
                desired_move_dir = Some(move_dir);
                self.last_move_dir = move_dir;
            }
        }

        if let Some(move_dir) = desired_move_dir {
            let desired_velocity = move_dir * self.max_speed;
            let horizontal = vec3(self.velocity.x, self.velocity.y, 0.0);
            let steer = desired_velocity - horizontal;
            self.velocity += steer * self.link_control_gain * dt;
            self.velocity += move_dir * (self.roll_force * 0.56) * dt;
            let torque_axis = up.cross(move_dir);
            self.angular_velocity += torque_axis * self.roll_torque * dt;
        } else {
            let brake = (1.0 - dt * self.link_brake_drag).max(0.0);
            self.velocity.x *= brake;
            self.velocity.y *= brake;
        }

        let damping = (1.0 - dt * self.linear_damping).max(0.0);
        self.velocity *= damping;
        let ang_damping = (1.0 - dt * self.angular_damping).max(0.0);
        self.angular_velocity *= ang_damping;

        let horizontal = vec2(self.velocity.x, self.velocity.y);
        let speed = horizontal.length();
        if speed > self.max_speed {
            let capped = horizontal.normalize() * self.max_speed;
            self.velocity.x = capped.x;
            self.velocity.y = capped.y;
        }

        if speed > 0.001 {
            self.roll_phase = (self.roll_phase + (speed / self.radius.max(0.01)) * dt) % std::f32::consts::TAU;
        }

        let grounded = self.position.z <= ground_z + self.radius + 0.06;
        if grounded {
            self.jumps_used = 0;
            let shift_hold = is_key_down(KeyCode::LeftShift) || is_key_down(KeyCode::RightShift);
            let friction = if shift_hold {
                self.friction_shift.max(0.0)
            } else {
                self.friction_default.max(0.0)
            };
            let friction_alpha = (1.0 - dt * (friction * 12.0)).max(0.0);
            self.velocity.x *= friction_alpha;
            self.velocity.y *= friction_alpha;
        }

        if self.jump_queued && (grounded || self.jumps_used < self.max_jumps) {
            let mut boost_dir = vec3(self.velocity.x, self.velocity.y, 0.0);
            if boost_dir.length_squared() < 1.0e-6 {
                boost_dir = self.last_move_dir;
            }
            if boost_dir.length_squared() > 1.0e-6 {
                boost_dir = boost_dir.normalize();
            }

            self.velocity += up * (self.jump_impulse * self.jump_rise_boost);
            if boost_dir.length_squared() > 1.0e-6 {
                self.velocity += boost_dir * (self.space_boost_impulse * 0.2);
            }
            if !grounded {
                self.jumps_used += 1;
            } else {
                self.jumps_used = 1;
            }
            self.jump_float_timer = self.jump_float_duration;
            self.jump_queued = false;
            jumped = true;
        }

        if !grounded {
            if self.jump_float_timer > 0.0 && self.velocity.z > 0.0 {
                let drag = (1.0 - dt * self.jump_float_drag).max(0.0);
                self.velocity.z *= drag;
            } else if self.velocity.z < 0.0 {
                let drag = (1.0 - dt * self.float_fall_drag).max(0.0);
                self.velocity.z *= drag;
            }
        }
        self.jump_float_timer = (self.jump_float_timer - dt).max(0.0);

        if let Some(water_h) = water_height {
            self.apply_water_buoyancy(dt, water_h);
        }

        self.velocity.z += self.gravity * dt;
        self.position += self.velocity * dt;

        let min_z = ground_z + self.radius + 0.06;
        if self.position.z < min_z {
            self.position.z = min_z;
            if self.velocity.z < 0.0 {
                self.velocity.z = 0.0;
            }
        }

        let (min_x, max_x, min_y, max_y) = bounds;
        if self.position.x < min_x + self.radius {
            self.position.x = min_x + self.radius;
            if self.velocity.x < 0.0 {
                self.velocity.x = 0.0;
            }
        } else if self.position.x > max_x - self.radius {
            self.position.x = max_x - self.radius;
            if self.velocity.x > 0.0 {
                self.velocity.x = 0.0;
            }
        }

        if self.position.y < min_y + self.radius {
            self.position.y = min_y + self.radius;
            if self.velocity.y < 0.0 {
                self.velocity.y = 0.0;
            }
        } else if self.position.y > max_y - self.radius {
            self.position.y = max_y - self.radius;
            if self.velocity.y > 0.0 {
                self.velocity.y = 0.0;
            }
        }

        jumped
    }

    pub fn draw(&self, color: Color) {
        draw_sphere(self.position, self.radius, self.ball_texture.as_ref(), color);
        self.draw_roll_marker();
    }

    fn draw_roll_marker(&self) {
        let mut move_dir = vec3(self.last_move_dir.x, self.last_move_dir.y, 0.0);
        if move_dir.length_squared() < 1.0e-6 {
            return;
        }
        move_dir = move_dir.normalize();
        let axis = vec3(-move_dir.y, move_dir.x, 0.0);
        let base = vec3(0.0, 0.0, 1.0);
        let offset = Self::rotate_around_axis(base, axis, self.roll_phase) * (self.radius * 0.9);
        let marker_pos = self.position + offset;
        draw_sphere(marker_pos, self.radius * 0.08, None, Color::from_rgba(220, 250, 255, 220));
    }

    fn rotate_around_axis(vec: Vec3, axis: Vec3, angle_rad: f32) -> Vec3 {
        let axis_n = axis.normalize();
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();
        vec * cos_a + axis_n.cross(vec) * sin_a + axis_n * (axis_n.dot(vec) * (1.0 - cos_a))
    }
    fn apply_water_buoyancy(&mut self, dt: f32, water_h: f32) {
        let bottom_z = self.position.z - self.radius;
        let depth = water_h - bottom_z;
        if depth <= 0.0 {
            return;
        }

        let up = vec3(0.0, 0.0, 1.0);
        let g_mag = self.gravity.abs().max(0.1);
        let submerge = (depth / (self.radius * 1.9).max(0.05)).clamp(0.0, 1.35);
        let buoy_force = self.mass
            * g_mag
            * (self.water_buoyancy_bias + submerge * self.water_buoyancy_strength);
        self.velocity += up * (buoy_force / self.mass) * dt;

        let v_up = self.velocity.dot(up);
        let mut v_planar = self.velocity - up * v_up;
        let planar_drag = (1.0
            - dt * self.water_drag_planar * (0.3 + submerge * 0.7))
            .max(0.0);
        let vertical_drag = (1.0
            - dt * self.water_drag_vertical * (0.4 + submerge * 0.9))
            .max(0.0);
        v_planar *= planar_drag;
        let v_up = v_up * vertical_drag;
        self.velocity = v_planar + up * v_up;
    }
}
