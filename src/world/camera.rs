use macroquad::prelude::*;

pub struct FollowCamera {
    position: Vec3,
    target: Vec3,
    smoothing: f32,
    distance: f32,
    heading_deg: f32,
    pitch_deg: f32,
    target_height: f32,
    min_pitch: f32,
    max_pitch: f32,
    min_distance: f32,
    max_distance: f32,
    collision_radius: f32,
    min_world_height: f32,
    fov_y: f32,
    mouse_look_enabled: bool,
    mouse_look_sensitivity_x: f32,
    mouse_look_sensitivity_y: f32,
    mouse_look_smooth: f32,
    mouse_look_invert_y: bool,
    mouse_turn_input: f32,
    mouse_pitch_input: f32,
    last_mouse_pos: Vec2,
    has_mouse_pos: bool,
}

impl FollowCamera {
    pub fn new() -> Self {
        Self {
            position: vec3(0.0, 0.0, 18.0),
            target: Vec3::ZERO,
            smoothing: 12.0,
            distance: 34.0,
            heading_deg: 0.0,
            pitch_deg: 14.0,
            target_height: 0.32,
            min_pitch: 6.0,
            max_pitch: 80.0,
            min_distance: 34.0,
            max_distance: 34.0,
            collision_radius: 0.18,
            min_world_height: 0.4,
            fov_y: 120.0,
            mouse_look_enabled: true,
            mouse_look_sensitivity_x: 0.16,
            mouse_look_sensitivity_y: 0.13,
            mouse_look_smooth: 0.62,
            mouse_look_invert_y: false,
            mouse_turn_input: 0.0,
            mouse_pitch_input: 0.0,
            last_mouse_pos: Vec2::ZERO,
            has_mouse_pos: false,
        }
    }

    pub fn update_input(&mut self, dt: f32) {
        self.apply_mouse_look();
        self.handle_input(dt);
    }

    pub fn update_follow(
        &mut self,
        target: Vec3,
        dt: f32,
        bounds: (f32, f32, f32, f32),
        floor_z: f32,
        water_height: f32,
    ) {
        self.target = target + vec3(0.0, 0.0, self.target_height);
        let orbit = orbit_position(self.target, self.heading_deg, self.pitch_deg, self.distance);
        let mut to_cam = orbit - self.target;
        let dist = to_cam.length();
        if dist < 1.0e-4 {
            to_cam = vec3(0.0, -1.0, 0.0);
        } else {
            to_cam /= dist;
        }
        if to_cam.z < 0.0 {
            to_cam.z = -to_cam.z;
            let len = to_cam.length().max(1.0e-4);
            to_cam /= len;
        }
        let mut desired = self.target + to_cam * self.distance.max(self.min_distance);

        let min_above = self.target.z + (self.distance * 0.35).max(0.2);
        let min_floor = floor_z + self.min_world_height;
        let min_water = water_height + 0.35;
        let min_height = min_above.max(min_floor).max(min_water);
        if desired.z < min_height {
            let mut to_cam = desired - self.target;
            to_cam.z = min_height - self.target.z;
            let len = to_cam.length().max(1.0e-4);
            to_cam /= len;
            desired = self.target + to_cam * self.distance.max(self.min_distance);
        }

        let (min_x, max_x, min_y, max_y) = bounds;
        let margin = self.collision_radius.max(0.0);
        let mut clamped = desired;
        clamped.x = clamped.x.clamp(min_x + margin, max_x - margin);
        clamped.y = clamped.y.clamp(min_y + margin, max_y - margin);
        let clamped_dist = (clamped - self.target).length();
        let tether = (self.distance * 0.9).max(self.min_distance);
        if clamped_dist >= tether && clamped.z >= min_height {
            desired = clamped;
        }
        let t = 1.0 - (-self.smoothing * dt).exp();
        self.position = self.position.lerp(desired, t);
    }

    #[allow(dead_code)]
    pub fn apply(&self) {
        self.apply_with_target(None);
    }

    pub fn apply_with_target(&self, render_target: Option<RenderTarget>) {
        set_camera(&Camera3D {
            position: self.position,
            target: self.target,
            up: vec3(0.0, 0.0, 1.0),
            fovy: self.fov_y,
            z_near: 0.012,
            z_far: 1500.0,
            render_target,
            ..Default::default()
        });
    }

    pub fn forward(&self) -> Vec3 {
        let dir = self.target - self.position;
        if dir.length_squared() < 1.0e-6 {
            vec3(0.0, 1.0, 0.0)
        } else {
            dir.normalize()
        }
    }

    pub fn forward_for_target(&self, target: Vec3) -> Vec3 {
        let target_pos = target + vec3(0.0, 0.0, self.target_height);
        let desired = orbit_position(target_pos, self.heading_deg, self.pitch_deg, self.distance);
        let dir = target_pos - desired;
        if dir.length_squared() < 1.0e-6 {
            vec3(0.0, 1.0, 0.0)
        } else {
            dir.normalize()
        }
    }

    fn handle_input(&mut self, dt: f32) {
        let rot_speed = 66.0;
        if is_key_down(KeyCode::Left) {
            self.heading_deg += rot_speed * dt;
        }
        if is_key_down(KeyCode::Right) {
            self.heading_deg -= rot_speed * dt;
        }
        if is_key_down(KeyCode::Up) {
            self.pitch_deg += rot_speed * dt;
        }
        if is_key_down(KeyCode::Down) {
            self.pitch_deg -= rot_speed * dt;
        }

        let (_wheel_x, _wheel_y) = mouse_wheel();

        if self.heading_deg < 0.0 {
            self.heading_deg += 360.0;
        } else if self.heading_deg > 360.0 {
            self.heading_deg -= 360.0;
        }

        self.pitch_deg = self.pitch_deg.clamp(self.min_pitch, self.max_pitch);
        self.distance = self.distance.clamp(self.min_distance, self.max_distance);
    }

    fn apply_mouse_look(&mut self) {
        if !self.mouse_look_enabled {
            return;
        }

        let (turn_deg, pitch_deg) = self.consume_mouse_look();
        if turn_deg.abs() > 0.0 || pitch_deg.abs() > 0.0 {
            self.heading_deg += turn_deg;
            self.pitch_deg += pitch_deg;
        }
    }

    fn consume_mouse_look(&mut self) -> (f32, f32) {
        if !self.mouse_look_enabled {
            return (0.0, 0.0);
        }

        let (mx, my) = mouse_position();
        let pos = vec2(mx, my);
        if !self.has_mouse_pos {
            self.last_mouse_pos = pos;
            self.has_mouse_pos = true;
            return (0.0, 0.0);
        }
        let delta = pos - self.last_mouse_pos;
        self.last_mouse_pos = pos;
        let dx = delta.x;
        let dy = delta.y;

        let turn_raw = -dx * self.mouse_look_sensitivity_x;
        let pitch_sign = if self.mouse_look_invert_y { -1.0 } else { 1.0 };
        let pitch_raw = -dy * self.mouse_look_sensitivity_y * pitch_sign;

        let smooth = self.mouse_look_smooth.clamp(0.0, 0.95);
        let keep = 1.0 - smooth;
        self.mouse_turn_input = self.mouse_turn_input * smooth + turn_raw * keep;
        self.mouse_pitch_input = self.mouse_pitch_input * smooth + pitch_raw * keep;
        (self.mouse_turn_input, self.mouse_pitch_input)
    }
}

fn rotate_around_axis(vec: Vec3, axis: Vec3, angle_rad: f32) -> Vec3 {
    let axis_n = axis.normalize();
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();
    vec * cos_a + axis_n.cross(vec) * sin_a + axis_n * (axis_n.dot(vec) * (1.0 - cos_a))
}

fn orbit_position(target: Vec3, heading_deg: f32, pitch_deg: f32, dist: f32) -> Vec3 {
    let up_axis = vec3(0.0, 0.0, 1.0);
    let mut base_forward = vec3(0.0, 1.0, 0.0);
    base_forward = base_forward - up_axis * base_forward.dot(up_axis);
    if base_forward.length_squared() < 1.0e-6 {
        base_forward = vec3(1.0, 0.0, 0.0) - up_axis * vec3(1.0, 0.0, 0.0).dot(up_axis);
    }
    base_forward = base_forward.normalize();

    let heading_rad = heading_deg.to_radians();
    let pitch_rad = pitch_deg.abs().clamp(0.0, 89.0).to_radians();

    let mut yaw_forward = rotate_around_axis(base_forward, up_axis, heading_rad);
    yaw_forward = yaw_forward.normalize();
    let back_dir = -yaw_forward;
    let mut offset = back_dir * (pitch_rad.cos() * dist) + up_axis * (pitch_rad.sin() * dist);

    let min_up_component = dist * 0.18;
    let up_component = offset.dot(up_axis);
    if up_component < min_up_component {
        offset += up_axis * (min_up_component - up_component);
    }

    target + offset
}
