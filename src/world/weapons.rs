use macroquad::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProjectileKind {
    Rocket,
    Bomb,
    SwordThrow,
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
struct Projectile {
    kind: ProjectileKind,
    position: Vec3,
    velocity: Vec3,
    age: f32,
    ttl: f32,
    radius: f32,
}

#[derive(Clone, Copy, Debug)]
struct Explosion {
    position: Vec3,
    age: f32,
    ttl: f32,
    max_radius: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
enum AttackMode {
    Idle,
    Swing,
    Spin,
    Throw,
}

pub struct WeaponsSystem {
    rockets: Vec<Projectile>,
    bombs: Vec<Projectile>,
    explosions: Vec<Explosion>,
    sword_throw: Option<Projectile>,
    attack_mode: AttackMode,
    attack_timer: f32,
    attack_cooldown: f32,
    sword_reach: f32,
    sword_forward_offset: f32,
    sword_side_offset: f32,
    sword_up_offset: f32,
    swing_duration: f32,
    spin_duration: f32,
    sword_throw_outbound_time: f32,
    sword_throw_distance: f32,
    sword_throw_total_time: f32,
    sword_throw_spin_speed: f32,
    sword_throw_origin: Option<Vec3>,
    sword_throw_dir: Vec3,
    attack_cooldown_multiplier: f32,
    sword_trail: Vec<Vec3>,
    sword_trail_timer: f32,
    sword_trail_interval: f32,
    sword_trail_max: usize,
    events: WeaponEvents,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WeaponEvents {
    pub rocket: bool,
    pub bomb: bool,
    pub spin: bool,
    pub throw_attack: bool,
}

impl WeaponsSystem {
    pub fn new() -> Self {
        Self {
            rockets: Vec::new(),
            bombs: Vec::new(),
            explosions: Vec::new(),
            sword_throw: None,
            attack_mode: AttackMode::Idle,
            attack_timer: 0.0,
            attack_cooldown: 0.0,
            sword_reach: 1.8,
            sword_forward_offset: 0.34,
            sword_side_offset: 0.22,
            sword_up_offset: 0.84,
            swing_duration: 0.18,
            spin_duration: 0.32,
            sword_throw_outbound_time: 0.22,
            sword_throw_distance: 5.0,
            sword_throw_total_time: 0.56,
            sword_throw_spin_speed: 1080.0,
            sword_throw_origin: None,
            sword_throw_dir: vec3(0.0, 1.0, 0.0),
            attack_cooldown_multiplier: 1.0,
            sword_trail: Vec::new(),
            sword_trail_timer: 0.0,
            sword_trail_interval: 1.0 / 85.0,
            sword_trail_max: 18,
            events: WeaponEvents::default(),
        }
    }

    pub fn update(
        &mut self,
        dt: f32,
        player_pos: Vec3,
        forward: Vec3,
        floor_z: f32,
        water_height: f32,
        bounds: (f32, f32, f32, f32),
    ) {
        if self.attack_cooldown > 0.0 {
            self.attack_cooldown = (self.attack_cooldown - dt).max(0.0);
        }

        if is_key_pressed(KeyCode::R) {
            self.spawn_rocket(player_pos, forward);
        }
        if is_mouse_button_pressed(MouseButton::Middle) {
            self.spawn_bomb(player_pos, forward);
        }

        if is_mouse_button_pressed(MouseButton::Right) {
            self.trigger_spin();
        }
        if is_mouse_button_pressed(MouseButton::Left) {
            self.trigger_throw(player_pos, forward);
        }

        self.update_attacks(dt, player_pos, forward, water_height);
        self.update_projectiles(dt, floor_z, bounds);
        self.update_explosions(dt);
    }

    pub fn draw(&self, player_pos: Vec3, forward: Vec3, water_height: f32) {
        self.draw_sword(player_pos, forward, water_height);
        for rocket in &self.rockets {
            let dir = rocket.velocity.normalize_or_zero();
            let tail = rocket.position - dir * 0.8;
            draw_line_3d(rocket.position, tail, Color::from_rgba(255, 120, 80, 255));
            draw_sphere(rocket.position, rocket.radius, None, Color::from_rgba(255, 120, 80, 255));
            draw_sphere(rocket.position - dir * 0.2, rocket.radius * 0.6, None, Color::from_rgba(255, 200, 140, 255));
        }
        for bomb in &self.bombs {
            draw_sphere(bomb.position, bomb.radius, None, Color::from_rgba(90, 200, 255, 255));
            draw_sphere(bomb.position, bomb.radius * 1.6, None, Color::from_rgba(80, 160, 255, 90));
        }
        if let Some(sword) = &self.sword_throw {
            draw_cube(sword.position, vec3(0.12, 0.9, 0.06), None, Color::from_rgba(150, 240, 255, 255));
            draw_cube_wires(sword.position, vec3(0.12, 0.9, 0.06), BLACK);
        }
        for fx in &self.explosions {
            let t = (fx.age / fx.ttl).clamp(0.0, 1.0);
            let radius = fx.max_radius * (0.2 + t);
            let alpha = (1.0 - t) * 0.8;
            let color = Color::new(1.0, 0.65, 0.25, alpha);
            draw_sphere(fx.position, radius, None, color);
        }
    }

    pub fn take_events(&mut self) -> WeaponEvents {
        let events = self.events;
        self.events = WeaponEvents::default();
        events
    }

    fn spawn_rocket(&mut self, player_pos: Vec3, forward: Vec3) {
        if self.attack_cooldown > 0.0 {
            return;
        }
        let dir = forward.normalize_or_zero();
        let spawn = player_pos + dir * 1.2 + vec3(0.0, 0.0, 0.72);
        self.rockets.push(Projectile {
            kind: ProjectileKind::Rocket,
            position: spawn,
            velocity: dir * 22.0 + vec3(0.0, 0.0, 1.4),
            age: 0.0,
            ttl: 3.0,
            radius: 0.16,
        });
        self.events.rocket = true;
        self.attack_cooldown = 0.12;
    }

    fn spawn_bomb(&mut self, player_pos: Vec3, forward: Vec3) {
        if self.attack_cooldown > 0.0 {
            return;
        }
        let dir = forward.normalize_or_zero();
        let spawn = player_pos + dir * 0.9 + vec3(0.0, 0.0, 0.68);
        self.bombs.push(Projectile {
            kind: ProjectileKind::Bomb,
            position: spawn,
            velocity: dir * 10.5 + vec3(0.0, 0.0, 6.8),
            age: 0.0,
            ttl: 4.5,
            radius: 0.22,
        });
        self.events.bomb = true;
        self.attack_cooldown = 0.18;
    }

    #[allow(dead_code)]
    fn trigger_swing(&mut self) {
        if self.attack_cooldown > 0.0 || self.attack_mode != AttackMode::Idle {
            return;
        }
        self.attack_mode = AttackMode::Swing;
        self.attack_timer = 0.0;
        self.attack_cooldown = 0.35;
    }

    fn trigger_spin(&mut self) {
        if self.attack_cooldown > 0.0 || self.attack_mode != AttackMode::Idle {
            return;
        }
        self.attack_mode = AttackMode::Spin;
        self.attack_timer = 0.0;
        self.attack_cooldown = 0.6;
        self.events.spin = true;
    }

    fn trigger_throw(&mut self, player_pos: Vec3, forward: Vec3) {
        if self.attack_cooldown > 0.0 || self.attack_mode != AttackMode::Idle {
            return;
        }
        let dir = forward.normalize_or_zero();
        let up = vec3(0.0, 0.0, 1.0);
        let mut right = up.cross(dir);
        if right.length_squared() < 1.0e-6 {
            right = vec3(1.0, 0.0, 0.0);
        } else {
            right = right.normalize();
        }
        let spawn = player_pos
            + dir * self.sword_forward_offset
            + right * self.sword_side_offset
            + up * self.sword_up_offset;
        self.sword_throw = Some(Projectile {
            kind: ProjectileKind::SwordThrow,
            position: spawn,
            velocity: dir,
            age: 0.0,
            ttl: self.sword_throw_total_time,
            radius: 0.15,
        });
        self.sword_throw_origin = Some(spawn);
        self.sword_throw_dir = dir;
        self.attack_mode = AttackMode::Throw;
        self.attack_timer = 0.0;
        self.attack_cooldown = 0.0;
        self.events.throw_attack = true;
    }

    fn update_attacks(&mut self, dt: f32, player_pos: Vec3, forward: Vec3, water_height: f32) {
        if self.attack_mode == AttackMode::Idle {
            self.sword_trail.clear();
            self.sword_trail_timer = 0.0;
            return;
        }
        self.attack_timer += dt;
        match self.attack_mode {
            AttackMode::Swing => {
                if self.attack_timer >= self.swing_duration {
                    self.attack_mode = AttackMode::Idle;
                    let cd_mult = self.attack_cooldown_multiplier.max(0.35);
                    self.attack_cooldown = 0.05 * cd_mult;
                }
            }
            AttackMode::Spin => {
                if self.attack_timer >= self.spin_duration {
                    self.attack_mode = AttackMode::Idle;
                    let cd_mult = self.attack_cooldown_multiplier.max(0.35);
                    self.attack_cooldown = 0.12 * cd_mult;
                }
            }
            AttackMode::Throw => {
                if let Some(throw_proj) = &mut self.sword_throw {
                    throw_proj.age += dt;
                    let total = self.sword_throw_total_time.max(0.2);
                    let outbound = self
                        .sword_throw_outbound_time
                        .max(0.06)
                        .min(total * 0.9);
                    let distance = self.sword_throw_distance.max(1.0);
                    let t = (throw_proj.age / total).clamp(0.0, 1.0);
                    let outbound_frac = (outbound / total).max(1.0e-6);

                    let (forward_amount, _vel_sign) = if t <= outbound_frac {
                        let out_t = t / outbound_frac;
                        (1.0 - (1.0 - out_t) * (1.0 - out_t), 1.0)
                    } else {
                        let back_t = (t - outbound_frac) / (1.0 - outbound_frac);
                        ((1.0 - back_t) * (1.0 - back_t), -1.0)
                    };

                    let dir = if self.sword_throw_dir.length_squared() < 1.0e-6 {
                        forward.normalize_or_zero()
                    } else {
                        self.sword_throw_dir.normalize()
                    };
                    let up = vec3(0.0, 0.0, 1.0);
                    let mut right_throw = up.cross(dir);
                    if right_throw.length_squared() < 1.0e-6 {
                        right_throw = vec3(1.0, 0.0, 0.0);
                    } else {
                        right_throw = right_throw.normalize();
                    }

                    let arc = (t * std::f32::consts::PI).sin() * distance * 0.16;
                    let hover = 0.08 + 0.12 * (t * std::f32::consts::PI).sin();
                    let origin = self.sword_throw_origin.unwrap_or(player_pos);
                    throw_proj.position = origin
                        + dir * (distance * forward_amount)
                        + right_throw * arc
                        + up * hover;
                }
                if self.attack_timer >= self.sword_throw_total_time {
                    self.sword_throw = None;
                    self.sword_throw_origin = None;
                    self.attack_mode = AttackMode::Idle;
                    let cd_mult = self.attack_cooldown_multiplier.max(0.35);
                    self.attack_cooldown = 0.14 * cd_mult;
                }
            }
            AttackMode::Idle => {}
        }

        if matches!(self.attack_mode, AttackMode::Swing | AttackMode::Spin) {
            self.sword_trail_timer -= dt;
            if self.sword_trail_timer <= 0.0 {
                let (_base, tip, _dir, _right, _up) = self.sword_pose(player_pos, forward, water_height);
                self.sword_trail.push(tip);
                if self.sword_trail.len() > self.sword_trail_max {
                    let overflow = self.sword_trail.len() - self.sword_trail_max;
                    self.sword_trail.drain(0..overflow);
                }
                self.sword_trail_timer = self.sword_trail_interval;
            }
        }
    }

    fn update_projectiles(&mut self, dt: f32, floor_z: f32, bounds: (f32, f32, f32, f32)) {
        let (min_x, max_x, min_y, max_y) = bounds;
        let mut next_rockets = Vec::new();
        for mut rocket in self.rockets.drain(..) {
            rocket.age += dt;
            rocket.position += rocket.velocity * dt;
            let out_of_bounds = rocket.position.x < min_x
                || rocket.position.x > max_x
                || rocket.position.y < min_y
                || rocket.position.y > max_y
                || rocket.position.z < floor_z - 2.0;
            if rocket.age >= rocket.ttl || out_of_bounds {
                self.explosions.push(Explosion {
                    position: rocket.position,
                    age: 0.0,
                    ttl: 0.5,
                    max_radius: 1.6,
                });
                continue;
            }
            next_rockets.push(rocket);
        }
        self.rockets = next_rockets;

        let mut next_bombs = Vec::new();
        for mut bomb in self.bombs.drain(..) {
            bomb.age += dt;
            bomb.velocity.z -= 12.0 * dt;
            bomb.position += bomb.velocity * dt;
            let out_of_bounds = bomb.position.x < min_x
                || bomb.position.x > max_x
                || bomb.position.y < min_y
                || bomb.position.y > max_y;
            if bomb.position.z <= floor_z + bomb.radius || bomb.age >= bomb.ttl || out_of_bounds {
                self.explosions.push(Explosion {
                    position: vec3(bomb.position.x, bomb.position.y, floor_z + bomb.radius),
                    age: 0.0,
                    ttl: 0.6,
                    max_radius: 2.2,
                });
                continue;
            }
            next_bombs.push(bomb);
        }
        self.bombs = next_bombs;
    }

    fn update_explosions(&mut self, dt: f32) {
        let mut keep = Vec::new();
        for mut fx in self.explosions.drain(..) {
            fx.age += dt;
            if fx.age < fx.ttl {
                keep.push(fx);
            }
        }
        self.explosions = keep;
    }

    fn draw_sword(&self, player_pos: Vec3, forward: Vec3, water_height: f32) {
        if self.attack_mode == AttackMode::Throw {
            return;
        }

        let (base, tip, dir, right, up) = self.sword_pose(player_pos, forward, water_height);
        let sword_scale = (0.68_f32 / 0.4).max(1.0);

        let glow_color = Color::from_rgba(80, 230, 255, 170);
        draw_line_3d(base, tip, glow_color);
        draw_line_3d(base, tip, Color::from_rgba(180, 245, 255, 255));

        let guard_left = base - right * (0.26 * sword_scale);
        let guard_right = base + right * (0.26 * sword_scale);
        draw_line_3d(guard_left, guard_right, Color::from_rgba(194, 214, 242, 255));

        let grip_end = base - dir * (0.18 * sword_scale);
        draw_line_3d(grip_end, base, Color::from_rgba(46, 56, 82, 255));

        let tip_cap = tip + up * (0.04 * sword_scale);
        draw_line_3d(tip, tip_cap, Color::from_rgba(230, 250, 255, 255));

        if self.sword_trail.len() >= 2 {
            let max_idx = self.sword_trail.len() - 1;
            for (i, window) in self.sword_trail.windows(2).enumerate() {
                let t = (i as f32) / (max_idx as f32).max(1.0);
                let alpha = (1.0 - t) * 0.65;
                let color = Color::new(0.28, 0.95, 1.0, alpha);
                draw_line_3d(window[0], window[1], color);
            }
        }
    }

    fn sword_pose(&self, player_pos: Vec3, forward: Vec3, water_height: f32) -> (Vec3, Vec3, Vec3, Vec3, Vec3) {
        let dir = forward.normalize_or_zero();
        let up = vec3(0.0, 0.0, 1.0);
        let mut right = up.cross(dir);
        if right.length_squared() < 1.0e-6 {
            right = vec3(1.0, 0.0, 0.0);
        } else {
            right = right.normalize();
        }
        let mut base = player_pos
            + dir * self.sword_forward_offset
            + right * self.sword_side_offset
            + up * self.sword_up_offset;
        let min_sword_z = water_height + 0.06;
        if base.z < min_sword_z {
            base.z = min_sword_z;
        }

        let angle_deg = match self.attack_mode {
            AttackMode::Swing => {
                let t = (self.attack_timer / self.swing_duration).clamp(0.0, 1.0);
                -96.0 + 192.0 * t
            }
            AttackMode::Spin => {
                let t = (self.attack_timer / self.spin_duration).clamp(0.0, 1.0);
                -180.0 + 540.0 * t
            }
            AttackMode::Idle | AttackMode::Throw => 0.0,
        };
        let angle = angle_deg.to_radians();
        let swing_dir = vec3(
            dir.x * angle.cos() - dir.y * angle.sin(),
            dir.x * angle.sin() + dir.y * angle.cos(),
            0.0,
        )
        .normalize_or_zero();
        let tip = base + swing_dir * self.sword_reach;
        (base, tip, dir, right, up)
    }
}

#[allow(dead_code)]
trait Vec3Ext {
    fn normalize_or_zero(self) -> Vec3;
}

impl Vec3Ext for Vec3 {
    fn normalize_or_zero(self) -> Vec3 {
        if self.length_squared() < 1.0e-6 {
            vec3(0.0, 1.0, 0.0)
        } else {
            self.normalize()
        }
    }
}
