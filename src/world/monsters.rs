use macroquad::prelude::*;
use macroquad::rand::gen_range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MonsterVariant {
    Normal,
    Giant,
    Fast,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MonsterState {
    Wandering,
    Guarding,
    Hunting,
    Attacking,
    Running,
}

#[derive(Clone, Copy, Debug)]
struct MonsterPart {
    base_offset: Vec3,
    min_scale: f32,
    max_scale: f32,
    phase: f32,
    speed: f32,
    color: Color,
}

#[derive(Clone, Copy, Debug)]
struct Monster {
    position: Vec3,
    velocity: Vec3,
    radius: f32,
    hp: f32,
    hp_max: f32,
    contact_damage: f32,
    state: MonsterState,
    state_timer: f32,
    variant: MonsterVariant,
    parts: Vec<MonsterPart>,
    outline_radius: f32,
    outline_phase: f32,
    speed_scale: f32,
    guard_range: f32,
    hunt_range: f32,
    attack_range: f32,
    ranged_enabled: bool,
    ranged_cooldown: f32,
}

#[derive(Clone, Copy, Debug)]
struct EnemyProjectile {
    position: Vec3,
    velocity: Vec3,
    age: f32,
    life: f32,
    damage: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MonsterEvents {
    pub damage_to_player: f32,
    pub monsters_slain: i32,
}

pub struct MonsterSystem {
    monsters: Vec<Monster>,
    projectiles: Vec<EnemyProjectile>,
    wave_index: i32,
    wave_respawn_timer: f32,
    time: f32,
    player_damage_cooldown: f32,
    monsters_total: i32,
    monsters_slain: i32,
}

impl MonsterSystem {
    pub fn new(bounds: (f32, f32, f32, f32), floor_z: f32, player_pos: Vec3) -> Self {
        let mut system = Self {
            monsters: Vec::new(),
            projectiles: Vec::new(),
            wave_index: 1,
            wave_respawn_timer: 0.0,
            time: 0.0,
            player_damage_cooldown: 0.0,
            monsters_total: 0,
            monsters_slain: 0,
        };
        system.spawn_wave(bounds, floor_z, player_pos);
        system
    }

    pub fn update(
        &mut self,
        dt: f32,
        player_pos: Vec3,
        player_radius: f32,
        bounds: (f32, f32, f32, f32),
        floor_z: f32,
    ) -> MonsterEvents {
        self.time += dt;
        self.player_damage_cooldown = (self.player_damage_cooldown - dt).max(0.0);

        if self.monsters.is_empty() {
            if self.wave_respawn_timer <= 0.0 {
                self.wave_respawn_timer = 3.0;
            }
        }
        if self.wave_respawn_timer > 0.0 {
            self.wave_respawn_timer = (self.wave_respawn_timer - dt).max(0.0);
            if self.wave_respawn_timer <= 0.0 {
                self.wave_index += 1;
                self.spawn_wave(bounds, floor_z, player_pos);
            }
        }

        let mut events = MonsterEvents::default();
        let mut alive = Vec::new();
        for mut monster in self.monsters.drain(..) {
            monster.state_timer += dt;
            let to_player = player_pos - monster.position;
            let dist_sq = to_player.length_squared();
            let guard_range = monster.guard_range;
            let hunt_range = monster.hunt_range;
            let attack_range = monster.attack_range;

            let desired_state = if dist_sq <= attack_range * attack_range {
                MonsterState::Attacking
            } else if dist_sq <= hunt_range * hunt_range {
                MonsterState::Hunting
            } else if dist_sq <= guard_range * guard_range {
                MonsterState::Guarding
            } else {
                MonsterState::Wandering
            };
            if desired_state != monster.state {
                monster.state = desired_state;
                monster.state_timer = 0.0;
            }

            let mut desired_dir = Vec3::ZERO;
            match monster.state {
                MonsterState::Attacking | MonsterState::Hunting => {
                    if dist_sq > 1.0e-6 {
                        desired_dir = to_player.normalize();
                    }
                }
                MonsterState::Running => {
                    if dist_sq > 1.0e-6 {
                        desired_dir = (-to_player).normalize();
                    }
                }
                MonsterState::Guarding | MonsterState::Wandering => {
                    let angle = self.time * 0.6 + monster.outline_phase;
                    desired_dir = vec3(angle.cos(), angle.sin(), 0.0);
                }
            }

            let base_speed = 2.4 + 1.2 * monster.speed_scale;
            let target_vel = desired_dir * base_speed;
            let blend = 1.0 - (-dt * 4.2).exp();
            monster.velocity = monster.velocity.lerp(target_vel, blend);
            monster.position += monster.velocity * dt;

            let (min_x, max_x, min_y, max_y) = bounds;
            monster.position.x = monster.position.x.clamp(min_x + monster.radius, max_x - monster.radius);
            monster.position.y = monster.position.y.clamp(min_y + monster.radius, max_y - monster.radius);
            monster.position.z = floor_z + monster.radius;

            if dist_sq <= (monster.radius + player_radius) * (monster.radius + player_radius) {
                if self.player_damage_cooldown <= 0.0 {
                    events.damage_to_player += monster.contact_damage;
                    self.player_damage_cooldown = 0.28;
                }
            }

            if monster.ranged_enabled {
                monster.ranged_cooldown = (monster.ranged_cooldown - dt).max(0.0);
                if monster.ranged_cooldown <= 0.0 {
                    let dist = dist_sq.sqrt();
                    if dist >= 6.0 && dist <= 38.0 {
                        let fire_chance = 0.7;
                        if gen_range(0.0, 1.0) < fire_chance * dt {
                            let origin = monster.position + vec3(0.0, 0.0, 0.6);
                            let mut shot_dir = (player_pos + vec3(0.0, 0.0, 0.35)) - origin;
                            if shot_dir.length_squared() > 1.0e-6 {
                                shot_dir = shot_dir.normalize();
                            } else {
                                shot_dir = vec3(0.0, 1.0, 0.0);
                            }
                            self.projectiles.push(EnemyProjectile {
                                position: origin,
                                velocity: shot_dir * 42.0,
                                age: 0.0,
                                life: 3.2,
                                damage: 16.0,
                            });
                            monster.ranged_cooldown = gen_range(0.6, 1.3);
                        }
                    }
                }
            }

            if monster.hp > 0.0 {
                alive.push(monster);
            } else {
                events.monsters_slain += 1;
            }
        }
        self.monsters = alive;

        let mut next_projectiles = Vec::new();
        for mut proj in self.projectiles.drain(..) {
            proj.age += dt;
            proj.position += proj.velocity * dt;
            if proj.age >= proj.life {
                continue;
            }
            let hit_dist = (proj.position - player_pos).length();
            if hit_dist <= player_radius + 0.35 {
                events.damage_to_player += proj.damage;
                continue;
            }
            next_projectiles.push(proj);
        }
        self.projectiles = next_projectiles;

        self.monsters_slain += events.monsters_slain;
        events
    }

    pub fn draw(&self, player_pos: Vec3) {
        for monster in &self.monsters {
            let dist_sq = (monster.position - player_pos).length_squared();
            let outline_on = dist_sq <= monster.outline_radius * monster.outline_radius;
            let t = self.time + monster.outline_phase;
            let (or, og, ob) = hsv_to_rgb((t * 0.38) % 1.0, 0.88, 1.0);

            for part in &monster.parts {
                let pulse = 0.5 + 0.5 * (t * part.speed + part.phase).sin();
                let scale = part.min_scale + (part.max_scale - part.min_scale) * pulse;
                let pos = monster.position + part.base_offset * (0.6 + pulse * 1.1);
                draw_cube(pos, vec3(scale, scale, scale), None, part.color);
                if outline_on {
                    draw_cube_wires(pos, vec3(scale * 1.18, scale * 1.18, scale * 1.18), Color::new(or, og, ob, 1.0));
                }
            }
        }

        for proj in &self.projectiles {
            draw_sphere(proj.position, 0.18, None, Color::from_rgba(255, 90, 90, 220));
            let tail = proj.position - proj.velocity.normalize_or_zero() * 0.6;
            draw_line_3d(proj.position, tail, Color::from_rgba(255, 180, 120, 200));
        }
    }

    pub fn stats(&self) -> (i32, i32) {
        (self.monsters_total, self.monsters_slain)
    }

    fn spawn_wave(&mut self, bounds: (f32, f32, f32, f32), floor_z: f32, player_pos: Vec3) {
        let (min_x, max_x, min_y, max_y) = bounds;
        let count_base = 8.0;
        let count_scale = 0.12;
        let wave_power = (self.wave_index - 1).max(0) as f32;
        let count = (count_base * (1.0 + wave_power * count_scale)).round().max(1.0) as i32;

        self.monsters.clear();
        self.projectiles.clear();
        self.monsters_total = count;
        self.monsters_slain = 0;

        for _ in 0..count {
            let mut pos = vec3(
                gen_range(min_x + 2.0, max_x - 2.0),
                gen_range(min_y + 2.0, max_y - 2.0),
                floor_z + 0.9,
            );
            if (pos - player_pos).length() < 5.0 {
                pos.x = (pos.x + (max_x - min_x) * 0.25).min(max_x - 2.0);
                pos.y = (pos.y + (max_y - min_y) * 0.25).min(max_y - 2.0);
            }

            let variant_roll = gen_range(0.0, 1.0);
            let mut variant = MonsterVariant::Normal;
            let mut speed_scale = gen_range(0.7, 1.4);
            let mut hp_scale = gen_range(0.8, 1.7);
            let mut radius = gen_range(0.85, 1.35);
            let mut ranged_enabled = gen_range(0.0, 1.0) < 0.22;

            if variant_roll < 0.08 {
                variant = MonsterVariant::Giant;
                hp_scale *= 4.8;
                speed_scale *= 0.6;
                radius *= 1.6;
                ranged_enabled = true;
            } else if variant_roll < 0.3 {
                variant = MonsterVariant::Fast;
                speed_scale *= gen_range(2.0, 3.4);
            }

            let mut parts = Vec::new();
            let part_count = gen_range(2, 5);
            for _ in 0..part_count {
                let hue = gen_range(0.0, 1.0);
                let (r, g, b) = hsv_to_rgb(hue, gen_range(0.55, 0.9), gen_range(0.72, 1.0));
                parts.push(MonsterPart {
                    base_offset: vec3(
                        gen_range(-0.5, 0.5),
                        gen_range(-0.5, 0.5),
                        gen_range(-0.2, 0.4),
                    ),
                    min_scale: gen_range(0.08, 0.16),
                    max_scale: gen_range(0.22, 0.5),
                    phase: gen_range(0.0, std::f32::consts::TAU),
                    speed: gen_range(2.0, 4.7),
                    color: Color::new(r, g, b, 1.0),
                });
            }

            let hp_max = 100.0 * hp_scale;
            let guard_range = gen_range(15.0, 20.0);
            let hunt_range = gen_range(9.0, 14.5);
            let attack_range = gen_range(1.5, 2.4);

            self.monsters.push(Monster {
                position: pos,
                velocity: vec3(gen_range(-2.2, 2.2), gen_range(-2.2, 2.2), 0.0),
                radius,
                hp: hp_max,
                hp_max,
                contact_damage: gen_range(8.0, 14.0) * 1.1,
                state: MonsterState::Wandering,
                state_timer: 0.0,
                variant,
                parts,
                outline_radius: gen_range(4.0, 7.4),
                outline_phase: gen_range(0.0, std::f32::consts::TAU),
                speed_scale,
                guard_range,
                hunt_range,
                attack_range,
                ranged_enabled,
                ranged_cooldown: gen_range(0.4, 1.4),
            });
        }
    }
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h = (h % 1.0 + 1.0) % 1.0;
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match i as i32 % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

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
