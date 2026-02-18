pub mod arena_fx;
pub mod camera;
pub mod level;
pub mod player;
pub mod weapons;

use arena_fx::ArenaFx;
use camera::FollowCamera;
use level::{Level, LevelKind};
use macroquad::prelude::*;
use player::Player;
use weapons::WeaponsSystem;
use std::path::Path;

pub struct World {
    level: Level,
    player: Player,
    camera: FollowCamera,
    weapons: WeaponsSystem,
    arena_fx: ArenaFx,
    time: f32,
    player_hp: f32,
    player_hp_max: f32,
    player_xp: f32,
    player_xp_next: f32,
    player_level: i32,
    monsters_total: i32,
    monsters_slain: i32,
    last_events: WorldEvents,
    last_level_kind: LevelKind,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WorldEvents {
    pub audio: crate::game::audio::AudioEvents,
}

impl World {
    pub async fn new() -> Self {
        let level = Level::main_arena();
        let level_kind = level.kind;
        let mut player = Player::new(level.spawn_point());
        let water_tex = load_texture("assets/graphics/water/water_base.png")
            .await
            .expect("Failed to load water texture");
        let ball_tex = load_random_ball_texture("assets/graphics/ball").await;
        if let Some(tex) = ball_tex.clone() {
            tex.set_filter(FilterMode::Linear);
            player.set_ball_texture(Some(tex));
        }

        Self {
            level,
            player,
            camera: FollowCamera::new(),
            weapons: WeaponsSystem::new(),
            arena_fx: ArenaFx::new(water_tex),
            time: 0.0,
            player_hp: 120.0,
            player_hp_max: 120.0,
            player_xp: 0.0,
            player_xp_next: 100.0,
            player_level: 1,
            monsters_total: 0,
            monsters_slain: 0,
            last_events: WorldEvents::default(),
            last_level_kind: level_kind,
        }
    }

    pub fn update(&mut self, dt: f32) {
        self.time += dt;
        if is_key_pressed(KeyCode::Key1) {
            self.set_level(LevelKind::MainArena);
        }
        if is_key_pressed(KeyCode::Key2) || is_key_pressed(KeyCode::B) {
            self.set_level(LevelKind::BossArena);
        }
        self.camera.update_input(dt);
        let camera_forward = self.camera.forward_for_target(self.player.position());
        let weapon_forward = -camera_forward;
        let water_height = self.level.floor_z() + 0.12 + 0.28;
        let jumped = self.player.update(
            dt,
            self.level.floor_z(),
            self.level.bounds(),
            -camera_forward,
            Some(water_height),
        );
        self.camera.update_follow(
            self.player.position(),
            dt,
            self.level.bounds(),
            self.level.floor_z(),
            water_height,
        );
        self.weapons.update(
            dt,
            self.player.position(),
            weapon_forward,
            self.level.floor_z(),
            water_height,
            self.level.bounds(),
        );
        let weapon_events = self.weapons.take_events();
        let boss_enter = self.level.kind == LevelKind::BossArena && self.last_level_kind != LevelKind::BossArena;
        self.last_level_kind = self.level.kind;
        self.last_events.audio = crate::game::audio::AudioEvents {
            jump: jumped,
            rocket: weapon_events.rocket,
            bomb: weapon_events.bomb,
            spin: weapon_events.spin,
            throw_attack: weapon_events.throw_attack,
            boss_enter,
        };
        let speed_ratio = self.player.speed_ratio();
        let compression = (1.0 - 0.65 * speed_ratio).clamp(0.0, 1.0);
        let thermal = 1.0;
        self.arena_fx.set_compression_factor(compression);
        self.arena_fx.set_thermal_strength(thermal);
        self.arena_fx.set_player_w(0.0);
        self.arena_fx
            .set_corridor_w(self.level.map_w.max(self.level.map_d).max(1.0));
        self.arena_fx.set_level_z_step(6.0);
        self.arena_fx.update(dt);
    }

    pub fn draw(&self, render_target: Option<RenderTarget>) {
        self.camera.apply_with_target(render_target);
        self.level.draw();
        let floor_z = self.level.floor_z();
        let center = vec3(self.level.map_w * 0.5, self.level.map_d * 0.5, floor_z + 0.12);
        let uv_scale = (self.level.map_w.max(self.level.map_d) / 16.0).max(0.2);
        self.arena_fx
            .draw_water(center, vec2(self.level.map_w, self.level.map_d), uv_scale);
        self.player.draw(Color::from_rgba(230, 200, 90, 255));
        let weapon_forward = -self.camera.forward();
        let water_height = self.level.floor_z() + 0.12 + 0.28;
        self.weapons
            .draw(self.player.position(), weapon_forward, water_height);
        set_default_camera();
    }

    pub fn hud_stats(&self) -> HudStats {
        HudStats {
            hp: self.player_hp,
            hp_max: self.player_hp_max,
            xp: self.player_xp,
            xp_next: self.player_xp_next,
            level: self.player_level,
            monsters_total: self.monsters_total,
            monsters_slain: self.monsters_slain,
            time: self.time,
        }
    }

    pub fn take_events(&mut self) -> WorldEvents {
        let events = self.last_events;
        self.last_events = WorldEvents::default();
        events
    }

    fn set_level(&mut self, kind: LevelKind) {
        if self.level.kind == kind {
            return;
        }

        self.level = match kind {
            LevelKind::MainArena => Level::main_arena(),
            LevelKind::BossArena => Level::boss_arena(),
        };
        self.player.set_position(self.level.spawn_point());
    }
}

async fn load_random_ball_texture(dir: &str) -> Option<Texture2D> {
    let mut files: Vec<String> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            if let Some(ext) = path.extension().and_then(|ext| ext.to_str()) {
                let ext_l = ext.to_lowercase();
                if !(ext_l == "png" || ext_l == "jpg" || ext_l == "jpeg" || ext_l == "bmp" || ext_l == "tga") {
                    continue;
                }
            } else {
                continue;
            }
            if let Some(path_str) = path.to_str() {
                files.push(path_str.to_string());
            }
        }
    }
    if files.is_empty() {
        return None;
    }
    let idx = macroquad::rand::gen_range(0, files.len());
    let pick = files.swap_remove(idx);
    if Path::new(&pick).exists() {
        if let Ok(tex) = load_texture(&pick).await {
            return Some(tex);
        }
    }
    None
}

#[derive(Clone, Copy, Debug)]
pub struct HudStats {
    pub hp: f32,
    pub hp_max: f32,
    pub xp: f32,
    pub xp_next: f32,
    pub level: i32,
    pub monsters_total: i32,
    pub monsters_slain: i32,
    pub time: f32,
}
