use macroquad::prelude::*;

use crate::world::World;
pub mod audio;
use audio::AudioSystem;
mod postfx;
use postfx::PostFx;

pub struct Game {
    world: World,
    postfx: PostFx,
    audio: AudioSystem,
    paused: bool,
    quit_requested: bool,
    frame: u64,
    hud_time: f32,
    stats_visible: bool,
    stats_update_timer: f32,
}

impl Game {
    pub async fn new() -> Self {
        Self {
            world: World::new().await,
            postfx: PostFx::new(),
            audio: AudioSystem::new().await,
            paused: false,
            quit_requested: false,
            frame: 0,
            hud_time: 0.0,
            stats_visible: false,
            stats_update_timer: 0.0,
        }
    }

    pub fn update(&mut self, dt: f32) {
        if is_key_pressed(KeyCode::Escape) {
            if self.paused {
                self.quit_requested = true;
            } else {
                self.paused = true;
            }
        }

        if is_key_pressed(KeyCode::P) {
            self.paused = !self.paused;
        }

        if is_key_pressed(KeyCode::K) {
            self.stats_visible = !self.stats_visible;
            self.stats_update_timer = 0.0;
        }

        if !self.paused {
            self.world.update(dt);
            self.postfx.update(dt);
            self.audio.update(dt);
            let events = self.world.take_events();
            self.audio.apply_events(events.audio);
        }

        self.hud_time += dt;
        if self.stats_visible {
            self.stats_update_timer = (self.stats_update_timer - dt).max(0.0);
        }

        self.frame = self.frame.wrapping_add(1);
    }

    pub fn should_quit(&self) -> bool {
        self.quit_requested
    }

    pub fn draw(&mut self) {
        self.postfx.ensure_size();
        self.world.draw(Some(self.postfx.target()));
        self.postfx.draw();
        self.draw_hud();
    }

    fn draw_hud(&self) {
        let stats = self.world.hud_stats();
        let w = screen_width().max(1.0);
        let h = screen_height().max(1.0);

        self.draw_health_ui(w, h, stats);
        self.draw_input_hud(w, h);
        self.draw_monster_hud(w, h, stats);
        if self.stats_visible {
            self.draw_stats_dialog(w, h, stats);
        }
    }

    fn draw_health_ui(&self, _w: f32, _h: f32, stats: crate::world::HudStats) {
        let base_x = 24.0;
        let base_y = 24.0;
        let bar_w = 260.0;
        let bar_h = 16.0;

        let hp_ratio = (stats.hp / stats.hp_max.max(1.0)).clamp(0.0, 1.0);
        let xp_ratio = (stats.xp / stats.xp_next.max(1.0)).clamp(0.0, 1.0);
        let pulse = 0.5 + 0.5 * (stats.time * 7.2).sin();
        let hue_shift = (stats.time * 0.42) % 1.0;

        let (hr, hg, hb) = hsv_to_rgb((0.0 + hp_ratio * 0.33 + hue_shift * 0.08) % 1.0, 0.88, 1.0);
        let (gr, gg, gb) = hsv_to_rgb((hue_shift + 0.52) % 1.0, 0.72, 1.0);
        let hp_bg = Color::new(0.05, 0.07, 0.10, 0.86);
        let hp_glow = Color::new(gr, gg, gb, 0.16 + 0.22 * pulse);
        let hp_fill = Color::new(hr, hg, hb, 0.88 + 0.1 * pulse);

        draw_rectangle(base_x, base_y, bar_w, bar_h, hp_bg);
        draw_rectangle(base_x - 2.0, base_y - 2.0, bar_w + 4.0, bar_h + 4.0, hp_glow);
        draw_rectangle(base_x + 2.0, base_y + 2.0, (bar_w - 4.0) * hp_ratio, bar_h - 4.0, hp_fill);
        draw_text("HP", base_x, base_y - 6.0, 18.0, Color::new(0.95, 0.98, 1.0, 0.95));

        let xp_y = base_y + 28.0;
        let (xr, xg, xb) = hsv_to_rgb((stats.time * 0.35 + 0.55) % 1.0, 0.85, 1.0);
        let (xgr, xgg, xgb) = hsv_to_rgb((stats.time * 0.22 + 0.82) % 1.0, 0.7, 1.0);
        let xp_bg = Color::new(0.04, 0.06, 0.09, 0.82);
        let xp_glow = Color::new(xgr, xgg, xgb, 0.12 + 0.18 * pulse);
        let xp_fill = Color::new(xr, xg, xb, 0.9);

        draw_rectangle(base_x, xp_y, bar_w, bar_h, xp_bg);
        draw_rectangle(base_x - 2.0, xp_y - 2.0, bar_w + 4.0, bar_h + 4.0, xp_glow);
        draw_rectangle(base_x + 2.0, xp_y + 2.0, (bar_w - 4.0) * xp_ratio, bar_h - 4.0, xp_fill);
        draw_text(
            &format!("LV {} XP", stats.level),
            base_x,
            xp_y + 20.0,
            16.0,
            Color::new(0.86, 0.96, 1.0, 0.92),
        );
    }

    fn draw_input_hud(&self, _w: f32, h: f32) {
        let base_x = 40.0;
        let base_y = h - 140.0;
        let spacing = 42.0;
        let key_w = 36.0;
        let key_h = 28.0;

        let btn = |label: &str, x: f32, y: f32, pressed: bool| {
            let hue = (self.hud_time * 0.28) % 1.0;
            let key_hue = (hue + 0.19 * (label.as_bytes()[0] as f32 % 7.0)) % 1.0;
            let (rr, rg, rb) = hsv_to_rgb(key_hue, 0.72, 1.0);
            let bg = if pressed {
                Color::new(0.14 + rr * 0.32, 0.15 + rg * 0.34, 0.18 + rb * 0.34, 0.94)
            } else {
                Color::new(0.06, 0.08, 0.12, 0.64)
            };
            let glow = if pressed {
                Color::new(rr, rg, rb, 0.52)
            } else {
                Color::new(0.28, 0.78, 1.0, 0.02)
            };
            draw_rectangle(x - 4.0, y - 4.0, key_w + 8.0, key_h + 8.0, glow);
            draw_rectangle(x, y, key_w, key_h, bg);
            draw_text(label, x + 10.0, y + 20.0, 18.0, Color::new(0.9, 0.94, 1.0, 0.92));
        };

        btn("W", base_x + spacing, base_y - spacing, is_key_down(KeyCode::W));
        btn("A", base_x, base_y, is_key_down(KeyCode::A));
        btn("S", base_x + spacing, base_y, is_key_down(KeyCode::S));
        btn("D", base_x + spacing * 2.0, base_y, is_key_down(KeyCode::D));

        let mouse_x = base_x + spacing * 4.2;
        let mouse_y = base_y - 6.0;
        draw_rectangle(mouse_x, mouse_y, 96.0, 70.0, Color::new(0.04, 0.06, 0.09, 0.72));
        draw_rectangle(mouse_x + 44.0, mouse_y + 10.0, 8.0, 22.0, Color::new(0.02, 0.03, 0.05, 0.88));

        btn("L", mouse_x + 8.0, mouse_y + 8.0, is_mouse_button_down(MouseButton::Left));
        btn("M", mouse_x + 36.0, mouse_y + 20.0, is_mouse_button_down(MouseButton::Middle));
        btn("R", mouse_x + 64.0, mouse_y + 8.0, is_mouse_button_down(MouseButton::Right));
        btn("R", mouse_x + 110.0, mouse_y + 8.0, is_key_down(KeyCode::R));
    }

    fn draw_monster_hud(&self, w: f32, _h: f32, stats: crate::world::HudStats) {
        let total = stats.monsters_total.max(0) as f32;
        let slain = stats.monsters_slain.max(0) as f32;
        let left = if total > 0.0 { (total - slain).max(0.0) } else { 0.0 };
        let text = format!("Slayed: {}\nLeft: {}", slain as i32, left as i32);
        let x = w - 160.0;
        let y = 36.0;
        draw_text(&text, x, y, 18.0, Color::new(0.95, 0.98, 1.0, 0.95));
    }

    fn draw_stats_dialog(&self, w: f32, h: f32, stats: crate::world::HudStats) {
        let panel_w = 420.0;
        let panel_h = 220.0;
        let x = (w - panel_w) * 0.5;
        let y = (h - panel_h) * 0.5;

        draw_rectangle(x, y, panel_w, panel_h, Color::new(0.02, 0.05, 0.08, 0.9));
        draw_text("STATS", x + 160.0, y + 36.0, 28.0, Color::new(0.6, 0.95, 1.0, 1.0));
        draw_text(
            &format!("Ball r {:.2}  pos {:.1},{:.1},{:.1}  w {:.2}", 0.68, 0.0, 0.0, 0.0, 0.0),
            x + 24.0,
            y + 90.0,
            18.0,
            Color::new(0.9, 0.98, 1.0, 1.0),
        );
        draw_text(
            &format!("Level {}  XP {:.0}/{:.0}  HP {:.0}/{:.0}", stats.level, stats.xp, stats.xp_next, stats.hp, stats.hp_max),
            x + 24.0,
            y + 120.0,
            18.0,
            Color::new(0.9, 0.98, 1.0, 1.0),
        );
        draw_text(
            "ATK 0  DEF 0  DEX 0  STA 0  INT 0",
            x + 24.0,
            y + 150.0,
            18.0,
            Color::new(0.9, 0.98, 1.0, 1.0),
        );
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
