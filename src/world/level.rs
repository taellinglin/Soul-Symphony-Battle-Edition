use macroquad::prelude::*;
use ::rand::{thread_rng, Rng};
use ::rand::seq::SliceRandom;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct GenerationConfig {
    pub scale: f32,
    pub average_room_size: f32,
    pub room_size_jitter: f32,
    pub room_height: f32,
    pub corridor_width: f32,
    pub room_density: f32,
    pub corridor_density: f32,
    pub max_rooms: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            scale: 1.0,
            average_room_size: 12.0,
            room_size_jitter: 0.2,
            room_height: 12.0,
            corridor_width: 9.6,
            room_density: 0.5,
            corridor_density: 0.5,
            max_rooms: 72,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LevelKind {
    MainArena,
    BossArena,
}

#[derive(Clone, Copy, Debug)]
pub struct Room {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Room {
    pub fn center(&self) -> Vec2 {
        vec2(self.x + self.w * 0.5, self.y + self.h * 0.5)
    }
}

#[allow(dead_code)]
pub struct Level {
    pub kind: LevelKind,
    pub config: GenerationConfig,
    pub rooms: Vec<Room>,
    pub edges: Vec<(usize, usize)>,
    pub map_w: f32,
    pub map_d: f32,
}

impl Level {
    pub fn main_arena() -> Self {
        let config = GenerationConfig {
            scale: 2.4,
            average_room_size: 13.0,
            room_size_jitter: 0.62,
            room_height: 4096.0,
            corridor_width: 15.5,
            room_density: 0.68,
            corridor_density: 0.64,
            max_rooms: 64,
        };

        let map_w = (176.0 * config.scale) as i32;
        let map_d = (176.0 * config.scale) as i32;
        let avg_scaled = config.average_room_size * config.scale;
        let cell_size = (avg_scaled * 0.96).clamp(22.0, 44.0) as i32;

        let generator = BspGenerator::new(map_w, map_d, config);
        let (rooms, edges) = generator.generate_hex_mixed(cell_size);

        Self {
            kind: LevelKind::MainArena,
            config,
            rooms,
            edges,
            map_w: map_w as f32,
            map_d: map_d as f32,
        }
    }

    pub fn boss_arena() -> Self {
        let config = GenerationConfig {
            scale: 1.6,
            average_room_size: 18.0,
            room_size_jitter: 0.0,
            room_height: 4096.0,
            corridor_width: 0.0,
            room_density: 0.0,
            corridor_density: 0.0,
            max_rooms: 1,
        };

        let map_w = 176.0 * config.scale;
        let map_d = 176.0 * config.scale;
        let room_w = map_w * 0.7;
        let room_h = map_d * 0.7;
        let room = Room {
            x: (map_w - room_w) * 0.5,
            y: (map_d - room_h) * 0.5,
            w: room_w,
            h: room_h,
        };

        Self {
            kind: LevelKind::BossArena,
            config,
            rooms: vec![room],
            edges: Vec::new(),
            map_w,
            map_d,
        }
    }

    pub fn spawn_point(&self) -> Vec3 {
        let center = if self.kind == LevelKind::MainArena {
            self.rooms
                .first()
                .map(|room| room.center())
                .unwrap_or_else(|| vec2(self.map_w * 0.5, self.map_d * 0.5))
        } else {
            self.rooms
                .first()
                .map(|room| room.center())
                .unwrap_or_else(|| vec2(self.map_w * 0.5, self.map_d * 0.5))
        };
        vec3(center.x, center.y, self.floor_z() + 0.70)
    }

    pub fn draw(&self) {
        let floor_z = self.floor_z();
        let floor_thickness = 0.2;
        let wall_height = 2.6;

        let floor_center = vec3(self.map_w * 0.5, self.map_d * 0.5, floor_z - floor_thickness * 0.5);
        let floor_size = vec3(self.map_w, self.map_d, floor_thickness);
        draw_cube(
            floor_center,
            floor_size,
            None,
            Color::from_rgba(18, 22, 30, 255),
        );

        let wall_pos = vec3(self.map_w * 0.5, self.map_d * 0.5, floor_z + wall_height * 0.5);
        let wall_size = vec3(self.map_w, self.map_d, wall_height);
        draw_cube_wires(wall_pos, wall_size, Color::from_rgba(70, 110, 190, 255));

        if self.kind == LevelKind::BossArena {
            for room in &self.rooms {
                let center = room.center();
                let ring_pos = vec3(center.x, center.y, floor_z + 0.08);
                let ring_size = vec3(room.w * 0.85, room.h * 0.85, 0.1);
                draw_cube_wires(ring_pos, ring_size, Color::from_rgba(120, 220, 255, 255));
            }
        }

        let inverted_z = floor_z - 6.5;
        let inverted_center = vec3(self.map_w * 0.5, self.map_d * 0.5, inverted_z - floor_thickness * 0.5);
        let inverted_size = vec3(self.map_w * 0.96, self.map_d * 0.96, floor_thickness);
        draw_cube(
            inverted_center,
            inverted_size,
            None,
            Color::from_rgba(10, 14, 20, 220),
        );
        draw_cube_wires(
            vec3(self.map_w * 0.5, self.map_d * 0.5, inverted_z + wall_height * 0.5),
            vec3(self.map_w * 0.96, self.map_d * 0.96, wall_height),
            Color::from_rgba(90, 40, 160, 200),
        );
    }

    pub fn floor_z(&self) -> f32 {
        0.0
    }

    pub fn bounds(&self) -> (f32, f32, f32, f32) {
        (0.0, self.map_w, 0.0, self.map_d)
    }
}

struct BspGenerator {
    width: i32,
    depth: i32,
    config: GenerationConfig,
}

impl BspGenerator {
    fn new(width: i32, depth: i32, config: GenerationConfig) -> Self {
        Self {
            width,
            depth,
            config,
        }
    }

    fn cube_unit(&self) -> f32 {
        1.0
    }

    fn snap(&self, value: f32) -> f32 {
        let unit = self.cube_unit();
        (value / unit).round() * unit
    }

    fn snap_room(&self, x: f32, y: f32, w: f32, h: f32) -> Room {
        let unit = self.cube_unit();
        let min_size = (unit * 3.0).max(2.5);
        let sw = self.snap(w).max(min_size);
        let sh = self.snap(h).max(min_size);
        let mut sx = self.snap(x);
        let mut sy = self.snap(y);

        let max_x = (self.width as f32 - sw).max(0.0);
        let max_y = (self.depth as f32 - sh).max(0.0);
        if sx < 0.0 {
            sx = 0.0;
        }
        if sy < 0.0 {
            sy = 0.0;
        }
        if sx > max_x {
            sx = max_x;
        }
        if sy > max_y {
            sy = max_y;
        }

        Room {
            x: sx,
            y: sy,
            w: sw,
            h: sh,
        }
    }

    fn room_budget(&self) -> usize {
        let configured = self.config.max_rooms;
        configured.clamp(8, 256)
    }

    fn generate_hex_mixed(&self, cell_size: i32) -> (Vec<Room>, Vec<(usize, usize)>) {
        let mut rng = thread_rng();
        let size = cell_size.max(12) as f32;
        let step_x = size * 0.9;
        let step_y = size * 0.78;
        let margin = size * 0.6;

        let rows = (((self.depth as f32) - margin * 2.0) / step_y)
            .floor()
            .max(3.0) as i32;
        let cols = (((self.width as f32) - margin * 2.0) / step_x)
            .floor()
            .max(3.0) as i32;

        let room_budget = self.room_budget();
        let mut cells: Vec<(i32, i32, f32, f32)> = Vec::new();
        for r in 0..rows {
            let x_offset = if r % 2 == 1 { step_x * 0.5 } else { 0.0 };
            for c in 0..cols {
                let cx = margin + c as f32 * step_x + x_offset;
                let cy = margin + r as f32 * step_y;
                if cx < margin
                    || cy < margin
                    || cx > (self.width as f32 - margin)
                    || cy > (self.depth as f32 - margin)
                {
                    continue;
                }
                cells.push((c, r, cx, cy));
            }
        }

        if cells.len() > room_budget {
            cells.shuffle(&mut rng);
            cells.truncate(room_budget);
        }

        let jitter = self.config.room_size_jitter.clamp(0.0, 0.75);
        let mut rooms: Vec<Room> = Vec::new();
        let mut index_map: HashMap<(i32, i32), usize> = HashMap::new();

        for (c, r, cx, cy) in &cells {
            let shape_roll: f32 = rng.gen();
            let (mut rw, mut rh) = if shape_roll < 0.54 {
                (size * rng.gen_range(0.72..0.96), size * rng.gen_range(0.66..0.9))
            } else if shape_roll < 0.68 {
                let base = size * rng.gen_range(0.58..0.82);
                (base, base)
            } else if shape_roll < 0.8 {
                (size * rng.gen_range(1.0..1.38), size * rng.gen_range(0.45..0.7))
            } else if shape_roll < 0.92 {
                (size * rng.gen_range(0.45..0.7), size * rng.gen_range(1.0..1.38))
            } else {
                (size * rng.gen_range(0.6..0.95), size * rng.gen_range(0.6..0.95))
            };

            rw *= rng.gen_range(1.0 - jitter * 0.45..1.0 + jitter * 0.4);
            rh *= rng.gen_range(1.0 - jitter * 0.45..1.0 + jitter * 0.4);

            let rx = cx - rw * 0.5;
            let ry = cy - rh * 0.5;
            let room = self.snap_room(rx, ry, rw, rh);
            let idx = rooms.len();
            rooms.push(room);
            index_map.insert((*c, *r), idx);
        }

        let mut edges: HashSet<(usize, usize)> = HashSet::new();
        for (c, r, _, _) in &cells {
            let Some(&a) = index_map.get(&(*c, *r)) else { continue };

            let neighbors: &[(i32, i32)] = if r % 2 == 0 {
                &[
                    (c - 1, *r),
                    (c + 1, *r),
                    (*c, r - 1),
                    (c - 1, r - 1),
                    (*c, r + 1),
                    (c - 1, r + 1),
                ]
            } else {
                &[
                    (c - 1, *r),
                    (c + 1, *r),
                    (*c, r - 1),
                    (c + 1, r - 1),
                    (*c, r + 1),
                    (c + 1, r + 1),
                ]
            };

            for (nc, nr) in neighbors {
                if let Some(&b) = index_map.get(&(*nc, *nr)) {
                    if a != b {
                        edges.insert(sorted_pair(a, b));
                    }
                }
            }

            if rng.gen::<f32>() < 0.32 {
                let leaps = [(c + 2, *r), (c - 2, *r), (*c, r + 2), (*c, r - 2)];
                for (nc, nr) in leaps {
                    if let Some(&b) = index_map.get(&(nc, nr)) {
                        if a != b {
                            edges.insert(sorted_pair(a, b));
                        }
                    }
                }
            }
        }

        if edges.is_empty() && rooms.len() > 1 {
            for i in 0..rooms.len() - 1 {
                edges.insert((i, i + 1));
            }
        }

        let mut edge_list: Vec<(usize, usize)> = edges.into_iter().collect();
        edge_list.sort_unstable();
        (rooms, edge_list)
    }
}

fn sorted_pair(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}
