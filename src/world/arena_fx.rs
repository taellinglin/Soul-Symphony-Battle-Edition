use macroquad::prelude::*;
use macroquad::window::miniquad::*;
use macroquad::models::{draw_mesh, Mesh, Vertex};

pub struct ArenaFx {
    water_material: Material,
    water_tex: Texture2D,
    room_tex: Texture2D,
    reflection_tex: Texture2D,
    time: f32,
    alpha: f32,
    rainbow_strength: f32,
    diffusion_strength: f32,
    spec_strength: f32,
    room_tex_strength: f32,
    room_tex_desat: f32,
    thermal_mode: bool,
    thermal_strength: f32,
    compression_factor: f32,
    compression_thermal_strength: f32,
    density_contrast: f32,
    density_gamma: f32,
    player_w: f32,
    corridor_w: f32,
    level_z_step: f32,
    static_uv: bool,
    fog_color: Vec3,
    fog_start: f32,
    fog_end: f32,
    reflection_strength: f32,
}

impl ArenaFx {
    pub fn new(water_tex: Texture2D) -> Self {
        let room_tex = water_tex.clone();
        let reflection_tex = solid_texture(Color::from_rgba(0, 0, 0, 255));
        let pipeline_params = PipelineParams {
            color_blend: Some(BlendState::new(
                Equation::Add,
                BlendFactor::Value(BlendValue::SourceAlpha),
                BlendFactor::OneMinusValue(BlendValue::SourceAlpha),
            )),
            cull_face: CullFace::Nothing,
            ..Default::default()
        };

        let water_material = load_material(
            ShaderSource::Glsl {
                vertex: WATER_VERTEX_SHADER,
                fragment: WATER_FRAGMENT_SHADER,
            },
            MaterialParams {
                uniforms: vec![
                    UniformDesc::new("u_time", UniformType::Float1),
                    UniformDesc::new("u_uv_scale", UniformType::Float1),
                    UniformDesc::new("u_alpha", UniformType::Float1),
                    UniformDesc::new("u_rainbow_strength", UniformType::Float1),
                    UniformDesc::new("u_diffusion_strength", UniformType::Float1),
                    UniformDesc::new("u_spec_strength", UniformType::Float1),
                    UniformDesc::new("u_room_tex_strength", UniformType::Float1),
                    UniformDesc::new("u_room_tex_desat", UniformType::Float1),
                    UniformDesc::new("u_thermal_mode", UniformType::Float1),
                    UniformDesc::new("u_thermal_strength", UniformType::Float1),
                    UniformDesc::new("u_compression_factor", UniformType::Float1),
                    UniformDesc::new("u_compression_thermal_strength", UniformType::Float1),
                    UniformDesc::new("u_density_contrast", UniformType::Float1),
                    UniformDesc::new("u_density_gamma", UniformType::Float1),
                    UniformDesc::new("u_player_w", UniformType::Float1),
                    UniformDesc::new("u_corridor_w", UniformType::Float1),
                    UniformDesc::new("u_level_z_step", UniformType::Float1),
                    UniformDesc::new("u_static_uv", UniformType::Float1),
                    UniformDesc::new("u_fog_color", UniformType::Float3),
                    UniformDesc::new("u_fog_start", UniformType::Float1),
                    UniformDesc::new("u_fog_end", UniformType::Float1),
                    UniformDesc::new("u_reflection_strength", UniformType::Float1),
                ],
                textures: vec![
                    "u_water_tex".to_string(),
                    "u_room_tex".to_string(),
                    "u_reflection_tex".to_string(),
                ],
                pipeline_params,
            },
        )
        .expect("Failed to load water shader");

        Self {
            water_material,
            water_tex,
            room_tex,
            reflection_tex,
            time: 0.0,
            alpha: 0.9,
            rainbow_strength: 1.0,
            diffusion_strength: 0.18,
            spec_strength: 0.72,
            room_tex_strength: 0.0,
            room_tex_desat: 0.85,
            thermal_mode: true,
            thermal_strength: 1.0,
            compression_factor: 1.0,
            compression_thermal_strength: 1.0,
            density_contrast: 1.35,
            density_gamma: 0.85,
            player_w: 0.0,
            corridor_w: 16.0,
            level_z_step: 6.0,
            static_uv: false,
            fog_color: vec3(0.0, 0.0, 0.0),
            fog_start: 2.0,
            fog_end: 140.0,
            reflection_strength: 0.0,
        }
    }

    pub fn update(&mut self, dt: f32) {
        self.time += dt;
    }

    pub fn set_compression_factor(&mut self, value: f32) {
        self.compression_factor = value.clamp(0.0, 1.0);
    }

    pub fn set_thermal_strength(&mut self, value: f32) {
        self.thermal_strength = value.clamp(0.0, 1.0);
    }

    pub fn set_player_w(&mut self, value: f32) {
        self.player_w = value;
    }

    pub fn set_corridor_w(&mut self, value: f32) {
        self.corridor_w = value.max(1.0);
    }

    pub fn set_level_z_step(&mut self, value: f32) {
        self.level_z_step = value.max(0.1);
    }

    #[allow(dead_code)]
    pub fn set_fog(&mut self, color: Vec3, start: f32, end: f32) {
        self.fog_color = color;
        self.fog_start = start;
        self.fog_end = end.max(start + 0.001);
    }

    pub fn draw_water(&self, center: Vec3, size: Vec2, uv_scale: f32) {
        let mesh = build_xy_plane(center, size, Color::from_rgba(255, 255, 255, 200));

        self.water_material.set_uniform("u_time", self.time);
        self.water_material.set_uniform("u_uv_scale", uv_scale);
        self.water_material.set_uniform("u_alpha", self.alpha);
        self.water_material
            .set_uniform("u_rainbow_strength", self.rainbow_strength);
        self.water_material
            .set_uniform("u_diffusion_strength", self.diffusion_strength);
        self.water_material
            .set_uniform("u_spec_strength", self.spec_strength);
        self.water_material
            .set_uniform("u_room_tex_strength", self.room_tex_strength);
        self.water_material
            .set_uniform("u_room_tex_desat", self.room_tex_desat);
        self.water_material
            .set_uniform("u_thermal_mode", if self.thermal_mode { 1.0 } else { 0.0 });
        self.water_material
            .set_uniform("u_thermal_strength", self.thermal_strength);
        self.water_material
            .set_uniform("u_compression_factor", self.compression_factor);
        self.water_material
            .set_uniform("u_compression_thermal_strength", self.compression_thermal_strength);
        self.water_material
            .set_uniform("u_density_contrast", self.density_contrast);
        self.water_material
            .set_uniform("u_density_gamma", self.density_gamma);
        self.water_material
            .set_uniform("u_player_w", self.player_w);
        self.water_material
            .set_uniform("u_corridor_w", self.corridor_w);
        self.water_material
            .set_uniform("u_level_z_step", self.level_z_step);
        self.water_material
            .set_uniform("u_static_uv", if self.static_uv { 1.0 } else { 0.0 });
        self.water_material
            .set_uniform("u_fog_color", self.fog_color);
        self.water_material
            .set_uniform("u_fog_start", self.fog_start);
        self.water_material
            .set_uniform("u_fog_end", self.fog_end);
        self.water_material
            .set_uniform("u_reflection_strength", self.reflection_strength);
        self.water_material
            .set_texture("u_water_tex", self.water_tex.clone());
        self.water_material
            .set_texture("u_room_tex", self.room_tex.clone());
        self.water_material
            .set_texture("u_reflection_tex", self.reflection_tex.clone());

        gl_use_material(&self.water_material);
        draw_mesh(&mesh);
        gl_use_default_material();
    }
}

fn solid_texture(color: Color) -> Texture2D {
    let image = Image::gen_image_color(1, 1, color);
    let tex = Texture2D::from_image(&image);
    tex.set_filter(FilterMode::Nearest);
    tex
}

fn build_xy_plane(center: Vec3, size: Vec2, color: Color) -> Mesh {
    let hx = size.x * 0.5;
    let hy = size.y * 0.5;

    let v1 = Vertex::new2(center + vec3(-hx, -hy, 0.0), vec2(0.0, 0.0), color);
    let v2 = Vertex::new2(center + vec3(-hx, hy, 0.0), vec2(0.0, 1.0), color);
    let v3 = Vertex::new2(center + vec3(hx, hy, 0.0), vec2(1.0, 1.0), color);
    let v4 = Vertex::new2(center + vec3(hx, -hy, 0.0), vec2(1.0, 0.0), color);

    Mesh {
        vertices: vec![v1, v2, v3, v4],
        indices: vec![0, 1, 2, 0, 2, 3],
        texture: None,
    }
}

const WATER_VERTEX_SHADER: &str = r#"#version 100
attribute vec3 position;
attribute vec2 texcoord;

uniform mat4 Model;
uniform mat4 Projection;

varying lowp vec2 v_uv;
varying lowp vec2 v_world_xy;
varying lowp vec3 v_world_pos;
varying lowp float v_eye_z;

void main() {
    vec4 world_pos = Model * vec4(position, 1.0);
    v_world_xy = world_pos.xy;
    v_world_pos = world_pos.xyz;
    v_uv = texcoord;
    v_eye_z = world_pos.z;
    gl_Position = Projection * world_pos;
}
"#;

const WATER_FRAGMENT_SHADER: &str = r#"#version 100
precision mediump float;

uniform sampler2D u_water_tex;
uniform sampler2D u_room_tex;
uniform sampler2D u_reflection_tex;
uniform float u_time;
uniform float u_uv_scale;
uniform float u_alpha;
uniform float u_rainbow_strength;
uniform float u_diffusion_strength;
uniform float u_spec_strength;
uniform float u_room_tex_strength;
uniform float u_room_tex_desat;
uniform float u_thermal_mode;
uniform float u_thermal_strength;
uniform float u_compression_factor;
uniform float u_compression_thermal_strength;
uniform float u_density_contrast;
uniform float u_density_gamma;
uniform float u_player_w;
uniform float u_corridor_w;
uniform float u_level_z_step;
uniform float u_static_uv;
uniform vec3 u_fog_color;
uniform float u_fog_start;
uniform float u_fog_end;
uniform float u_reflection_strength;

varying lowp vec2 v_uv;
varying lowp vec2 v_world_xy;
varying lowp vec3 v_world_pos;
varying lowp float v_eye_z;

float hash12(vec2 p) {
    float h = dot(p, vec2(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

float noise2(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);

    float a = hash12(i + vec2(0.0, 0.0));
    float b = hash12(i + vec2(1.0, 0.0));
    float c = hash12(i + vec2(0.0, 1.0));
    float d = hash12(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p) {
    float f = 0.0;
    float amp = 0.5;
    mat2 m = mat2(1.6, 1.2, -1.2, 1.6);
    for (int i = 0; i < 4; ++i) {
        f += amp * noise2(p);
        p = m * p;
        amp *= 0.5;
    }
    return f;
}

float round_nearest(float x) {
    return (x >= 0.0) ? floor(x + 0.5) : ceil(x - 0.5);
}

float compute_level_w_like(vec3 p) {
    vec2 n0 = p.xy * 0.018;
    vec2 n1 = p.xy * 0.043 + vec2(13.7, -8.9);
    float f0 = fbm(n0);
    float f1 = fbm(n1);
    float n = (f0 * 0.68 + f1 * 0.32);
    float w = (n * 2.0 - 1.0) * 2.25;
    return clamp(w, -2.25, 2.25);
}

vec3 roygbiv_thermal(float t) {
    float x = clamp(t, 0.0, 1.0);
    vec3 c0;
    vec3 c1;
    float local;
    float band = 1.0 / 7.0;
    if (x < band) {
        c0 = vec3(1.0, 0.0, 0.0);
        c1 = vec3(1.0, 0.5, 0.0);
        local = x / band;
    } else if (x < band * 2.0) {
        c0 = vec3(1.0, 0.5, 0.0);
        c1 = vec3(1.0, 1.0, 0.0);
        local = (x - band) / band;
    } else if (x < band * 3.0) {
        c0 = vec3(1.0, 1.0, 0.0);
        c1 = vec3(0.0, 1.0, 0.0);
        local = (x - band * 2.0) / band;
    } else if (x < band * 4.0) {
        c0 = vec3(0.0, 1.0, 0.0);
        c1 = vec3(0.0, 0.0, 1.0);
        local = (x - band * 3.0) / band;
    } else if (x < band * 5.0) {
        c0 = vec3(0.0, 0.0, 1.0);
        c1 = vec3(0.29, 0.0, 0.51);
        local = (x - band * 4.0) / band;
    } else if (x < band * 6.0) {
        c0 = vec3(0.29, 0.0, 0.51);
        c1 = vec3(0.56, 0.0, 1.0);
        local = (x - band * 5.0) / band;
    } else {
        c0 = vec3(0.56, 0.0, 1.0);
        c1 = vec3(0.85, 0.45, 1.0);
        local = (x - band * 6.0) / band;
    }
    local = smoothstep(0.0, 1.0, local);
    return mix(c0, c1, local);
}

vec3 radar_palette(float t) {
    float x = clamp(t, 0.0, 1.0);
    vec3 c0;
    vec3 c1;
    float local;
    if (x < 0.25) {
        c0 = vec3(0.05, 0.08, 0.2);
        c1 = vec3(0.05, 0.25, 0.6);
        local = x / 0.25;
    } else if (x < 0.55) {
        c0 = vec3(0.05, 0.25, 0.6);
        c1 = vec3(0.0, 0.65, 0.35);
        local = (x - 0.25) / 0.3;
    } else if (x < 0.8) {
        c0 = vec3(0.0, 0.65, 0.35);
        c1 = vec3(0.85, 0.85, 0.2);
        local = (x - 0.55) / 0.25;
    } else {
        c0 = vec3(0.85, 0.85, 0.2);
        c1 = vec3(0.95, 0.2, 0.2);
        local = (x - 0.8) / 0.2;
    }
    local = smoothstep(0.0, 1.0, local);
    return mix(c0, c1, local);
}

vec3 hue_shift(vec3 c, float a) {
    float s = sin(a);
    float co = cos(a);
    mat3 m = mat3(
        0.299 + 0.701 * co + 0.168 * s, 0.587 - 0.587 * co + 0.330 * s, 0.114 - 0.114 * co - 0.497 * s,
        0.299 - 0.299 * co - 0.328 * s, 0.587 + 0.413 * co + 0.035 * s, 0.114 - 0.114 * co + 0.292 * s,
        0.299 - 0.300 * co + 1.250 * s, 0.587 - 0.588 * co - 1.050 * s, 0.114 + 0.886 * co - 0.203 * s
    );
    return clamp(m * c, 0.0, 1.0);
}

void main() {
    float local_w = compute_level_w_like(v_world_pos);
    float density = (local_w + 2.25) / 4.5;
    if (u_static_uv > 0.5) {
        vec3 static_pos = v_world_pos;
        static_pos.xy *= 6.0;
        local_w = compute_level_w_like(static_pos);
        density = (local_w + 2.25) / 4.5;
        float micro = fbm(static_pos.xy * 0.35 + vec2(9.1, -4.3));
        density = clamp(density * 1.25 + (micro - 0.5) * 0.35 + 0.05, 0.0, 1.0);
    } else {
        float micro = fbm(v_world_xy * 0.12 + vec2(5.4, -2.2));
        density = clamp(density + (micro - 0.5) * 0.4, 0.0, 1.0);
    }
    density = clamp(density * u_density_contrast, 0.0, 1.0);
    density = pow(density, u_density_gamma);
    density = smoothstep(0.0, 1.0, density);
    vec3 thermal_col = roygbiv_thermal(density);
    float cycle_mix = clamp(u_rainbow_strength, 0.0, 1.0);
    if (cycle_mix > 0.001) {
        vec3 cycle_col = roygbiv_thermal(density);
        thermal_col = mix(thermal_col, cycle_col, cycle_mix);
    }

    vec2 uv = v_world_xy * max(0.02, u_uv_scale * 0.08);
    vec2 flow_a = vec2(u_time * 0.09, -u_time * 0.05);
    vec2 flow_b = vec2(-u_time * 0.06, u_time * 0.07);

    float n0 = fbm(uv * 1.25 + flow_a);
    float n1 = fbm(uv * 2.35 + flow_b + vec2(11.3, -4.7));
    float h = n0 * 0.62 + n1 * 0.38;

    float eps = 0.06;
    float hx = fbm((uv + vec2(eps, 0.0)) * 1.25 + flow_a) * 0.62 + fbm((uv + vec2(eps, 0.0)) * 2.35 + flow_b + vec2(11.3, -4.7)) * 0.38;
    float hy = fbm((uv + vec2(0.0, eps)) * 1.25 + flow_a) * 0.62 + fbm((uv + vec2(0.0, eps)) * 2.35 + flow_b + vec2(11.3, -4.7)) * 0.38;
    vec3 normal = normalize(vec3((hx - h) * 6.8, (hy - h) * 6.8, 1.0));

    vec3 light_dir = normalize(vec3(0.35, 0.28, 0.89));
    vec3 view_dir = vec3(0.0, 0.0, 1.0);
    vec3 half_vec = normalize(light_dir + view_dir);

    float ndotl = max(dot(normal, light_dir), 0.0);
    float ndotv = max(dot(normal, view_dir), 0.0);
    float spec = pow(max(dot(normal, half_vec), 0.0), 84.0);

    float sparkle_noise = fbm(uv * 9.4 + vec2(u_time * 0.34, -u_time * 0.27));
    float sparkle = smoothstep(0.78, 0.95, sparkle_noise) * smoothstep(0.45, 1.0, spec);

    float spec_strength = max(0.2, u_spec_strength);
    vec3 base = vec3(0.05, 0.08, 0.11) + vec3(0.10, 0.14, 0.18) * ndotl;
    float fresnel = pow(1.0 - ndotv, 3.0);
    vec3 highlights = vec3(1.0) * (spec * (0.7 + spec_strength * 0.9) + sparkle * (0.22 + spec_strength * 0.55));
    vec3 water_col = clamp(base + highlights, 0.0, 1.0);
    if (u_spec_strength <= 0.01 && u_room_tex_strength <= 0.01 && u_diffusion_strength <= 0.01 && u_rainbow_strength <= 0.01) {
        water_col = vec3(0.0);
    }

    vec3 room_tex = texture2D(u_room_tex, v_uv).rgb;
    float room_luma = dot(room_tex, vec3(0.299, 0.587, 0.114));
    vec3 room_desat = mix(room_tex, vec3(room_luma), clamp(u_room_tex_desat, 0.0, 1.0));

    float thermal_mix = clamp(u_thermal_mode, 0.0, 1.0) * clamp(u_thermal_strength * 0.6, 0.0, 1.0);
    bool thermal_only = (u_thermal_mode > 0.5 && u_thermal_strength > 0.01 && u_room_tex_strength <= 0.01
        && u_spec_strength <= 0.01 && u_diffusion_strength <= 0.01 && u_rainbow_strength <= 0.01);
    vec3 final_col = thermal_only ? thermal_col : mix(water_col, thermal_col, thermal_mix);
    float compression_intensity = clamp((1.0 - u_compression_factor) / 0.65, 0.0, 1.0);
    vec3 compression_col = roygbiv_thermal(compression_intensity);
    float compression_mix = clamp(u_compression_thermal_strength, 0.0, 1.0);
    final_col = mix(final_col, compression_col, compression_mix);
    if (!thermal_only) {
        final_col = clamp(final_col + room_desat * clamp(u_room_tex_strength, 0.0, 1.0), 0.0, 1.0);
    }
    final_col = clamp(final_col, 0.0, 1.0);

    if (u_thermal_mode > 0.5) {
        float compression_intensity = clamp((1.0 - u_compression_factor) / 0.65, 0.0, 1.0);
        float field_a = fbm(v_world_xy * 0.08 + vec2(13.2, -7.4));
        float field_b = fbm(v_world_xy * 0.18 + vec2(-4.7, 9.1));
        float field_c = fbm(v_world_xy * 0.35 + vec2(2.1, -3.6));
        float field = clamp(0.15 + field_a * 0.55 + field_b * 0.28 + field_c * 0.12, 0.0, 1.0);
        float radar_val = clamp(field + compression_intensity * 0.6, 0.0, 1.0);
        float band_steps = 9.0;
        float band_pos = radar_val * band_steps;
        float banded = floor(band_pos) / band_steps;
        float band_edge = smoothstep(0.35, 0.6, fract(band_pos));
        float band_val = mix(banded, clamp(banded + 1.0 / band_steps, 0.0, 1.0), band_edge);
        vec3 band_col = radar_palette(band_val);
        float contour = smoothstep(0.48, 0.52, fract(band_pos));
        vec3 thermal_band = mix(band_col, vec3(0.95, 0.98, 1.0), contour * 0.2);
        float thermal_blend = clamp(u_thermal_strength * 0.6, 0.0, 1.0);
        final_col = mix(final_col, thermal_band, thermal_blend);
    }

    float fog_range = max(0.001, u_fog_end - u_fog_start);
    float fog_factor = clamp((u_fog_end - v_eye_z) / fog_range, 0.0, 1.0);
    final_col = mix(u_fog_color, final_col, fog_factor);
    if (u_spec_strength > 0.01) {
        final_col = clamp(final_col + vec3(0.55) * fresnel + highlights * (0.35 + fresnel * 0.85), 0.0, 1.0);
    }
    if (u_reflection_strength > 0.001) {
        vec2 reflect_uv = v_uv * vec2(1.0, -1.0) + vec2(0.0, 1.0);
        reflect_uv = clamp(reflect_uv, vec2(0.0), vec2(1.0));
        vec3 reflect_col = texture2D(u_reflection_tex, reflect_uv).rgb;
        float reflect_mix = clamp(u_reflection_strength * (0.65 + fresnel * 0.75), 0.0, 1.0);
        final_col = mix(final_col, reflect_col, reflect_mix);
        final_col = clamp(final_col + reflect_col * (0.08 + fresnel * 0.12), 0.0, 1.0);
    }
    if (u_spec_strength > 0.01) {
        final_col = hue_shift(final_col, u_time * 0.18);
    }
    float out_alpha = (u_spec_strength > 0.01) ? clamp(u_alpha * 0.35, 0.08, 0.8) : clamp(u_alpha, 0.0, 1.0);
    if (u_thermal_mode > 0.5) {
        out_alpha = clamp(out_alpha + 0.18, 0.0, 1.0);
    }
    gl_FragColor = vec4(final_col, out_alpha);
}
"#;
