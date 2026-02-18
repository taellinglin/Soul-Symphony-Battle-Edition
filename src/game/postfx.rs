use macroquad::prelude::*;

pub struct PostFx {
    render_target: RenderTarget,
    material: Material,
    size: (u32, u32),
    time: f32,
    strength: f32,
    speed: f32,
    bloom_strength: f32,
    bloom_radius: f32,
    bloom_threshold: f32,
}

impl PostFx {
    pub fn new() -> Self {
        let width = screen_width().max(1.0) as u32;
        let height = screen_height().max(1.0) as u32;
        let render_target = render_target(width, height);
        render_target.texture.set_filter(FilterMode::Linear);

        let material = load_material(
            ShaderSource::Glsl {
                vertex: POST_VERTEX_SHADER,
                fragment: POST_FRAGMENT_SHADER,
            },
            MaterialParams {
                uniforms: vec![
                    UniformDesc::new("u_time", UniformType::Float1),
                    UniformDesc::new("u_resolution", UniformType::Float2),
                    UniformDesc::new("u_speed", UniformType::Float1),
                    UniformDesc::new("u_strength", UniformType::Float1),
                    UniformDesc::new("u_bloom_strength", UniformType::Float1),
                    UniformDesc::new("u_bloom_radius", UniformType::Float1),
                    UniformDesc::new("u_bloom_threshold", UniformType::Float1),
                ],
                ..Default::default()
            },
        )
        .expect("Failed to load postfx shader");

        Self {
            render_target,
            material,
            size: (width, height),
            time: 0.0,
            strength: 0.75,
            speed: 0.65,
            bloom_strength: 0.55,
            bloom_radius: 1.25,
            bloom_threshold: 0.75,
        }
    }

    pub fn update(&mut self, dt: f32) {
        self.time += dt;
    }

    pub fn ensure_size(&mut self) {
        let width = screen_width().max(1.0) as u32;
        let height = screen_height().max(1.0) as u32;
        if (width, height) == self.size {
            return;
        }
        let render_target = render_target(width, height);
        render_target.texture.set_filter(FilterMode::Linear);
        self.render_target = render_target;
        self.size = (width, height);
    }

    pub fn target(&self) -> RenderTarget {
        self.render_target.clone()
    }

    pub fn draw(&self) {
        set_default_camera();
        clear_background(BLACK);

        self.material.set_uniform("u_time", self.time);
        self.material
            .set_uniform("u_resolution", vec2(self.size.0 as f32, self.size.1 as f32));
        self.material.set_uniform("u_speed", self.speed);
        self.material.set_uniform("u_strength", self.strength);
        self.material
            .set_uniform("u_bloom_strength", self.bloom_strength);
        self.material.set_uniform("u_bloom_radius", self.bloom_radius);
        self.material
            .set_uniform("u_bloom_threshold", self.bloom_threshold);

        gl_use_material(&self.material);
        draw_texture_ex(
            &self.render_target.texture,
            0.0,
            0.0,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(screen_width(), screen_height())),
                ..Default::default()
            },
        );
        gl_use_default_material();
    }
}

const POST_VERTEX_SHADER: &str = r#"#version 100
attribute vec3 position;
attribute vec2 texcoord;
attribute vec4 color0;

varying lowp vec2 uv;

uniform mat4 Model;
uniform mat4 Projection;

void main() {
    gl_Position = Projection * Model * vec4(position, 1.0);
    uv = texcoord;
}
"#;

const POST_FRAGMENT_SHADER: &str = r#"#version 100
precision lowp float;

varying lowp vec2 uv;

uniform sampler2D Texture;
uniform vec2 u_resolution;
uniform float u_time;
uniform float u_speed;
uniform float u_strength;
uniform float u_bloom_strength;
uniform float u_bloom_radius;
uniform float u_bloom_threshold;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}

void main() {
    vec2 frag_uv = clamp(uv, vec2(0.0), vec2(1.0));
    float t = u_time;
    float spd = clamp(u_speed, 0.0, 1.0);
    float strength = max(0.0, u_strength) * (0.45 + 0.95 * spd);

    vec2 c1 = vec2(0.35 + 0.12 * sin(t * 0.67), 0.56 + 0.10 * cos(t * 0.51));
    vec2 c2 = vec2(0.66 + 0.09 * cos(t * 0.83), 0.38 + 0.11 * sin(t * 0.59));

    vec2 p1 = frag_uv - c1;
    vec2 p2 = frag_uv - c2;
    float r1 = length(p1) + 1e-5;
    float r2 = length(p2) + 1e-5;

    float bulge1 = exp(-r1 * 8.0) * sin(t * 2.7 - r1 * 34.0);
    float bulge2 = exp(-r2 * 9.2) * cos(t * 2.1 - r2 * 30.0);

    vec2 radial1 = p1 / r1;
    vec2 radial2 = p2 / r2;

    float flow = sin((frag_uv.x * 11.0 + frag_uv.y * 14.0) + t * 1.8)
               + cos((frag_uv.x * 17.0 - frag_uv.y * 9.0) - t * 1.35);
    flow *= 0.5;

    float n = hash21(frag_uv * 42.0 + t * 0.08) - 0.5;

    vec2 warp = radial1 * bulge1 * 0.036 * strength;
    warp -= radial2 * bulge2 * 0.03 * strength;
    warp += vec2(flow * 0.01, -flow * 0.009) * strength;
    warp += vec2(n, -n) * 0.0032 * strength;

    vec2 uv2 = clamp(frag_uv + warp, vec2(0.001), vec2(0.999));
    vec3 col = texture2D(Texture, uv2).rgb;

    vec2 px = vec2(1.0) / max(vec2(1.0), u_resolution);
    float bloom_radius = max(0.3, u_bloom_radius) * (0.65 + 0.8 * spd);
    vec3 bloom_acc = vec3(0.0);
    float bloom_wsum = 0.0;
    for (int ix = -2; ix <= 2; ++ix) {
        for (int iy = -2; iy <= 2; ++iy) {
            vec2 tap_off = vec2(float(ix), float(iy)) * px * bloom_radius;
            vec3 tap = texture2D(Texture, clamp(uv2 + tap_off, vec2(0.001), vec2(0.999))).rgb;
            float luma = dot(tap, vec3(0.2126, 0.7152, 0.0722));
            float bright = smoothstep(u_bloom_threshold, 1.0, luma);
            float w = exp(-(float(ix * ix + iy * iy)) * 0.42) * bright;
            bloom_acc += tap * w;
            bloom_wsum += w;
        }
    }

    vec3 bloom = (bloom_wsum > 1e-4) ? (bloom_acc / bloom_wsum) : vec3(0.0);
    float bloom_mix = max(0.0, u_bloom_strength) * (0.7 + 0.6 * strength);
    vec3 final_rgb = col * (1.0 + 0.16 * strength);
    final_rgb += bloom * bloom_mix;
    gl_FragColor = vec4(clamp(final_rgb, 0.0, 1.0), 1.0);
}
"#;
