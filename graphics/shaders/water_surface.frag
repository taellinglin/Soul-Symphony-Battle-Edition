#version 120

uniform sampler2D u_water_tex;
uniform float u_time;
uniform float u_uv_scale;
uniform float u_alpha;
uniform float u_rainbow_strength;
uniform float u_diffusion_strength;
uniform float u_spec_strength;

varying vec2 v_uv;
varying vec2 v_world_xy;

void main() {
    vec2 tiled = v_world_xy * u_uv_scale;
    vec2 uv0 = tiled + vec2(u_time * 0.02, -u_time * 0.015);
    vec2 uv1 = tiled * 0.82 + vec2(-u_time * 0.01, u_time * 0.013);
    vec2 uv2 = tiled * 1.21 + vec2(u_time * 0.028, u_time * 0.019);

    vec3 tex0 = texture2D(u_water_tex, uv0).rgb;
    vec3 tex1 = texture2D(u_water_tex, uv1).rgb;
    vec3 tex2 = texture2D(u_water_tex, uv2).rgb;
    vec3 base = mix(tex0, tex1, 0.4);
    base = mix(base, tex2, 0.2);
    base = clamp(base, 0.0, 1.0);

    float lum = dot(base, vec3(0.2126, 0.7152, 0.0722));
    vec3 detail = base * 0.45 + vec3(lum * 0.55);
    detail = clamp(detail * 0.9, 0.0, 0.95);

    float wave = sin((v_world_xy.x + v_world_xy.y) * 1.8 + u_time * 1.7) * 0.5 + 0.5;
    float ripple = sin((v_world_xy.x - v_world_xy.y) * 2.9 - u_time * 2.3) * 0.5 + 0.5;
    float diffusion_mask = clamp((wave * 0.55 + ripple * 0.45) * u_diffusion_strength, 0.0, 1.0);

    vec3 rainbow = vec3(
        0.5 + 0.5 * sin(u_time * 1.35 + v_world_xy.x * 2.1 + 0.0),
        0.5 + 0.5 * sin(u_time * 1.35 + v_world_xy.y * 2.1 + 2.094),
        0.5 + 0.5 * sin(u_time * 1.35 + (v_world_xy.x + v_world_xy.y) * 1.1 + 4.188)
    );

    vec3 water_tint = vec3(0.14, 0.42, 0.78);
    vec3 textured_water = detail * water_tint * 1.05;

    float spec_a = pow(max(0.0, sin((v_world_xy.x + v_world_xy.y) * 12.0 + u_time * 2.8) * 0.5 + 0.5), 14.0);
    float spec_b = pow(max(0.0, sin((v_world_xy.x - v_world_xy.y) * 16.0 - u_time * 3.6) * 0.5 + 0.5), 10.0);
    float spec = clamp(spec_a * 0.62 + spec_b * 0.38, 0.0, 1.0) * u_spec_strength;
    vec3 spec_col = vec3(0.84, 0.96, 1.0) * spec;

    float overlay_mix = clamp(u_rainbow_strength * diffusion_mask * 0.7, 0.0, 0.62);
    vec3 rainbow_tinted = textured_water * 0.72 + rainbow * 0.62;
    vec3 final_col = mix(textured_water, rainbow_tinted, overlay_mix);
    final_col += spec_col;
    final_col = clamp(final_col * vec3(0.94, 0.99, 1.0), 0.0, 1.0);

    gl_FragColor = vec4(final_col, clamp(u_alpha, 0.0, 1.0));
}
