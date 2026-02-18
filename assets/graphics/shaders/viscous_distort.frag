#version 130

out vec4 fragColor;

in vec2 v_uv;

uniform sampler2D tx;
uniform sampler2D screen_tex;
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
    vec2 uv = clamp(gl_FragCoord.xy / max(vec2(1.0), u_resolution), vec2(0.0), vec2(1.0));
    float t = u_time;
    float spd = clamp(u_speed, 0.0, 1.0);
    float strength = max(0.0, u_strength) * (0.45 + 0.95 * spd);

    vec2 c1 = vec2(0.35 + 0.12 * sin(t * 0.67), 0.56 + 0.10 * cos(t * 0.51));
    vec2 c2 = vec2(0.66 + 0.09 * cos(t * 0.83), 0.38 + 0.11 * sin(t * 0.59));

    vec2 p1 = uv - c1;
    vec2 p2 = uv - c2;
    float r1 = length(p1) + 1e-5;
    float r2 = length(p2) + 1e-5;

    float bulge1 = exp(-r1 * 8.0) * sin(t * 2.7 - r1 * 34.0);
    float bulge2 = exp(-r2 * 9.2) * cos(t * 2.1 - r2 * 30.0);

    vec2 radial1 = p1 / r1;
    vec2 radial2 = p2 / r2;

    float flow = sin((uv.x * 11.0 + uv.y * 14.0) + t * 1.8)
               + cos((uv.x * 17.0 - uv.y * 9.0) - t * 1.35);
    flow *= 0.5;

    float n = hash21(uv * 42.0 + t * 0.08) - 0.5;

    vec2 warp = radial1 * bulge1 * 0.036 * strength;
    warp -= radial2 * bulge2 * 0.03 * strength;
    warp += vec2(flow * 0.01, -flow * 0.009) * strength;
    warp += vec2(n, -n) * 0.0032 * strength;

    vec2 uv2 = clamp(uv + warp, vec2(0.001), vec2(0.999));
    vec4 col_tx = texture(tx, uv2);
    vec4 col_scr = texture(screen_tex, uv2);
    vec4 col = (dot(col_tx.rgb, vec3(1.0)) > 0.0001 || col_tx.a > 0.0001) ? col_tx : col_scr;

    vec2 px = vec2(1.0) / max(vec2(1.0), u_resolution);
    float bloom_radius = max(0.3, u_bloom_radius) * (0.65 + 0.8 * spd);
    vec3 bloom_acc = vec3(0.0);
    float bloom_wsum = 0.0;
    for (int ix = -2; ix <= 2; ++ix) {
        for (int iy = -2; iy <= 2; ++iy) {
            vec2 tap_off = vec2(float(ix), float(iy)) * px * bloom_radius;
            vec3 tap = texture(screen_tex, clamp(uv2 + tap_off, vec2(0.001), vec2(0.999))).rgb;
            float luma = dot(tap, vec3(0.2126, 0.7152, 0.0722));
            float bright = smoothstep(u_bloom_threshold, 1.0, luma);
            float w = exp(-(float(ix * ix + iy * iy)) * 0.42) * bright;
            bloom_acc += tap * w;
            bloom_wsum += w;
        }
    }

    vec3 bloom = (bloom_wsum > 1e-4) ? (bloom_acc / bloom_wsum) : vec3(0.0);
    float bloom_mix = max(0.0, u_bloom_strength) * (0.7 + 0.6 * strength);
    vec3 final_rgb = col.rgb * (1.0 + 0.16 * strength);
    final_rgb += bloom * bloom_mix;
    fragColor = vec4(clamp(final_rgb, 0.0, 1.0), 1.0);
}
