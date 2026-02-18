#version 120

uniform sampler2D p3d_Texture0;
uniform float u_time;
uniform float u_intensity;
uniform float u_variant;
uniform float u_hyper_w;

varying vec2 v_uv;
varying vec3 v_world;

void main() {
    float t = u_time * (0.65 + u_intensity * 0.25);
    float w = u_hyper_w * 0.35;

    vec2 uv = v_uv * (1.8 + 0.22 * u_variant);
    uv += vec2(
        0.14 * sin(v_world.x * 0.18 + t * 1.8 + w),
        0.14 * cos(v_world.y * 0.21 - t * 1.6 - w)
    );

    vec3 base = texture2D(p3d_Texture0, uv).rgb;

    float n1 = sin(v_world.x * 0.22 + t * 2.2 + w);
    float n2 = cos(v_world.y * 0.28 - t * 1.9 - w * 0.7);
    float n3 = sin((v_world.x + v_world.y + v_world.z) * 0.14 + t * 1.3);
    float pulse = 0.5 + 0.5 * sin(t * 3.2 + v_world.z * 0.6);
    float noise = (n1 + n2 + n3) / 3.0;

    vec3 c0 = vec3(0.20, 0.95, 1.00);
    vec3 c1 = vec3(1.00, 0.25, 0.85);
    vec3 c2 = vec3(0.95, 1.00, 0.20);

    float m0 = 0.5 + 0.5 * sin(t + noise * 2.2 + u_variant);
    float m1 = 0.5 + 0.5 * cos(t * 0.8 - noise * 2.7 - u_variant);

    vec3 trippy = mix(c0, c1, m0);
    trippy = mix(trippy, c2, m1 * 0.45 + pulse * 0.2);

    vec3 color = mix(base, trippy, 0.58 + 0.22 * u_intensity);
    color += trippy * (0.14 + 0.12 * pulse);

    gl_FragColor = vec4(color, 1.0);
}
