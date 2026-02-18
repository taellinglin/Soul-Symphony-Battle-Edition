#version 130

uniform sampler2D p3d_Texture0;
uniform float u_time;
uniform vec2 u_contact_uv;
uniform float u_wake_strength;
uniform float u_room_uv_scale;

in vec2 v_world_xy;
out vec4 fragColor;

void main() {
    vec2 uv = v_world_xy * u_room_uv_scale;

    vec2 flow_a = vec2(u_time * 0.038, -u_time * 0.023);
    vec2 flow_b = vec2(-u_time * 0.027, u_time * 0.031);

    vec3 base_a = texture(p3d_Texture0, uv + flow_a).rgb;
    vec3 base_b = texture(p3d_Texture0, uv * 1.31 + flow_b).rgb;
    vec3 base_mix = mix(base_a, base_b, 0.52);

    float dist_uv = distance(uv, u_contact_uv);
    float ring = sin(dist_uv * 66.0 - u_time * 15.0);
    float envelope = exp(-dist_uv * 7.8);
    float wake = (0.5 + 0.5 * ring) * envelope * u_wake_strength;

    float shimmer = 0.5 + 0.5 * sin((uv.x + uv.y) * 10.5 + u_time * 1.8);

    vec3 water = vec3(0.06, 0.13, 0.2) + base_mix * 0.62;
    water += vec3(0.16, 0.25, 0.34) * (wake * 1.2 + shimmer * 0.18);

    fragColor = vec4(clamp(water, 0.0, 1.0), 0.96);
}
