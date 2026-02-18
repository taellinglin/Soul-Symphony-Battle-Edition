#version 120

attribute vec4 p3d_Vertex;
attribute vec2 p3d_MultiTexCoord0;
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
uniform mat4 p3d_ModelViewMatrix;

varying vec2 v_uv;
varying vec2 v_world_xy;
varying vec3 v_world_pos;
varying float v_eye_z;

void main() {
    vec4 world_pos = p3d_ModelMatrix * p3d_Vertex;
    vec4 view_pos = p3d_ModelViewMatrix * p3d_Vertex;
    v_world_xy = world_pos.xy;
    v_world_pos = world_pos.xyz;
    v_uv = p3d_MultiTexCoord0;
    v_eye_z = -view_pos.z;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
