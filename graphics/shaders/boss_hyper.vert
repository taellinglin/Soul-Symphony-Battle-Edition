#version 120

attribute vec4 p3d_Vertex;
attribute vec2 p3d_MultiTexCoord0;
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;

varying vec2 v_uv;
varying vec3 v_world;

void main() {
    vec4 world_pos = p3d_ModelMatrix * p3d_Vertex;
    v_world = world_pos.xyz;
    v_uv = p3d_MultiTexCoord0;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
