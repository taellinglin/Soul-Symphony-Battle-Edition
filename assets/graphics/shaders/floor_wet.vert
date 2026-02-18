#version 130

in vec4 p3d_Vertex;
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;

out vec2 v_world_xy;

void main() {
    vec4 world_pos = p3d_ModelMatrix * p3d_Vertex;
    v_world_xy = world_pos.xy;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
