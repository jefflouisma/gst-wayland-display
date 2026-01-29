// Solid color vertex shader (GLSL 450 for SPIR-V)
// Compile with: glslc -fshader-stage=vertex solid.vert.glsl -o solid.vert.spv

#version 450

// Push constants for transform and color
layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 color;
} push;

// Quad vertices (CCW winding)
vec2 positions[6] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 1.0, -1.0),
    vec2( 1.0,  1.0),
    vec2(-1.0, -1.0),
    vec2( 1.0,  1.0),
    vec2(-1.0,  1.0)
);

void main() {
    vec2 pos = positions[gl_VertexIndex];
    gl_Position = push.mvp * vec4(pos, 0.0, 1.0);
}
