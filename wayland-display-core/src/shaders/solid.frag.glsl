// Solid color fragment shader (GLSL 450 for SPIR-V)
// Compile with: glslc -fshader-stage=fragment solid.frag.glsl -o solid.frag.spv

#version 450

// Push constants for color
layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 color;
} push;

// Output color
layout(location = 0) out vec4 outColor;

void main() {
    outColor = push.color;
}
