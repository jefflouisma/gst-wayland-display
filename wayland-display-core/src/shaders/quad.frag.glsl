// Fragment shader for textured quad rendering (GLSL 450 for SPIR-V)
// Compile with: glslc -fshader-stage=fragment quad.frag.glsl -o quad.frag.spv

#version 450

// Push constants for alpha
layout(push_constant) uniform PushConstants {
    mat4 mvp;
    float alpha;
} push;

// Texture sampler binding
layout(set = 0, binding = 0) uniform sampler2D texSampler;

// Input from vertex shader
layout(location = 0) in vec2 fragTexCoord;

// Output color
layout(location = 0) out vec4 outColor;

void main() {
    vec4 texColor = texture(texSampler, fragTexCoord);
    outColor = vec4(texColor.rgb, texColor.a * push.alpha);
}
