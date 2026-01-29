// Vertex shader for textured quad rendering (GLSL 450 for SPIR-V)
// Compile with: glslc -fshader-stage=vertex quad.vert.glsl -o quad.vert.spv

#version 450

// Push constants for transform
layout(push_constant) uniform PushConstants {
    mat4 mvp;
    float alpha;
} push;

// Output to fragment shader
layout(location = 0) out vec2 fragTexCoord;

// Full-screen quad vertices (CCW winding)
vec2 positions[6] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 1.0, -1.0),
    vec2( 1.0,  1.0),
    vec2(-1.0, -1.0),
    vec2( 1.0,  1.0),
    vec2(-1.0,  1.0)
);

vec2 texCoords[6] = vec2[](
    vec2(0.0, 0.0),
    vec2(1.0, 0.0),
    vec2(1.0, 1.0),
    vec2(0.0, 0.0),
    vec2(1.0, 1.0),
    vec2(0.0, 1.0)
);

void main() {
    vec2 pos = positions[gl_VertexIndex];
    fragTexCoord = texCoords[gl_VertexIndex];
    
    gl_Position = push.mvp * vec4(pos, 0.0, 1.0);
}
