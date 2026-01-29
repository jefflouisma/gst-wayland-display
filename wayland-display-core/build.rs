use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    // Rerun if build script or shaders change
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/shaders/");

    // Check if the cuda feature is enabled
    #[cfg(feature = "cuda")]
    {
        // Link GStreamer CUDA library
        if let Err(e) = pkg_config::Config::new()
            .atleast_version("1.24")
            .probe("gstreamer-cuda-1.0")
        {
            eprintln!(
                "Warning: gstreamer-cuda-1.0 not found via pkg-config: {}",
                e
            );
        }
    }

    // Compile shaders if vulkan feature is enabled
    #[cfg(feature = "vulkan")]
    compile_shaders();
}

#[cfg(feature = "vulkan")]
fn compile_shaders() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let shader_dir = Path::new(&manifest_dir).join("src/shaders");

    if !shader_dir.exists() {
        println!("cargo:warning=Shader directory not found, skipping compilation");
        return;
    }

    let shaders = [
        ("quad.vert.glsl", "quad.vert.spv", "vertex"),
        ("quad.frag.glsl", "quad.frag.spv", "fragment"),
        ("solid.vert.glsl", "solid.vert.spv", "vertex"),
        ("solid.frag.glsl", "solid.frag.spv", "fragment"),
    ];

    for (input, output, stage) in shaders {
        let input_path = shader_dir.join(input);
        let output_path = shader_dir.join(output);

        if !input_path.exists() {
            continue;
        }

        // Try glslc (Vulkan SDK)
        let result = Command::new("glslc")
            .arg(format!("-fshader-stage={}", stage))
            .arg(&input_path)
            .arg("-o")
            .arg(&output_path)
            .status();

        match result {
            Ok(status) if status.success() => {}
            _ => {
                println!("cargo:warning=glslc unavailable, using pre-compiled shaders");
            }
        }
    }
}
