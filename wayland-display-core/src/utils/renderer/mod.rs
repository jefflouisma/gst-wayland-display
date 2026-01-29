//! Renderer setup and abstraction
//!
//! This module provides a unified renderer interface that can use either
//! GlesRenderer (EGL-based) or VulkanRenderer depending on feature flags.

use once_cell::sync::Lazy;
use smithay::backend::drm::{DrmNode, NodeType};
use smithay::backend::egl::{EGLContext, EGLDevice, EGLDisplay};
use smithay::backend::renderer::gles::GlesRenderer;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

static EGL_DISPLAYS: Lazy<Mutex<HashMap<Option<DrmNode>, Weak<EGLDisplay>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub fn get_egl_device_for_node(drm_node: &DrmNode) -> EGLDevice {
    let drm_node = drm_node
        .node_with_type(NodeType::Render)
        .and_then(Result::ok)
        .unwrap_or(drm_node.clone());
    EGLDevice::enumerate()
        .expect("Failed to enumerate EGLDevices")
        .find(|d| d.try_get_render_node().unwrap_or_default() == Some(drm_node))
        .expect("Unable to find EGLDevice for drm-node")
}

/// Setup the renderer (GlesRenderer for now, VulkanRenderer when vulkan feature is enabled)
///
/// This is the default EGL/OpenGL ES renderer setup.
pub fn setup_renderer(render_node: Option<DrmNode>) -> GlesRenderer {
    let mut displays = EGL_DISPLAYS.lock().unwrap();
    let maybe_display = displays
        .get(&render_node)
        .and_then(|weak_display| weak_display.upgrade());

    let egl = match maybe_display {
        Some(display) => display,
        None => {
            let device = match render_node.as_ref() {
                Some(render_node) => get_egl_device_for_node(render_node),
                None => EGLDevice::enumerate()
                    .expect("Failed to enumerate EGLDevices")
                    .find(|device| {
                        device
                            .extensions()
                            .iter()
                            .any(|e| e == "EGL_MESA_device_software")
                    })
                    .expect("Failed to find software device"),
            };
            let egl = unsafe { EGLDisplay::new(device).expect("Failed to create EGLDisplay") };
            let display = Arc::new(egl);
            displays.insert(render_node, Arc::downgrade(&display));
            display
        }
    };
    let context = EGLContext::new(&egl).expect("Failed to initialize EGL context");
    let renderer = unsafe { GlesRenderer::new(context) }.expect("Failed to initialize renderer");
    renderer
}

/// Setup Vulkan renderer (when vulkan feature is enabled)
#[cfg(feature = "vulkan")]
pub fn setup_vulkan_renderer() -> Result<crate::vulkan_renderer::VulkanRenderer, crate::vulkan_renderer::VulkanError> {
    crate::vulkan_renderer::VulkanRenderer::new()
}

/// Check if Vulkan rendering is available and should be used
#[cfg(feature = "vulkan")]
pub fn is_vulkan_available() -> bool {
    // Try to create a Vulkan renderer to check if it's available
    match setup_vulkan_renderer() {
        Ok(_) => {
            tracing::info!("Vulkan renderer available");
            true
        }
        Err(e) => {
            tracing::warn!("Vulkan renderer not available: {}", e);
            false
        }
    }
}

#[cfg(not(feature = "vulkan"))]
pub fn is_vulkan_available() -> bool {
    false
}
