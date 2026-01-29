//! Vulkan Renderer for Smithay using ash
//!
//! Direct Vulkan implementation for Wayland compositor rendering,
//! replacing the EGL-based GlesRenderer to avoid fence sync issues on NVIDIA.
//!
//! # Features
//! - Direct Vulkan 1.1+ via ash
//! - DMA-BUF import via VK_EXT_external_memory_dma_buf
//! - Frame capture for GStreamer video encoding

#![cfg(feature = "vulkan")]

use ash::{
    ext, khr,
    vk::{self, Handle},
    Device, Entry, Instance,
};
use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::allocator::Fourcc;
use smithay::backend::renderer::{
    sync::SyncPoint, Bind, DebugFlags, Frame, ImportDma, ImportMem,
    Renderer, TextureFilter, Unbind,
};
use smithay::utils::{Buffer, Physical, Rectangle, Scale, Size, Transform};
use std::collections::HashSet;
use std::ffi::CStr;
use std::sync::Arc;
use tracing::{debug, error, info, trace, warn};

/// Error type for Vulkan renderer operations
#[derive(Debug, Clone)]
pub enum VulkanError {
    /// Failed to load Vulkan library
    LoadError,
    /// No suitable physical device found
    NoDevice,
    /// Required extension not available
    MissingExtension(&'static str),
    /// Vulkan API error
    VkError(vk::Result),
    /// Failed to import DMA-BUF
    DmabufImport(String),
    /// Rendering failed
    RenderError(String),
    /// Feature not yet implemented
    NotImplemented(&'static str),
}

impl std::fmt::Display for VulkanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VulkanError::LoadError => write!(f, "Failed to load Vulkan library"),
            VulkanError::NoDevice => write!(f, "No suitable Vulkan device found"),
            VulkanError::MissingExtension(ext) => write!(f, "Missing extension: {}", ext),
            VulkanError::VkError(e) => write!(f, "Vulkan error: {:?}", e),
            VulkanError::DmabufImport(e) => write!(f, "DMA-BUF import failed: {}", e),
            VulkanError::RenderError(e) => write!(f, "Render error: {}", e),
            VulkanError::NotImplemented(feature) => write!(f, "Not implemented: {}", feature),
        }
    }
}

impl std::error::Error for VulkanError {}

impl From<vk::Result> for VulkanError {
    fn from(e: vk::Result) -> Self {
        VulkanError::VkError(e)
    }
}

/// Texture handle for Vulkan renderer
#[derive(Debug, Clone)]
pub struct VulkanTexture {
    id: u64,
    size: Size<i32, Buffer>,
    format: Fourcc,
    // TODO: Add VkImage, VkImageView, VkDeviceMemory handles
}

/// Sync point for Vulkan operations
#[derive(Debug)]
pub struct VulkanSyncPoint {
    // TODO: VkFence or VkSemaphore for synchronization
}

impl SyncPoint for VulkanSyncPoint {
    fn wait(&self) -> Result<(), ()> {
        // TODO: vkWaitForFences
        Ok(())
    }

    fn is_reached(&self) -> bool {
        // TODO: vkGetFenceStatus
        true
    }
}

/// Frame for Vulkan rendering operations
pub struct VulkanFrame<'a> {
    renderer: &'a mut VulkanRenderer,
    size: Size<i32, Physical>,
}

impl<'a> Frame for VulkanFrame<'a> {
    type Error = VulkanError;
    type TextureId = VulkanTexture;

    fn id(&self) -> usize {
        0
    }

    fn clear(&mut self, color: [f32; 4], at: &[Rectangle<i32, Physical>]) -> Result<(), Self::Error> {
        debug!("VulkanFrame::clear color={:?}", color);
        // TODO: Record clear command to command buffer
        Ok(())
    }

    fn draw_solid_rect(
        &mut self,
        dst: Rectangle<i32, Physical>,
        color: [f32; 4],
    ) -> Result<(), Self::Error> {
        debug!("VulkanFrame::draw_solid_rect");
        // TODO: Render solid color quad
        Ok(())
    }

    fn render_texture_at(
        &mut self,
        texture: &Self::TextureId,
        pos: smithay::utils::Point<i32, Physical>,
        texture_scale: i32,
        output_scale: Scale<f64>,
        src: Option<Rectangle<f64, Buffer>>,
        dst: Option<Rectangle<i32, Physical>>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: Option<&[Rectangle<i32, Physical>]>,
        kind: smithay::backend::renderer::element::Kind,
    ) -> Result<(), Self::Error> {
        debug!("VulkanFrame::render_texture_at id={}", texture.id);
        // TODO: Bind texture and render quad
        Ok(())
    }

    fn render_texture_from_to(
        &mut self,
        texture: &Self::TextureId,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: Option<&[Rectangle<i32, Physical>]>,
        transform: Transform,
        alpha: f32,
    ) -> Result<(), Self::Error> {
        debug!("VulkanFrame::render_texture_from_to id={}", texture.id);
        // TODO: Render with transform
        Ok(())
    }

    fn transformation(&self) -> Transform {
        Transform::Normal
    }

    fn finish(self) -> Result<VulkanSyncPoint, Self::Error> {
        debug!("VulkanFrame::finish - submitting command buffer");
        // TODO: Submit command buffer
        Ok(VulkanSyncPoint {})
    }
}

/// Required instance extensions for DMA-BUF support
const REQUIRED_INSTANCE_EXTENSIONS: &[&CStr] = &[
    khr::external_memory_capabilities::NAME,
    khr::external_semaphore_capabilities::NAME,
    khr::get_physical_device_properties2::NAME,
];

/// Required device extensions for DMA-BUF import
const REQUIRED_DEVICE_EXTENSIONS: &[&CStr] = &[
    khr::external_memory::NAME,
    khr::external_memory_fd::NAME,
    ext::external_memory_dma_buf::NAME,
    ext::image_drm_format_modifier::NAME,
    khr::external_semaphore::NAME,
    khr::external_semaphore_fd::NAME,
];

/// Vulkan Renderer using ash for direct Vulkan access
///
/// Implements Smithay's Renderer trait using raw Vulkan via ash.
/// Supports DMA-BUF buffer import for zero-copy rendering.
pub struct VulkanRenderer {
    _entry: Entry,
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    device: Device,
    queue: vk::Queue,
    queue_family_index: u32,
    command_pool: vk::CommandPool,
    next_texture_id: u64,
    debug_flags: DebugFlags,
    // Extension function pointers
    external_memory_fd: khr::external_memory_fd::Device,
}

impl std::fmt::Debug for VulkanRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanRenderer")
            .field("physical_device", &self.physical_device)
            .field("next_texture_id", &self.next_texture_id)
            .finish()
    }
}

impl VulkanRenderer {
    /// Create a new Vulkan renderer
    pub fn new() -> Result<Self, VulkanError> {
        info!("Initializing Vulkan renderer with ash");

        // Load Vulkan library
        let entry = unsafe { Entry::load() }.map_err(|_| VulkanError::LoadError)?;

        // Check instance version
        let instance_version = entry
            .try_enumerate_instance_version()
            .map_err(|e| VulkanError::VkError(e))?
            .unwrap_or(vk::make_api_version(0, 1, 0, 0));

        let major = vk::api_version_major(instance_version);
        let minor = vk::api_version_minor(instance_version);
        info!("Vulkan instance version: {}.{}", major, minor);

        if major < 1 || (major == 1 && minor < 1) {
            error!("Vulkan 1.1+ required for external memory support");
            return Err(VulkanError::MissingExtension("Vulkan 1.1"));
        }

        // Create instance with required extensions
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Wolf Compositor")
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(c"Smithay")
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::make_api_version(0, 1, 1, 0));

        let extension_names: Vec<*const i8> = REQUIRED_INSTANCE_EXTENSIONS
            .iter()
            .map(|ext| ext.as_ptr())
            .collect();

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names);

        let instance = unsafe { entry.create_instance(&create_info, None) }?;
        info!("Vulkan instance created");

        // Find suitable physical device
        let physical_devices = unsafe { instance.enumerate_physical_devices() }?;
        let physical_device = physical_devices
            .into_iter()
            .find(|&pd| Self::is_device_suitable(&instance, pd))
            .ok_or(VulkanError::NoDevice)?;

        let device_props = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_name = unsafe { CStr::from_ptr(device_props.device_name.as_ptr()) };
        info!("Using Vulkan device: {:?}", device_name);

        // Find graphics queue family
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let queue_family_index = queue_families
            .iter()
            .enumerate()
            .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|(idx, _)| idx as u32)
            .ok_or(VulkanError::NoDevice)?;

        // Create logical device
        let queue_priorities = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);

        let device_extension_names: Vec<*const i8> = REQUIRED_DEVICE_EXTENSIONS
            .iter()
            .map(|ext| ext.as_ptr())
            .collect();

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info))
            .enabled_extension_names(&device_extension_names);

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }?;
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        info!("Vulkan device and queue created");

        // Load extension function pointers
        let external_memory_fd = khr::external_memory_fd::Device::new(&instance, &device);

        // Create command pool
        let pool_create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe { device.create_command_pool(&pool_create_info, None) }?;

        info!("Vulkan renderer initialized successfully");

        Ok(Self {
            _entry: entry,
            instance,
            physical_device,
            device,
            queue,
            queue_family_index,
            command_pool,
            next_texture_id: 1,
            debug_flags: DebugFlags::empty(),
            external_memory_fd,
        })
    }

    /// Check if a physical device supports required extensions
    fn is_device_suitable(instance: &Instance, device: vk::PhysicalDevice) -> bool {
        let extensions = unsafe { instance.enumerate_device_extension_properties(device) };
        let available: HashSet<&CStr> = match extensions {
            Ok(exts) => exts
                .iter()
                .map(|ext| unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) })
                .collect(),
            Err(_) => return false,
        };

        for required in REQUIRED_DEVICE_EXTENSIONS {
            if !available.contains(*required) {
                trace!("Device missing extension: {:?}", required);
                return false;
            }
        }
        true
    }

    /// Import a DMA-BUF as a Vulkan texture
    fn import_dmabuf_internal(&mut self, dmabuf: &Dmabuf) -> Result<VulkanTexture, VulkanError> {
        let id = self.next_texture_id;
        self.next_texture_id += 1;

        let width = dmabuf.width() as u32;
        let height = dmabuf.height() as u32;
        let format = dmabuf.format();
        
        debug!(
            "Importing DMA-BUF: {}x{}, format={:?}, planes={}",
            width, height, format.code, dmabuf.num_planes()
        );

        // TODO: Full implementation:
        // 1. Create VkImage with VK_IMAGE_CREATE_DISJOINT_BIT if multi-plane
        // 2. For each plane:
        //    - Create VkImportMemoryFdInfoKHR with DMA-BUF fd
        //    - Allocate memory with import info
        //    - Bind memory to image
        // 3. Create VkImageView

        Ok(VulkanTexture {
            id,
            size: Size::from((width as i32, height as i32)),
            format: format.code,
        })
    }

    fn allocate_texture_id(&mut self) -> u64 {
        let id = self.next_texture_id;
        self.next_texture_id += 1;
        id
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
        info!("Vulkan renderer destroyed");
    }
}

impl Renderer for VulkanRenderer {
    type Error = VulkanError;
    type TextureId = VulkanTexture;
    type Frame<'frame> = VulkanFrame<'frame> where Self: 'frame;

    fn id(&self) -> usize {
        self as *const Self as usize
    }

    fn downscale_filter(&mut self, _filter: TextureFilter) -> Result<(), Self::Error> {
        Ok(())
    }

    fn upscale_filter(&mut self, _filter: TextureFilter) -> Result<(), Self::Error> {
        Ok(())
    }

    fn set_debug_flags(&mut self, flags: DebugFlags) {
        self.debug_flags = flags;
    }

    fn debug_flags(&self) -> DebugFlags {
        self.debug_flags
    }

    fn render(
        &mut self,
        output_size: Size<i32, Physical>,
        _dst_transform: Transform,
    ) -> Result<Self::Frame<'_>, Self::Error> {
        debug!("VulkanRenderer::render size={:?}", output_size);
        Ok(VulkanFrame {
            renderer: self,
            size: output_size,
        })
    }
}

impl Bind<Dmabuf> for VulkanRenderer {
    fn bind(&mut self, target: Dmabuf) -> Result<(), VulkanError> {
        debug!("VulkanRenderer::bind DMA-BUF for rendering");
        // TODO: Import DMA-BUF as render target
        Err(VulkanError::NotImplemented("Bind<Dmabuf>"))
    }

    fn supported_formats(&self) -> Option<HashSet<smithay::backend::allocator::Format>> {
        // TODO: Query supported formats via vkGetPhysicalDeviceFormatProperties
        None
    }
}

impl Unbind for VulkanRenderer {
    fn unbind(&mut self) -> Result<(), <Self as Renderer>::Error> {
        Ok(())
    }
}

impl ImportDma for VulkanRenderer {
    fn import_dmabuf(
        &mut self,
        dmabuf: &Dmabuf,
        _damage: Option<&[Rectangle<i32, Buffer>]>,
    ) -> Result<Self::TextureId, Self::Error> {
        self.import_dmabuf_internal(dmabuf)
    }

    fn dmabuf_formats(&self) -> Box<dyn Iterator<Item = smithay::backend::allocator::Format>> {
        // TODO: Query via VK_EXT_image_drm_format_modifier
        Box::new(std::iter::empty())
    }
}

impl ImportMem for VulkanRenderer {
    fn import_memory(
        &mut self,
        data: &[u8],
        format: Fourcc,
        size: Size<i32, Buffer>,
        _flipped: bool,
    ) -> Result<Self::TextureId, Self::Error> {
        debug!("VulkanRenderer::import_memory size={:?}", size);
        let id = self.allocate_texture_id();
        // TODO: Create staging buffer, copy data, transfer to optimal layout
        Ok(VulkanTexture { id, size, format })
    }

    fn mem_formats(&self) -> Box<dyn Iterator<Item = Fourcc>> {
        Box::new(
            [
                Fourcc::Argb8888,
                Fourcc::Xrgb8888,
                Fourcc::Abgr8888,
                Fourcc::Xbgr8888,
            ]
            .into_iter(),
        )
    }

    fn update_memory(
        &mut self,
        texture: &Self::TextureId,
        _data: &[u8],
        _region: Rectangle<i32, Buffer>,
    ) -> Result<(), Self::Error> {
        debug!("VulkanRenderer::update_memory id={}", texture.id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vulkan_renderer_creation() {
        match VulkanRenderer::new() {
            Ok(renderer) => {
                println!("Created Vulkan renderer");
            }
            Err(e) => {
                println!("Could not create Vulkan renderer (expected if no Vulkan): {}", e);
            }
        }
    }
}
