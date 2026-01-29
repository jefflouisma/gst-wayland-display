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
use smithay::backend::allocator::{Format, Fourcc, Modifier};
use smithay::backend::renderer::{
    sync::SyncPoint, Bind, Color32F, ContextId, DebugFlags, Frame, ImportDma, ImportMem,
    Renderer, RendererSuper, Texture, TextureFilter,
};
use smithay::utils::{Buffer, Physical, Rectangle, Scale, Size, Transform};
use std::collections::{HashMap, HashSet};
use std::ffi::CStr;
use std::os::fd::AsRawFd;
use std::sync::Arc;
use tracing::{debug, error, info, trace, warn};

// ============================================================================
// Error Types
// ============================================================================

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
    /// Unsupported DRM format
    UnsupportedFormat(Fourcc),
    /// Rendering failed
    RenderError(String),
    /// No render target bound
    NoRenderTarget,
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
            VulkanError::UnsupportedFormat(f) => write!(f, "Unsupported format: {:?}", f),
            VulkanError::RenderError(e) => write!(f, "Render error: {}", e),
            VulkanError::NoRenderTarget => write!(f, "No render target bound"),
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

// ============================================================================
// Vulkan Texture
// ============================================================================

/// Internal Vulkan resources for a texture
struct VulkanTextureInner {
    image: vk::Image,
    memory: Vec<vk::DeviceMemory>, // One per plane for multi-planar formats
    view: vk::ImageView,
    sampler: vk::Sampler,
    size: Size<i32, Buffer>,
    format: Fourcc,
    owns_image: bool, // False for imported DMA-BUFs
}

/// Texture handle for Vulkan renderer
#[derive(Debug, Clone)]
pub struct VulkanTexture {
    id: u64,
    size: Size<i32, Buffer>,
    format: Fourcc,
}

impl Texture for VulkanTexture {
    fn width(&self) -> u32 {
        self.size.w as u32
    }

    fn height(&self) -> u32 {
        self.size.h as u32
    }

    fn format(&self) -> Option<Fourcc> {
        Some(self.format)
    }
}

/// Sync point for Vulkan operations
pub struct VulkanSyncPoint {
    fence: Option<vk::Fence>,
    device: Option<Device>,
}

impl std::fmt::Debug for VulkanSyncPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanSyncPoint")
            .field("fence", &self.fence)
            .finish()
    }
}

impl SyncPoint for VulkanSyncPoint {
    fn wait(&self) -> Result<(), ()> {
        if let (Some(fence), Some(device)) = (&self.fence, &self.device) {
            unsafe {
                device
                    .wait_for_fences(&[*fence], true, u64::MAX)
                    .map_err(|_| ())?;
            }
        }
        Ok(())
    }

    fn is_reached(&self) -> bool {
        if let (Some(fence), Some(device)) = (&self.fence, &self.device) {
            unsafe { device.get_fence_status(*fence).unwrap_or(false) }
        } else {
            true
        }
    }
}

// ============================================================================
// Render Target
// ============================================================================

/// A render target (framebuffer) for the Vulkan renderer
struct RenderTarget {
    image: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
    size: Size<i32, Physical>,
    format: vk::Format,
    owns_image: bool,
}

// ============================================================================
// Vulkan Frame
// ============================================================================

/// Frame for Vulkan rendering operations
pub struct VulkanFrame<'a, 'buffer> {
    renderer: &'a mut VulkanRenderer,
    size: Size<i32, Physical>,
    command_buffer: vk::CommandBuffer,
    _phantom: std::marker::PhantomData<&'buffer ()>,
}

impl<'a, 'buffer> Frame for VulkanFrame<'a, 'buffer> {
    type Error = VulkanError;
    type TextureId = VulkanTexture;

    fn context_id(&self) -> ContextId<Self::TextureId> {
        ContextId::new(std::any::TypeId::of::<VulkanRenderer>(), self.command_buffer.as_raw() as usize)
    }

    fn clear(&mut self, color: Color32F, at: &[Rectangle<i32, Physical>]) -> Result<(), Self::Error> {
        debug!("VulkanFrame::clear color={:?}", color);
        
        let clear_value = vk::ClearColorValue { 
            float32: [color.r, color.g, color.b, color.a] 
        };
        
        // If no specific regions, clear the whole target
        if at.is_empty() {
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);
            
            if let Some(target) = &self.renderer.current_target {
                unsafe {
                    self.renderer.device.cmd_clear_color_image(
                        self.command_buffer,
                        target.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &clear_value,
                        &[range],
                    );
                }
            }
        } else {
            // Clear specific regions
            for rect in at {
                let clear_rect = vk::ClearRect::default()
                    .rect(vk::Rect2D {
                        offset: vk::Offset2D { x: rect.loc.x, y: rect.loc.y },
                        extent: vk::Extent2D { 
                            width: rect.size.w as u32, 
                            height: rect.size.h as u32 
                        },
                    })
                    .base_array_layer(0)
                    .layer_count(1);
                
                let clear_attachment = vk::ClearAttachment::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .color_attachment(0)
                    .clear_value(vk::ClearValue { color: clear_value });
                
                unsafe {
                    self.renderer.device.cmd_clear_attachments(
                        self.command_buffer,
                        &[clear_attachment],
                        &[clear_rect],
                    );
                }
            }
        }
        
        Ok(())
    }

    fn draw_solid(
        &mut self,
        dst: Rectangle<i32, Physical>,
        _damage: &[Rectangle<i32, Physical>],
        color: Color32F,
    ) -> Result<(), Self::Error> {
        debug!("VulkanFrame::draw_solid dst={:?} color={:?}", dst, color);
        // TODO: Use push constants to set color and render a quad
        Ok(())
    }

    fn render_texture_from_to(
        &mut self,
        texture: &Self::TextureId,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        _damage: &[Rectangle<i32, Physical>],
        _opaque_regions: &[Rectangle<i32, Physical>],
        _transform: Transform,
        _alpha: f32,
    ) -> Result<(), Self::Error> {
        debug!(
            "VulkanFrame::render_texture_from_to id={} src={:?} dst={:?}",
            texture.id, src, dst
        );
        
        if let Some(_tex_inner) = self.renderer.textures.get(&texture.id) {
            // TODO: Bind texture, set up push constants for src/dst/transform/alpha, draw quad
        }
        
        Ok(())
    }

    fn transformation(&self) -> Transform {
        Transform::Normal
    }

    fn wait(&mut self, sync: &smithay::backend::renderer::sync::SyncPoint) -> Result<(), Self::Error> {
        sync.wait();
        Ok(())
    }

    fn finish(self) -> Result<smithay::backend::renderer::sync::SyncPoint, Self::Error> {
        debug!("VulkanFrame::finish - submitting command buffer");
        
        // End command buffer
        unsafe {
            self.renderer.device.end_command_buffer(self.command_buffer)?;
        }
        
        // Create fence for synchronization
        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { self.renderer.device.create_fence(&fence_info, None)? };
        
        // Submit command buffer
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&[self.command_buffer]);
        
        unsafe {
            self.renderer.device.queue_submit(
                self.renderer.queue,
                &[submit_info],
                fence,
            )?;
        }
        
        // Wait for completion and return signaled sync point
        unsafe {
            self.renderer.device.wait_for_fences(&[fence], true, u64::MAX).ok();
        }
        
        Ok(smithay::backend::renderer::sync::SyncPoint::signaled())
    }
}

// ============================================================================
// Extension and Format Constants
// ============================================================================

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

/// Map DRM fourcc to Vulkan format
fn fourcc_to_vk_format(fourcc: Fourcc) -> Option<vk::Format> {
    match fourcc {
        Fourcc::Argb8888 => Some(vk::Format::B8G8R8A8_UNORM),
        Fourcc::Xrgb8888 => Some(vk::Format::B8G8R8A8_UNORM),
        Fourcc::Abgr8888 => Some(vk::Format::R8G8B8A8_UNORM),
        Fourcc::Xbgr8888 => Some(vk::Format::R8G8B8A8_UNORM),
        Fourcc::Rgb888 => Some(vk::Format::R8G8B8_UNORM),
        Fourcc::Bgr888 => Some(vk::Format::B8G8R8_UNORM),
        Fourcc::Argb2101010 => Some(vk::Format::A2R10G10B10_UNORM_PACK32),
        Fourcc::Abgr2101010 => Some(vk::Format::A2B10G10R10_UNORM_PACK32),
        Fourcc::Nv12 => Some(vk::Format::G8_B8R8_2PLANE_420_UNORM),
        _ => None,
    }
}

// ============================================================================
// Vulkan Renderer
// ============================================================================

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
    
    // Textures
    textures: HashMap<u64, VulkanTextureInner>,
    next_texture_id: u64,
    
    // Render state
    current_target: Option<RenderTarget>,
    debug_flags: DebugFlags,
    
    // Pipeline resources
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    
    // Extension function pointers
    external_memory_fd: khr::external_memory_fd::Device,
    image_drm_format_modifier: ext::image_drm_format_modifier::Device,
    
    // Supported formats cache
    supported_formats: HashSet<Format>,
}

impl std::fmt::Debug for VulkanRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanRenderer")
            .field("physical_device", &self.physical_device)
            .field("next_texture_id", &self.next_texture_id)
            .field("texture_count", &self.textures.len())
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
        let image_drm_format_modifier = ext::image_drm_format_modifier::Device::new(&instance, &device);

        // Create command pool
        let pool_create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe { device.create_command_pool(&pool_create_info, None) }?;

        // Create render pass
        let render_pass = Self::create_render_pass(&device)?;
        
        // Create descriptor set layout
        let descriptor_set_layout = Self::create_descriptor_set_layout(&device)?;
        
        // Create pipeline layout
        let pipeline_layout = Self::create_pipeline_layout(&device, descriptor_set_layout)?;
        
        // Create graphics pipeline
        let pipeline = Self::create_pipeline(&device, render_pass, pipeline_layout)?;
        
        // Create descriptor pool
        let descriptor_pool = Self::create_descriptor_pool(&device)?;
        
        // Query supported formats
        let supported_formats = Self::query_supported_formats(&instance, physical_device);

        info!("Vulkan renderer initialized successfully with {} supported formats", 
              supported_formats.len());

        Ok(Self {
            _entry: entry,
            instance,
            physical_device,
            device,
            queue,
            queue_family_index,
            command_pool,
            textures: HashMap::new(),
            next_texture_id: 1,
            current_target: None,
            debug_flags: DebugFlags::empty(),
            render_pass,
            pipeline_layout,
            pipeline,
            descriptor_set_layout,
            descriptor_pool,
            external_memory_fd,
            image_drm_format_modifier,
            supported_formats,
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

    /// Create render pass for compositing
    fn create_render_pass(device: &Device) -> Result<vk::RenderPass, VulkanError> {
        let attachment = vk::AttachmentDescription::default()
            .format(vk::Format::B8G8R8A8_UNORM)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);

        let color_attachment_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_attachment_ref));

        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(std::slice::from_ref(&attachment))
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(std::slice::from_ref(&dependency));

        unsafe { device.create_render_pass(&render_pass_info, None).map_err(Into::into) }
    }

    /// Create descriptor set layout for texture sampling
    fn create_descriptor_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout, VulkanError> {
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(std::slice::from_ref(&binding));

        unsafe { device.create_descriptor_set_layout(&layout_info, None).map_err(Into::into) }
    }

    /// Create pipeline layout with push constants for transforms
    fn create_pipeline_layout(
        device: &Device,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> Result<vk::PipelineLayout, VulkanError> {
        // Push constants for: MVP matrix (64 bytes) + alpha (4 bytes)
        let push_constant = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(68);

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant));

        unsafe { device.create_pipeline_layout(&layout_info, None).map_err(Into::into) }
    }

    /// Create graphics pipeline for quad rendering
    fn create_pipeline(
        device: &Device,
        render_pass: vk::RenderPass,
        pipeline_layout: vk::PipelineLayout,
    ) -> Result<vk::Pipeline, VulkanError> {
        // Embedded SPIR-V shaders (fallback if files not compiled)
        // These are minimal shaders for basic quad rendering
        static QUAD_VERT_SPV: &[u8] = include_bytes!("shaders/quad.vert.spv");
        static QUAD_FRAG_SPV: &[u8] = include_bytes!("shaders/quad.frag.spv");

        // Create shader modules
        let vert_module = Self::create_shader_module(device, QUAD_VERT_SPV)?;
        let frag_module = Self::create_shader_module(device, QUAD_FRAG_SPV)?;

        let entry_name = c"main";

        let vert_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(entry_name);

        let frag_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(entry_name);

        let shader_stages = [vert_stage, frag_stage];

        // Vertex input - empty, using gl_VertexIndex for fullscreen quad
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(1920.0)
            .height(1080.0)
            .min_depth(0.0)
            .max_depth(1.0);

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width: 1920, height: 1080 },
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(std::slice::from_ref(&viewport))
            .scissors(std::slice::from_ref(&scissor));

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&color_blend_attachment));

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&dynamic_states);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let pipelines = unsafe {
            device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            )
        }.map_err(|(_, e)| e)?;

        // Cleanup shader modules
        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
        }

        info!("Graphics pipeline created successfully");
        Ok(pipelines[0])
    }

    /// Create a shader module from SPIR-V bytecode
    fn create_shader_module(device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule, VulkanError> {
        // SPIR-V requires 4-byte alignment
        let code: Vec<u32> = bytecode
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let create_info = vk::ShaderModuleCreateInfo::default().code(&code);

        unsafe { device.create_shader_module(&create_info, None).map_err(Into::into) }
    }

    /// Create descriptor pool for texture bindings
    fn create_descriptor_pool(device: &Device) -> Result<vk::DescriptorPool, VulkanError> {
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(100); // Support up to 100 textures

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(std::slice::from_ref(&pool_size))
            .max_sets(100);

        unsafe { device.create_descriptor_pool(&pool_info, None).map_err(Into::into) }
    }

    /// Query supported DRM formats from the device
    fn query_supported_formats(instance: &Instance, physical_device: vk::PhysicalDevice) -> HashSet<Format> {
        let mut formats = HashSet::new();
        
        // Common formats that most Vulkan implementations support
        let common_fourccs = [
            Fourcc::Argb8888,
            Fourcc::Xrgb8888,
            Fourcc::Abgr8888,
            Fourcc::Xbgr8888,
        ];
        
        for fourcc in common_fourccs {
            if let Some(vk_format) = fourcc_to_vk_format(fourcc) {
                let props = unsafe {
                    instance.get_physical_device_format_properties(physical_device, vk_format)
                };
                
                // Check if format supports color attachment and sampling
                if props.optimal_tiling_features.contains(
                    vk::FormatFeatureFlags::COLOR_ATTACHMENT | vk::FormatFeatureFlags::SAMPLED_IMAGE
                ) {
                    formats.insert(Format {
                        code: fourcc,
                        modifier: Modifier::Linear,
                    });
                    formats.insert(Format {
                        code: fourcc,
                        modifier: Modifier::Invalid, // Driver-preferred
                    });
                }
            }
        }
        
        formats
    }

    /// Import a DMA-BUF as a Vulkan texture
    fn import_dmabuf_internal(&mut self, dmabuf: &Dmabuf) -> Result<VulkanTexture, VulkanError> {
        let id = self.next_texture_id;
        self.next_texture_id += 1;

        let width = dmabuf.width() as u32;
        let height = dmabuf.height() as u32;
        let format = dmabuf.format();
        
        let vk_format = fourcc_to_vk_format(format.code)
            .ok_or(VulkanError::UnsupportedFormat(format.code))?;
        
        debug!(
            "Importing DMA-BUF: {}x{}, format={:?}, modifier={:?}, planes={}",
            width, height, format.code, format.modifier, dmabuf.num_planes()
        );

        // Create image with external memory
        let mut external_memory_info = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external_memory_info);

        let image = unsafe { self.device.create_image(&image_info, None)? };

        // Get memory requirements
        let mem_reqs = unsafe { self.device.get_image_memory_requirements(image) };

        // Import DMA-BUF fd as memory
        let fd = dmabuf.handles().next()
            .ok_or_else(|| VulkanError::DmabufImport("No file descriptor".into()))?;
        
        let mut import_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(fd.as_raw_fd());

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(self.find_memory_type(
                mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?)
            .push_next(&mut import_info);

        let memory = unsafe { self.device.allocate_memory(&alloc_info, None)? };
        
        // Bind memory to image
        unsafe { self.device.bind_image_memory(image, memory, 0)? };

        // Create image view
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk_format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = unsafe { self.device.create_image_view(&view_info, None)? };

        // Create sampler
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);

        let sampler = unsafe { self.device.create_sampler(&sampler_info, None)? };

        // Store texture
        self.textures.insert(id, VulkanTextureInner {
            image,
            memory: vec![memory],
            view,
            sampler,
            size: Size::from((width as i32, height as i32)),
            format: format.code,
            owns_image: false, // DMA-BUF import
        });

        debug!("DMA-BUF imported as texture id={}", id);

        Ok(VulkanTexture {
            id,
            size: Size::from((width as i32, height as i32)),
            format: format.code,
        })
    }

    /// Find suitable memory type
    fn find_memory_type(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<u32, VulkanError> {
        let mem_props = unsafe {
            self.instance.get_physical_device_memory_properties(self.physical_device)
        };

        for i in 0..mem_props.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && mem_props.memory_types[i as usize].property_flags.contains(properties)
            {
                return Ok(i);
            }
        }

        Err(VulkanError::NoDevice)
    }

    /// Allocate a command buffer
    fn allocate_command_buffer(&self) -> Result<vk::CommandBuffer, VulkanError> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let buffers = unsafe { self.device.allocate_command_buffers(&alloc_info)? };
        Ok(buffers[0])
    }

    /// Begin a command buffer
    fn begin_command_buffer(&self, cmd: vk::CommandBuffer) -> Result<(), VulkanError> {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { self.device.begin_command_buffer(cmd, &begin_info)? };
        Ok(())
    }

    /// Create render target from DMA-BUF for rendering into
    fn create_render_target(&mut self, dmabuf: &Dmabuf) -> Result<RenderTarget, VulkanError> {
        let width = dmabuf.width() as u32;
        let height = dmabuf.height() as u32;
        let format = dmabuf.format();
        
        let vk_format = fourcc_to_vk_format(format.code)
            .ok_or(VulkanError::UnsupportedFormat(format.code))?;

        // Create image with external memory for render target
        let mut external_memory_info = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external_memory_info);

        let image = unsafe { self.device.create_image(&image_info, None)? };

        // Import DMA-BUF memory
        let mem_reqs = unsafe { self.device.get_image_memory_requirements(image) };
        let fd = dmabuf.handles().next()
            .ok_or_else(|| VulkanError::DmabufImport("No file descriptor".into()))?;
        
        let mut import_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(fd.as_raw_fd());

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(self.find_memory_type(
                mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?)
            .push_next(&mut import_info);

        let memory = unsafe { self.device.allocate_memory(&alloc_info, None)? };
        unsafe { self.device.bind_image_memory(image, memory, 0)? };

        // Create image view for framebuffer
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk_format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = unsafe { self.device.create_image_view(&view_info, None)? };

        Ok(RenderTarget {
            image,
            view,
            memory,
            size: Size::from((width as i32, height as i32)),
            format: vk_format,
            owns_image: false,
        })
    }

    fn allocate_texture_id(&mut self) -> u64 {
        let id = self.next_texture_id;
        self.next_texture_id += 1;
        id
    }

    /// Destroy a texture and free its resources
    fn destroy_texture(&mut self, id: u64) {
        if let Some(tex) = self.textures.remove(&id) {
            unsafe {
                self.device.destroy_sampler(tex.sampler, None);
                self.device.destroy_image_view(tex.view, None);
                if tex.owns_image {
                    self.device.destroy_image(tex.image, None);
                }
                for mem in tex.memory {
                    self.device.free_memory(mem, None);
                }
            }
        }
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            
            // Destroy textures
            let texture_ids: Vec<_> = self.textures.keys().copied().collect();
            for id in texture_ids {
                self.destroy_texture(id);
            }
            
            // Destroy render target
            if let Some(target) = self.current_target.take() {
                self.device.destroy_image_view(target.view, None);
                if target.owns_image {
                    self.device.destroy_image(target.image, None);
                }
                self.device.free_memory(target.memory, None);
            }
            
            // Destroy pipeline resources
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            if self.pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.pipeline, None);
            }
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
            
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
        info!("Vulkan renderer destroyed");
    }
}

// ============================================================================
// Framebuffer wrapper for Vulkan render target
// ============================================================================

/// Vulkan framebuffer wrapper that implements Texture
pub struct VulkanFramebuffer {
    size: Size<i32, Physical>,
    format: vk::Format,
}

impl std::fmt::Debug for VulkanFramebuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanFramebuffer")
            .field("size", &self.size)
            .finish()
    }
}

impl Texture for VulkanFramebuffer {
    fn width(&self) -> u32 {
        self.size.w as u32
    }

    fn height(&self) -> u32 {
        self.size.h as u32
    }

    fn format(&self) -> Option<Fourcc> {
        // Map back from Vulkan format to DRM fourcc
        match self.format {
            vk::Format::B8G8R8A8_UNORM => Some(Fourcc::Argb8888),
            vk::Format::R8G8B8A8_UNORM => Some(Fourcc::Abgr8888),
            _ => None,
        }
    }
}

// ============================================================================
// Smithay Renderer Trait Implementation
// ============================================================================

impl RendererSuper for VulkanRenderer {
    type Error = VulkanError;
    type TextureId = VulkanTexture;
    type Framebuffer<'buffer> = VulkanFramebuffer;
    type Frame<'frame, 'buffer> = VulkanFrame<'frame, 'buffer> where 'buffer: 'frame, Self: 'frame;
}

impl Renderer for VulkanRenderer {
    fn context_id(&self) -> ContextId<Self::TextureId> {
        ContextId::new(std::any::TypeId::of::<VulkanRenderer>(), self as *const Self as usize)
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

    fn render<'frame, 'buffer>(
        &'frame mut self,
        framebuffer: &'frame mut Self::Framebuffer<'buffer>,
        output_size: Size<i32, Physical>,
        _dst_transform: Transform,
    ) -> Result<Self::Frame<'frame, 'buffer>, Self::Error>
    where
        'buffer: 'frame,
    {
        debug!("VulkanRenderer::render size={:?}", output_size);
        
        // Allocate and begin command buffer
        let command_buffer = self.allocate_command_buffer()?;
        self.begin_command_buffer(command_buffer)?;
        
        Ok(VulkanFrame {
            renderer: self,
            size: output_size,
            command_buffer,
            _phantom: std::marker::PhantomData,
        })
    }

    fn wait(&mut self, sync: &smithay::backend::renderer::sync::SyncPoint) -> Result<(), Self::Error> {
        sync.wait();
        Ok(())
    }
}

impl Bind<Dmabuf> for VulkanRenderer {
    fn bind<'a>(&mut self, target: &'a mut Dmabuf) -> Result<Self::Framebuffer<'a>, VulkanError> {
        debug!("VulkanRenderer::bind DMA-BUF {}x{}", target.width(), target.height());
        
        // Create render target from DMA-BUF
        let render_target = self.create_render_target(target)?;
        let format = render_target.format;
        let size = render_target.size;
        self.current_target = Some(render_target);
        
        Ok(VulkanFramebuffer { size, format })
    }

    fn supported_formats(&self) -> Option<smithay::backend::renderer::FormatSet> {
        Some(smithay::backend::renderer::FormatSet::from_iter(self.supported_formats.clone()))
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

    fn dmabuf_formats(&self) -> Box<dyn Iterator<Item = Format>> {
        Box::new(self.supported_formats.clone().into_iter())
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
        
        let vk_format = fourcc_to_vk_format(format)
            .ok_or(VulkanError::UnsupportedFormat(format))?;
        
        let id = self.allocate_texture_id();
        let width = size.w as u32;
        let height = size.h as u32;

        // Create image
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { self.device.create_image(&image_info, None)? };

        // Allocate memory
        let mem_reqs = unsafe { self.device.get_image_memory_requirements(image) };
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(self.find_memory_type(
                mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?);

        let memory = unsafe { self.device.allocate_memory(&alloc_info, None)? };
        unsafe { self.device.bind_image_memory(image, memory, 0)? };

        // Create staging buffer and copy data
        let staging_size = data.len() as vk::DeviceSize;
        let staging_info = vk::BufferCreateInfo::default()
            .size(staging_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let staging_buffer = unsafe { self.device.create_buffer(&staging_info, None)? };
        let staging_reqs = unsafe { self.device.get_buffer_memory_requirements(staging_buffer) };
        
        let staging_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(staging_reqs.size)
            .memory_type_index(self.find_memory_type(
                staging_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?);

        let staging_memory = unsafe { self.device.allocate_memory(&staging_alloc, None)? };
        unsafe { self.device.bind_buffer_memory(staging_buffer, staging_memory, 0)? };

        // Copy data to staging buffer
        unsafe {
            let ptr = self.device.map_memory(staging_memory, 0, staging_size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            self.device.unmap_memory(staging_memory);
        }

        // Record copy commands
        let cmd = self.allocate_command_buffer()?;
        self.begin_command_buffer(cmd)?;

        // Transition image to transfer dst
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);

        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }

        // Copy buffer to image
        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D { width, height, depth: 1 });

        unsafe {
            self.device.cmd_copy_buffer_to_image(
                cmd,
                staging_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }

        // Transition to shader read
        let barrier2 = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);

        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier2],
            );
            
            self.device.end_command_buffer(cmd)?;
        }

        // Submit and wait
        let submit_info = vk::SubmitInfo::default().command_buffers(&[cmd]);
        unsafe {
            self.device.queue_submit(self.queue, &[submit_info], vk::Fence::null())?;
            self.device.queue_wait_idle(self.queue)?;
        }

        // Cleanup staging
        unsafe {
            self.device.destroy_buffer(staging_buffer, None);
            self.device.free_memory(staging_memory, None);
        }

        // Create image view
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk_format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = unsafe { self.device.create_image_view(&view_info, None)? };

        // Create sampler
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);

        let sampler = unsafe { self.device.create_sampler(&sampler_info, None)? };

        // Store texture
        self.textures.insert(id, VulkanTextureInner {
            image,
            memory: vec![memory],
            view,
            sampler,
            size,
            format,
            owns_image: true,
        });

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
        // TODO: Similar to import_memory but only update region
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vulkan_renderer_creation() {
        match VulkanRenderer::new() {
            Ok(renderer) => {
                println!("Created Vulkan renderer with {} supported formats", 
                         renderer.supported_formats.len());
            }
            Err(e) => {
                println!("Could not create Vulkan renderer (expected if no Vulkan): {}", e);
            }
        }
    }
    
    #[test]
    fn test_fourcc_to_vk_format() {
        assert_eq!(fourcc_to_vk_format(Fourcc::Argb8888), Some(vk::Format::B8G8R8A8_UNORM));
        assert_eq!(fourcc_to_vk_format(Fourcc::Abgr8888), Some(vk::Format::R8G8B8A8_UNORM));
        assert_eq!(fourcc_to_vk_format(Fourcc::Nv12), Some(vk::Format::G8_B8R8_2PLANE_420_UNORM));
    }
}
