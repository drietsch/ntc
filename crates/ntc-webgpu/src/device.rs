//! Adapter/device acquisition and a capability probe.
//!
//! V1 kernels are portable f32 with default limits — nothing here *requires*
//! `shader-f16`; the probe only records whether the adapter offers it (for a
//! future f16 weight path).

use ntc_core::NtcError;

/// What the selected adapter offers (recorded at context creation; V1 does
/// not require any of the optional features).
#[derive(Debug, Clone)]
pub struct GpuCaps {
    pub adapter_name: String,
    /// Backend the adapter runs on (Metal, Vulkan, Dx12, Gl, BrowserWebGpu).
    pub backend: wgpu::Backend,
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u32,
    /// Whether the adapter supports the `shader-f16` feature. Informational
    /// only in V1 — the device is requested WITHOUT it.
    pub shader_f16: bool,
}

/// An acquired wgpu device + queue with its capability probe.
pub struct WgpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub caps: GpuCaps,
}

impl WgpuContext {
    /// Acquire an adapter from the default instance (all enabled backends)
    /// and request a device with default (downlevel-friendly) limits and no
    /// optional features.
    pub async fn new() -> Result<Self, NtcError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok_or_else(|| NtcError::Inference("no wgpu adapter available".into()))?;

        let info = adapter.get_info();
        let limits = adapter.limits();
        let caps = GpuCaps {
            adapter_name: info.name.clone(),
            backend: info.backend,
            max_buffer_size: limits.max_buffer_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            shader_f16: adapter.features().contains(wgpu::Features::SHADER_F16),
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("ntc-webgpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| NtcError::Inference(format!("wgpu device request failed: {e}")))?;

        Ok(Self {
            device,
            queue,
            caps,
        })
    }
}
