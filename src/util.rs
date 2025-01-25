use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{
        AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryTypeFilter,
    },
    VulkanLibrary,
};

pub struct VkDevice<T> {
    pub device: Arc<Device>,
    pub queues: T,
    pub queue_index: u32,
}

pub fn create_device() -> VkDevice<impl ExactSizeIterator<Item = Arc<Queue>>> {
    let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
    .expect("failed to create instance");
    let physical_device = instance
        .enumerate_physical_devices()
        .expect("could not enumerate devices")
        .next()
        .expect("no devices available");

    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_queue_family_index, queue_family_properties)| {
            queue_family_properties
                .queue_flags
                .contains(QueueFlags::GRAPHICS)
        })
        .expect("couldn't find a graphical queue family") as u32;

    let (device, queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            // here we pass the desired queue family to use by index
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .expect("failed to create device");

    VkDevice {
        device,
        queues,
        queue_index: queue_family_index,
    }
}

pub fn create_buffer(
    source: Vec<i32>,
    allocator: &Arc<GenericMemoryAllocator<FreeListAllocator>>,
    usage: BufferUsage,
    type_filter: MemoryTypeFilter,
) -> vulkano::buffer::Subbuffer<[i32]> {
    Buffer::from_iter(
        allocator.clone(),
        BufferCreateInfo {
            usage,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: type_filter,
            ..Default::default()
        },
        source,
    )
    .expect("failed to create buffer")
}
