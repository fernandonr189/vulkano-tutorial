use std::sync::Arc;

use vulkano::{
    buffer::BufferUsage,
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo,
    },
    memory::allocator::{MemoryTypeFilter, StandardMemoryAllocator},
    sync::{self, GpuFuture},
};

mod util;

fn main() {
    let mut vk_device = util::create_device();
    let queue = vk_device.queues.next().unwrap();
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(
        vk_device.device.clone(),
    ));

    let source_buffer = util::create_buffer(
        (0..64).collect(),
        &memory_allocator,
        BufferUsage::TRANSFER_SRC,
        MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
    );

    let destination_buffer = util::create_buffer(
        (0..64).map(|_| 0).collect(),
        &memory_allocator,
        BufferUsage::TRANSFER_DST,
        MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
    );

    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        vk_device.device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        vk_device.queue_index,
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    builder
        .copy_buffer(CopyBufferInfo::buffers(
            source_buffer.clone(),
            destination_buffer.clone(),
        ))
        .unwrap();

    let command_buffer = builder.build().unwrap();

    let future = sync::now(vk_device.device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let src_content = source_buffer.read().unwrap();
    let destination_content = destination_buffer.read().unwrap();
    assert_eq!(&*src_content, &*destination_content);

    println!("Everything succeeded!");
}
