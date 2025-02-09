use std::sync::Arc;

use image::{ImageBuffer, Rgba};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, ClearColorImageInfo, CommandBufferUsage, CopyImageToBufferInfo,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    format::{ClearColorValue, Format},
    image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{Pipeline, PipelineBindPoint},
    sync::{self, GpuFuture},
};

use crate::util;

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
        #version 460

        layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

        layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;

        void main() {
            vec2 norm_coordinates = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(img));
            vec2 c = (norm_coordinates - vec2(0.5)) * 2.0 - vec2(1.0, 0.0);

            vec2 z = vec2(0.0, 0.0);
            float i;
            for (i = 0.0; i < 1.0; i += 0.005) {
                z = vec2(
                    z.x * z.x - z.y * z.y + c.x,
                    z.y * z.x + z.x * z.y + c.y
                );

                if (length(z) > 4.0) {
                    break;
                }
            }

            vec4 to_write = vec4(vec3(i), 1.0);
            imageStore(img, ivec2(gl_GlobalInvocationID.xy), to_write);
        }

        ",
    }
}

pub fn using_images() {
    let mut vk_device = util::create_device();
    let device = vk_device.device;
    let queue = vk_device.queues.next().unwrap();
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    let image = util::create_image(
        &memory_allocator,
        ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC,
        MemoryTypeFilter::PREFER_DEVICE,
    );

    let buf = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        (0..1024 * 1024 * 4).map(|_| 0u8),
    )
    .expect("failed to create buffer");

    builder
        .clear_color_image(ClearColorImageInfo {
            clear_value: ClearColorValue::Float([0.0, 0.0, 1.0, 1.0]),
            ..ClearColorImageInfo::image(image.clone())
        })
        .unwrap()
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            image.clone(),
            buf.clone(),
        ))
        .unwrap();

    let command_buffer = builder.build().unwrap();

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let buffer_content = buf.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();

    image.save("clear_image.png").unwrap();

    println!("Clear image creation successful");
}

pub fn mandelbrot_set() {
    let mut vk_device = util::create_device();
    let device = vk_device.device;
    let queue = vk_device.queues.next().unwrap();

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_UNORM,
            extent: [1024, 1024, 1],
            usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
    .unwrap();

    let view = ImageView::new_default(image.clone()).unwrap();

    let shader = cs::load(device.clone()).expect("failed to create shader module");

    let compute_pipeline = util::create_compute_pipeline(device.clone(), shader);

    let buf = util::create_buffer(
        (0..1024 * 1024 * 4).map(|_| 0u8),
        &memory_allocator,
        BufferUsage::TRANSFER_DST,
        MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
    );

    let descriptor_set_allocator =
        StandardDescriptorSetAllocator::new(device.clone(), Default::default());

    let layout = compute_pipeline.layout().set_layouts().get(0).unwrap();
    let set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        layout.clone(),
        [WriteDescriptorSet::image_view(0, view.clone())], // 0 is the binding
        [],
    )
    .unwrap();

    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            0,
            set,
        )
        .unwrap()
        .dispatch([1024 / 8, 1024 / 8, 1])
        .unwrap()
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            image.clone(),
            buf.clone(),
        ))
        .unwrap();

    let command_buffer = builder.build().unwrap();
    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let buffer_content = buf.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    image.save("mandelbrot.png").unwrap();

    println!("Mandelbrot set creatio successful!");
}
