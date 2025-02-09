use std::sync::Arc;

use image::{ImageBuffer, Rgba};
use vulkano::buffer::{BufferContents, BufferUsage};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo, RenderPassBeginInfo,
    SubpassBeginInfo, SubpassContents, SubpassEndInfo,
};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::ImageUsage;
use vulkano::memory::allocator::{MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, Subpass};
use vulkano::sync::{self, GpuFuture};

use crate::util::{self, create_buffer};

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 460

            layout(location = 0) in vec2 position;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 460

            layout(location = 0) out vec4 f_color;

            void main() {
                f_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        ",
    }
}

pub fn graphics_pipeline() {
    let mut vk_device = util::create_device();
    let queue = vk_device.queues.next().unwrap();
    let device = vk_device.device.clone();
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let image = util::create_image(
        &memory_allocator,
        ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
        MemoryTypeFilter::PREFER_DEVICE,
    );

    let buf = util::create_buffer(
        (0..1024 * 1024 * 4).map(|_| 0u8),
        &memory_allocator,
        BufferUsage::TRANSFER_DST,
        MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
    );

    let vertex1 = MyVertex {
        position: [-0.5, -0.5],
    };
    let vertex2 = MyVertex {
        position: [0.0, 0.5],
    };
    let vertex3 = MyVertex {
        position: [0.5, -0.25],
    };

    let vertex_buffer = create_buffer(
        vec![vertex1, vertex2, vertex3],
        &memory_allocator,
        BufferUsage::VERTEX_BUFFER,
        MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
    );

    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                format: Format::R8G8B8A8_UNORM,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    )
    .unwrap();

    let view = ImageView::new_default(image.clone()).unwrap();
    let framebuffer = Framebuffer::new(
        render_pass.clone(),
        FramebufferCreateInfo {
            attachments: vec![view],
            ..Default::default()
        },
    )
    .unwrap();

    let vs = vs::load(device.clone()).expect("failed to create shader module");
    let fs = fs::load(device.clone()).expect("failed to create shader module");

    let viewport = Viewport {
        offset: [0.0, 0.0],
        extent: [1024.0, 1024.0],
        depth_range: 0.0..=1.0,
    };
    let pipeline = {
        let vs = vs.entry_point("main").unwrap();
        let fs = fs.entry_point("main").unwrap();

        let vertex_input_state = MyVertex::per_vertex()
            .definition(&vs.info().input_interface)
            .unwrap();

        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                // The stages of our pipeline, we have vertex and fragment stages.
                stages: stages.into_iter().collect(),
                // Describes the layout of the vertex input and how should it behave.
                vertex_input_state: Some(vertex_input_state),
                // Indicate the type of the primitives (the default is a list of triangles).
                input_assembly_state: Some(InputAssemblyState::default()),
                // Set the fixed viewport.
                viewport_state: Some(ViewportState {
                    viewports: [viewport].into_iter().collect(),
                    ..Default::default()
                }),
                // Ignore these for now.
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                // This graphics pipeline object concerns the first pass of the render pass.
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    };

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
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
            },
            SubpassBeginInfo {
                contents: SubpassContents::Inline,
                ..Default::default()
            },
        )
        .unwrap()
        .bind_pipeline_graphics(pipeline.clone())
        .unwrap()
        .bind_vertex_buffers(0, vertex_buffer.clone())
        .unwrap()
        .draw(
            3, 1, 0, 0, // 3 is the number of vertices, 1 is the number of instances
        )
        .unwrap()
        .end_render_pass(SubpassEndInfo::default())
        .unwrap()
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(image, buf.clone()))
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
    image.save("vertex_image.png").unwrap();

    println!("Graphics pipeline successful!");
}
