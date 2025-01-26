mod lessons;
mod util;

use lessons::buffer_creation::buffer_creation;
use lessons::compute_pipeline::compute_pipeline;
use lessons::using_images::using_images;

fn main() {
    buffer_creation();
    compute_pipeline();
    using_images();
}
