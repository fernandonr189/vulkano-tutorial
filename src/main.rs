mod lessons;
mod util;

use lessons::buffer_creation::buffer_creation;
use lessons::compute_pipeline::compute_pipeline;

fn main() {
    buffer_creation();
    compute_pipeline();
}
