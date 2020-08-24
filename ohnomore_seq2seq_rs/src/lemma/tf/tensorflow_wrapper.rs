use super::tf_config::config::ConfigProto;
use protobuf::Message;
use std::fs::File;
use std::io::Read;
use tensorflow::{
    Graph, ImportGraphDefOptions, Operation, Session, SessionOptions, SessionRunArgs, Tensor,
};

use util::Ops;

/// Wrapper struct that holds the Tensorflow session and performs steps with the computation graph.
pub struct TensorflowWrapper {
    session: Session,
    input_op: Operation,
    input_length_op: Operation,
    start_token_op: Operation,
    morph_op: Operation,
    length_op: Operation,
    output_op: Operation,
}

impl TensorflowWrapper {
    /// Construct a new TensorflowWrapper.
    ///
    /// * `model_path` is the location of the frozen graph in binary protobuf format.
    /// * `intra_op_threads` defines the amount of intra operation threads in Tensorflow.
    /// * `inter_op_threads` defines the amount of inter operation threads in Tensorflow.
    /// * `ops` struct holding the op names in the graph.
    ///
    pub fn new(
        model_path: &str,
        intra_op_threads: usize,
        inter_op_threads: usize,
        ops: &Ops,
    ) -> Self {
        let mut graph = Graph::new();
        let mut proto = Vec::new();

        let input_op = &ops.input.input;
        let input_length_op = &ops.input.input_lengths;
        let pos_op = &ops.input.pos;
        let morph_op = &ops.input.morph;

        let output_op = &ops.output.chars;
        let length_op = &ops.output.seq_length;

        let table_init_op = &ops.init_ops.table;

        File::open(model_path)
            .expect("Didn't find file")
            .read_to_end(&mut proto)
            .expect("IO Error");
        graph
            .import_graph_def(&proto, &ImportGraphDefOptions::new())
            .expect("Error while importing the graph.");

        let mut so = SessionOptions::new();

        let mut cp = ConfigProto::new();
        cp.inter_op_parallelism_threads = inter_op_threads as i32;
        cp.intra_op_parallelism_threads = intra_op_threads as i32;

        so.set_config(&cp.write_to_bytes().unwrap()).expect("wat");

        let mut session = Session::new(&so, &graph).expect("Error while creating the session.");

        let table_init = graph
            .operation_by_name_required(table_init_op)
            .expect("Didn't find init tables op.");
        let mut step = SessionRunArgs::new();
        step.add_target(&table_init);
        session.run(&mut step).expect("Couldn't initialize tables");

        let input_op = graph
            .operation_by_name_required(input_op)
            .expect("Didn't find input op");
        let input_length_op = graph
            .operation_by_name_required(input_length_op)
            .expect("Didn't find input length op");
        let start_token_op = graph
            .operation_by_name_required(pos_op)
            .expect("Didn't find start op");
        let morph_op = graph
            .operation_by_name_required(morph_op)
            .expect("Didn't find start op");
        let output_op = graph
            .operation_by_name_required(output_op)
            .expect("Didn't find output op");
        let length_op = graph
            .operation_by_name_required(length_op)
            .expect("Didn't find output op");

        TensorflowWrapper {
            session,
            input_op,
            input_length_op,
            start_token_op,
            morph_op,
            length_op,
            output_op,
        }
    }
    /// Performs a step with the graph.
    ///
    /// forms, pos and morph tags as Tensor<String> and input lenghts as Tensor<i32>.
    ///
    /// Returns a tuple of tensors (Predictions : Tensor<String>, Sequence Lengths :Tensor<i32>).
    pub fn step(
        &mut self,
        input: &Tensor<String>,
        input_length: &Tensor<i32>,
        pos: &Tensor<String>,
        morph: &Tensor<String>,
    ) -> (Tensor<String>, Tensor<i32>) {
        let mut step = SessionRunArgs::new();
        step.add_feed(&self.input_op, 0, input);
        step.add_feed(&self.input_length_op, 0, input_length);
        step.add_feed(&self.start_token_op, 0, pos);
        step.add_feed(&self.morph_op, 0, morph);

        let outp = step.request_fetch(&self.output_op, 0);
        let lens = step.request_fetch(&self.length_op, 0);

        self.session
            .run(&mut step)
            .expect("Stepping the graph failed.");
        (
            step.fetch(outp)
                .expect("Fetching the lemmas from Tensorflow failed."),
            step.fetch(lens)
                .expect("Fetching the lemma lengths from Tensorflow failed."),
        )
    }
}
