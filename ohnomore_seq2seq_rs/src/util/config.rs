use std::collections::HashSet;
use std::fs::read_to_string;
use std::path::Path;
use stdinout::OrExit;
use toml;

/// Struct to hold parameters and meta information of the trained model in python.
///
/// Saved by `train.py` alongside `<model>.pb`.
#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct Config {
    pub max_morph_tags: usize,
    pub morph_feats: HashSet<String>,
    pad_symbol: String,
    pub ops: Ops,
}

impl Config {
    pub fn pad_symbol(&self) -> &str {
        self.pad_symbol.as_ref()
    }
}

/// Relevant ops of the Tensorflow graph.
#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct Ops {
    pub input: Input,
    pub output: Output,
    pub init_ops: InitOps,
}

/// Input ops of the Tensorflow graph.
#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct Input {
    pub input: String,
    pub input_lengths: String,
    pub morph: String,
    pub pos: String,
}

/// Output ops of the Tensorflow graph.
#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct Output {
    pub ids: String,
    pub chars: String,
    pub seq_length: String,
}

/// Init ops of the Tensorflow graph. Table inits etc.
#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct InitOps {
    pub table: String,
}

impl<P: AsRef<Path>> From<P> for Config {
    /// Deserializes config from path.
    fn from(path: P) -> Self {
        let data = read_to_string(path).or_exit("Reading the config failed.", 1);
        toml::from_str(&data).or_exit("Parsing the config failed!", 1)
    }
}

#[cfg(test)]
mod tests {
    use super::{Config, InitOps, Input, Ops, Output};
    use std::collections::HashSet;

    #[test]
    pub fn test_config() {
        let target = Config {
            max_morph_tags: 5,
            pad_symbol: "<PAD>".to_string(),
            morph_feats: ["A:A", "B:B", "A:B", "B:B", "B:A"]
                .iter()
                .map(|x| String::from(*x))
                .collect(),
            ops: Ops {
                input: Input {
                    input: "pred/model/inputs".to_string(),
                    input_lengths: "pred/model/input_lengths".to_string(),
                    morph: "pred/model/morph_input".to_string(),
                    pos: "pred/model/pos_input".to_string(),
                },
                output: Output {
                    ids: "pred/model/decoder/decoder/transpose_1".to_string(),
                    chars: "pred/model/hash_table_Lookup".to_string(),
                    seq_length: "pred/model/decoder/decoder/while/Exit_13".to_string(),
                },
                init_ops: InitOps {
                    table: "init_tables".to_string(),
                },
            },
        };
        let config: Config = Config::from("testdata/config.toml");
        assert_eq!(target, config);
    }
}
