//! A library providing an interface to Sequence2Sequence lemmatization in Rust.
//!
//! Example usage:
//!
//! ```
//! extern crate stdinout;
//! extern crate conllx;
//! extern crate ohnomore_seq2seq_rs;
//!
//! use conllx::ReadSentence;
//! use ohnomore_seq2seq_rs::Config;
//! use ohnomore_seq2seq_rs::{Lemmatizer,Lemmatize};
//! use stdinout::OrExit;
//!
//! // set up input reader
//! let input = stdinout::Input::from(Some("testdata/wiki-sample.conll"));
//! let mut reader = conllx::Reader::new(input.buf_read().or_exit("Cannot open input file", 1));
//!
//! // read the inference_config that accompanies the frozen graph file
//! let config = Config::from("testdata/inference_config.toml");
//!
//! // create a lemmatizer struct with the path to the frozen graph, batch size and number of intra
//! // and inter op parallelism in tensorflow.
//! let mut lemmatizer = Lemmatizer::new(config,"testdata/ohnomore_test.pb",30,0,0);
//!
//! // lemmatize the sentences
//! while let Some(sentence) = reader.read_sentence().or_exit("Reading sentence failed!",1) {
//!     let sent: Vec<conllx::Token> = lemmatizer
//!         .lemmatize_batch(&sentence)
//!         .zip(sentence)
//!         .map(|(lemma, mut token)| {
//!             token.set_lemma(Some(lemma));
//!             token
//!         }).collect();
//!     println!("{:#?}",sent);
//! }
//! ```

extern crate conllx;
extern crate fasthash;
extern crate stdinout;
extern crate tensorflow;
extern crate toml;
#[macro_use]
extern crate serde;
extern crate core;
extern crate protobuf;

mod lemma;
pub use lemma::{LemmaFeatures, Lemmatize, Lemmatizer};

mod util;
pub use util::Config;
