//! Sequence2Sequence lemmatization in rust.
extern crate clap;
extern crate conllx;
extern crate indicatif;
extern crate lru_cache;
extern crate stdinout;

use stdinout::OrExit;

use clap::{App, AppSettings, Arg, ArgMatches};

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
];

extern crate ohnomore_seq2seq_rs;
use ohnomore_seq2seq_rs::{Config, LemmaFeatures, Lemmatize, Lemmatizer};

fn main() {
    let parsed = args();

    let config = Config::from(parsed.value_of("CONFIG").unwrap());

    let model_path = parsed.value_of("MODEL").unwrap();

    let input = stdinout::Input::from(parsed.value_of("INPUT"));
    let reader = conllx::Reader::new(input.buf_read().or_exit("Cannot open input file", 1));
    let output = stdinout::Output::from(parsed.value_of("OUTPUT"));

    let out = conllx::Writer::new(output.write().or_exit("Couldn't find output", 1));

    let batch_size = parse_usize_arg(&parsed, &"BATCH_SIZE");
    let cache_size = parse_usize_arg(&parsed, "CACHE_SIZE");
    let intra_op_threads = parse_usize_arg(&parsed, "INTRA_OP_THREADS");
    let inter_op_threads = parse_usize_arg(&parsed, "INTER_OP_THREADS");
    let max_length = parse_usize_arg(&parsed, "MAX_LENGTH");

    let force = parsed.is_present("FORCE");
    let verbose = match output {
        stdinout::Output::File(_) => parsed.is_present("VERBOSE"),
        _ => false,
    };

    let lemmatizer: Lemmatizer = Lemmatizer::new(
        config,
        model_path,
        batch_size,
        intra_op_threads,
        inter_op_threads,
    );
    let mut processor = Processor::new(
        lemmatizer, out, reader, batch_size, cache_size, max_length, force,
    );
    let pb = indicatif::ProgressBar::new(0);
    pb.set_style(
        indicatif::ProgressStyle::default_spinner().template("Time: {elapsed_precise} ::: {msg}"),
    );

    if verbose {
        pb.enable_steady_tick(200);
    }

    // call process batch until
    let time = std::time::Instant::now();
    let mut cnt = 0;
    let mut batch_cnt = 0;
    while let Some(statistics) = processor.prepare_and_process_batch() {
        if verbose {
            cnt += statistics.processed_tokens;
            batch_cnt += 1;
            pb.set_message(&format!(
                "Processed: {:?} ::: avg. tokens/batch: {:?} ::: avg. time/token: {:?}",
                cnt,
                cnt as f32 / batch_cnt as f32,
                time.elapsed() / cnt as u32,
            ));
        }
    }
}

// Struct to pass on metrics of a processed batch.
#[derive(Debug)]
struct Stats {
    processed_tokens: usize,
    known_tokens: usize,
}

// Wrapper struct bringing together necessary parts to perform batched lemmatization.
struct Processor<T, R>
where
    T: conllx::WriteSentence,
    R: conllx::ReadSentence,
{
    lengths: Vec<usize>,
    result: Vec<conllx::Token>,
    buffer: Vec<conllx::Token>,
    unknown_idx: Vec<usize>,
    writer: T,
    reader: R,
    lemmatizer: Lemmatizer,
    cache: lru_cache::LruCache<Box<LemmaFeatures>, String>,
    batch_size: usize,
    max_length: usize,
    force: bool,
}

impl<T, R> Processor<T, R>
where
    T: conllx::WriteSentence,
    R: conllx::ReadSentence,
{
    fn new(
        lemmatizer: Lemmatizer,
        writer: T,
        reader: R,
        batch_size: usize,
        cache_size: usize,
        max_length: usize,
        force: bool,
    ) -> Self {
        let max_length = if max_length == 0 {
            usize::max_value()
        } else {
            max_length
        };
        Processor {
            lengths: Vec::new(),
            result: Vec::new(),
            buffer: Vec::with_capacity(batch_size),
            unknown_idx: Vec::new(),
            reader,
            writer,
            lemmatizer,
            cache: lru_cache::LruCache::new(cache_size),
            batch_size,
            max_length,
            force,
        }
    }

    // Returns None if the last sentence was read.
    fn prepare_and_process_batch(&mut self) -> Option<Stats> {
        let mut cnt = 0;

        // since we process sentence based we cannot guarantee exact batch size
        while self.buffer.len() < self.batch_size {
            let sentence = match self
                .reader
                .read_sentence()
                .or_exit("Failed reading sentence!", 1)
            {
                Some(sentence) => sentence,
                None => {
                    return None;
                }
            };

            self.lengths.push(sentence.len());
            for mut t in sentence {
                // force means overwrite lemmas in input tokens with a non-empty lemma
                if self.force || t.lemma().is_none() {
                    match self.cache.get_mut(&t as &LemmaFeatures) {
                        // cache hit -> we got lucky
                        Some(result) => {
                            t.set_lemma(Some(result.as_str()));
                            self.result.push(t);
                        }
                        // cache miss :(
                        None => {
                            // only lemmatize tokens up to max length
                            if t.form().len() <= self.max_length {
                                self.unknown_idx.push(cnt);
                                self.buffer.push(t.clone());
                            } else {
                                let form: String = t.form().to_owned();
                                t.set_lemma(Some(form));
                            }
                            self.result.push(t)
                        }
                    }
                }
                cnt += 1;
            }
        }
        Some(self.process_batch())
    }

    // process a batch of tokens and return statistics
    fn process_batch(&mut self) -> Stats {
        for (lemma, idx) in self
            .lemmatizer
            .lemmatize_batch(&self.buffer)
            .zip(&self.unknown_idx)
        {
            self.result[*idx].set_lemma(Some(lemma.as_str()));
            self.cache
                .insert(Box::new(self.result[*idx].clone()), lemma);
        }

        let diff = self.result.len() - self.buffer.len();

        let stats = Stats {
            processed_tokens: self.result.len(),
            known_tokens: diff,
        };

        for length in &self.lengths {
            self.writer
                .write_sentence(&self.result[..*length])
                .expect("Writing to the output failed!");
        }
        self.lengths.clear();
        self.result.clear();
        self.unknown_idx.clear();
        self.buffer.clear();
        stats
    }
}

impl<T, R> Drop for Processor<T, R>
where
    T: conllx::WriteSentence,
    R: conllx::ReadSentence,
{
    fn drop(&mut self) {
        self.process_batch();
    }
}

fn parse_usize_arg(args: &ArgMatches, name: &str) -> usize {
    args.value_of(name)
        .unwrap()
        .parse::<usize>()
        .or_exit(format!("{} not a positive integer!", name), 1)
}

fn args() -> ArgMatches<'static> {
    App::new("ohnomore_seq2seq")
        .settings(DEFAULT_CLAP_SETTINGS)
        .arg(
            Arg::with_name("MODEL")
                .help("Model file.")
                .long_help(
                    "Frozen graph in binary protobuf format. Produced by train.py. To be\
                     used together with inference_config.toml.",
                ).index(1)
                .required(true),
        ).arg(
        Arg::with_name("CONFIG")
            .long("configuration")
            .index(2)
            .help("Config in toml format produced by train.py")
            .long_help("The configuration file contains important meta information about the graph. \
                            It is automatically produced by train.py and should not be altered.")
            .takes_value(true)
            .required(true),
    ).arg(
        Arg::with_name("INPUT")
            .help("Input file in conll-x format. If not provided input reads from stdin.")
            .long("input")
            .long_help("The input file contains the input in conll-x format. If no input file is \
                            provided the input is read from stdin.")
            .index(3)
            .takes_value(true)
            .required(false),
    ).arg(
        Arg::with_name("OUTPUT")
            .help("Output File. If not provided output writes to stdout.")
            .long("output")
            .short("o")
            .takes_value(true)
            .required(false),
    ).arg(
        Arg::with_name("BATCH_SIZE")
            .help("Approximate batch size used for Tensorflow.")
            .long_help("The batch size determines how many tokens are passed into the computation \
                            graph. The batch size is only approximate as we process full sentences.")
            .short("b")
            .long("batch_size")
            .takes_value(true)
            .required(false)
            .default_value("256"),
    ).arg(
        Arg::with_name("CACHE_SIZE")
            .help("Cache size.")
            .short("c")
            .long("cache_size")
            .long_help("Defines the size of the LRU-cache. Speeds up processing significantly. \
                            Size of 0 deactivates the cache.")
            .takes_value(true)
            .required(false)
            .default_value("100000"),
    ).arg(
        Arg::with_name("VERBOSE")
            .help("Prints run metrics. Only available if '-o' is specified.")
            .long("verbose")
            .long_help("Prints throughput measures and elapsed time to stderr. Can only be used \
                            if an output file is specified using '-o'.")
            .short("-v")
            .requires("OUTPUT")
            .required(false),
    ).arg(
        Arg::with_name("FORCE")
            .help("Writes lemmas for tokens that already have lemma.")
            .long_help("By default tokens with an entry in the lemma column are not lemmatized. \
                         This enables preprocessing using dictionary lookups for closed class words etc.")
            .long("force")
            .short("-f")
            .required(false)

    ).arg(
        Arg::with_name("INTRA_OP_THREADS")
            .help("Amount of intra op threads in Tensorflow. Defaults to Tensorflow defaults.")
            .long("intra")
            .takes_value(true)
            .required(false)
            .default_value("0"),
    ).arg(
        Arg::with_name("INTER_OP_THREADS")
            .help("Amount of inter op threads in Tensorflow. Defaults to Tensorflow defaults.")
            .long("inter")
            .takes_value(true)
            .required(false)
            .default_value("0"),
    ).arg(
        Arg::with_name("MAX_LENGTH")
            .help("Maximum length before assuming form == lemma. 0 means ")
            .long_help("Input forms longer than max_length are assumed to be the lemma. This can \
                            be helpful when processing e.g. a web corpus with long links.")
            .long("max_length")
            .short("l")
            .takes_value(true)
            .required(false)
            .default_value("0")
    ).get_matches()
}
