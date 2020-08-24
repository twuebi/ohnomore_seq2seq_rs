# Ohnomore_seq2seq_rs
Sequence2Sequence lemmatization in rust.

## Requirements

1. python 3.x, Tensorflow 1.9, numpy, toml, argparse
2. an up-to-date rust toolchain (tested with 1.29)
3. (optional) it is recommended to use an optimized build of Tensorflow as the binary obtained by cargo slows things down.
4. a conllx treebank containing with Lemma, POS, and morphological feature annotations for training a model

## Training the model

This lemmatizer requires a pretrained model. The model can be trained using the python script `train.py` in the python folder. The input files are expected to be in conllx format where the feature column holds morphological features.

    python train.py validation-file.conll train-file.conll config.toml --word_filter filter.txt --pos_filter pos_filter.txt
 
The two filter arguments are optional and can be used to filter either words or pos.

Before running `train.py` some adjustments can be made to `config.toml` (However, it contains sensible defaults for Dutch and German). An requirement is to set the max_morph_tags attribute in the config. The `python` folder contains a script `morph_cnt.sh` which extracts the maximum amount of morph tags in the provided file and writes it into `config.toml`.

    cat <train_file.conll> | morph_cnt.sh // morph_cnt.sh <train_file.conll>

The script needs to be executed in the same folder as config.toml with the last line either being empty or containing an earlier entry for max_morph_tags.

Both the morph-tag counting script and the training procedure assume that all entries in the feature column are to be used as input features to the network. Also it is assumed that the order of the features is fixed.

Running `train.py` saves the frozen graph as `<train_file>.pb` alongside `inference_config.toml` in the save_dir specified in `config.toml`. `inference_config.toml` should not be changed as it contains important meta data and the names of the operations in the graph.

## Input format:

The input is assumed to be in conll-x format. The morphological features are assumed to be in the same order as during training. 

Caching either requires that only morphological features are stored in the feature column or that there is an entry with the key `morph` that has a unique value for each combination of morphological features.


## Running the lemmatizer

The lemmatizer can be executed using:

    ./ohnomore_seq2seq_rs <model_file.pb> <inference_config.toml> <input_file>

Further it is possible to read input from stdin by omitting the input file:

    cat <input_file> | ./ohnomore_seq2seq_rs <model_file.pb> <inference_config.toml>

It should be noted that by default tokens with a non-empty lemma column are not lemmatized. This allows for a pipeline which includes first a dictionary to efficiently lemmatize closed class words. The parameter `-f` enables replacing existing lemmas. 

Optional parameters:

    -o // --output      <file_name> File to write results to, defaults to stdout
    -b // --batch_size  <int>       Adjust batch size.
    -c // --cache_size  <int>       Adjust the size of the cache.
    -f // --force                   Also lemmatize tokens that already have a lemma.
    -l // --max_length  <int>       Form length up to which lemmatization is done. For longer forms we assume form=lemma.
    -v // --verbose                 Print throughput measures. Only available with -o.
          --inter       <int>       Amount of inter OP threads in Tensorflow.
          --intra       <int>       Amount of intra OP threads in Tensorflow.
    -h // --help                    Print detailed help.

## Using the library

Besides the standalone binary there is also an API. Example usage can be found in the rust doc of the crate *ohnomore_seq2seq_rs*.

## Under the Hood

### Tensorflow

The lemmatizer is a wrapper around a Tensorflow computation graph. The model is a Sequence2Sequence model utilizing morphological information.

### Cache

Words in natural language follow a [Zipfian Distribution](https://en.wikipedia.org/wiki/Zipf%27s_law). Since we process a text sequentially we will naturally see the same tokens over and over again. To exploit this we employ a cache in which we save previous lemmatizations. To avoid saving all seen tokens we use a LRU-cache based on form, part of speech and morphological features. 

With a batch size of 256 and a cache size of 100 000 we process 1000-1500 tokens per batch. 

### Batching

Since we perform a lookup in the cache before batching the tokens we avoid known tokens from wasting space in the costly calls to Tensorflow. We achieve this while maintaining the sentence structure of conllx by flattening the incoming Vec<Vec\<Token\>> while saving sentence offsets and the indices of unknown tokens. The lemmas of the known tokens are set immediately using the cache. After lemmatizing the unknowns, the lemmas are mapped to the tokens using the saved indices. The results are then written sentence by sentence using the conllx writer.
