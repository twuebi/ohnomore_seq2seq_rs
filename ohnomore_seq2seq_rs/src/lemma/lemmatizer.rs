use tensorflow::Tensor;

use lemma::tf::tensorflow_wrapper::TensorflowWrapper;
use lemma::{LemmaFeatures, Lemmatize};

use std::iter;
use std::iter::Iterator;
use util::Config;

/// Struct to perform lemmatization.
pub struct Lemmatizer {
    vectorizer: Vectorizer,
    tf_wrapper: TensorflowWrapper,
}

impl Lemmatizer {
    /// Constructs a new Lemmatizer.
    pub fn new(
        config: Config,
        model_path: &str,
        batch_size: usize,
        intra_op_threads: usize,
        inter_op_threads: usize,
    ) -> Self {
        Lemmatizer {
            tf_wrapper: TensorflowWrapper::new(
                model_path,
                intra_op_threads,
                inter_op_threads,
                &config.ops,
            ),
            vectorizer: Vectorizer::new(config, batch_size),
        }
    }
}

/// Implementation of the Lemmatize Trait for the Lemmatizer Struct.
impl<R> Lemmatize<R> for Lemmatizer
where
    R: LemmaFeatures,
{
    /// Lemmatizes a Vec<conllx::Token> using the loaded model.
    fn lemmatize_batch(&mut self, tokens: &[R]) -> Box<Iterator<Item = String>> {
        let (forms, lengths, pos, morph) = self.vectorizer.vectorize(tokens);
        let (preds, lens): (Tensor<String>, Tensor<i32>) =
            self.tf_wrapper.step(&forms, &lengths, &pos, &morph);
        let offset = preds.dims()[1] as usize;

        Box::new((0..tokens.len()).into_iter().map(move |cnt: usize| {
            preds[cnt * offset..(cnt + 1) * offset][..(lens[cnt] as usize) - 1]
                .iter()
                .map(|x| x.to_owned())
                .collect::<String>()
        }))
    }
}

/// Vectorizer that converts input into tensors
struct Vectorizer {
    config: Config,
    batch_size: usize,
}

impl Vectorizer {
    pub fn new(config: Config, batch_size: usize) -> Self {
        Vectorizer { config, batch_size }
    }

    /// Function to vectorize inputs. Takes a Vec<LemmaFeatures>, and returns a tuple consisting of
    /// 4 tensors.
    /// The 4 tensors are:
    ///          forms : Tensor<String>,
    ///          form lengths : Tensor<i32>
    ///          pos : Tensor<String>
    ///          morph tags : Tensor<String>
    ///
    ///
    pub fn vectorize<R>(
        &self,
        tokens: &[R],
    ) -> (Tensor<String>, Tensor<i32>, Tensor<String>, Tensor<String>)
    where
        R: LemmaFeatures,
    {
        let mut forms: Vec<String> = Vec::with_capacity(self.batch_size);
        let mut lengths: Vec<i32> = Vec::with_capacity(self.batch_size);
        let mut pos: Vec<String> = Vec::with_capacity(self.batch_size);
        let mut morphs: Vec<String> =
            Vec::with_capacity(self.batch_size * self.config.max_morph_tags);
        let mut max_length = 0usize;

        for token in tokens {
            let p = token.pos().to_string();
            let form = token.form();
            let morph = token.morph();

            if form.len() > max_length {
                max_length = form.len();
            }

            pos.push(p);
            forms.push(form.to_lowercase());
            lengths.push(form.len() as i32);
            morphs.extend(
                morph
                    .split('|')
                    .map(|x| x.to_string())
                    .filter(|x| self.config.morph_feats.contains(x))
                    .chain(iter::repeat(self.config.pad_symbol().to_string()))
                    .take(self.config.max_morph_tags),
            );
        }
        let actual_batch_size = lengths.len();

        // need to run through inputs again as we only know the max length after going through
        // all of them
        let inp: Vec<String> = forms.iter().fold(
            Vec::with_capacity(actual_batch_size * max_length),
            |mut acc: Vec<String>, x| {
                acc.extend(pad_to(&x, max_length, self.config.pad_symbol()));
                acc
            },
        );
        let lengths: Tensor<i32> = Tensor::new(&[actual_batch_size as u64])
            .with_values(&lengths)
            .unwrap();
        let pos: Tensor<String> = Tensor::new(&[actual_batch_size as u64])
            .with_values(&pos)
            .unwrap();
        let inputs = Tensor::new(&[actual_batch_size as u64, max_length as u64])
            .with_values(&inp)
            .unwrap();
        let morphs: Tensor<String> =
            Tensor::new(&[actual_batch_size as u64, self.config.max_morph_tags as u64])
                .with_values(&morphs)
                .unwrap();

        (inputs, lengths, pos, morphs)
    }
}

#[inline]
fn pad_to(val: &str, length: usize, symbol: &str) -> Vec<String> {
    let mut res: Vec<String> = val.chars().map(|x| x.to_string()).collect();
    let ext = vec![symbol.to_string(); length - res.len()];
    res.extend(ext);
    res
}

#[cfg(test)]
mod tests {
    use super::Vectorizer;
    use conllx::{Features, TokenBuilder};
    use tensorflow::Tensor;
    use util::Config;

    #[test]
    pub fn test_vectorizer() {
        let config = Config::from("testdata/config.toml");
        let vectorizer = Vectorizer::new(config, 25usize);

        let token = TokenBuilder::new("Hallo")
            .pos("NN")
            .features(Features::from_string("mpos:x|A:A|A:B"))
            .token();
        let token2 = TokenBuilder::new("Welt")
            .pos("NN")
            .features(Features::from_string("mpos:x|B:A|B:B"))
            .token();

        let exp_form = Tensor::new(&[2u64, 5u64])
            .with_values(&[
                "h".to_string(),
                "a".to_string(),
                "l".to_string(),
                "l".to_string(),
                "o".to_string(),
                "w".to_string(),
                "e".to_string(),
                "l".to_string(),
                "t".to_string(),
                "<PAD>".to_string(),
            ]).unwrap();
        let exp_pos = Tensor::new(&[2u64])
            .with_values(&["NN".to_string(), "NN".to_string()])
            .unwrap();
        let exp_lengths = Tensor::new(&[2u64]).with_values(&[5i32, 4i32]).unwrap();
        let exp_morph = Tensor::new(&[2u64, 5])
            .with_values(&[
                "A:A".to_string(),
                "A:B".to_string(),
                "<PAD>".to_string(),
                "<PAD>".to_string(),
                "<PAD>".to_string(),
                "B:A".to_string(),
                "B:B".to_string(),
                "<PAD>".to_string(),
                "<PAD>".to_string(),
                "<PAD>".to_string(),
            ]).unwrap();
        let mut sent = vec![token, token2];

        let (form, lengths, pos, morph) = vectorizer.vectorize(&mut sent);

        assert_eq!(exp_form, form);
        assert_eq!(exp_pos, pos);
        assert_eq!(exp_morph, morph);
        assert_eq!(exp_lengths, lengths);
    }
}