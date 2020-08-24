mod lemmatizer;
pub use self::lemmatizer::Lemmatizer;

mod tf;

use conllx;
use std::hash::Hash;
use std::hash::Hasher;

/// Trait for lemmatization
pub trait Lemmatize<T>
where
    T: LemmaFeatures,
{
    fn lemmatize_batch(&mut self, tokens: &[T]) -> Box<Iterator<Item = String>>;
}

/// Trait defining the minimal accessories to run inference using the ohnomore_seq2seq morph model.
pub trait LemmaFeatures {
    fn form(&self) -> &str;
    fn pos(&self) -> &str;
    fn morph(&self) -> &str;
}

/// LemmaFeatures for conllx::Token.
impl LemmaFeatures for conllx::Token {
    fn form(&self) -> &str {
        self.form()
    }

    fn pos(&self) -> &str {
        self.pos().unwrap()
    }

    fn morph(&self) -> &str {
        match self.features() {
            // use str and not map since linear order matters
            Some(feats) => feats.as_str(),
            None => "_",
        }
    }
}

impl Hash for LemmaFeatures {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(self.form().to_lowercase().as_bytes());
        state.write(self.pos().as_bytes());
    }
}

/// We consider form + pos + morph for equality to allow for the cache lookups.
///
/// Since there might be irrelevant features or meta information in the features column there are
/// 3 cases for comparing morph features:
///     1. the two strings are identical                            -> equal
///     2. both strings contain the key morph and match             -> equal
///     3. the strings are not identical and don't contain morph    -> equal
///
/// We make the third assumption to be able to use this trait for cache lookups. If the morph key
/// is absent we assume that there are no relevant features in the column.
impl PartialEq for LemmaFeatures {
    fn eq(&self, other: &LemmaFeatures) -> bool {
        let own_morph = self.morph();
        let other_morph = other.morph();
        self.pos() == other.pos()
            && self.form().to_lowercase() == other.form().to_lowercase()
            && own_morph == other_morph
            || own_morph
                .split('|')
                .filter(|x| x.starts_with("morph"))
                .eq(other.morph().split('|').filter(|x| x.starts_with("morph")))
    }
}

impl Eq for LemmaFeatures {}
