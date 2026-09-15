#!/usr/bin/env python3
"""Dataset class for machine translation"""
import transformers
from setup import load_pt2en


class Dataset:
    """Loads and preps a Portuguese-English dataset for machine
    translation.
    """

    def __init__(self):
        """Initialize the train/validation splits and the tokenizers
        trained on them.
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        # On entraine les tokenizers uniquement sur le train pour eviter
        # toute fuite d'information venant du split de validation.
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """Create sub-word tokenizers for the Portuguese and English
        sentences of a dataset.

        Args:
            data: tf.data.Dataset of (pt, en) tf.string pairs.

        Returns:
            tuple: (tokenizer_pt, tokenizer_en) trained tokenizers.
        """
        # Tokenizers pre-entraines specifiques a chaque langue : le
        # portugais a sa propre casse/accentuation, l'anglais courant
        # tourne en minuscules (uncased) pour reduire la taille du
        # vocabulaire.
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased")
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased")
        # Les tenseurs tf.string doivent etre decodes en str Python avant
        # d'etre passes au tokenizer, qui ne comprend pas les bytes.
        pt_sentences = (pt.numpy().decode('utf-8') for pt, en in data)
        en_sentences = (en.numpy().decode('utf-8') for pt, en in data)
        # On reentraine sur notre corpus (au lieu de garder le vocabulaire
        # d'origine) pour que le tokenizer connaisse les mots specifiques
        # a ce jeu de donnees, tout en gardant une taille de vocabulaire
        # reduite (2**13) adaptee a un petit dataset.
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_sentences, vocab_size=2**13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_sentences, vocab_size=2**13)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encode a Portuguese-English sentence pair into token id
        lists, wrapped with start/end tokens.

        Args:
            pt: tf.Tensor containing the Portuguese sentence.
            en: tf.Tensor containing the corresponding English
                sentence.

        Returns:
            tuple: (pt_tokens, en_tokens) lists of token ids.
        """
        vocab_size_pt = self.tokenizer_pt.vocab_size
        vocab_size_en = self.tokenizer_en.vocab_size
        pt = pt.numpy().decode('utf-8')
        en = en.numpy().decode('utf-8')
        # add_special_tokens=False : on ne veut pas des [CLS]/[SEP] du
        # tokenizer d'origine, on ajoute nos propres tokens de
        # debut/fin juste apres pour que le modele repere clairement
        # ou commence et ou finit chaque phrase.
        pt_tokens = self.tokenizer_pt.encode(pt, add_special_tokens=False)
        en_tokens = self.tokenizer_en.encode(en, add_special_tokens=False)
        # On reserve les ids vocab_size et vocab_size + 1 (hors du
        # vocabulaire existant) comme tokens start/end, pour ne pas
        # entrer en collision avec un token deja utilise.
        pt_tokens = [vocab_size_pt] + pt_tokens + [vocab_size_pt + 1]
        en_tokens = [vocab_size_en] + en_tokens + [vocab_size_en + 1]
        return pt_tokens, en_tokens
