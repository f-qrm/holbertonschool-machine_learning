#!/usr/bin/env python3
"""Find a snippet of text within a reference document to answer a question."""
# Operations sur les tenseurs pour preparer les entrees du modele
import tensorflow as tf
# Permet de charger le modele BERT pre-entraine depuis TF Hub
import tensorflow_hub as hub
# Le tokenizer transforme le texte brut en sous-tokens compris par BERT
from transformers import BertTokenizer

# BERT-SQuAD ne genere pas de texte, il predit juste la position de
# debut/fin de la reponse dans le contexte fourni
model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
# Le tokenizer doit venir du meme checkpoint que le modele, sinon les
# ids de tokens ne correspondent plus au vocabulaire attendu par BERT
tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad")


def question_answer(question, reference):
    """
    Find a snippet of text within a reference document that answers
    a question.

    Args:
        question (str): the question to answer.
        reference (str): the reference document from which to find
            the answer.

    Returns:
        str: the snippet of text containing the answer, or None if no
            answer is found.
    """
    # BERT ne travaille pas sur des mots entiers mais sur des
    # sous-tokens de son propre vocabulaire
    question_tokens = tokenizer.tokenize(question)
    # Meme decoupage pour le contexte, pour que les deux parties
    # parlent le meme "langage" de tokens
    reference_tokens = tokenizer.tokenize(reference)

    # Le format d'entree de BERT impose ces marqueurs speciaux pour
    # regrouper question et contexte dans une seule sequence
    tokens = (['[CLS]'] + question_tokens + ['[SEP]']
              + reference_tokens + ['[SEP]'])
    # Le modele ne prend pas du texte mais des indices numeriques
    # dans son vocabulaire
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    # Il n'y a pas de padding ici donc tous les tokens sont reels : le
    # mask vaut 1 partout pour dire a BERT de tous les considerer
    input_mask = [1] * len(input_word_ids)

    # Position ou commence le contexte (+2 pour [CLS] et le premier
    # [SEP]), utile pour ignorer plus tard les tokens de la question
    question_len = len(question_tokens) + 2
    # Longueur totale, pour savoir combien de tokens sont du contexte
    input_WId_len = len(input_word_ids)

    # segment 0 = question : permet a BERT de savoir quelle partie de
    # la sequence concatenee est la question
    input_type_ids = [0] * question_len
    # segment 1 = contexte : tout ce qui reste apres la question
    input_type_ids += [1] * (input_WId_len - question_len)
    # Le modele attend une dimension de batch, meme pour une seule
    # sequence, d'ou l'ajout d'un axe supplementaire
    input_word_ids = tf.expand_dims(
        tf.convert_to_tensor(input_word_ids, dtype=tf.int32), 0)
    # Meme ajout de l'axe batch pour le mask, les 3 entrees doivent
    # avoir la meme forme
    input_mask = tf.expand_dims(
        tf.convert_to_tensor(input_mask, dtype=tf.int32), 0)
    # Idem pour les segments, toujours pour garder des formes alignees
    input_type_ids = tf.expand_dims(
        tf.convert_to_tensor(input_type_ids, dtype=tf.int32), 0)

    # Le modele attend precisement ces 3 entrees, dans cet ordre
    outputs = model([input_word_ids, input_mask, input_type_ids])

    # outputs[0] donne un score par position pour un debut de reponse
    # probable
    start_logits = outputs[0]
    # outputs[1] donne le meme score mais pour une fin de reponse
    end_logits = outputs[1]

    # On ignore la portion "question" des logits car la reponse ne
    # peut se trouver que dans le contexte, jamais dans la question
    context_start_logits = start_logits[0][question_len:]
    # Meme decoupage pour les logits de fin
    context_end_logits = end_logits[0][question_len:]
    # argmax donne un indice relatif au contexte : on rajoute
    # question_len pour revenir a un indice dans tokens
    short_start = tf.math.argmax(context_start_logits) + question_len
    # Meme recalage d'indice pour la position de fin
    short_end = tf.math.argmax(context_end_logits) + question_len

    # Ce modele n'a pas d'option "pas de reponse" explicite : meme
    # pour une question sans rapport avec reference, il renvoie
    # toujours la meilleure position possible. Le softmax donne la
    # confiance du modele en cette position precise plutot qu'une
    # autre : proche de 0 signifie que le modele hesite entre plein
    # de positions, donc qu'aucune ne ressort vraiment comme reponse
    start_prob = tf.reduce_max(tf.nn.softmax(context_start_logits))
    # Meme mesure de confiance pour la position de fin
    end_prob = tf.reduce_max(tf.nn.softmax(context_end_logits))

    # Seuil choisi empiriquement : les vraies reponses observees
    # depassent largement 0.1 sur les deux scores, alors que les
    # questions hors-sujet ou incomprehensibles restent en dessous
    low_confidence = start_prob < 0.1 or end_prob < 0.1
    # Si la fin predite precede le debut, la prediction est
    # incoherente : on considere qu'il n'y a pas de reponse valable
    incoherent_span = short_end < short_start
    # Un seul des deux problemes suffit a rendre la reponse douteuse
    if low_confidence or incoherent_span:
        # None permet a l'appelant d'afficher un message d'excuse
        return None
    else:
        # +1 car la borne de fin d'un slice Python est exclue, alors
        # que short_end fait partie de la reponse
        answer_tokens = tokens[short_start:short_end + 1]
        # La reponse est souvent decoupee en sous-tokens (##...), il
        # faut les recoller pour obtenir du texte lisible
        answer = tokenizer.convert_tokens_to_string(answer_tokens)
        # Renvoie le texte final lisible par l'utilisateur
        return answer
