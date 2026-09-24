# QA Bot: Extractive Question Answering with BERT

This project implements a question-answering system that finds the answer to a question inside a given reference document, using a BERT model pre-trained on SQuAD.

## Overview

Extractive question answering does not generate new text — it locates the span of the reference document that already contains the answer. BERT was fine-tuned on SQuAD to do exactly this: given a question and a context concatenated into a single sequence, it outputs, for every token position, a score for "this token starts the answer" and a score for "this token ends the answer". Picking the highest-scoring start and end positions (restricted to the context, never the question) gives the answer span.

## Contents

| File | Description |
| --- | --- |
| `0-qa.py` | `question_answer(question, reference)` — tokenizes the question and reference, builds the `[CLS] question [SEP] reference [SEP]` input expected by BERT, runs it through the QA model, and decodes the highest-scoring answer span back into text. |
| `1-loop.py` | A standalone script implementing the input/exit loop on its own: prompts with `Q: `, echoes a placeholder `A:`, and exits with `A: Goodbye` on `exit`/`quit`/`goodbye`/`bye` (case-insensitive). A stepping stone toward `2-qa.py`, not part of the deliverable API. |
| `2-qa.py` | `answer_loop(reference)` — wraps the same loop around `question_answer` from `0-qa.py`: each question is answered against `reference`, printing `A: Sorry, I do not understand your question.` whenever `question_answer` returns `None`. |

The folder also contains `0-main.py` / `2-main.py`, driver scripts that exercise `question_answer` and `answer_loop` against `ZendeskArticles/PeerLearningDays.md`, and the `ZendeskArticles/` reference documents themselves. These are usage examples/data, not part of the deliverable API.

## How It Works

`0-qa.py` loads two pre-trained artifacts once at import time, so repeated calls to `question_answer` don't reload them:

- The QA model itself, `bert-uncased-tf2-qa`, from TensorFlow Hub.
- The matching `BertTokenizer`, `bert-large-uncased-whole-word-masking-finetuned-squad`, from the `transformers` library.

`question_answer(question, reference)` then:

1. Tokenizes `question` and `reference` separately with BERT's WordPiece tokenizer.
2. Builds a single token sequence: `['[CLS]'] + question_tokens + ['[SEP]'] + reference_tokens + ['[SEP]']`, and converts it to vocabulary ids.
3. Builds the two auxiliary inputs BERT needs alongside the ids: an attention mask (all `1`s, since there is no padding) and segment ids (`0` for the question span, `1` for the reference span), each expanded with a leading batch dimension.
4. Runs the three tensors through the model to get `start_logits` and `end_logits` — one score per token position for "answer starts here" / "answer ends here".
5. Restricts the search to the reference portion of the logits (slicing off everything up to `question_len`) and takes the `argmax` of each, since the answer can only live in the reference, never in the question itself.
6. If the predicted end position comes before the start position, the prediction is incoherent and the function returns `None`. Otherwise, it slices the token span out of the original `tokens` list and reassembles it into a readable string with `tokenizer.convert_tokens_to_string`, which also recombines WordPiece sub-tokens (e.g. `##ing`) into whole words.

`2-qa.py`'s `answer_loop(reference)` turns `question_answer` into an interactive chatbot: it reads one question at a time with `input('Q: ')`, checks it against a tuple of exit words *before* calling the model (so typing `exit` doesn't trigger an unnecessary BERT forward pass), and otherwise forwards the question straight to `question_answer(user_input, reference)`. A `None` result — the "incoherent span" case described above — is surfaced to the user as `A: Sorry, I do not understand your question.` instead of leaking the internal `None`.

## Requirements

- Python 3.9
- numpy 1.25.2
- tensorflow 2.15
- tensorflow-hub 0.15.0
- transformers 4.44.2

## Usage

```python
#!/usr/bin/env python3
question_answer = __import__('0-qa').question_answer

with open('ZendeskArticles/PeerLearningDays.md') as f:
    reference = f.read()

print(question_answer('When are PLDs?', reference))
# on - site days from 9 : 00 am to 3 : 00 pm
```

For the interactive chatbot:

```python
#!/usr/bin/env python3
answer_loop = __import__('2-qa').answer_loop

with open('ZendeskArticles/PeerLearningDays.md') as f:
    reference = f.read()

answer_loop(reference)
# Q: When are PLDs?
# A: on - site days from 9 : 00 am to 3 : 00 pm
# Q: EXIT
# A: Goodbye
```

## Design Notes

- The model and tokenizer are loaded at module import time rather than inside `question_answer`, so a caller answering many questions against the same or different references only pays the (large) download/load cost once per process.
- The answer search is explicitly restricted to the reference span of the logits (`start_logits[0][question_len:]`) rather than searched over the full sequence — without this, the model could technically "answer" with a span from inside the question itself, which is never a valid extractive answer.
- Returning `None` when `short_end < short_start` guards against a genuinely unanswerable question: BERT-SQuAD has no explicit "no answer" output here, so an incoherent (end-before-start) span is treated as the closest available signal that no good answer was found in the reference.
- `answer_loop` checks the exit words before calling `question_answer` rather than after, so quitting the chatbot never costs a BERT forward pass.
- The extractive model has no true "I don't know" output — it always picks *some* start/end positions, even for a question unrelated to the reference. Whether a given off-topic question produces `None` (end before start) or a coherent-but-wrong span depends on the exact logits, so `answer_loop`'s "Sorry, I do not understand" fallback is best read as "no coherent span was found", not a guarantee that every irrelevant question will be caught.

## Author

Fjolla Qerimi
