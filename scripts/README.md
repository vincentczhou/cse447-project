# Scripts

Utility scripts for training data augmentation, prediction, and evaluation.

## `gemini_predictor.py`

Calls the Gemini API to predict the three most likely next characters for each partial input string. Supports async concurrency, exponential-backoff retries, a persistent prediction cache, and optional accuracy grading against gold answers. Can also run in `--eval-only` mode to grade an existing predictions file without making API calls. Used for Checkpoint 3 and Distillation purposes.

## `grade.py`

Evaluates a predictions file against gold answers. A prediction is counted correct if the gold character appears within the first `--top-k` characters (case-insensitive, default top-3). Prints overall accuracy.

## `extend_inputs.py`

Takes an input file and a Gemini predictions file, then produces three output files where each line is the original input concatenated with the 1st, 2nd, or 3rd predicted character respectively. Used to generate augmented inputs for downstream re-scoring. Used for Distillation purposes.

## `augment_train.py`

Appends tokenized extended-input lines to an existing `train.txt`. Each line from the provided `--ext` files is normalized and character-tokenized (matching `preprocess.py` format) before being written `--repeat` times. Used to augment training data with Gemini-predicted continuations. Used for Distillation purposes.

## `parse_predictions_table.py`

Converts a Weights & Biases predictions table JSON export into flat `input.txt` and (optionally) `pred.txt` files, one entry per line. Useful for extracting model inputs/outputs logged during a W&B run.

## `oov_test.py`

Diagnostic script that checks whether a given token is in the KenLM model vocabulary and prints its base score. Useful for debugging out-of-vocabulary behavior in the character-level language model.
