# KenLM Research Notes

## Our Setup

- **Model**: Character-level 6-gram, binarized as `work/char6.binary` (probing hash tables)
- **ARPA source**: `src/char6_pruned.arpa` / `work/char6_pruned.arpa`
- **Vocab**: 13,694 unique character tokens in `src/data/data/madlad_multilang_clean_1k_optionB_kenlm/vocab.json`
- **Training data**: ~141M tokens from MADLAD multilingual dataset
- **Pruning**: `--prune 0 0 0 1 2 2` (prune 4/5/6-gram singletons/rare)
- **Ngram counts**: 1=13,685 | 2=713,843 | 3=2,980,433 | 4=3,165,582 | 5=4,486,426 | 6=7,096,536

## Task

For each input string (e.g. `"Happ"`), predict the top 3 most likely next characters.
- Grader lowercases both prediction and answer, checks `gold in pred` (substring containment in 3-char output)
- Example: input `"Happ"` → answer `"y"` → prediction `"yea"` ✓

---

## KenLM Python API Reference

| Method                                           | Signature                                      | Description                                     |
| ------------------------------------------------ | ---------------------------------------------- | ----------------------------------------------- |
| `Model(path)`                                    | Constructor                                    | Load ARPA or binary model                       |
| `model.order`                                    | Property                                       | Returns n-gram order (6)                        |
| `model.score(sentence, bos, eos)`                | → `float`                                      | log10 P(sentence). Splits on whitespace.        |
| `model.perplexity(sentence)`                     | → `float`                                      | 10^(-score/words)                               |
| `model.full_scores(sentence, bos, eos)`          | → generator of `(log_prob, ngram_length, oov)` | Per-token scores                                |
| `model.BeginSentenceWrite(state)`                | Mutates `State`                                | Sets state to BOS context (`<s>`)               |
| `model.NullContextWrite(state)`                  | Mutates `State`                                | Sets state to empty/null context                |
| `model.BaseScore(in_state, word, out_state)`     | → `float`                                      | log10 P(word \| in_state), writes to out_state  |
| `model.BaseFullScore(in_state, word, out_state)` | → `FullScoreReturn`                            | Returns `(log_prob, ngram_length, oov)`         |
| `word in model`                                  | → `bool`                                       | Checks if word is in vocabulary                 |
| `State()`                                        | Constructor                                    | Creates a new state object                      |
| `FullScoreReturn`                                | Object                                         | `.log_prob`, `.ngram_length`, `.oov` properties |

**Key detail**: `BaseScore` accepts a **string** for `word` (e.g., `"h"`, `"<sp>"`). The `in_state` is **not modified** — only `out_state` is written to, so you can reuse `in_state` across multiple candidates.

---

## Optimal Prediction Strategy: State-Based BaseScore

### Why not `model.score()`?

`model.score()` takes a full sentence string, retokenizes, and rescores from scratch. Scoring all V candidates = O(V × context_length). **Way too slow.**

### The right approach: build context state once, score candidates in O(1) each

```python
import kenlm

model = kenlm.Model("work/char6.binary")

# 1. Build context state from input tokens
state = kenlm.State()
model.BeginSentenceWrite(state)

context_tokens = ["h", "a", "p", "p"]  # from input "Happ" (lowercased)
tmp_state = kenlm.State()
for token in context_tokens:
    model.BaseScore(state, token, tmp_state)
    state, tmp_state = tmp_state, state  # swap

# 2. Score every candidate next token (state is NOT modified)
out_state = kenlm.State()
scores = {}
for candidate in vocab:
    log_prob = model.BaseScore(state, candidate, out_state)
    scores[candidate] = log_prob

# 3. Top 3
top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
```

### Complexity Comparison

| Approach                    | Time per input            | Practical?                   |
| --------------------------- | ------------------------- | ---------------------------- |
| Naive `model.score()`       | O(V × context_length)     | ❌ Very slow                  |
| **State-based `BaseScore`** | **O(context_length + V)** | **✅ Best**                   |
| `full_scores()`             | O(sentence_length)        | ❌ Can't enumerate candidates |

### Performance Estimate

- 13,685 candidates × ~2µs per `BaseScore` call ≈ **~27ms per input line**
- Even with thousands of test inputs → well under the 30-minute time limit
- **No vocab pruning needed**

---

## Implementation Pseudocode

```python
class MyModel:
    def load(work_dir):
        model = kenlm.Model(os.path.join(work_dir, "char6.binary"))
        vocab = list(json.load(open("vocab.json")).keys())  # all tokens
        # Exclude <s>, </s> from candidates
        return model, vocab

    def input_to_tokens(inp):
        text = normalize_text(inp)  # lowercase, NFC, collapse whitespace
        return [SPACE_TOKEN if ch == " " else ch for ch in text]

    def build_state(model, tokens):
        state = kenlm.State()
        model.BeginSentenceWrite(state)
        out = kenlm.State()
        for token in tokens:
            model.BaseScore(state, token, out)
            state, out = out, state
        return state

    def predict_top3(model, state, vocab):
        out = kenlm.State()
        scored = [(model.BaseScore(state, tok, out), tok) for tok in vocab]
        top3 = heapq.nlargest(3, scored)
        return "".join(" " if t == "<sp>" else t for _, t in top3)
```

---

## Important Gotchas

1. **Token format must match training**: Tokens are single characters or `<sp>`. Must match exactly what was in `train.txt`.

2. **OOV handling**: Unknown chars get `<unk>` score with backoff. `BaseScore` still works — KenLM handles this gracefully. Check with `token in model` → `False` if OOV.

3. **BOS vs NullContext**:
   - `BeginSentenceWrite` → conditions on `<s>` (sentence start). Use for sentence-initial inputs.
   - `NullContextWrite` → no context. Use for mid-sentence fragments if needed.

4. **Case sensitivity**: Our model is lowercased. Always `.lower()` test inputs before tokenizing.

5. **Exclude `</s>` from candidates**: It's a special KenLM token, not a printable character.

6. **Space prediction**: If `<sp>` is top-scored, output a literal space character `" "`. The grader handles this.

7. **No trie enumeration**: KenLM has NO method to enumerate valid continuations from a state. Must loop through all candidates explicitly.

8. **State is compact**: For 6-gram, stores at most 5 word indices. Swapping is cheap.

9. **Log base 10**: All KenLM scores are log10 (not ln). Doesn't matter for ranking.

10. **NFC normalization**: Our preprocessing does `unicodedata.normalize("NFC", text)` — apply this to test inputs too for consistency.

---

## Files

| File                           | Purpose                                                      |
| ------------------------------ | ------------------------------------------------------------ |
| `work/char6.binary`            | Binarized KenLM model (probing hash tables)                  |
| `work/char6_pruned.arpa`       | ARPA text-format model                                       |
| `src/data/data/.../vocab.json` | `{token: count}` sorted by frequency descending              |
| `src/myprogram.py`             | Main program (needs KenLM integration)                       |
| `src/predict.sh`               | Entry point: `uv run python src/myprogram.py test ...`       |
| `src/data/preprocess.py`       | Preprocessing: normalize → char tokenize → train/valid/vocab |
