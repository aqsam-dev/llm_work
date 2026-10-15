# Prompt Strategy Benchmark

An experiment in LLM engineering using a **local model (Ollama, `llama3.2`)** and plain Python.

> **Question:** Does few-shot or chain-of-thought prompting beat zero-shot on a classification task, and what does it cost?

The same model gets the same 50 support tickets three times, once per prompt style. The only thing that changes is the prompt, so any difference in score comes from the prompt.

## Files in this folder

| File | What it is |
|---|---|
| `benchmark.py` | The main experiment: runs three prompt strategies on 50 tickets and scores them |
| `chat.py` | A terminal chat with the local model (warm-up: how a conversation works) |
| `scraper.py` | Scrapes a web page and has the model summarise it (warm-up: giving outside text to a model) |
| `tickets.csv` | The data used by `benchmark.py` (60 labelled tickets) |
| `results.md`, `results.jsonl` | Created when you run `benchmark.py` (scoreboard and every raw answer) |

---

## `benchmark.py`: the experiment

### What it does
It classifies customer-support tickets into one of four labels: `billing`, `bug`, `feature_request`, `other`.

For every ticket it asks the model three ways:

| Strategy | What the prompt contains |
|---|---|
| **Zero-shot** | The label list and "reply with only the label" |
| **Few-shot** | The same, plus 6 solved example tickets |
| **Chain-of-thought** | The same, but the model first explains its reasoning, then ends with `LABEL: <label>` |

### What happens when you run it
1. **Loads `tickets.csv`.** 10 rows are the `pool` (used only as few-shot examples) and 50 are the `test` set (used for scoring). The model is never graded on tickets it was shown as examples.
2. **Builds the prompts** (function `build_messages`). Each call gets a fresh message list, so one ticket cannot influence the next.
3. **Calls the model** through the OpenAI-compatible API that Ollama exposes at `http://localhost:11434/v1`, with `temperature=0` so results are as repeatable as possible. That is 50 tickets x 3 strategies = 150 calls.
4. **Parses each answer** into a label (functions `to_label`, `parse`). Models do not always reply cleanly, so unreadable answers are counted as *parse failures*.
5. **Scores** overall accuracy, accuracy on the 12 deliberately ambiguous tickets, confusion matrices, token counts and time.
6. **Saves** `results.md` and `results.jsonl`.

### Why it is built this way
- **A fixed answer key** (`tickets.csv`) turns "this prompt feels better" into a number.
- **Same model, same tickets, temperature 0** means the prompt is the only variable.
- **Held-out test tickets** prevent the few-shot examples from inflating the score.
- **Ambiguous tickets** (e.g. "I was charged twice after the app crashed") are where prompt strategies differ most.

### Results (llama3.2, one run)

| Strategy | Accuracy | Ambiguous (12) | Parse fails | Avg output tokens | Total time |
|---|---|---|---|---|---|
| Zero-shot | 70% (35/50) | 5/12 | 0 | 2 | 1.3 min |
| Few-shot | 78% (39/50) | 9/12 | 0 | 2 | 1.1 min |
| Chain-of-thought | 80% (40/50) | 9/12 | 0 | 76 | 8.4 min |

Correct answers per label:

| Label | Zero-shot | Few-shot | CoT |
|---|---|---|---|
| billing (13) | 6 | 8 | 10 |
| bug (13) | 12 | 9 | 13 |
| feature_request (12) | 11 | 12 | 12 |
| other (12) | 6 | 10 | 5 |

### What the results suggest
- **Few-shot was the best trade-off.** It beat zero-shot by 4 tickets (ambiguous: 5/12 to 9/12) with no extra time.
- **Chain-of-thought is only 1 ticket ahead of few-shot** and took about 7x longer. That gap is within noise on 50 tickets.
- **Similar totals, different mistakes.** CoT was best on `bug` but weakest on `other`; few-shot was strong on `other` but lost ground on `bug`.
- **Some tickets were missed by every strategy**, which points at my labels as much as at the model.

### Limitations
- One run, one small model (3B), 50 tickets: each ticket is worth 2 points, so small gaps are not evidence.
- I wrote and labelled the tickets myself; real tickets are messier and some labels are debatable.
- Results may differ for larger models.

---

## `chat.py`: talk to the model

A terminal chat. Type a message, the model streams its reply, type `quit` to exit.

**What it teaches:**
- **Models are stateless.** The `history` list is the only memory. It is re-sent in full on every turn, so deleting it makes the model forget.
- **Streaming** (`stream=True`) prints tokens as they are generated instead of waiting for the full answer.
- **Message roles:** a `system` message sets behaviour, `user` and `assistant` messages form the conversation.

This is the same API call `benchmark.py` uses, so it is the foundation of the experiment.

---

## `scraper.py`: scrape a page and summarise it

```
python scraper.py https://example.com
```

**What happens:**
1. **Download** the page HTML with `requests`.
2. **Clean it** with BeautifulSoup: remove scripts, styles, menus and footers, keep the readable text.
3. **Trim** the text to 3000 characters, because small local models have a limited *context window*.
4. **Prompt the model** with a system message and the page text: "summarise in 5 bullets, use only the text below."
5. **Print** the summary.

**What it teaches:**
- Putting outside text into a prompt, which is the core idea behind RAG.
- The context window limit and why input must be trimmed.
- Limits of scraping: pages built with JavaScript return little text, and some sites block scrapers.

---

## Setup

1. Install [Ollama](https://ollama.com/download) and download a model:
   ```
   ollama pull llama3.2
   ```
2. Create and activate a virtual environment, then install the packages:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install openai requests beautifulsoup4
   ```
   (Mac/Linux: `source .venv/bin/activate`)
3. Run:
   ```
   python benchmark.py --limit 5     # quick test: 5 tickets
   python benchmark.py               # full run (chain-of-thought is the slow part)
   python chat.py
   python scraper.py https://example.com
   ```

`benchmark.py` options: `--model`, `--shots`, `--limit`, `--strategies`, `--base-url`, `--api-key`.

The same code can point at hosted models (OpenAI, OpenRouter) by changing `--base-url`, `--api-key` and `--model`.

