# LLM Engineering Lab: Prompt Strategy Benchmark

Hands-on LLM engineering experiments using **local models (Ollama)** and plain Python.
The main project asks one question and answers it with numbers instead of opinions:

> **Does few-shot or chain-of-thought prompting beat zero-shot on a classification task, and what does it cost?**

![Results table](assets/results_table.png)

*Model: `llama3.2` (local, via Ollama) | 50 self-labelled test tickets | temperature 0 | single run.*

---

## Why this repo exists

I'm learning LLM engineering (RAG and agents come next). Reading about prompt techniques is easy; checking whether they help on a real task is the part most tutorials skip. Each project in this repo answers a specific question and produces a measurable result.

## Roadmap

| # | Project | Question it answers | Status |
|---|---|---|---|
| 1 | **Prompt strategy benchmark** | Does few-shot / chain-of-thought beat zero-shot, and at what cost? | Done (this repo) |
| 2 | Messy text to JSON extractor | How often is structured output valid and correct? | Planned |
| 3 | Mini eval harness with LLM-as-judge | Does an LLM judge agree with my own labels? | Planned |
| 4 | Chunking experiment | How do chunk size and overlap change retrieval? | Planned |
| 5 | Keyword vs semantic vs hybrid + reranker | Which retrieval setup wins? | Planned |
| 6 | End-to-end RAG with failure analysis | Where does RAG fail and how often? | Planned |
| 7 | Tool-calling agent without a framework | What happens inside a ReAct loop? | Planned |
| 8 | Agent reliability test | What is the success rate, cost, and top failure types? | Planned |
| 9 | Capstone: domain agent (RAG + tools) | Can I combine everything into something usable? | Planned |

---

## Project 1: Prompt strategy benchmark

### The task
Classify customer-support tickets into one of four labels: `billing`, `bug`, `feature_request`, `other`.

### The data (`tickets.csv`)
60 tickets I wrote and labelled myself:

| Split | Rows | Used for |
|---|---|---|
| `pool` | 10 | Few-shot examples only (never scored) |
| `test` | 50 | Scoring (13 billing, 13 bug, 12 feature_request, 12 other) |

12 of the test tickets are marked `ambiguous` (they plausibly fit two labels, e.g. *"I was charged twice after the app crashed"*). They are where strategies differ most.

### The three prompts
All three share the system message *"You are a support-ticket classifier."* and the instruction *"Classify the support ticket into exactly one of these labels: billing, bug, feature_request, other."*

| Strategy | What is added |
|---|---|
| **Zero-shot** | "Reply with only the label, nothing else." |
| **Few-shot** | The same, plus 6 solved examples from the pool (balanced across labels) |
| **Chain-of-thought** | "First explain your reasoning in 2-3 short sentences... then on the last line write exactly: LABEL: <label>" |

The prompts live in `build_messages()` in `benchmark.py`.

### What happens when you run `python benchmark.py`
1. **Load** `tickets.csv` and split it into pool and test.
2. **Build** one prompt per ticket per strategy (50 x 3 = 150 calls).
3. **Call** the model through the OpenAI-compatible endpoint Ollama exposes at `http://localhost:11434/v1`. Temperature is 0 and every call uses a fresh message list, so no ticket can influence another.
4. **Parse** the reply into a label. For chain-of-thought it reads the last `LABEL:` line. Anything unreadable counts as a *parse failure*.
5. **Score** accuracy, accuracy on ambiguous tickets, confusion matrices, token counts and time.
6. **Save** `results.md` (tables) and `results.jsonl` (every raw answer, for error analysis).

### Results

| Strategy | Accuracy | Ambiguous (12) | Parse fails | Avg output tokens | Total time |
|---|---|---|---|---|---|
| Zero-shot | 70% (35/50) | 5/12 | 0 | 2 | 1.3 min |
| Few-shot | 78% (39/50) | 9/12 | 0 | 2 | 1.1 min |
| Chain-of-thought | 80% (40/50) | 9/12 | 0 | 76 | 8.4 min |

Correct per label:

| Label | Zero-shot | Few-shot | CoT |
|---|---|---|---|
| billing (13) | 6 | 8 | 10 |
| bug (13) | 12 | 9 | 13 |
| feature_request (12) | 11 | 12 | 12 |
| other (12) | 6 | 10 | 5 |

### What I take from it
- **Few-shot was the best trade-off here.** +4 tickets over zero-shot and ambiguous accuracy went from 5/12 to 9/12, at no extra time (only more input tokens).
- **Chain-of-thought's lead is within noise.** It is one ticket ahead of few-shot (40 vs 39) and took about 7x longer.
- **Equal totals hide different behaviour.** CoT got every bug right but was weakest on `other`; few-shot recovered `other` but lost ground on `bug`.
- **Several tickets were missed by all three strategies.** That points at my labels or label definitions as much as at the model. Run `python inspect_errors.py` to list them.

### Limitations (read before trusting the numbers)
- One run, one small model, 50 tickets: each ticket is worth 2 points, so gaps of 1-2 tickets are not evidence.
- I wrote the tickets and labels; real tickets are messier, and some labels are debatable.
- Results are for `llama3.2` (3B). Larger models may behave differently.
- Local models can vary slightly between runs even at temperature 0.
- I have not yet tested why `other` is hard. My hypothesis is that the label is not defined in the prompt; adding definitions is the next experiment.

---

## Warm-up scripts

Small scripts written to learn the building blocks.

| File | What it does | What it teaches |
|---|---|---|
| `chat.py` | Terminal chat with a local model, streaming tokens | Models are stateless: the `history` list is the "memory" and is re-sent every turn |
| `summarize_site.py` | Scrapes a page (requests + BeautifulSoup) and summarises it | Putting outside text into a prompt; trimming to fit the context window |
| `brochure.py` | Scrapes a site, then chains **two LLM calls**: facts first, brochure copy second | Prompt chaining; different temperature per step (0 for facts, 0.7 for copy); each step can use a different model |

---

## Key concepts practised

- **Message roles:** system, user, assistant, tool.
- **Prompting techniques:** zero-shot, few-shot, chain-of-thought (plus role, structured output, grounding and chaining in the warm-ups).
- **Context window:** limited input size, which is why pages are trimmed and long chats get truncated.
- **Statelessness:** nothing is remembered unless you re-send it.
- **Fair evaluation:** same model, temperature 0, held-out test data, defensive output parsing.
- **OpenAI-compatible API:** the same `client.chat.completions.create` code works with Ollama, and can be pointed at OpenAI or OpenRouter by changing `base_url`, `api_key` and `model`.

---

## Setup

1. Install [Ollama](https://ollama.com/download) and pull a model:
   ```
   ollama pull llama3.2
   ```
2. Create and activate a virtual environment, then install dependencies:
   ```
   python -m venv .venv
   .venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```
3. Run:
   ```
   python benchmark.py --limit 5      # quick smoke test
   python benchmark.py                # full run (chain-of-thought is the slow part)
   python inspect_errors.py           # list tickets any strategy got wrong
   python chat.py
   python brochure.py https://example.com
   ```
   

