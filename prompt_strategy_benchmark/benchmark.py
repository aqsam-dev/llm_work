import argparse, csv, json, os, re, time
from collections import Counter, defaultdict

LABELS = ["billing", "bug", "feature_request", "other"]
SYSTEM = "You are a support-ticket classifier."
TASK = f"Classify the support ticket into exactly one of these labels: {', '.join(LABELS)}.\n"


# ---------- 1. data ----------
def load(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pool = [r for r in rows if r["split"] == "pool"]
    test = [r for r in rows if r["split"] == "test"]
    return pool, test


def pick_shots(pool, k):
    """Round-robin over labels so examples are balanced."""
    by = defaultdict(list)
    for r in pool:
        by[r["label"]].append(r)
    shots, i = [], 0
    while len(shots) < min(k, len(pool)):
        for lab in LABELS:
            if i < len(by[lab]) and len(shots) < k:
                shots.append(by[lab][i])
        i += 1
    return shots


# ---------- 2. the three prompt strategies ----------
def build_messages(strategy, text, shots):
    if strategy == "zero_shot":
        user = TASK + "Reply with only the label, nothing else.\n\nTicket: " + text
    elif strategy == "few_shot":
        ex = "\n\n".join(f"Ticket: {s['text']}\nLabel: {s['label']}" for s in shots)
        user = (TASK + "Reply with only the label, nothing else.\n\nExamples:\n\n" + ex
                + f"\n\nNow classify this one:\nTicket: {text}\nLabel:")
    elif strategy == "cot":
        user = (TASK + "First explain your reasoning in 2-3 short sentences, considering which "
                "labels could fit. Then, on the last line, write exactly: LABEL: <label>\n\nTicket: " + text)
    else:
        raise ValueError(strategy)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


# ---------- 3. calling the model ----------
def make_client(base_url, api_key):
    from openai import OpenAI  # imported here so the file can be tested without the package
    return OpenAI(base_url=base_url, api_key=api_key)


def ask(client, model, messages, max_tokens):
    r = client.chat.completions.create(model=model, messages=messages,
                                       temperature=0, max_tokens=max_tokens)
    u = r.usage
    return (r.choices[0].message.content or "",
            getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0)


# ---------- 4. parsing ----------
def to_label(s):
    norm = re.sub(r"[\s\-]+", "_", s.lower())
    found = [l for l in LABELS if l in norm]
    return found[0] if len(found) == 1 else None


def parse(strategy, out):
    if strategy == "cot":
        m = re.findall(r"label\s*:\s*\**\s*([a-zA-Z_ \-]+)", out, flags=re.I)
        return to_label(m[-1]) if m else None
    return to_label(out)


# ---------- 5. run + score ----------
def run(args):
    pool, test = load(args.data)
    if args.limit:
        test = test[: args.limit]
    shots = pick_shots(pool, args.shots)
    client = make_client(args.base_url, args.api_key)
    results = []
    for strat in args.strategies:
        max_tok = 400 if strat == "cot" else 20
        for i, row in enumerate(test, 1):
            msgs = build_messages(strat, row["text"], shots)
            t0 = time.time()
            try:
                out, pt, ct = ask(client, args.model, msgs, max_tok)
            except Exception as e:
                out, pt, ct = f"ERROR: {e}", 0, 0
            dt = time.time() - t0
            pred = parse(strat, out)
            results.append(dict(strategy=strat, id=row["id"], text=row["text"], true=row["label"],
                                pred=pred, ok=pred == row["label"], ambiguous=int(row["ambiguous"]),
                                prompt_tokens=pt, completion_tokens=ct, seconds=round(dt, 2), raw=out))
            print(f"[{strat}] {i}/{len(test)}  true={row['label']:<16} pred={pred}", flush=True)
    with open("results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report(results, args)


def report(results, args):
    lines = [f"Model: {args.model}   Tickets: {len({r['id'] for r in results})}   Shots: {args.shots}\n",
             "| Strategy | Accuracy | Ambiguous acc. | Parse fails | Avg in tok | Avg out tok | Avg sec | Total min |",
             "|---|---|---|---|---|---|---|---|"]
    for s in dict.fromkeys(r["strategy"] for r in results):
        rs = [r for r in results if r["strategy"] == s]
        amb = [r for r in rs if r["ambiguous"]]
        n = len(rs)
        lines.append(f"| {s} | {sum(r['ok'] for r in rs)/n:.1%} | "
                     f"{(sum(r['ok'] for r in amb)/len(amb) if amb else 0):.1%} | "
                     f"{sum(r['pred'] is None for r in rs)} | "
                     f"{sum(r['prompt_tokens'] for r in rs)/n:.0f} | {sum(r['completion_tokens'] for r in rs)/n:.0f} | "
                     f"{sum(r['seconds'] for r in rs)/n:.2f} | {sum(r['seconds'] for r in rs)/60:.1f} |")
    lines.append("")
    for s in dict.fromkeys(r["strategy"] for r in results):
        cm = Counter((r["true"], r["pred"] or "none") for r in results if r["strategy"] == s)
        cols = LABELS + ["none"]
        lines.append(f"Confusion matrix: {s} (rows = true, columns = predicted)")
        lines.append("| true \\ pred | " + " | ".join(cols) + " |")
        lines.append("|---|" + "---|" * len(cols))
        for t in LABELS:
            lines.append(f"| {t} | " + " | ".join(str(cm[(t, c)]) for c in cols) + " |")
        lines.append("")
    text = "\n".join(lines)
    print("\n" + text)
    open("results.md", "w", encoding="utf-8").write(text)
    print("Saved results.md and results.jsonl (open results.jsonl to read every wrong answer).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llama3.2")
    p.add_argument("--base-url", default="http://localhost:11434/v1")   # Ollama
    p.add_argument("--api-key", default=os.getenv("LLM_API_KEY", "ollama"))
    p.add_argument("--data", default="tickets.csv")
    p.add_argument("--shots", type=int, default=6)
    p.add_argument("--limit", type=int, default=0, help="only first N test tickets (quick check)")
    p.add_argument("--strategies", nargs="+", default=["zero_shot", "few_shot", "cot"])
    run(p.parse_args())
