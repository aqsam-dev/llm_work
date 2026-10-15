import sys
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

FACTS_MODEL = "llama3.2"     
WRITER_MODEL = "llama3.2"    
MAX_CHARS = 3000             # small models have small context windows


def get_page_text(url):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    for tag in soup(["head", "script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    body = soup.get_text(separator="\n", strip=True)
    return f"Title: {title}\n\n{body}"[:MAX_CHARS]


def chat(model, system, user, temperature):
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=temperature,
    )
    return r.choices[0].message.content


def extract_facts(page):
    """LLM call 1: turn messy page text into clean facts. Temperature 0 = no creativity."""
    return chat(
        FACTS_MODEL,
        "You extract facts from web pages. You never invent anything.",
        "From the text below, list the key facts as short bullets: what the organisation "
        "does, what it offers, who it is for, and any notable details. "
        "Use only the text. If something is not stated, leave it out.\n\n" + page,
        temperature=0,
    )


def write_brochure(facts, url):
    """LLM call 2: turn the facts into a brochure. Higher temperature = more varied wording."""
    return chat(
        WRITER_MODEL,
        "You are a professional brand copywriter and brochure designer.",
        f"Design a one-page brochure in Markdown for the website {url}, "
        "using ONLY the facts below.\n\n"
        "Layout:\n"
        "- A headline and a one-line tagline\n"
        "- A short 'About us' paragraph\n"
        "- 3 to 4 highlighted offerings or benefits (bold title + one sentence each)\n"
        "- 'Who it is for' section\n"
        "- A closing call to action\n\n"
        "Rules: do not invent prices, statistics, awards, names or contact details. "
        "If a section has no facts, skip it. Friendly, confident tone.\n\n"
        "FACTS:\n" + facts,
        temperature=0.7,
    )


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"

    print(f"Scraping {url} ...")
    page = get_page_text(url)

    print("\n=== STEP 1: facts ===")
    facts = extract_facts(page)
    print(facts)

    print("\n=== STEP 2: brochure ===")
    brochure = write_brochure(facts, url)
    print(brochure)

    filename = urlparse(url).netloc.replace(".", "_") + "_brochure.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(brochure)
    print(f"\nSaved to {filename}")