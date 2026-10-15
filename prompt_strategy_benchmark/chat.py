from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "llama3.2"

SYSTEM_PROMPT = {"role": "system", "content": "You are a helpful, concise assistant."}
history = [SYSTEM_PROMPT]

print("Commands: 'reset' clears memory, 'quit' exits.")

while True:
    user_text = input("\nYou: ")
    command = user_text.strip().lower()

    if command in ("quit", "exit"):
        break

    if command == "reset":
        history = [SYSTEM_PROMPT]          # throw away everything except the system prompt
        print("Memory cleared. Starting a fresh conversation.")
        continue                           # go back to the top of the loop

    history.append({"role": "user", "content": user_text})

    print("Llama: ", end="", flush=True)
    reply = ""
    stream = client.chat.completions.create(model=MODEL, messages=history, stream=True)
    for chunk in stream:
        piece = chunk.choices[0].delta.content or ""
        print(piece, end="", flush=True)
        reply += piece
    print()

    history.append({"role": "assistant", "content": reply})