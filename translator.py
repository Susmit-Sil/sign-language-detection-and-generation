"""
translator.py
─────────────
Translate English sentences into ASL gloss notation using Ollama (Llama 3.1).

Includes content-moderation detection: if the model refuses to translate
(e.g. hateful/inappropriate input), a ContentRejectedError is raised
so the UI can show a proper error screen.
"""

import ast
import ollama


class ContentRejectedError(Exception):
    """Raised when the LLM refuses to translate due to content policy."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


# Phrases that indicate the model is refusing the request
_REFUSAL_INDICATORS = [
    "i cannot",
    "i can't",
    "i'm not able to",
    "i am not able to",
    "i won't",
    "i will not",
    "not appropriate",
    "inappropriate",
    "offensive",
    "harmful",
    "hate speech",
    "violent",
    "i'm unable",
    "i am unable",
    "against my guidelines",
    "cannot assist",
    "can't assist",
    "cannot help",
    "can't help",
    "not able to assist",
    "sorry, but i",
    "apologies, but",
    "i must decline",
    "cannot generate",
    "can't generate",
    "cannot translate",
    "can't translate",
    "not comfortable",
]


def _is_refusal(text: str) -> bool:
    """Check if the model's response is a content refusal."""
    lower = text.lower().strip()

    # If it starts with a bracket, it's probably a valid list
    if lower.startswith("["):
        return False

    # Check for refusal phrases
    for phrase in _REFUSAL_INDICATORS:
        if phrase in lower:
            return True

    return False


def _clean_refusal_message(text: str) -> str:
    """Extract a clean, user-friendly reason from the model's refusal."""
    # Take the first sentence as the reason
    text = text.strip()
    for sep in [".", "!", "\n"]:
        if sep in text:
            text = text[:text.index(sep) + 1]
            break

    # Cap length
    if len(text) > 200:
        text = text[:200] + "..."

    return text


def translate_to_gloss(text: str) -> list[str]:
    """
    Translates an English sentence into a list of core ASL glosses
    using a local Llama 3.1 model via Ollama.

    Args:
        text: An English sentence to translate.

    Returns:
        A Python list of ASL gloss strings in ALL CAPS.

    Raises:
        ContentRejectedError: If the model refuses due to content policy.
    """
    prompt = (
        "You are an American Sign Language (ASL) gloss translator. "
        "Convert the following English sentence into ASL gloss notation.\n\n"
        "Rules:\n"
        "- Return ONLY a valid Python list of strings.\n"
        "- Each string must be a single ASL gloss word in ALL CAPS.\n"
        "- Drop articles (a, an, the), linking verbs (am, is, are), "
        "and other words not used in ASL.\n"
        "- Use base/root sign forms (e.g., 'going' -> 'GO').\n"
        "- Do NOT include any explanation, greeting, or extra text.\n\n"
        f'English: "{text}"\n'
        "ASL Gloss List:"
    )

    response = ollama.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )

    raw = response["message"]["content"].strip()

    # ── Check for content refusal ─────────────────────────────────────────
    if _is_refusal(raw):
        reason = _clean_refusal_message(raw)
        raise ContentRejectedError(reason)

    # ── Parse the gloss list ──────────────────────────────────────────────
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]

    try:
        result = ast.literal_eval(raw)
        if isinstance(result, list) and all(isinstance(g, str) for g in result):
            return [g.upper() for g in result]
    except (ValueError, SyntaxError):
        pass

    # Fallback: try to clean up a comma-separated response
    cleaned = raw.strip("[]").replace("'", "").replace('"', "")
    return [word.strip().upper() for word in cleaned.split(",") if word.strip()]


if __name__ == "__main__":
    test_sentences = [
        "I am going to the store",
        "What is your name?",
        "She likes to eat pizza",
    ]
    for sentence in test_sentences:
        try:
            glosses = translate_to_gloss(sentence)
            print(f"Input:  {sentence!r}")
            print(f"Output: {glosses}\n")
        except ContentRejectedError as e:
            print(f"Input:  {sentence!r}")
            print(f"REJECTED: {e.reason}\n")
