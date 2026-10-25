#
# Long-term memory: after each conversation the LLM updates a few notes about the user,
# one file per persona (memory/<persona>.json, git-ignored). Delete the file to forget.
#

import json
import os
from datetime import datetime
from pathlib import Path

from loguru import logger

MEMORY_DIR = Path(__file__).parent / "memory"
MIN_USER_TURNS = 2  # don't bother remembering a hello and goodbye

UPDATE_PROMPT = """Eres la memoria de {name}, un personaje de voz que habla con un usuario. Actualiza tus notas sobre el usuario con la conversación de hoy ({date}).

Guarda solo lo que sirva en futuras conversaciones: su nombre, lo que haya contado de sí mismo, sus gustos, sus objetivos, lo que quedó pendiente y de qué hablasteis la última vez. Conserva lo importante de las notas anteriores y quita lo que ya no sea cierto. Escribe frases cortas en español, como máximo 150 palabras, sin títulos ni listas. Responde solo con las notas.

Notas anteriores:
{notes}

Conversación de hoy:
{transcript}"""


def load_notes(persona_id: str) -> str:
    try:
        return json.loads((MEMORY_DIR / f"{persona_id}.json").read_text())["notes"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return ""


def _transcript(messages: list, name: str) -> tuple[str, int]:
    """Plain user/assistant text of a conversation, skipping the prompt and images."""
    lines, user_turns = [], 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(part.get("text", "") for part in content if part.get("type") == "text")
        if not isinstance(content, str) or not content.strip():
            continue
        content = content.strip()
        if message.get("role") == "user":
            if content.startswith("(") or len(content) > 1500:  # connection note or the prompt itself
                continue
            user_turns += 1
            lines.append(f"Usuario: {content}")
        elif message.get("role") == "assistant":
            lines.append(f"{name}: {content}")
    return "\n".join(lines), user_turns


async def _complete(prompt: str) -> str:
    """One-off completion with the same LLM provider as the bot."""
    if os.getenv("LLM_PROVIDER", "local") == "anthropic":
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = await client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
            max_tokens=1024,
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")

    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url="http://127.0.0.1:1234/v1", api_key="dummyKey")
    response = await client.chat.completions.create(
        model="google/gemma-4-e4b",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"reasoning_effort": "none"},
    )
    return response.choices[0].message.content or ""


async def update_notes(persona_id: str, name: str, messages: list):
    transcript, user_turns = _transcript(messages, name)
    if user_turns < MIN_USER_TURNS:
        logger.debug(f"Memory: conversation too short to remember ({user_turns} user turns)")
        return
    try:
        notes = await _complete(
            UPDATE_PROMPT.format(
                name=name,
                date=datetime.now().strftime("%d/%m/%Y"),
                notes=load_notes(persona_id) or "(ninguna, es la primera conversación)",
                transcript=transcript,
            )
        )
    except Exception as e:
        logger.warning(f"Memory: could not update the notes: {e}")
        return
    if notes.strip():
        MEMORY_DIR.mkdir(exist_ok=True)
        (MEMORY_DIR / f"{persona_id}.json").write_text(
            json.dumps({"updated": datetime.now().isoformat(timespec="seconds"), "notes": notes.strip()},
                       ensure_ascii=False, indent=2)
        )
        logger.info(f"Memory: notes for {persona_id} updated")
