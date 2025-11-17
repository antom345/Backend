# backend_server.py
# FastAPI + OpenAI backend for language tutor

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import json

from openai import OpenAI

# ------------ OpenAI client ------------

# Ключ берётся из переменной окружения OPENAI_API_KEY
# пример: export OPENAI_API_KEY="sk-...."
client = OpenAI()

# ------------ Pydantic модели ------------

class Message(BaseModel):
    role: str   # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    language: Optional[str] = "English"
    topic: Optional[str] = "General conversation"
    level: Optional[str] = "B1"
    user_gender: Optional[str] = "unspecified"  # male / female / unspecified
    user_age: Optional[int] = None
    partner_gender: Optional[str] = "female"    # male / female

class ChatResponse(BaseModel):
    reply: str
    corrections_text: str

class TranslateRequest(BaseModel):
    word: str
    language: Optional[str] = "English"          # язык слова
    target_language: Optional[str] = "Russian"   # язык перевода

class TranslateResponse(BaseModel):
    word: str
    translation: str
    example: str


# ------------ Вспомогательная логика ------------

def build_system_prompt(req: ChatRequest) -> str:
    """
    Создаёт системный промпт для собеседника.
    Здесь задаём стиль общения, уровень, роль и т.д.
    """

    lang = req.language or "English"
    topic = req.topic or "General conversation"
    level = req.level or "B1"

    user_gender = req.user_gender or "unspecified"
    user_age = req.user_age
    partner_gender = req.partner_gender or "female"

    partner_gender_text = "a woman" if partner_gender == "female" else "a man"

    profile_bits = []
    if user_gender != "unspecified":
        profile_bits.append(f"user gender: {user_gender}")
    if user_age is not None:
        profile_bits.append(f"user age: {user_age}")
    profile_text = ", ".join(profile_bits) if profile_bits else "no extra profile data"

    return f"""
You are a friendly {partner_gender_text} who is a native speaker of {lang}.
You are chatting with a learner in a NATURAL, realistic way, like a new friend,
NOT like a generic assistant or AI model.

User profile: {profile_text}
User level in this language: {level} (A1–C2 scale).

Your main goals:
- Keep the conversation casual, personal and interesting.
- Stay in character as a real person at all times. Never say that you are an AI or a language model.
- Always reply ONLY in {lang}. Do not switch to other languages.
- Keep replies short and conversational (1–3 sentences).
- Ask questions sometimes, but not after every single message.
- If the user writes something off-topic, you may gently bring the conversation back to the main topic,
  but in a natural way.

Error correction rules:
- If the user's message contains IMPORTANT mistakes in grammar, word choice or word order,
  you must correct them.
- Ignore tiny things like capitalization and minor punctuation unless they make the sentence confusing.
- When you correct mistakes, you MUST provide:
  1) A normal conversational reply in {lang}.
  2) A SHORT list of corrections in Russian: e.g. "1) I wants → I want (ошибка в согласовании)."
- Если существенных ошибок нет, напиши пустую строку для corrections_text.

Conversation topic: {topic}.

VERY IMPORTANT OUTPUT FORMAT:
You MUST answer STRICTLY as JSON, no extra text, no explanations.
The JSON format is:

{{
  "reply": "your short answer in {lang}",
  "corrections_text": "краткий список исправлений на русском или пустая строка"
}}

Special case – first turn:
- If there are NO user messages yet (the messages array is empty),
  you should START the conversation yourself with a simple engaging question
  related to the topic. Still use the same JSON format.
""".strip()


def call_openai_chat(system_prompt: str, history: List[Message]) -> ChatResponse:
    """
    Вызывает OpenAI Chat API и возвращает ChatResponse.
    Парсим JSON, который вернул модель.
    """

    # История диалога в формате OpenAI
    history_messages = [
        {"role": msg.role, "content": msg.content}
        for msg in history
    ]

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_messages)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
    )

    content = completion.choices[0].message.content or ""

    # Пытаемся распарсить JSON
    reply_text = ""
    corrections_text = ""

    try:
        data = json.loads(content)
        reply_text = data.get("reply", "") or ""
        corrections_text = data.get("corrections_text", "") or ""
    except json.JSONDecodeError:
        # если вдруг модель не смогла соблюсти формат — хоть что-то отдадим
        reply_text = content
        corrections_text = ""

    return ChatResponse(reply=reply_text.strip(), corrections_text=corrections_text.strip())


def call_openai_translate(req: TranslateRequest) -> TranslateResponse:
    """
    Вызывает OpenAI для перевода отдельного слова + пример.
    """

    system_prompt = f"""
You are a concise bilingual dictionary.
The user will give you ONE word or a very short phrase in {req.language}.
Your task:

1) Translate it into {req.target_language}.
2) Give ONE simple example sentence in {req.language} that naturally uses this word/phrase.
3) The example should be understandable for a language learner (A2–B1 level).

VERY IMPORTANT: respond STRICTLY as JSON, without any extra text, in this format:

{{
  "translation": "перевод на {req.target_language}",
  "example": "Example sentence in {req.language}."
}}
""".strip()

    user_content = req.word.strip()

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )

    content = completion.choices[0].message.content or ""
    translation = ""
    example = ""

    try:
        data = json.loads(content)
        translation = data.get("translation", "") or ""
        example = data.get("example", "") or ""
    except json.JSONDecodeError:
        translation = content.strip()
        example = ""

    return TranslateResponse(
        word=req.word,
        translation=translation.strip(),
        example=example.strip(),
    )


# ------------ FastAPI приложение ------------

app = FastAPI(title="Language Tutor Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # при желании можно ограничить
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    """
    Основной эндпоинт диалога.
    """
    system_prompt = build_system_prompt(payload)
    result = call_openai_chat(system_prompt, payload.messages)
    return result


@app.post("/translate", response_model=TranslateResponse)
async def translate_endpoint(payload: TranslateRequest):
    """
    Перевод отдельного слова + пример.
    """
    result = call_openai_translate(payload)
    return result


# ------------ Локальный запуск ------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
