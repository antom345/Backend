# backend_server.py
# Backend для языкового собеседника (FastAPI + OpenAI)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Literal
from openai import OpenAI
import os
import json


# ---------- OpenAI клиент ----------

# Ключ должен быть в переменной окружения OPENAI_API_KEY
# Пример: export OPENAI_API_KEY="sk-...."
client = OpenAI()


# ---------- Pydantic-модели ----------

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message] = []
    language: Optional[str] = "English"
    level: Optional[str] = "B1"              # A1–C2
    topic: Optional[str] = "General conversation"
    user_gender: Optional[str] = "unspecified"   # "male" / "female" / "unspecified"
    user_age: Optional[int] = None
    partner_gender: Optional[str] = "female"     # "male" / "female"


class ChatResponse(BaseModel):
    reply: str
    corrections_text: str
    partner_name: str


class TranslateRequest(BaseModel):
    word: str
    language: Optional[str] = "English"


class TranslateResponse(BaseModel):
    translation: str
    example: str
    example_translation: str


# ---------- Вспомогательные функции ----------

def get_partner_name(language: str, partner_gender: str) -> str:
    """Подбираем имя собеседника под язык и пол."""
    female_names = {
        "English": "Emily",
        "German": "Anna",
        "French": "Marie",
        "Spanish": "Sofía",
        "Italian": "Giulia",
        "Korean": "Ji-woo",
    }
    male_names = {
        "English": "Jack",
        "German": "Lukas",
        "French": "Pierre",
        "Spanish": "Carlos",
        "Italian": "Luca",
        "Korean": "Min-jun",
    }

    if partner_gender == "male":
        return male_names.get(language, "Alex")
    else:
        return female_names.get(language, "Emily")


def topics_for_language(language: str) -> List[str]:
    """Список готовых топиков для выбора во фронтенде."""
    base_topics = [
        "Daily life",
        "Friends and relationships",
        "Studies and university",
        "Work and career",
        "Travel and countries",
        "Hobbies and free time",
        "Movies, books and music",
        "Plans for the future",
    ]

    # Можно при желании делать локализацию,
    # пока просто возвращаем один и тот же список
    return base_topics


def build_system_prompt(req: ChatRequest) -> str:
    """
    Системный промпт: описывает поведение собеседника
    и формат JSON-ответа.
    """

    lang = req.language or "English"
    topic = req.topic or "General conversation"
    level = req.level or "B1"

    user_gender = (req.user_gender or "unspecified").lower()
    user_age = req.user_age
    partner_gender = (req.partner_gender or "female").lower()

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

ERROR CORRECTION RULES (VERY IMPORTANT):

- For EVERY user message you MUST check grammar, word choice and word order carefully.
- If there is AT LEAST ONE non-trivial mistake (for example:
  "I likes bananas", wrong preposition, wrong verb form, missing article,
  wrong word order, wrong tense, etc.), you MUST add corrections to
  the field "corrections_text".
- Write all corrections in Russian.
- Use this style for corrections_text:

  "1) I likes bananas → I like bananas (ошибка в согласовании подлежащего и сказуемого)."
  "2) He live in London → He lives in London (ошибка в форме глагола)."

- DO NOT mix corrections into the main reply. The main reply should be a normal,
  natural answer in {lang}.
- If the user's message is completely correct and there are NO meaningful errors,
  set corrections_text to an empty string "".

Conversation topic: {topic}.

FIRST MESSAGE RULE:

- If there are NO user messages yet (messages array is empty),
  you should START the conversation yourself with a simple engaging question
  related to the topic. Still use the same JSON format.

VERY IMPORTANT OUTPUT FORMAT:

You MUST answer STRICTLY as JSON, no extra text, no explanations.
The JSON format is:

{{
  "reply": "your short answer in {lang}",
  "corrections_text": "краткий список исправлений на русском или пустая строка"
}}
""".strip()


def call_openai_chat(req: ChatRequest, partner_name: str) -> ChatResponse:
    """
    Вызывает OpenAI Chat Completions и парсит JSON-ответ.
    """

    system_prompt = build_system_prompt(req)

    history_messages = [
        {"role": msg.role, "content": msg.content}
        for msg in req.messages
    ]

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_messages)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.4,
    )

    content = completion.choices[0].message.content or ""

    # Пытаемся распарсить JSON, как мы попросили в промпте
    reply_text = ""
    corrections_text = ""

    try:
        data = json.loads(content)
        reply_text = str(data.get("reply", "")).strip()
        corrections_text = str(data.get("corrections_text", "")).strip()
    except Exception:
        # если модель вдруг ответила не JSON
        reply_text = content.strip()
        corrections_text = ""

    if not reply_text:
        reply_text = "Sorry, something went wrong. Could you write that again?"

    return ChatResponse(
        reply=reply_text,
        corrections_text=corrections_text,
        partner_name=partner_name,
    )


def call_openai_translate(language: str, word: str) -> TranslateResponse:
    """
    Перевод одного слова/фразы на русский + пример и перевод примера.
    Ответ тоже в JSON.
    """

    system_prompt = f"""
You are a translator.
Your task: translate ONE word or a very short phrase from {language} to Russian
and give ONE short example sentence in {language} with this word,
AND also provide a Russian translation of this example sentence.

Answer STRICTLY as JSON, without any extra text:

{{
  "translation": "перевод на русский",
  "example": "short example sentence in {language} with this word",
  "example_translation": "перевод этого примера на русский"
}}
""".strip()


    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": word},
        ],
        temperature=0.3,
    )

    content = completion.choices[0].message.content.strip()

    try:
        data = json.loads(content)
        translation = str(data.get("translation", "")).strip()
        example = str(data.get("example", "")).strip()
        example_translation = str(data.get("example_translation", "")).strip()
    except Exception:
        # fallback: просто отдать весь текст в перевод
        translation = content.strip()
        example = ""
        example_translation = ""

    if not translation:
        translation = word

    # если по какой-то причине модель не дала перевод примера,
    # но дала сам пример, чтобы не было пустоты
    if not example_translation:
        example_translation = "перевод примера не указан"

    return TranslateResponse(
        translation=translation,
        example=example,
        example_translation=example_translation,
    )



# ---------- FastAPI приложение ----------

app = FastAPI(title="Language Tutor Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # при желании можно сузить
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Эндпоинты ----------

@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/topics")
async def get_topics(language: str = "English"):
    return {
        "language": language,
        "topics": topics_for_language(language),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    partner_name = get_partner_name(payload.language or "English",
                                    payload.partner_gender or "female")
    return call_openai_chat(payload, partner_name)


@app.post("/translate-word", response_model=TranslateResponse)
async def translate_word_endpoint(payload: TranslateRequest):
    lang = payload.language or "English"
    return call_openai_translate(lang, payload.word)


# ---------- Локальный запуск ----------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
