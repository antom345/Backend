# backend_server.py
# Backend для языкового собеседника (FastAPI + OpenAI)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Literal
from openai import OpenAI
import os
import json
import base64
import requests


# ---------- OpenAI клиент ----------

# Ключ должен быть в переменной окружения OPENAI_API_KEY
# Пример: export OPENAI_API_KEY="sk-...."
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Разрешаем запросы с фронтенда (Flutter Web / мобильный)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # при желании можно сузить до конкретных доменов
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Модели запросов/ответов ----------


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    language: str
    level: Optional[str] = "B1"              # A1–C2
    topic: Optional[str] = "General conversation"
    user_gender: Optional[str] = "unspecified"   # "male" / "female" / "unspecified"
    user_age: Optional[int] = None
    partner_gender: Optional[str] = "female"     # "male" / "female"
    messages: List[ChatMessage]


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
    # base64-encoded audio (mp3) for the word pronunciation
    audio_base64: Optional[str] = None


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
        return female_names.get(language, "Alex")


def build_system_prompt(
    language: str,
    level: Optional[str],
    topic: Optional[str],
    user_gender: Optional[str],
    user_age: Optional[int],
    partner_gender: Optional[str],
    partner_name: str,
) -> str:
    """Формируем system prompt для беседы с учетом параметров пользователя."""
    lang = language or "English"
    level = level or "B1"
    topic = topic or "General conversation"
    partner_gender = partner_gender or "female"

    if user_gender in ("male", "female"):
        user_gender_text = user_gender
    else:
        user_gender_text = "unspecified"

    age_text = f"{user_age} years old" if user_age else "age unspecified"

    if partner_gender == "male":
        partner_gender_text = "male friend"
    else:
        partner_gender_text = "female friend"

    profile_text = (
        f"The user is {age_text}, gender: {user_gender_text}. "
        f"They are learning {lang} at level {level}. Preferred topic: {topic}."
    )

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
- If the user writes something off-topic, you may gently bring the conversation back to the main topic
  ({topic}), but do it naturally.

VERY IMPORTANT:
1) You correct ONLY the learner's messages (messages with role "user").
2) Never correct or comment on the AI assistant's messages (role "assistant"),
   including your own previous replies.
3) For each turn, give corrections ONLY for the learner's last message.
4) If there are any mistakes (grammar, vocabulary, word order, etc.), correct them
   and explain very briefly.
5) If there are no important mistakes, still give the user 1–2 small suggestions
   how to sound more natural.

Respond STRICTLY as valid JSON with two fields:
{{
  "reply": "your short reply in {lang}",
  "corrections_text": "short corrections and tips in {lang} (or partly in user's native language if needed)"
}}
"""


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


def call_openai_chat(req: ChatRequest) -> ChatResponse:
    """Вызов OpenAI для чат-диалога с коррекциями."""
    partner_name = get_partner_name(req.language, req.partner_gender or "female")
    system_prompt = build_system_prompt(
        language=req.language,
        level=req.level,
        topic=req.topic,
        user_gender=req.user_gender,
        user_age=req.user_age,
        partner_gender=req.partner_gender,
        partner_name=partner_name,
    )

    history_messages = [
        {"role": msg.role, "content": msg.content}
        for msg in req.messages
    ]

    # Было ли хотя бы одно сообщение ученика?
    has_user_message = any(msg.role == "user" for msg in req.messages)


    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_messages)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.4,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "chatReply",
                "schema": {
                    "type": "object",
                    "properties": {
                        "reply": {"type": "string"},
                        "corrections_text": {"type": "string"},
                    },
                    "required": ["reply", "corrections_text"],
                    "additionalProperties": False,
                },
            },
        },
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

    # Если ученик ещё ни разу не писал (только первое приветствие Эмили) —
    # не показываем никаких исправлений
    if not has_user_message:
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
    Озвучка слова через OpenAI TTS.
    """

    # ---------- 1. Получаем перевод и пример через ChatGPT ----------
    system_prompt = f"""
You are a translator.
Your task: translate ONE word or a very short phrase from {language} to Russian
and give ONE short example sentence in {language} with this word,
AND also provide a Russian translation of this example sentence.

Answer STRICTLY as JSON, without any extra text:

{{
  "translation": "перевод на русский",
  "example": "пример предложения на {language}",
  "example_translation": "перевод примера на русский"
}}
"""

    user_prompt = f"Word: {word}\nLanguage: {language}\nTarget: Russian"

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "word_translation_with_example",
                "schema": {
                    "type": "object",
                    "properties": {
                        "translation": {"type": "string"},
                        "example": {"type": "string"},
                        "example_translation": {"type": "string"},
                    },
                    "required": [
                        "translation",
                        "example",
                        "example_translation",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        temperature=0.2,
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

    if not example_translation:
        example_translation = "перевод примера не указан"

    # ---- Генерация озвучки через OpenAI TTS (gpt-4o-mini-tts) ----
    audio_b64: Optional[str] = None
    try:
        text = (word or "").strip()
        if text:
            print(f"[TTS] OpenAI TTS for word={text!r}, language={language!r}")

            # БЕЗ format / response_format — твоя версия клиента этого не понимает
            tts_response = client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice="alloy",
                input=text,
            )

            # В разных версиях клиента ответ может быть либо байтами,
            # либо объектом с методом .read() — обработаем оба варианта
            audio_bytes = (
                tts_response
                if isinstance(tts_response, (bytes, bytearray))
                else tts_response.read()
            )

            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        print("TTS ERROR (OpenAI):", e)
        audio_b64 = None



    return TranslateResponse(
        translation=translation,
        example=example,
        example_translation=example_translation,
        audio_base64=audio_b64,
    )




# ---------- Эндпоинты FastAPI ----------


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
    return call_openai_chat(payload)


@app.post("/translate-word", response_model=TranslateResponse)
async def translate_word_endpoint(payload: TranslateRequest):
    lang = payload.language or "English"
    return call_openai_translate(lang, payload.word)



# ---------- Локальный запуск ----------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
