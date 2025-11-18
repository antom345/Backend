# backend_server.py
# Backend для языкового собеседника (FastAPI + OpenAI)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import json

from openai import OpenAI


# ---------- Настройки OpenAI ----------

# Ключ должен быть в переменной окружения OPENAI_API_KEY
# пример: export OPENAI_API_KEY="sk-...."
client = OpenAI()


# ---------- Pydantic модели ----------

class Message(BaseModel):
    role: str   # "user" или "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    language: str = "English"           # язык общения
    level: str = "B1"                   # A1–C2
    user_gender: Optional[str] = None   # "male" / "female" / None
    partner_gender: Optional[str] = None
    user_age: Optional[int] = None
    topic: str = "General conversation" # тема диалога


class ChatResponse(BaseModel):
    reply: str
    partner_name: str
    corrections: Optional[str] = None   # отдельный блок с ошибками (может быть null)


class TranslateRequest(BaseModel):
    word: str
    language: str = "English"


class TranslateResponse(BaseModel):
    translation: str
    example: str


class TopicsRequest(BaseModel):
    language: str = "English"
    level: str = "B1"


class TopicsResponse(BaseModel):
    topics: List[str]


# ---------- Вспомогательные функции ----------

def get_partner_name(language: str, partner_gender: Optional[str]) -> str:
    """
    Подбираем имя собеседника под язык и пол.
    """
    lang = language.lower()
    gender = (partner_gender or "").lower()

    names: Dict[str, Dict[str, str]] = {
        "english": {"female": "Emily", "male": "James"},
        "german": {"female": "Lena", "male": "Lukas"},
        "french": {"female": "Marie", "male": "Pierre"},
        "spanish": {"female": "Sofia", "male": "Carlos"},
        "italian": {"female": "Giulia", "male": "Luca"},
        "korean": {"female": "Ji-eun", "male": "Min-jun"},
    }

    by_lang = names.get(lang)
    if by_lang:
        if gender in by_lang:
            return by_lang[gender]
        # если пол не указан — берём первое имя
        return list(by_lang.values())[0]

    # запасной вариант
    return "Alex"


def create_system_prompt(
    language: str,
    level: str,
    user_gender: Optional[str],
    partner_gender: Optional[str],
    user_age: Optional[int],
    partner_name: str,
    topic: str,
) -> str:
    """
    Создаёт системный промпт, который описывает поведение собеседника.
    """

    level = level.upper()
    if level not in ["A1", "A2", "B1", "B2", "C1", "C2"]:
        level = "B1"

    user_profile_parts = []
    if user_gender:
        user_profile_parts.append(f"gender: {user_gender}")
    if user_age:
        user_profile_parts.append(f"age: {user_age}")
    user_profile = ", ".join(user_profile_parts) if user_profile_parts else "no extra profile info"

    return f"""
You are {partner_name}, a friendly and realistic native speaker of {language}.
You are chatting with a Russian learner of {language}.
The user profile: {user_profile}.
Stay in character as a real person, not as an AI or language model.

MAIN ROLE:
- You are a CONVERSATION PARTNER, not a general assistant.
- Keep the dialogue natural and engaging.
- Ask interesting follow-up questions connected to the topic.
- Do not talk about prompts, tokens, or AI stuff.
- Never say things like "as an AI" or "as a language model".

LANGUAGE & LEVEL:
- Conversation language: {language}.
- Approximate level of the learner: {level}.
- Adjust your vocabulary and grammar difficulty to this level.
  * A1–A2: very simple words and short sentences.
  * B1–B2: everyday vocabulary, natural but not too complex.
  * C1–C2: natural, rich language but still clear.

TOPIC:
- Main topic: {topic}.
- Keep most questions and comments related to this topic,
  but you may naturally connect it to everyday life and emotions.

STYLE:
- Write normal text with spaces between words (no joined words).
- Answers should be short and conversational: usually 1–3 sentences.
- Vary your questions so the dialogue feels alive and personal.
- Sometimes share a short opinion or small story about yourself.

ERROR CORRECTION:
- Your second task is to correct the learner's mistakes.
- Focus on important grammar, word choice and unnatural phrases.
- Ignore small things like capitalization or minor punctuation.
- When you correct, be clear and specific:
  * show the incorrect fragment,
  * show the corrected version,
  * optionally give a short Russian explanation.
- If there are no important mistakes, say that there are no serious errors.

OUTPUT FORMAT (VERY IMPORTANT):
Always answer in this exact structure, in English (or in the target language + Russian explanations):

REPLY: <your natural reply to the user>

CORRECTIONS: <corrections list OR "No major mistakes.">

Rules:
- Keep the word REPLY: and CORRECTIONS: exactly like this in English.
- Use normal spaces between all words.
- Do not add any other sections or headings.
"""


def build_topics(language: str, level: str) -> List[str]:
    """
    Немного разных тем в зависимости от языка / уровня.
    Можно потом расширять.
    """
    lang = language.lower()
    level = level.upper()

    generic = [
        "Daily routine",
        "Studies and work",
        "Travel and holidays",
        "Friends and relationships",
        "Hobbies and free time",
        "Food and cooking",
        "Plans for the future",
    ]

    if lang in ["english", "german", "french", "spanish", "italian", "korean"]:
        topics = generic.copy()
        if level in ["B2", "C1", "C2"]:
            topics.extend([
                "Cultural differences",
                "Social media and technology",
                "Big life decisions",
                "Dream job and career paths",
            ])
        return topics

    return generic


# ---------- FastAPI приложение ----------

app = FastAPI(title="Language Tutor Backend")

# Разрешаем CORS (чтобы можно было обращаться из Flutter-приложения и т.п.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # при желании можно ограничить
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Эндпоинты ----------

@app.get("/health")
async def health_check():
    """
    Простой эндпоинт для проверки, что сервер жив.
    """
    return {"status": "ok"}


@app.post("/topics", response_model=TopicsResponse)
async def topics_endpoint(payload: TopicsRequest):
    """
    Возвращает список возможных тем для выбранного языка и уровня.
    """
    topics = build_topics(payload.language, payload.level)
    return TopicsResponse(topics=topics)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    """
    Основной эндпоинт для диалога.
    Клиент отправляет историю сообщений, параметры пользователя,
    язык, уровень и тему.
    Сервер добавляет системный промпт и обращается к OpenAI.
    """

    partner_name = get_partner_name(payload.language, payload.partner_gender)

    system_prompt = create_system_prompt(
        language=payload.language,
        level=payload.level,
        user_gender=payload.user_gender,
        partner_gender=payload.partner_gender,
        user_age=payload.user_age,
        partner_name=partner_name,
        topic=payload.topic,
    )

    # История сообщений в формате OpenAI
    history_messages = [
        {"role": msg.role, "content": msg.content}
        for msg in payload.messages
    ]

    # Если история пустая — просим модель начать диалог первой
    if not history_messages:
        history_messages.append({
            "role": "user",
            "content": (
                "Start the conversation with me according to the instructions. "
                "Ask an engaging first question about the main topic."
            ),
        })

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_messages)

    # Запрос к OpenAI Chat Completions
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    raw_text = response.choices[0].message.content.strip()

    # Парсим структуру REPLY / CORRECTIONS
    reply_text = raw_text
    corrections_text: Optional[str] = None

    upper = raw_text.upper()
    if "REPLY:" in upper and "CORRECTIONS:" in upper:
        # пытаемся выделить части
        # сначала найдём позицию "REPLY:" и "CORRECTIONS:"
        idx_reply = upper.index("REPLY:")
        idx_corr = upper.index("CORRECTIONS:")

        reply_part = raw_text[idx_reply:idx_corr]
        corr_part = raw_text[idx_corr:]

        reply_text = reply_part.replace("REPLY:", "", 1).strip()
        corrections_text = corr_part.replace("CORRECTIONS:", "", 1).strip()

        # Если модель написала, что ошибок нет, возвращаем None
        if corrections_text.lower() in [
            "no mistakes.",
            "no major mistakes.",
            "no major mistakes",
            "no important mistakes.",
            "no important mistakes",
            "нет серьёзных ошибок.",
            "нет серьёзных ошибок",
        ]:
            corrections_text = None

    return ChatResponse(
        reply=reply_text,
        partner_name=partner_name,
        corrections=corrections_text,
    )


@app.post("/translate_word", response_model=TranslateResponse)
async def translate_word_endpoint(payload: TranslateRequest):
    """
    Перевод отдельного слова + пример предложения.
    """
    word = payload.word.strip()
    language = payload.language

    if not word:
        return TranslateResponse(translation="", example="")

    system_prompt = f"""
You are a bilingual dictionary for Russian speakers who learn {language}.
For the given word or short phrase in {language}:

- Give a natural Russian translation (1–3 words).
- Give ONE short example sentence in {language} that shows how to use this word.
- Use normal spaces between words.
- Answer ONLY in valid JSON with keys "translation" and "example".
Example:

{{"translation": "пример", "example": "This is an example sentence."}}
"""

    user_message = f'WORD: "{word}"\nLANGUAGE: {language}'

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    content = response.choices[0].message.content.strip()

    translation = ""
    example = ""

    # Пытаемся распарсить JSON из ответа
    try:
        data = json.loads(content)
        translation = str(data.get("translation", "")).strip()
        example = str(data.get("example", "")).strip()
    except Exception:
        # если модель вдруг не вернула JSON — просто возвращаем текст как перевод
        translation = content
        example = ""

    return TranslateResponse(translation=translation, example=example)


# ---------- Точка входа (локальный запуск) ----------

if __name__ == "__main__":
    import uvicorn

    # Запуск сервера: 0.0.0.0:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
