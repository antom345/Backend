# backend_server.py
# Backend для языкового собеседника (FastAPI + OpenAI)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import json

from openai import OpenAI

# ---------- Настройки OpenAI ----------

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set in environment")

client = OpenAI(api_key=api_key)


# ---------- Pydantic модели ----------

class Message(BaseModel):
    role: str   # "user" или "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    language: Optional[str] = "English"
    topic: Optional[str] = "General conversation"

    # доп. параметры
    level: Optional[str] = "B1"                 # A1, A2, B1, B2, C1, C2
    user_gender: Optional[str] = "unspecified"  # "male", "female", "unspecified"
    user_age: Optional[int] = None
    partner_gender: Optional[str] = "female"    # "male" или "female"


class ChatResponse(BaseModel):
    reply: str
    partner_name: str


class TranslationRequest(BaseModel):
    word: str        # слово или короткая фраза на изучаемом языке
    language: str    # English / German / Italian / Korean и т.д.


class TranslationResponse(BaseModel):
    translation: str  # перевод на русский
    example: str      # пример на изучаемом языке


# ---------- Логика промптов и выбора собеседника ----------

def get_partner_name(language: str, partner_gender: str) -> str:
    """
    Подбирает имя собеседника под язык и пол.
    partner_gender: "male" или "female"
    """
    gender = (partner_gender or "").lower()
    names = {
        "English": {
            "male": "James",
            "female": "Emily",
        },
        "German": {
            "male": "Lukas",
            "female": "Anna",
        },
        "French": {
            "male": "Pierre",
            "female": "Marie",
        },
        "Spanish": {
            "male": "Carlos",
            "female": "Sofia",
        },
        "Italian": {
            "male": "Marco",
            "female": "Giulia",
        },
        "Korean": {
            "male": "Minjun",
            "female": "Jisoo",
        },
    }

    lang_map = names.get(language, names["English"])
    if gender not in ("male", "female"):
        gender = "female"
    return lang_map.get(gender, "Emily")


def create_system_prompt(
    language: str,
    partner_name: str,
    topic: str,
    level: str,
    user_gender: str,
    user_age: Optional[int],
    partner_gender: str,
) -> str:
    """
    Создаёт системный промпт, который описывает поведение собеседника.
    """

    # Описание пользователя для контекста
    user_descr_parts = []
    if user_age:
        user_descr_parts.append(f"{user_age} years old")
    if user_gender and user_gender.lower() in ("male", "female"):
        user_descr_parts.append(user_gender.lower())
    user_descr = ", ".join(user_descr_parts) if user_descr_parts else "no specific profile"

    gender_word = "man" if (partner_gender or "").lower() == "male" else "woman"

    return f"""
You are {partner_name}, a friendly {gender_word} and native speaker of {language}.
You are ONLY a human conversation partner for language practice, not an assistant and not a tool.
The user is a language learner (level approximately {level}, {user_descr}).

Your role:
- Have an interesting, engaging, slightly provocative (but polite) conversation.
- Stay in character as {partner_name} at all times. Never talk about models, prompts or AI.
- Do NOT answer questions like "what model are you", "are you ChatGPT", "write code", "translate this text",
  "explain complex math" etc. Politely refuse and redirect back to casual conversation in {language}.
- Use ONLY {language} in your replies.

Conversation style:
- Adapt to the user's level: {level}.
  * A1–A2: use simple sentences, basic vocabulary, speak slowly.
  * B1–B2: use richer vocabulary, but still clear and not too long.
  * C1–C2: speak naturally, but still keep sentences relatively short and clear.
- Keep answers short and conversational (about 1–3 sentences).
- Make the dialogue engaging: react to what the user says, share small personal details,
  ask follow-up questions from time to time (but not after every single sentence).
- Ask about opinions, experiences, feelings, hobbies – make the user talk more.

Error correction (VERY IMPORTANT):
- Carefully read the user's latest message and check it for errors.
- You MUST correct all important errors in:
  * verb forms (e.g. "he go", "i wants"),
  * basic grammar (articles, prepositions, word order),
  * clearly wrong word choices that change the meaning.
- You may IGNORE only very minor things such as:
  * punctuation that does not change the meaning,
  * capitalization alone, except for very important things like "I" in English.
- You must BOTH:
  1) answer as a normal person in {language}, reacting to the content,
  2) give a separate, clear block with corrections if there are important errors.

Output format for every reply:
1) First, write a natural, short reply (1–3 sentences) in {language}, continuing the conversation.
2) Then, IF there are important mistakes in the user's last message, add:

   A blank line, then:
   Corrections:
   - "wrong text" → "correct text" – short explanation in {language}

   Example:
   Corrections:
   - "i wants" → "I want" – we say "I want", not "I wants". Also "I" is always capitalized.

3) If there are NO important mistakes, do NOT add the "Corrections:" block at all.

Topic:
- Main topic: {topic}.
- You may naturally move a bit around the topic to keep the conversation interesting,
  but always stay within normal, safe, everyday topics.

Remember:
- You are {partner_name}, a real person, not a teacher in a classroom and not an AI assistant.
- Keep the user emotionally involved and comfortable speaking {language}.
- Answer ONLY in {language}.
"""


# ---------- FastAPI приложение ----------

app = FastAPI(title="Language Tutor Backend")

# CORS, чтобы приложение с любых устройств могло подключаться
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # при желании можно сузить
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


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    """
    Основной эндпоинт для диалога.
    Клиент отправляет:
      - messages: история диалога
      - language: язык (English, German, French, Spanish, Italian, Korean)
      - topic: тема разговора
      - level: уровень (A1–C2)
      - user_gender: пол пользователя
      - user_age: возраст пользователя
      - partner_gender: пол собеседника
    Сервер добавляет системный промпт и обращается к OpenAI.
    """
    language = payload.language or "English"
    topic = payload.topic or "General conversation"
    level = payload.level or "B1"
    user_gender = payload.user_gender or "unspecified"
    partner_gender = payload.partner_gender or "female"
    user_age = payload.user_age

    partner_name = get_partner_name(language, partner_gender)

    # Создаём системный промпт
    system_prompt = create_system_prompt(
        language=language,
        partner_name=partner_name,
        topic=topic,
        level=level,
        user_gender=user_gender,
        user_age=user_age,
        partner_gender=partner_gender,
    )

    # История сообщений в формате OpenAI
    history_messages = [
        {"role": msg.role, "content": msg.content}
        for msg in payload.messages
    ]

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_messages)

    # Запрос к OpenAI Chat Completions
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # быстрая и относительно дешёвая модель
        messages=messages,
    )

    reply_text = response.choices[0].message.content

    return ChatResponse(reply=reply_text, partner_name=partner_name)


@app.post("/translate_word", response_model=TranslationResponse)
async def translate_word(payload: TranslationRequest):
    """
    Перевод одного слова/фразы на русский + пример использования.
    """
    word = payload.word.strip()
    language = payload.language or "English"

    if not word:
        return TranslationResponse(translation="", example="")

    system_prompt = f"""
You are a bilingual dictionary for learners of {language} who speak Russian.
User gives you ONE word or short phrase in {language}.

You MUST respond ONLY in JSON with keys "translation" and "example":
- "translation": short natural translation into Russian.
- "example": one short, natural example sentence in {language} using this word.
Do NOT translate the example sentence into Russian.
Do NOT add any extra keys or text.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": word},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content

    try:
        data = json.loads(content)
    except Exception:
        data = {"translation": content, "example": ""}

    translation = data.get("translation", "")
    example = data.get("example", "")

    return TranslationResponse(translation=translation, example=example)


# ---------- Точка входа (для локального запуска) ----------

if __name__ == "__main__":
    import uvicorn

    # Запуск сервера: 0.0.0.0:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
