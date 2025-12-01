# backend_server.py
# Backend для языкового собеседника (FastAPI + OpenAI)

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Literal
from openai import OpenAI
import os
import json
import base64
import requests
from typing import Dict
from io import BytesIO

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

class STTResponse(BaseModel):
    text: str          # распознанный текст
    language: str      # язык, который мы ожидали



class TranslateRequest(BaseModel):
    word: str
    language: Optional[str] = "English"
    with_audio: Optional[bool] = False


class TranslateResponse(BaseModel):
    translation: str
    example: str
    example_translation: str
    # base64-encoded audio (mp3) for the word pronunciation
    audio_base64: Optional[str] = None


class CoursePreferences(BaseModel):
    """Параметры ученика для генерации плана курса."""
    language: str                       # например: "English", "German"
    level_hint: Optional[str] = None    # например: "A2", "beginner"
    age: Optional[int] = None
    gender: Optional[Literal["male", "female", "other"]] = None
    goals: Optional[str] = None         # свободный текст: "переезд", "экзамен" и т.д.


class Lesson(BaseModel):
    """Один урок внутри уровня."""
    id: str
    title: str
    type: Literal["dialog", "vocab", "grammar", "mixed"]
    description: str


class CourseLevel(BaseModel):
    """Один уровень курса (ступень)."""
    level_index: int                    # 1, 2, 3...
    title: str
    description: str
    target_grammar: List[str]
    target_vocab: List[str]
    lessons: List[Lesson]


class CoursePlan(BaseModel):
    """Полный план курса из нескольких уровней."""
    language: str
    overall_level: str                  # например "A2", "B1"
    levels: List[CourseLevel]



class LessonRequest(BaseModel):
    """Запрос на генерацию конкретного урока."""
    language: str
    level_hint: Optional[str] = None
    lesson_title: str                 # название из CoursePlan
    grammar_topics: Optional[List[str]] = None
    vocab_topics: Optional[List[str]] = None


class LessonExercise(BaseModel):
    """
    Одно упражнение в уроке.

    type:
      - multiple_choice      — выбор правильного варианта
      - translate_sentence   — перевод предложения целиком
      - fill_in_blank        — пропуск в предложении
      - reorder_words        — расставить слова в правильном порядке
    """
    id: str
    type: Literal[
        "multiple_choice",
        "translate_sentence",
        "fill_in_blank",
        "reorder_words",
    ]

    # Общие поля
    question: str                          # что показываем пользователю (основный текст задания)
    explanation: str                       # короткое объяснение / разбор

    # Для multiple_choice
    options: Optional[List[str]] = None    # варианты ответа
    correct_index: Optional[int] = None    # индекс правильного варианта

    # Для translate_sentence / fill_in_blank
    correct_answer: Optional[str] = None   # правильный ответ / правильный перевод

    # Для fill_in_blank
    sentence_with_gap: Optional[str] = None  # строка с пропуском, например: "I ____ to school yesterday."

    # Для reorder_words
    reorder_words: Optional[List[str]] = None      # список слов в случайном порядке
    reorder_correct: Optional[List[str]] = None    # тот же список, но в правильном порядке


class LessonContent(BaseModel):
    """Контент целого урока: список упражнений."""
    lesson_id: str
    lesson_title: str
    description: str
    exercises: List[LessonExercise]


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
    partner_gender: Optional[str],
    partner_name: str,
) -> str:
    """Короткий system prompt без лишних персональных деталей."""
    lang = language or "English"
    level = level or "B1"
    topic = topic or "General conversation"
    partner_gender = partner_gender or "female"

    if partner_gender == "male":
        partner_role = "male friend"
    else:
        partner_role = "female friend"

    return f"""
You talk to a {level} {lang} learner. Your character name is {partner_name}. You are a friendly {partner_role} and native {lang} speaker. Keep the chat casual about {topic}.

Rules:
- Reply ONLY in {lang}, 1–3 sentences, natural and human-like.
- Stay in character; never say you are an AI.
- Correct ONLY the learner's last user message, never assistant messages.
- Put conversation text in "reply" only; put all corrections in "corrections_text" only.
- Correct grammar/word choice/word order; ignore capitalization and harmless punctuation.
- If there are no real mistakes, set "corrections_text" to an empty string.
- Do not repeat corrections or copy the user's original sentence.

Return STRICT JSON:
{{"reply":"...","corrections_text":"..."}}
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
        partner_gender=req.partner_gender,
        partner_name=partner_name,
    )

    history_messages = [
        {"role": msg.role, "content": msg.content}
        for msg in req.messages
    ]

    # Берём только последние 5 сообщений, чтобы экономить токены
    history_messages = history_messages[-5:]

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


def call_openai_translate(
    language: str,
    word: str,
    include_audio: bool = False,
) -> TranslateResponse:
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
    if include_audio:
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
    return call_openai_translate(
        lang,
        payload.word,
        include_audio=bool(payload.with_audio),
    )


@app.post("/stt", response_model=STTResponse)
async def stt_endpoint(
    language_code: str = "en",   # en, de, fr, es, it, ko, ru
    file: UploadFile = File(...),
):
    """
    Принимает аудиофайл, отправляет его в OpenAI на распознавание
    и возвращает текст.
    """
    audio_bytes = await file.read()
    print(f"[STT] len={len(audio_bytes)}, language_code={language_code}, filename={file.filename}")

    text = ""

    try:
        # Делаем "файл" из байтов — так, как ждёт openai-клиент
        audio_file = BytesIO(audio_bytes)
        audio_file.name = file.filename or "input.m4a"

        result = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file,
            language=language_code,   # жёстко задаём язык
        )

        # В новой библиотеке у результата есть поле .text
        text = getattr(result, "text", "") or ""
        print("[STT] recognized:", text)

    except Exception as e:
        print("[STT] error:", e)

    # Возвращаем нормальный объект, а не класс!
    return STTResponse(text=text, language=language_code)



@app.post("/generate_course_plan", response_model=CoursePlan)
def generate_course_plan(prefs: CoursePreferences):
    """
    Генерирует поуровневый план курса на основе предпочтений ученика.
    Пока НИЧЕГО не сохраняем, просто отдаём план фронтенду.
    """
    system_prompt = """
Ты методист по иностранным языкам и составляешь учебную программу.
Нужно сделать поуровневый план курса для ученика.

ДАЙ ОТВЕТ СТРОГО В ВИДЕ JSON СЛЕДУЮЩЕЙ СТРУКТУРЫ (БЕЗ ОБЪЯСНЕНИЙ ВНЕ JSON):

{
  "language": "...",
  "overall_level": "...",
  "levels": [
    {
      "level_index": 1,
      "title": "...",
      "description": "...",
      "target_grammar": ["..."],
      "target_vocab": ["..."],
      "lessons": [
        {
          "id": "1-1",
          "title": "...",
          "type": "dialog | vocab | grammar | mixed",
          "description": "..."
        }
      ]
    }
  ]
}

Требования:
- 3–5 уровней.
- В каждом уровне 3–6 уроков.
- Описание и названия на английском языке (кроме language, там просто название языка).
- Учитывай возраст, пол, цели и примерный уровень ученика.
"""

    # prefs.dict() превращаем в JSON-строку и отправляем как контекст
    user_content = json.dumps(prefs.dict(), ensure_ascii=False)

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",  # или та модель, которую ты используешь
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Вот данные ученика в JSON:\n{user_content}"}
        ],
        response_format={"type": "json_object"},
    )

    content = completion.choices[0].message.content or ""

    try:
        data = json.loads(content)
    except Exception:
        # на всякий случай, если модель сделает фигню
        raise ValueError("Failed to parse course plan JSON from model")

    # Pydantic сам проверит, что структура корректна
    return CoursePlan(**data)


@app.post("/generate_lesson", response_model=LessonContent)
def generate_lesson(req: LessonRequest):
    """
    Генерирует набор УМНЫХ упражнений для конкретного урока.
    Типы заданий: multiple_choice, translate_sentence, fill_in_blank, reorder_words.
    """
    system_prompt = """
Ты опытный преподаватель иностранного языка и методист.
Твоя задача — создать СТРУКТУРИРОВАННЫЙ УРОК с упражнениями наподобие Duolingo.

ВАЖНО:
- НЕ вести диалог, а выдать СТРОГО JSON-структуру.
- Урок должен быть связан с темой, уровнем и грамматикой/лексикой, которые я передам.
- Количество заданий: 6–8.
- Обязательно должны встретиться разные типы заданий: multiple_choice, translate_sentence, fill_in_blank, reorder_words.
- Язык target — это язык, который изучает пользователь (например, English, German и т.д.).
- Можно использовать исходный язык пользователя (например, Russian) для переводов, если это логично.

ТИПЫ ЗАДАНИЙ:

1) multiple_choice
   - Покажи пользователю вопрос (question).
   - Дай 3–5 вариантов (options).
   - Укажи индекс правильного ответа (correct_index).
   - В explanation коротко объясни, почему ответ верный (на target языке или кратко на языке пользователя).

2) translate_sentence
   - В question выведи предложение на target языке ИЛИ на языке пользователя.
   - Пользователь должен перевести его на другой язык (ты сам выбираешь направление, исходя из урока).
   - В correct_answer запиши идеальный/ожидаемый перевод.
   - В explanation объясни основные моменты (сложные слова, грамматику).

3) fill_in_blank
   - В sentence_with_gap сделай одно предложение target-языка с пропуском (например: "I ____ to school yesterday.").
   - В question можно кратко указать инструкцию: "Заполни пропуск правильной формой глагола".
   - В correct_answer укажи правильное слово/форму.
   - В explanation объясни, почему именно такая форма.

4) reorder_words
   - В reorder_words укажи список слов в ПЕРЕМЕШАННОМ порядке.
   - В reorder_correct укажи этот же список, но в ПРАВИЛЬНОМ порядке.
   - В question можно написать: "Собери правильное предложение из слов".
   - В explanation кратко объясни порядок слов, важные моменты.

СТРОГО ВАЖНЫЕ ПРАВИЛА:
- Выдай ТОЛЬКО JSON без комментариев и без дополнительного текста.
- Следуй СТРОГО заданной схеме JSON (типы полей, обязательные поля).
- Все строки должны быть валидными, без многоточий и без "..." внутри.
"""

    # Что передаём в модель как входные данные для урока
    user_payload = {
        "language": req.language,
        "level_hint": req.level_hint or "",
        "lesson_title": req.lesson_title,
        "grammar_topics": req.grammar_topics or [],
        "vocab_topics": req.vocab_topics or [],
    }

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Сгенерируй урок по следующим параметрам:\n"
                           + json.dumps(user_payload, ensure_ascii=False),
            },
        ],
        temperature=0.4,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "lessonContent",
                "schema": {
                    "type": "object",
                    "properties": {
                        "lesson_id": {"type": "string"},
                        "lesson_title": {"type": "string"},
                        "description": {"type": "string"},
                        "exercises": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "type": {
                                        "type": "string",
                                        "enum": [
                                            "multiple_choice",
                                            "translate_sentence",
                                            "fill_in_blank",
                                            "reorder_words",
                                        ],
                                    },
                                    "question": {"type": "string"},
                                    "explanation": {"type": "string"},
                                    "options": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "correct_index": {"type": "integer"},
                                    "correct_answer": {"type": "string"},
                                    "sentence_with_gap": {"type": "string"},
                                    "reorder_words": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "reorder_correct": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": [
                                    "id",
                                    "type",
                                    "question",
                                    "explanation",
                                ],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": [
                        "lesson_id",
                        "lesson_title",
                        "description",
                        "exercises",
                    ],
                    "additionalProperties": False,
                },
            },
        },
    )

    content = completion.choices[0].message.content or ""
    try:
        data = json.loads(content)
    except Exception:
        raise ValueError("Failed to parse lesson JSON from model")

    return LessonContent(**data)



# ---------- Локальный запуск ----------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
