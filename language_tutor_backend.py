# backend_server.py
# Backend для языкового собеседника (FastAPI + OpenAI)

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from openai import OpenAI
import os
import json
import base64
import requests
from typing import Dict
from io import BytesIO


# ---------- System prompts ----------

COURSE_PLAN_SYSTEM_PROMPT = """
Ты — методист международной языковой школы и автор современных игровых курсов по иностранным языкам.

Твоя задача — создать СТРУКТУРИРОВАННЫЙ, ИНТЕРЕСНЫЙ и ЖИВОЙ ПЛАН КУРСА для конкретного ученика.

ВХОДНЫЕ ДАННЫЕ (ты получаешь их в user-сообщении в формате JSON):
- language: какой язык изучает ученик (например, "English", "German").
- level_hint: примерный уровень ученика (A1, A2, B1, B2, C1, C2 или пусто).
- age, gender: возраст и пол (нужны только для стилистики примеров).
- goals: цели ученика (например: переезд, учеба за границей, общение в путешествиях).
- interests (если есть в запросе): темы, которые нравятся ученику (путешествия, IT, спорт, фильмы, музыка и т.д.).

ТВОЯ ЗАДАЧА:
1. Разбить обучение на уровни (CourseLevel), согласованные с входным level_hint.
2. Для каждого уровня:
   - Придумать понятный title и description.
   - Задать общий список target_grammar (грамматические темы уровня).
   - Задать общий список target_vocab (лексические темы уровня).
   - Сформировать список lessons (уроков), где каждый урок — отдельная коммуникативная ситуация.

3. Для КАЖДОГО урока (Lesson) внутри уровня ОБЯЗАТЕЛЬНО укажи:
   - id: уникальная строка (можно просто "L1", "L2" и т.п. — главное, чтобы были разные).
   - title: короткое название урока.
   - type: "dialog", "vocab", "grammar" или "mixed" — в соответствии с текущей моделью кода.
   - description: краткое описание реальной жизненной ситуации с лёгким юмором (например, неловкое знакомство, смешной заказ в кафе и т.п.).
   - grammar_topics: список конкретных грамматических ПОДТЕМ именно этого урока (подмножество или детализация target_grammar).
   - vocab_topics: список конкретных лексических ПОДТЕМ именно этого урока (подмножество или детализация target_vocab).

ВАЖНО:
- target_grammar и target_vocab в CourseLevel — это ОБЩИЕ темы всего уровня.
- lesson.grammar_topics и lesson.vocab_topics — это ТЕМЫ КОНКРЕТНОГО УРОКА.
- Уроки одного уровня НЕ должны иметь идентичные списки grammar_topics и vocab_topics. Каждый урок отвечает за свою часть материала.
- Если level_hint относится к C1 или C2:
    * В target_grammar и lesson.grammar_topics НЕ добавляй базовую грамматику.
    * Основной фокус — на лексике и стилях: устойчивые выражения, идиомы, коллокации, академическая и профессиональная лексика, нюансы значений.
    * В vocab_topics делай акцент на сложных темах: общество, культура, технологии, профессия, дискуссии, искусство.

УЧЁТ ИНТЕРЕСОВ УЧЕНИКА:
- Если interests переданы, старайся, чтобы как минимум 60–70% уроков напрямую или косвенно касались этих интересов.
  Например:
    * "путешествия" → аэропорт, отель, экскурсии, переезд, новые города;
    * "IT" → встречи, проекты, удалённая работа, приложения, стартапы;
    * "спорт" → тренировки, соревнования, обсуждение матчей;
    * "фильмы и сериалы" → сюжеты, герои, жанры, рекомендации и т.д.
- Оставшиеся уроки могут покрывать общие бытовые и общественные темы, важные для языка.

СТИЛЬ:
- План курса должен быть живым и мотивирующим.
- Описания уроков допускают лёгкий юмор и жизненные ситуации, но без грубости и токсичности.
- Темы должны быть практичными: чтобы ученик понимал, где он сможет использовать этот язык в жизни.

ЛЕКСИЧЕСКАЯ НАГРУЗКА (ориентиры, чтобы равномерно распределять material по урокам):
- A1–A2: в каждом уроке 15–25 новых слов/выражений (определяешь через vocab_topics).
- B1: 25–40 слов/выражений.
- B2: 40–60 слов/выражений.
- C1–C2: 70–100 слов/выражений, в т.ч. устойчивые фразы, профессионализмы, идиомы.

ВЫВОД:
- Ты ВСЕГДА возвращаешь СТРОГО JSON БЕЗ какого-либо пояснительного текста.
- Структура JSON должна точно соответствовать Pydantic-моделям CoursePlan, CourseLevel и Lesson, которые уже есть в коде:
    * CoursePlan содержит список levels.
    * Каждый CourseLevel содержит level_index, title, description, target_grammar, target_vocab, lessons.
    * Каждый Lesson содержит id, title, type, description, grammar_topics, vocab_topics.
"""


LESSON_SYSTEM_PROMPT = """
Ты — опытный преподаватель иностранного языка и автор интерактивных упражнений в стиле Duolingo.

Ты СОЗДАЁШЬ ПОЛНОЕ СОДЕРЖАНИЕ ОДНОГО КОНКРЕТНОГО УРОКА.

В user-сообщении ты получаешь JSON с такими полями (названия могут совпадать с моделями из кода):
- language: язык урока (например, "English").
- level_hint: примерный уровень (A1, A2, B1, B2, C1, C2 или пусто).
- lesson_title: название урока.
- grammar_topics: список грамматических подтем ИМЕННО ЭТОГО урока.
- vocab_topics: список лексических подтем ИМЕННО ЭТОГО урока.
- при необходимости могут быть goals / interests (если они добавлены в код и передаются).

ТВОЯ ЗАДАЧА:
- На основе lesson_title, grammar_topics, vocab_topics, уровня и (при наличии) интересов ученика сгенерировать LessonContent:
  * описание урока;
  * список упражнений (LessonExercise) в строгом JSON-формате, который ожидает backend.

СТИЛЬ УРОКА:
- Урок должен быть живым, с лёгким юмором, но без грубостей.
- Все ситуации и предложения должны быть реалистичными: кафе, работа, путешествия, общение с друзьями, учёба и т.п.
- При наличии interests в данных урок можно слегка подстраивать под эти интересы (например, делать примеры про спорт, IT, путешествия и т.д.).

УРОВЕНЬ ЯЗЫКА:
- Уровни A1–A2:
    * Простые предложения.
    * Базовая лексика.
    * Грамматические темы — очень простые (настоящее время, артикли, простые местоимения).
    * Инструкции к упражнениям — максимально короткие и понятные.
- Уровень B1:
    * Более сложные предложения, но всё ещё без перегруза.
    * Появляются времена, модальные глаголы, более сложные структуры.
    * Лексика — бытовая и общественная (работа, хобби, путешествия).
- Уровень B2:
    * Длинные предложения, сложносочинённые конструкции.
    * Больше абстрактных тем (мнения, планы, обсуждение проблем).
    * Лексика — более продвинутая, но всё ещё практичная.
- Уровни C1 и C2:
    * НЕ СОЗДАВАЙ базовых грамматических упражнений.
    * Grammar_topics можно игнорировать или использовать только как описания уже известных структур.
    * Фокус на:
        - сложной лексике,
        - устойчивых выражениях (collocations),
        - идиомах,
        - перефразировании,
        - стилистике (формальный / неформальный регистр),
        - аргументации и выражении мнения,
        - понимании подтекста.
    * Даже если используются те же типы упражнений (multiple_choice, translate_sentence, fill_in_blank, reorder_words), содержание внутри них должно быть насыщенным и "взрослым".

ТИПЫ УПРАЖНЕНИЙ:
- Используй ТОЛЬКО те типы, которые уже поддерживаются текущей моделью данных в коде (не добавляй новые типы).
- Минимальный набор, который нужно обязательно задействовать:
    * multiple_choice
    * translate_sentence
    * fill_in_blank
    * reorder_words
- Для fill_in_blank:
    * ВСЕГДА заполняй sentence_with_gap — предложение с «___» на месте пропуска.
    * correct_answer — одно слово или короткое устойчивое выражение.
    * explanation — НЕ пустая строка, чётко объясняет, что вставить (часть речи, число слов, смысл), чтобы ответ был однозначен.
    * избегай двусмысленных предложений: естественный правильный ответ должен быть один.
- В пределах этих типов старайся делать задания разнообразными:
    * менять темы предложений;
    * использовать и диалоги, и отдельные фразы;
    * иногда добавлять лёгкий юмор или неловкие бытовые ситуации.

ОБЪЁМ:
- В одном уроке должно быть примерно 6–10 упражнений.
- Лексика из vocab_topics должна активно повторяться в разных заданиях, чтобы ученик её закреплял.

ВЫВОД:
- Ты ВСЕГДА возвращаешь СТРОГО JSON БЕЗ пояснительного текста.
- Структура JSON должна строго соответствовать Pydantic-модели LessonContent и вложенным моделям упражнений, которые уже есть в коде:
    * LessonContent содержит общую информацию об уроке и список exercises.
    * Каждый элемент в exercises имеет поля type, instruction, варианты ответов/правильный ответ и т.п. — ровно так, как это сейчас ожидается backend’ом.
"""

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
    interests: List[str] = Field(
        default_factory=list,
        description="Темы и сферы, которые интересны ученику (например: путешествия, работа, отношения, спорт, IT, искусство)",
    )


class Lesson(BaseModel):
    """Один урок внутри уровня."""
    id: str
    title: str
    type: Literal["dialog", "vocab", "grammar", "mixed"]
    description: str
    grammar_topics: List[str] = []
    vocab_topics: List[str] = []


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
    interests: Optional[List[str]] = None


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
    instruction: Optional[str] = None
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

    # prefs.dict() превращаем в JSON-строку и отправляем как контекст
    user_content = json.dumps(prefs.dict(), ensure_ascii=False)

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",  # или та модель, которую ты используешь
        messages=[
            {"role": "system", "content": COURSE_PLAN_SYSTEM_PROMPT},
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

    # Подстраховка: если модель не вернула обязательные поля, добиваем из prefs
    if (
        "language" not in data
        or not isinstance(data.get("language"), str)
        or not data.get("language", "").strip()
    ):
        data["language"] = prefs.language

    if (
        "overall_level" not in data
        or not isinstance(data.get("overall_level"), str)
    ):
        data["overall_level"] = (prefs.level_hint or "").strip()

    if "levels" not in data or not isinstance(data["levels"], list):
        raise ValueError("Model did not provide 'levels' list in course plan")

    # Pydantic сам проверит, что структура корректна
    return CoursePlan(**data)


@app.post("/generate_lesson", response_model=LessonContent)
def generate_lesson(req: LessonRequest):
    """
    Генерирует набор УМНЫХ упражнений для конкретного урока.
    Типы заданий: multiple_choice, translate_sentence, fill_in_blank, reorder_words.
    """

    # Что передаём в модель как входные данные для урока
    user_payload = {
        "language": req.language,
        "level_hint": req.level_hint or "",
        "lesson_title": req.lesson_title,
        "grammar_topics": req.grammar_topics or [],
        "vocab_topics": req.vocab_topics or [],
        "interests": req.interests or [],
    }

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": LESSON_SYSTEM_PROMPT},
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
                                    "instruction": {"type": "string"},
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

    # Подстраховка: проверяем обязательные поля и чистим упражнения
    if not isinstance(data, dict):
        raise ValueError("Lesson JSON root must be an object")

    if (
        "lesson_id" not in data
        or not isinstance(data.get("lesson_id"), str)
        or not data.get("lesson_id", "").strip()
    ):
        data["lesson_id"] = (req.lesson_title or "lesson").strip() or "lesson"

    if (
        "lesson_title" not in data
        or not isinstance(data.get("lesson_title"), str)
        or not data.get("lesson_title", "").strip()
    ):
        data["lesson_title"] = req.lesson_title

    if "description" not in data or not isinstance(data.get("description"), str):
        data["description"] = ""

    exercises = data.get("exercises")
    if not isinstance(exercises, list):
        raise ValueError("Model did not provide 'exercises' list in lesson JSON")

    fixed_exercises = []
    for idx, ex in enumerate(exercises):
        if not isinstance(ex, dict):
            continue

        if (
            "id" not in ex
            or not isinstance(ex.get("id"), str)
            or not ex.get("id", "").strip()
        ):
            ex["id"] = f"ex_{idx + 1}"

        if ex.get("type") not in (
            "multiple_choice",
            "translate_sentence",
            "fill_in_blank",
            "reorder_words",
        ):
            ex["type"] = "multiple_choice"

        if "question" not in ex or not isinstance(ex.get("question"), str):
            ex["question"] = ""

        if "explanation" not in ex or not isinstance(ex.get("explanation"), str):
            ex["explanation"] = ""

        fixed_exercises.append(ex)

    if not fixed_exercises:
        raise ValueError("Lesson has no valid exercises after cleanup")

    data["exercises"] = fixed_exercises

    return LessonContent(**data)



# ---------- Локальный запуск ----------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
