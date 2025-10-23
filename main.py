import os
from dotenv import load_dotenv
load_dotenv()
import telebot
from telebot.types import InlineQueryResultArticle, InputTextMessageContent
import google.generativeai as genai
import uuid
from datetime import datetime, timedelta, timezone
import time
import json
from contextlib import contextmanager

@contextmanager
def typing_action(chat_id):
    try:
        bot.send_chat_action(chat_id, 'typing')
        yield
    finally:
        pass

def send_long(chat_id, text, chunk=3500):
    i = 0
    n = len(text)
    while i < n:
        bot.send_message(chat_id, text[i:i+chunk])
        i += chunk

def strip_mention(message):
    text = message.text or ""
    entities = getattr(message, 'entities', None)
    if not entities:
        return text.strip()
    # remove all @mentions safely using entity offsets
    offset_shift = 0
    for e in entities:
        if e.type in ('mention', 'text_mention'):
            s = e.offset - offset_shift
            l = e.length
            text = text[:s] + text[s+l:]
            offset_shift += l
    return text.strip()

def extract_messages_json(text: str):
    s = (text or "").strip()
    # Удаляем возможные тройные кавычки ```json ... ``` или просто ``` ... ```
    if s.startswith("```json"):
        s = s[len("```json"):].strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    elif s.startswith("```"):
        s = s[3:].strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
            return obj["messages"]
    except Exception:
        return None
    return None

# === Language preference & detection ===
LANG_KEYWORDS = {
    'en': [' in english', ' english', 'англ', 'по-англ', 'english plz', 'answer in english'],
    'ru': [' по-рус', ' russian', 'на русском', 'ответь по-русски', 'по русски'],
    'kk': ['қазақ', 'kazakh', 'қазақша', 'по-казахски', 'qazaq', 'qazaqsha', 'kz', 'kaz'],
    'es': [' en español', ' spanish', ' en espanol', 'respuesta en español'],
}

def detect_explicit_lang(text: str):
    t = (text or '').lower()
    for code, phrases in LANG_KEYWORDS.items():
        if any(p in t for p in phrases):
            return code
    return None

def guess_lang_by_chars(text: str):
    # Heuristic: detect basic script and Kazakh-specific letters
    kazakh_letters = set("әіңғүұқөһӘІҢҒҮҰҚӨҺ")
    latin = sum(1 for c in text if 'A' <= c <= 'Z' or 'a' <= c <= 'z')
    cyr   = sum(1 for c in text if 'А' <= c <= 'Я' or 'а' <= c <= 'я' or c in 'Ёё')
    has_kazakh = any(c in kazakh_letters for c in text)
    if has_kazakh and cyr > 0:
        return 'kk'
    if latin > cyr * 1.2:
        return 'en'
    if cyr > latin * 1.2:
        return 'ru'
    return None

# === КОНФИГ ===
BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# === НАСТРОЙКА Gemini ===
genai.configure(api_key=GEMINI_API_KEY)

def pick_gemini_model():
    try:
        models = list(genai.list_models())
        supported = [m.name for m in models if 'generateContent' in getattr(m, 'supported_generation_methods', [])]
        preferred = [
            'models/gemini-1.5-flash-latest',
            'models/gemini-1.5-flash',
            'models/gemini-1.5-flash-8b',
            'models/gemini-1.5-pro-latest',
            'models/gemini-1.5-pro',
        ]
        for p in preferred:
            if p in supported:
                return p
        # tolerate bare names without the 'models/' prefix
        bare_supported = {n.split('/')[-1]: n for n in supported}
        for p in [s.split('/')[-1] for s in preferred]:
            if p in bare_supported:
                return bare_supported[p]
        if supported:
            return supported[0]
    except Exception:
        pass
    # offline/default fallback
    return 'gemini-1.5-flash-latest'

MODEL_NAME = pick_gemini_model()
print(f"🤖 Using Gemini model: {MODEL_NAME}")
model = genai.GenerativeModel(MODEL_NAME)

# Инструкция для Gemini о формате ответа
FORMAT_POLICY = (
    "You are a Telegram assistant. If `Context: reply_language=XX` is provided, respond FULLY in that BCP‑47 language code (e.g., en, ru, kk, es) regardless of the system/history language. "
    "Otherwise, detect the language from the LAST 'User:' message and keep the ENTIRE reply in that language. "
    "For a normal reply, produce ONE single plain‑text message (no JSON, no code fences) composed of three lines:\n"
    "1) A short greeting + user's full name (I'll pass it as full_name). Use a natural greeting for the detected language.\n"
    "2) The user's request quoted verbatim but neatly (fix only obvious typos that do not change meaning). Do NOT prepend labels like 'Your question:' — just the quote itself. Use quotation marks typical for that language.\n"
    "3) The answer.\n"
    "Return plain text for this case — exactly three lines as described. No extra commentary, no JSON. "
    "ONLY when the user explicitly asks to send many separate Telegram messages (e.g., each step/number as its own message), return STRICT JSON without any fences/prefixes: {\"messages\": [\"msg1\", \"msg2\", ...]}. Each array element is one Telegram message. Max 100 messages."
)

# === БОТ ===
bot = telebot.TeleBot(BOT_TOKEN)

user_state = {}

@bot.inline_handler(lambda query: len(query.query) > 0)
def inline_query_handler(inline_query):
    query_text = inline_query.query

    try:
        # Запрос к Gemini
        response = model.generate_content(query_text)
        reply = response.text.strip()

        # Ответ в Telegram inline
        result = InlineQueryResultArticle(
            id=str(uuid.uuid4()),
            title='Ответ от Gemini',
            description=reply[:50],  # короткое описание
            input_message_content=InputTextMessageContent(reply)
        )

        if inline_query.id:
            bot.answer_inline_query(inline_query.id, [result])
    except Exception as e:
        error_result = InlineQueryResultArticle(
            id='error',
            title='Ошибка',
            description=str(e),
            input_message_content=InputTextMessageContent("Произошла ошибка при обращении к Gemini.")
        )
        if inline_query.id:
            bot.answer_inline_query(inline_query.id, [error_result])

# === Обычные сообщения (личка, группы) ===
@bot.message_handler(func=lambda message: message.text and (f"@{bot.get_me().username}" in message.text or message.chat.type == "private"))
def handle_text_message(message):
    original_user_prompt = strip_mention(message)

    # Если это reply на другое сообщение — подмешиваем текст того сообщения как контекст
    replied_text = ""
    if getattr(message, 'reply_to_message', None):
        replied_text = (getattr(message.reply_to_message, 'text', None) or
                        getattr(message.reply_to_message, 'caption', None) or "")
        replied_text = replied_text.strip()
        if replied_text:
            # Ограничим длину контекста, чтобы не раздувать промпт
            MAX_CTX = 1500
            ctx = replied_text[-MAX_CTX:]
            original_user_prompt = f"(Context from replied message: {ctx})\n{original_user_prompt}"

    chat_id = message.chat.id
    now = datetime.now(timezone.utc)
    state = user_state.setdefault(chat_id, {"mode": "assistant", "history": []})

    prefs = state.setdefault('prefs', {})
    # check explicit language request in the current message
    explicit = detect_explicit_lang(original_user_prompt)
    if explicit:
        prefs['reply_language'] = explicit
    # fallbacks: saved preference, or quick heuristic from chars
    reply_language = prefs.get('reply_language') or guess_lang_by_chars(original_user_prompt) or ''

    # Если текущий запрос очень короткий, попробуем определить язык по сообщению-контексту
    if (not reply_language or len(reply_language) == 0) and replied_text:
        guessed_ctx_lang = guess_lang_by_chars(replied_text)
        if guessed_ctx_lang:
            reply_language = guessed_ctx_lang
            prefs['reply_language'] = reply_language

    # Имя и фамилия пользователя для приветствия (передаём в Gemini)
    first_name = (message.from_user.first_name or "").strip()
    last_name = (message.from_user.last_name or "").strip()
    full_name = (first_name + " " + last_name).strip()

    # Очистка устаревшей истории (старше 20 минут)
    twenty_minutes_ago = now - timedelta(minutes=20)
    state["history"] = [item for item in state["history"] if item["timestamp"] >= twenty_minutes_ago]

    history_prompt = ""
    for item in state["history"]:
        history_prompt += f"User: {item['user']}\nAssistant: {item['bot']}\n"
    user_text = (f"System: {FORMAT_POLICY}\n"
                 f"Context: full_name={full_name}; reply_language={reply_language}\n"
                 f"{history_prompt}User: {original_user_prompt}\nAssistant:")

    print(f"[{message.chat.type}] {message.from_user.username}: {message.text}")
    try:
        response = model.generate_content(user_text)
        reply = response.text.strip()

        msgs = extract_messages_json(reply)
        if msgs:
            msgs = msgs[:100]
            with typing_action(chat_id):
                for m in msgs:
                    bot.send_message(chat_id, str(m))
                    time.sleep(0.2)
            state["history"].append({
                "timestamp": now,
                "user": message.text,
                "bot": f"[multi x{len(msgs)}]"
            })
            return
        # Иначе — одиночный ответ одним сообщением (как сформировал Gemini)
        with typing_action(chat_id):
            if len(reply) > 3500:
                send_long(chat_id, reply)
            else:
                bot.send_message(chat_id, reply)
        state["history"].append({
            "timestamp": now,
            "user": message.text,
            "bot": reply
        })
        return
    except Exception as e:
        bot.reply_to(message, f"⚠️ Ошибка: {str(e)}")

@bot.message_handler(commands=["history"])
def get_history(message):
    chat_id = message.chat.id
    history = user_state.get(chat_id, {}).get("history", [])
    if not history:
        bot.reply_to(message, "📭 История пуста.")
    else:
        text = "\n\n".join([f"🧍 {h['user']}\n🤖 {h['bot']}" for h in history[-5:]])
        bot.reply_to(message, f"🕓 История за 20 минут:\n\n{text}")

@bot.message_handler(commands=["clearhistory"])
def clear_history(message):
    chat_id = message.chat.id
    if chat_id in user_state:
        user_state[chat_id]["history"] = []
    bot.reply_to(message, "🧹 История очищена.")

print("✅ Bot run:")
bot.remove_webhook()
bot.infinity_polling(skip_pending=True, allowed_updates=['message', 'inline_query'])