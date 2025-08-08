import os
import uuid
import google.generativeai as genai
from flask import Flask, request
from telegram import Update, InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import (
    Application, ContextTypes,
    MessageHandler, InlineQueryHandler,
    filters
)

# === Load tokens from environment ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# === Gemini setup ===
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-pro")

# === Flask for Render hosting ===
app = Flask(__name__)

# === Telegram setup ===
telegram_app = Application.builder().token(TELEGRAM_TOKEN).build()

# === Handle normal messages ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    try:
        response = model.generate_content(user_input)
        await update.message.reply_text(response.text)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Gemini Error: {str(e)}")

# === Handle inline queries ===
async def handle_inline_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.inline_query.query.strip()
    if not query:
        return

    try:
        response = model.generate_content(query)
        answer = response.text
        preview = answer[:50] + "..." if len(answer) > 50 else answer

        result = InlineQueryResultArticle(
            id=str(uuid.uuid4()),
            title="💡 Gemini AI Answer",
            description=preview,
            input_message_content=InputTextMessageContent(answer)
        )

        await update.inline_query.answer([result], cache_time=1)
    except Exception as e:
        print("Gemini error:", e)

# === Webhook route ===
@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), telegram_app.bot)
    telegram_app.update_queue.put_nowait(update)
    return "ok"

@app.route("/")
def home():
    return "✅ Gemini Inline Bot is Online!"

# === Register handlers ===
telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
telegram_app.add_handler(InlineQueryHandler(handle_inline_query))

# === Set webhook ===
async def set_webhook():
    # Change this to your domain on Render.com
    webhook_url = os.getenv("WEBHOOK_URL") or f"https://your-render-service.onrender.com/{TELEGRAM_TOKEN}"
    await telegram_app.bot.set_webhook(url=webhook_url)

telegram_app.run_task(set_webhook)

# === Start Flask app ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)