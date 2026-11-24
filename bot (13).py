import asyncio
import logging
import os
import re
import sqlite3

import httpx
from aiohttp import web
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv

load_dotenv()
BOT_TOKEN = os.environ["BOT_TOKEN"]
ADMIN_ID = int(os.environ["ADMIN_ID"])
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
DB_PATH = os.getenv("DB_PATH", "bot.db")
ANALYSIS_LIMIT = int(os.getenv("ANALYSIS_LIMIT", "300"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "100"))
PORT = int(os.getenv("PORT", "10000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("buxbot")

db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.executescript("""
CREATE TABLE IF NOT EXISTS companies (
    chat_id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL COLLATE NOCASE
);
CREATE TABLE IF NOT EXISTS messages (
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    sent_at TEXT NOT NULL,
    sender TEXT NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (chat_id, message_id)
);
""")
db.commit()

dp = Dispatcher()

ANALYSIS_INSTRUCTIONS = """Quyida kompaniya guruhidagi xabarlar berilgan. Format: [xabar_id] vaqt, yuboruvchi: matn.
Faqat shu xabarlarga asoslanib, o'zbek tilida quyidagi bo'limlarni yoz:
1. OCHIQ MUAMMOLAR - hali hal qilinmagan muammolar
2. HAL QILINGAN MUAMMOLAR - yechilgani aniq ko'rinib turgan masalalar
3. JAVOBSIZ SAVOLLAR - javob berilmagan savollar
4. MAS'UL SHAXSLAR - kim qaysi ishga javobgar ekani
Har bir band oxirida manba xabar id'sini [xabar_id] ko'rinishida ko'rsat.
Agar bo'lim bo'sh bo'lsa, "yo'q" deb yoz. Taxmin qilma."""

MERGE_INSTRUCTIONS = """Quyida bir kompaniya guruhi xabarlarining qismlar bo'yicha tahlil natijalari berilgan.
Ularni birlashtirib, bir xil formatda yakuniy hisobot yoz (o'zbek tilida, 4 ta bo'lim:
OCHIQ MUAMMOLAR, HAL QILINGAN MUAMMOLAR, JAVOBSIZ SAVOLLAR, MAS'UL SHAXSLAR).
Takrorlarni olib tashla, manba [xabar_id] larini saqla. Faqat berilgan ma'lumotga asoslan."""


def message_link(chat_id: int, message_id: int) -> str:
    return f"https://t.me/c/{str(chat_id)[4:]}/{message_id}"


async def ollama_chat(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=900) as client:
        r = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0.2},
            },
        )
        r.raise_for_status()
        return r.json()["message"]["content"]


def format_chunk(rows) -> str:
    return "\n".join(f"[{mid}] {sent}, {sender}: {text}" for mid, sent, sender, text in rows)


async def analyze(chat_id: int) -> str:
    rows = db.execute(
        "SELECT message_id, sent_at, sender, text FROM messages "
        "WHERE chat_id = ? ORDER BY message_id DESC LIMIT ?",
        (chat_id, ANALYSIS_LIMIT),
    ).fetchall()
    if not rows:
        return "Bu kompaniya guruhidan saqlangan xabar topilmadi."

    rows.reverse()
    partials = []
    for i in range(0, len(rows), CHUNK_SIZE):
        chunk = rows[i : i + CHUNK_SIZE]
        prompt = f"{ANALYSIS_INSTRUCTIONS}\n\n{format_chunk(chunk)}"
        partials.append(await ollama_chat(prompt))
        log.info("chunk %d/%d done", i // CHUNK_SIZE + 1, -(-len(rows) // CHUNK_SIZE))

    if len(partials) == 1:
        return partials[0]
    merged_input = "\n\n---\n\n".join(partials)
    return await ollama_chat(f"{MERGE_INSTRUCTIONS}\n\n{merged_input}")


def split_text(text: str, limit: int = 4000):
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        cut = cut if cut > 0 else limit
        yield text[:cut]
        text = text[cut:].lstrip("\n")
    if text:
        yield text


@dp.message(Command("kompaniya"))
async def register_company(message: types.Message):
    if message.from_user is None or message.from_user.id != ADMIN_ID:
        return
    if message.chat.type not in ("group", "supergroup"):
        await message.answer("Bu buyruq faqat kompaniya guruhida ishlaydi.")
        return
    name = (message.text or "").partition(" ")[2].strip()
    if not name:
        await message.answer("Foydalanish: /kompaniya Bux Prime")
        return
    try:
        db.execute(
            "INSERT INTO companies (chat_id, name) VALUES (?, ?) "
            "ON CONFLICT(chat_id) DO UPDATE SET name = excluded.name",
            (message.chat.id, name),
        )
        db.commit()
    except sqlite3.IntegrityError:
        await message.answer(f"'{name}' nomi boshqa guruhga biriktirilgan.")
        return
    await message.answer(f"Guruh '{name}' kompaniyasi sifatida saqlandi.")


@dp.message(Command("tahlil"))
async def run_analysis(message: types.Message):
    if message.from_user is None or message.from_user.id != ADMIN_ID:
        return
    if message.chat.type != "private":
        return
    name = (message.text or "").partition(" ")[2].strip()
    if not name:
        await message.answer("Foydalanish: /tahlil Bux Prime")
        return
    row = db.execute("SELECT chat_id FROM companies WHERE name = ?", (name,)).fetchone()
    if row is None:
        await message.answer(f"'{name}' topilmadi. Avval guruhda /kompaniya {name} deb yozing.")
        return

    await message.answer(f"'{name}' bo'yicha tahlil boshlandi. Bir necha daqiqa vaqt olishi mumkin.")
    try:
        report = await analyze(row[0])
    except httpx.HTTPError as e:
        log.exception("ollama error")
        await message.answer(f"LLM serveriga ulanishda xato: {e}. Ollama ishlayotganini tekshiring.")
        return
    report = re.sub(
        r"\[(\d+)\]",
        lambda m: message_link(row[0], int(m.group(1))),
        report,
    )
    for part in split_text(f"Tahlil: {name}\n\n{report}"):
        await message.answer(part)


@dp.message()
async def save_group_message(message: types.Message):
    if message.chat.type not in ("group", "supergroup"):
        return
    if not message.text or message.text.startswith("/"):
        return
    sender = message.from_user.full_name if message.from_user else "noma'lum"
    db.execute(
        "INSERT OR IGNORE INTO messages (chat_id, message_id, sent_at, sender, text) VALUES (?, ?, ?, ?, ?)",
        (message.chat.id, message.message_id, message.date.strftime("%Y-%m-%d %H:%M"), sender, message.text),
    )
    db.commit()


async def health(_: web.Request) -> web.Response:
    return web.Response(text="ok")


async def main():
    app = web.Application()
    app.router.add_get("/", health)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", PORT).start()
    log.info("web server on port %d", PORT)

    bot = Bot(BOT_TOKEN, default=DefaultBotProperties(parse_mode=None))
    log.info("bot started, model=%s", OLLAMA_MODEL)
    try:
        await dp.start_polling(bot)
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
