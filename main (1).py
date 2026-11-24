import asyncio
import html
import logging
import os
from datetime import datetime, timezone

import asyncpg
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import load_dotenv
from aiohttp import web
from google import genai

from open_items_service import format_open_items_html, get_open_problems_and_questions
from report_builder import build_structured_report
from resolve_service import mark_problem_solved
from scheduler_service import daily_report_scheduler_loop
from stats_service import format_daily_stats_html, get_daily_stats, save_analysis_report
from weekly_scheduler_service import weekly_summary_scheduler_loop
from weekly_summary_service import (
    format_all_companies_overview_html,
    format_company_weekly_html,
    get_weekly_company_summaries,
)

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN Environment Variable sozlanmagan.")
PORT = int(os.getenv("PORT", "10000"))
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
DATABASE_URL = os.environ["DATABASE_URL"]
SUPER_ADMIN_ID = int(os.environ["SUPER_ADMIN_ID"])
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

AUTO_REPORT_HOUR = int(os.getenv("AUTO_REPORT_HOUR", "19"))
AUTO_REPORT_MINUTE = int(os.getenv("AUTO_REPORT_MINUTE", "0"))
AUTO_REPORT_TZ = os.getenv("AUTO_REPORT_TZ", "Asia/Tashkent")
AUTO_REPORT_HOURS = int(os.getenv("AUTO_REPORT_HOURS", "24"))

WEEKLY_REPORT_WEEKDAY = os.getenv("WEEKLY_REPORT_WEEKDAY", "monday")
WEEKLY_REPORT_HOUR = int(os.getenv("WEEKLY_REPORT_HOUR", "9"))
WEEKLY_REPORT_MINUTE = int(os.getenv("WEEKLY_REPORT_MINUTE", "0"))
WEEKLY_REPORT_DAYS = int(os.getenv("WEEKLY_REPORT_DAYS", "7"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("group_analyzer_bot")

genai_client = genai.Client(api_key=GEMINI_API_KEY)


async def init_db(pool: asyncpg.Pool) -> None:
    """Barcha zarur jadvallar va indekslarni yaratadi."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tracked_groups (
                chat_id BIGINT PRIMARY KEY,
                company_name TEXT NOT NULL,
                group_title TEXT,
                report_chat_id BIGINT NOT NULL,
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS group_messages (
                id BIGSERIAL PRIMARY KEY,
                chat_id BIGINT NOT NULL,
                message_id BIGINT NOT NULL,
                reply_to_message_id BIGINT,
                sender_id BIGINT,
                sender_name TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (chat_id, message_id)
            );

            CREATE INDEX IF NOT EXISTS idx_group_messages_chat_time
                ON group_messages (chat_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS analysis_reports (
                id BIGSERIAL PRIMARY KEY,
                chat_id BIGINT NOT NULL,
                company_name TEXT NOT NULL,
                period_hours INT NOT NULL DEFAULT 24,
                messages_analyzed INT NOT NULL DEFAULT 0,
                problems_count INT NOT NULL DEFAULT 0,
                solved_count INT NOT NULL DEFAULT 0,
                in_progress_count INT NOT NULL DEFAULT 0,
                unsolved_count INT NOT NULL DEFAULT 0,
                unanswered_count INT NOT NULL DEFAULT 0,
                report_payload JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_analysis_reports_chat_created
                ON analysis_reports (chat_id, created_at DESC);
            """
        )


async def upsert_group_info(
    pool: asyncpg.Pool, chat_id: int, group_title: str | None
) -> bool:
    """Guruh faol bo'lsa nomini yangilaydi va True qaytaradi."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT is_active, group_title FROM tracked_groups WHERE chat_id = $1",
            chat_id,
        )
        if not row or not row["is_active"]:
            return False
        if group_title and row["group_title"] != group_title:
            await conn.execute(
                "UPDATE tracked_groups SET group_title = $1 WHERE chat_id = $2",
                group_title,
                chat_id,
            )
        return True


async def save_message(pool: asyncpg.Pool, msg: Message) -> None:
    """Guruhdagi matn yoki izohli xabarni bazaga yozadi."""
    content = (msg.text or msg.caption or "").strip()
    if not content:
        return

    sender_name = "Noma'lum"
    if msg.from_user:
        full_name = f"{msg.from_user.first_name or ''} {msg.from_user.last_name or ''}".strip()
        sender_name = full_name or (
            f"@{msg.from_user.username}" if msg.from_user.username else str(msg.from_user.id)
        )

    reply_id = msg.reply_to_message.message_id if msg.reply_to_message else None
    msg_time = msg.date.astimezone(timezone.utc) if msg.date else datetime.now(timezone.utc)

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO group_messages (
                chat_id, message_id, reply_to_message_id,
                sender_id, sender_name, text, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (chat_id, message_id) DO UPDATE
            SET text = EXCLUDED.text
            """,
            msg.chat.id,
            msg.message_id,
            reply_id,
            msg.from_user.id if msg.from_user else None,
            sender_name,
            content,
            msg_time,
        )


async def send_long_message(bot: Bot, chat_id: int, text: str) -> None:
    """Telegram 4096 belgilik limitidan oshmaslik uchun uzun xabarni bo'lib yuboradi."""
    max_len = 4000
    for i in range(0, len(text), max_len):
        chunk = text[i : i + max_len]
        await bot.send_message(chat_id=chat_id, text=chunk, parse_mode="HTML")


async def analyze_and_persist_group(
    pool: asyncpg.Pool,
    chat_id: int,
    hours: int,
) -> tuple[str, int, int]:
    """Bitta guruhni tahlil qiladi, natijani bazaga saqlaydi va hisobot hamda yuborish manzilini qaytaradi."""
    report_text, msg_count, report_target, parsed_analysis = await build_structured_report(
        pool=pool,
        genai_client=genai_client,
        model_name=GEMINI_MODEL,
        chat_id=chat_id,
        hours=hours,
    )

    if parsed_analysis is not None:
        async with pool.acquire() as conn:
            company_name = (
                await conn.fetchval(
                    "SELECT company_name FROM tracked_groups WHERE chat_id = $1",
                    chat_id,
                )
                or str(chat_id)
            )

        await save_analysis_report(
            pool=pool,
            chat_id=chat_id,
            company_name=company_name,
            period_hours=hours,
            messages_analyzed=msg_count,
            analysis=parsed_analysis,
        )

    return report_text, msg_count, report_target


def _is_super_admin(message: Message) -> bool:
    return bool(message.from_user and message.from_user.id == SUPER_ADMIN_ID)


def _format_user_display(message: Message) -> str:
    if not message.from_user:
        return "Admin"
    full_name = f"{message.from_user.first_name or ''} {message.from_user.last_name or ''}".strip()
    if full_name:
        return full_name
    if message.from_user.username:
        return f"@{message.from_user.username}"
    return str(message.from_user.id)


async def health_check(request: web.Request) -> web.Response:
    """UptimeRobot va Render health-check uchun oddiy endpoint."""
    return web.json_response({"status": "ok", "service": "telegram-bot"})


async def start_health_server() -> web.AppRunner:
    """aiohttp serverini polling bilan bir event loop ichida ishga tushiradi."""
    app = web.Application()
    app.router.add_get("/", health_check)
    app.router.add_get("/health", health_check)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=PORT)
    await site.start()

    logger.info("Health server ishga tushdi: 0.0.0.0:%s", PORT)
    return runner


async def main() -> None:
    pool = await asyncpg.create_pool(dsn=DATABASE_URL, min_size=1, max_size=10)
    await init_db(pool)

    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()

    @dp.message(Command("start"))
    async def cmd_start(message: Message) -> None:
        if not _is_super_admin(message):
            uid = message.from_user.id if message.from_user else "noma'lum"
            await message.answer(
                f"Ushbu bot faqat tizim administratori uchun.\nSizning ID: <code>{uid}</code>",
                parse_mode="HTML",
            )
            return

        await message.answer(
            "🤖 <b>Guruh Tahlil Boti Tayyor</b>\n\n"
            "<b>Guruh boshqaruvi:</b>\n"
            "• <code>/add_group &lt;chat_id&gt; &lt;report_chat_id&gt; &lt;Kompaniya&gt;</code>\n"
            "• <code>/groups</code>\n\n"
            "<b>Hisobot va tahlil:</b>\n"
            "• <code>/tahlil &lt;chat_id&gt; [soat]</code>\n"
            "• <code>/tahlil_shu [soat]</code> (guruh ichida)\n"
            "• <code>/hisobot [soat]</code> (barcha faol guruhlar)\n"
            "• <code>/haftalik [kun] [yubor]</code>\n\n"
            "<b>Statistika va nazorat:</b>\n"
            "• <code>/statistika [kun] [chat_id]</code>\n"
            "• <code>/ochiq [kun] [chat_id]</code>\n"
            "• <code>/yechildi &lt;chat_id&gt; &lt;kalit_so'z&gt; | [izoh]</code>",
            parse_mode="HTML",
        )

    @dp.message(Command("add_group"))
    async def cmd_add_group(message: Message) -> None:
        if not _is_super_admin(message):
            return

        parts = (message.text or "").split(maxsplit=3)
        if len(parts) < 4:
            await message.answer(
                "Format noto'g'ri.\nNamuna:\n"
                "<code>/add_group -1001234567890 987654321 Acme Corp</code>",
                parse_mode="HTML",
            )
            return

        try:
            chat_id = int(parts[1])
            report_chat_id = int(parts[2])
        except ValueError:
            await message.answer(
                "<code>chat_id</code> va <code>report_chat_id</code> butun son bo'lishi kerak.",
                parse_mode="HTML",
            )
            return

        company_name = parts[3].strip()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO tracked_groups (chat_id, company_name, report_chat_id, is_active)
                VALUES ($1, $2, $3, TRUE)
                ON CONFLICT (chat_id) DO UPDATE
                SET company_name = EXCLUDED.company_name,
                    report_chat_id = EXCLUDED.report_chat_id,
                    is_active = TRUE
                """,
                chat_id,
                company_name,
                report_chat_id,
            )

        await message.answer(
            f"✅ Guruh qo'shildi:\n"
            f"🏢 Kompaniya: <b>{html.escape(company_name)}</b>\n"
            f"💬 Guruh ID: <code>{chat_id}</code>\n"
            f"📨 Hisobot manzili: <code>{report_chat_id}</code>",
            parse_mode="HTML",
        )

    @dp.message(Command("groups"))
    async def cmd_groups(message: Message) -> None:
        if not _is_super_admin(message):
            return

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT chat_id, company_name, group_title, report_chat_id, is_active
                FROM tracked_groups
                ORDER BY company_name ASC
                """
            )

        if not rows:
            await message.answer("Hozircha birorta guruh ulanmagan.")
            return

        out = ["<b>📋 Ulangan guruhlar:</b>\n"]
        for r in rows:
            status = "Faol" if r["is_active"] else "O'chirilgan"
            out.append(
                f"• <b>{html.escape(r['company_name'])}</b> ({html.escape(r['group_title'] or 'Nomsiz')})\n"
                f"  Guruh: <code>{r['chat_id']}</code> | Hisobot: <code>{r['report_chat_id']}</code> | {status}"
            )
        await send_long_message(bot, message.chat.id, "\n".join(out))

    @dp.message(Command("tahlil"))
    async def cmd_tahlil(message: Message) -> None:
        if not _is_super_admin(message):
            return

        parts = (message.text or "").split()
        if len(parts) < 2:
            await message.answer("Namuna: <code>/tahlil -1001234567890 24</code>", parse_mode="HTML")
            return

        try:
            target_chat_id = int(parts[1])
            hours = int(parts[2]) if len(parts) >= 3 else 24
            if not (1 <= hours <= 720):
                raise ValueError
        except ValueError:
            await message.answer("Guruh ID va soat (1..720) to'g'ri butun son bo'lishi kerak.")
            return

        wait_msg = await message.answer("⏳ Xabarlar o'qilmoqda va tahlil qilinmoqda...")
        try:
            report_text, msg_count, report_target = await analyze_and_persist_group(
                pool=pool, chat_id=target_chat_id, hours=hours
            )
            await send_long_message(bot, report_target, report_text)
            await wait_msg.edit_text(
                f"✅ Tahlil tayyor ({msg_count} ta xabar), bazaga saqlandi va <code>{report_target}</code> ga yuborildi.",
                parse_mode="HTML",
            )
        except Exception as e:
            logger.exception("Tahlil buyrug'ida xato | chat_id=%s", target_chat_id)
            await wait_msg.edit_text(f"❌ Xatolik: {html.escape(str(e))}", parse_mode="HTML")

    @dp.message(Command("tahlil_shu"))
    async def cmd_tahlil_shu(message: Message) -> None:
        if not _is_super_admin(message):
            return
        if message.chat.type not in {"group", "supergroup"}:
            await message.answer("Bu buyruq guruh ichida ishlatiladi.")
            return

        parts = (message.text or "").split()
        try:
            hours = int(parts[1]) if len(parts) >= 2 else 24
            if not (1 <= hours <= 720):
                raise ValueError
        except ValueError:
            await message.answer("Namuna: <code>/tahlil_shu 24</code>", parse_mode="HTML")
            return

        wait_msg = await message.answer("⏳ Ushbu guruh yozishmalari tahlil qilinmoqda...")
        try:
            report_text, msg_count, report_target = await analyze_and_persist_group(
                pool=pool, chat_id=message.chat.id, hours=hours
            )
            await send_long_message(bot, report_target, report_text)
            await wait_msg.edit_text(
                f"✅ Tahlil tayyor ({msg_count} ta xabar) va <code>{report_target}</code> ga yuborildi.",
                parse_mode="HTML",
            )
        except Exception as e:
            logger.exception("tahlil_shu xatosi | chat_id=%s", message.chat.id)
            await wait_msg.edit_text(f"❌ Xatolik: {html.escape(str(e))}", parse_mode="HTML")

    @dp.message(Command("hisobot"))
    async def cmd_hisobot_all(message: Message) -> None:
        if not _is_super_admin(message):
            return

        parts = (message.text or "").split()
        try:
            hours = int(parts[1]) if len(parts) >= 2 else 24
            if not (1 <= hours <= 720):
                raise ValueError
        except ValueError:
            await message.answer("Namuna: <code>/hisobot 24</code>", parse_mode="HTML")
            return

        async with pool.acquire() as conn:
            groups = await conn.fetch(
                """
                SELECT chat_id, company_name
                FROM tracked_groups
                WHERE is_active = TRUE
                ORDER BY company_name ASC
                """
            )

        if not groups:
            await message.answer("Faol guruhlar topilmadi.")
            return

        status_msg = await message.answer(
            f"⏳ {len(groups)} ta faol guruh bo'yicha hisobot tayyorlanmoqda..."
        )

        done = 0
        errors: list[str] = []
        for g in groups:
            g_chat_id = int(g["chat_id"])
            try:
                report_text, _, report_target = await analyze_and_persist_group(
                    pool=pool, chat_id=g_chat_id, hours=hours
                )
                await send_long_message(bot, report_target, report_text)
                done += 1
                await asyncio.sleep(0.4)
            except Exception as e:
                logger.exception("hisobot buyrug'ida xato | chat_id=%s", g_chat_id)
                errors.append(
                    f"• {html.escape(g['company_name'])} (<code>{g_chat_id}</code>): {html.escape(str(e))}"
                )

        summary = f"✅ Yakunlandi: <b>{done}</b>/{len(groups)} ta guruh hisoboti yuborildi."
        if errors:
            summary += "\n\n❌ <b>Xatoliklar:</b>\n" + "\n".join(errors)
        await status_msg.edit_text(summary, parse_mode="HTML")

    @dp.message(Command("statistika"))
    async def cmd_statistika(message: Message) -> None:
        if not _is_super_admin(message):
            return

        parts = (message.text or "").split()
        try:
            days = int(parts[1]) if len(parts) >= 2 else 7
            target_chat = int(parts[2]) if len(parts) >= 3 else None
            if not (1 <= days <= 90):
                raise ValueError
        except ValueError:
            await message.answer(
                "Namuna:\n"
                "<code>/statistika 7</code>\n"
                "<code>/statistika 7 -1001234567890</code>",
                parse_mode="HTML",
            )
            return

        rows = await get_daily_stats(
            pool=pool, days=days, chat_id=target_chat, tz_offset_hours=5
        )
        text = format_daily_stats_html(rows, days=days)
        await send_long_message(bot, message.chat.id, text)

    @dp.message(Command("ochiq"))
    async def cmd_ochiq(message: Message) -> None:
        if not _is_super_admin(message):
            return

        parts = (message.text or "").split()
        try:
            days = int(parts[1]) if len(parts) >= 2 else 3
            target_chat = int(parts[2]) if len(parts) >= 3 else None
            if not (1 <= days <= 90):
                raise ValueError
        except ValueError:
            await message.answer(
                "Namuna:\n"
                "<code>/ochiq 3</code>\n"
                "<code>/ochiq 7 -1001234567890</code>",
                parse_mode="HTML",
            )
            return

        open_problems, unanswered_list = await get_open_problems_and_questions(
            pool=pool, days=days, chat_id=target_chat
        )
        text = format_open_items_html(
            open_problems=open_problems,
            unanswered_list=unanswered_list,
            days=days,
            chat_id=target_chat,
        )
        await send_long_message(bot, message.chat.id, text)

    @dp.message(Command("yechildi"))
    async def cmd_yechildi(message: Message) -> None:
        if not _is_super_admin(message):
            return

        parts = (message.text or "").strip().split(maxsplit=2)
        if len(parts) < 3:
            await message.answer(
                "Namuna:\n"
                "<code>/yechildi -1001234567890 1C | Server qayta ishga tushirildi</code>",
                parse_mode="HTML",
            )
            return

        try:
            target_chat_id = int(parts[1])
        except ValueError:
            await message.answer("<code>chat_id</code> butun son bo'lishi kerak.", parse_mode="HTML")
            return

        rest = parts[2].strip()
        if "|" in rest:
            query_part, note_part = rest.split("|", maxsplit=1)
            problem_query = query_part.strip()
            note = note_part.strip() or "Admin tomonidan yechilgan deb belgilandi"
        else:
            problem_query = rest
            note = "Admin tomonidan yechilgan deb belgilandi"

        try:
            updated_reports, updated_titles = await mark_problem_solved(
                pool=pool,
                chat_id=target_chat_id,
                problem_query=problem_query,
                note=note,
                resolved_by=_format_user_display(message),
                days=14,
            )
        except ValueError as e:
            await message.answer(f"❌ {html.escape(str(e))}")
            return

        if updated_reports == 0:
            await message.answer(
                f"ℹ️ Oxirgi 14 kunda <code>{target_chat_id}</code> guruhida "
                f"«<b>{html.escape(problem_query)}</b>» bo'yicha ochiq muammo topilmadi.",
                parse_mode="HTML",
            )
            return

        titles_html = "\n".join(f"• {html.escape(t)}" for t in updated_titles)
        await message.answer(
            f"✅ <b>YECHILDI deb belgilandi</b>\n"
            f"💬 Guruh: <code>{target_chat_id}</code>\n"
            f"📄 Yangilangan hisobotlar: <b>{updated_reports}</b> ta\n"
            f"📝 Izoh: {html.escape(note)}\n\n"
            f"<b>Muammolar:</b>\n{titles_html}",
            parse_mode="HTML",
        )

    @dp.message(Command("haftalik"))
    async def cmd_haftalik(message: Message) -> None:
        if not _is_super_admin(message):
            return

        parts = (message.text or "").split()
        days = WEEKLY_REPORT_DAYS
        should_send_to_companies = False

        for token in parts[1:]:
            low = token.lower()
            if low in {"yubor", "send", "kompaniyalarga"}:
                should_send_to_companies = True
            else:
                try:
                    parsed_days = int(token)
                    if 1 <= parsed_days <= 90:
                        days = parsed_days
                    else:
                        raise ValueError
                except ValueError:
                    await message.answer(
                        "Namuna:\n"
                        "<code>/haftalik 7</code>\n"
                        "<code>/haftalik 7 yubor</code>",
                        parse_mode="HTML",
                    )
                    return

        summaries = await get_weekly_company_summaries(pool=pool, days=days)
        overview_text = format_all_companies_overview_html(summaries, days=days)
        await send_long_message(bot, message.chat.id, overview_text)

        if should_send_to_companies and summaries:
            sent_ok = 0
            sent_fail = 0
            for s in summaries:
                if not s.report_chat_id:
                    sent_fail += 1
                    continue
                comp_text = format_company_weekly_html(s, days=days)
                try:
                    await send_long_message(bot, s.report_chat_id, comp_text)
                    sent_ok += 1
                    await asyncio.sleep(0.3)
                except Exception:
                    logger.exception("Kompaniyaga haftalik xulosa yuborilmadi: %s", s.company_name)
                    sent_fail += 1

            await message.answer(
                f"📤 Kompaniyalarga yuborildi: <b>{sent_ok}</b> ta | Xato/manzilsiz: <b>{sent_fail}</b> ta.",
                parse_mode="HTML",
            )

    @dp.message(F.chat.type.in_({"group", "supergroup"}))
    async def handle_group_message(message: Message) -> None:
        if message.from_user and message.from_user.is_bot:
            return
        is_tracked = await upsert_group_info(pool, message.chat.id, message.chat.title)
        if not is_tracked:
            return
        await save_message(pool, message)

    daily_task = asyncio.create_task(
        daily_report_scheduler_loop(
            bot=bot,
            pool=pool,
            genai_client=genai_client,
            model_name=GEMINI_MODEL,
            super_admin_id=SUPER_ADMIN_ID,
            run_hour=AUTO_REPORT_HOUR,
            run_minute=AUTO_REPORT_MINUTE,
            tz_name=AUTO_REPORT_TZ,
            hours=AUTO_REPORT_HOURS,
        )
    )

    weekly_task = asyncio.create_task(
        weekly_summary_scheduler_loop(
            bot=bot,
            pool=pool,
            super_admin_id=SUPER_ADMIN_ID,
            run_weekday=WEEKLY_REPORT_WEEKDAY,
            run_hour=WEEKLY_REPORT_HOUR,
            run_minute=WEEKLY_REPORT_MINUTE,
            tz_name=AUTO_REPORT_TZ,
            days=WEEKLY_REPORT_DAYS,
        )
    )

    web_runner = None
    try:
        web_runner = await start_health_server()
        logger.info("Bot polling rejimida ishga tushdi...")
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        if web_runner is not None:
            await web_runner.cleanup()
        weekly_task.cancel()
        daily_task.cancel()
        await asyncio.gather(weekly_task, daily_task, return_exceptions=True)
        await pool.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())