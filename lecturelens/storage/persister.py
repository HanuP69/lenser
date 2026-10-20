import asyncio
import logging
from typing import Any, Callable, Optional

from lecturelens.core.event_loop import ensure_event_loop_policy

ensure_event_loop_policy()

from lecturelens.config.settings import Settings
from lecturelens.core.agent_runtime import (
    PERSISTER_STOP_FLAG_NAME,
    AgentControl,
    create_redis_bus,
    execute_agent_main,
    parse_runtime_arguments
)
from lecturelens.core.redis_bus import RedisBus
from lecturelens.schemas.events import GeneralNoteSegment, TimelineEntry
from lecturelens.storage.postgres import PostgresStorage

logger = logging.getLogger(__name__)

PERSISTER_GROUP_NAME = "group:persister"


class PostgresPersister:
    def __init__(
        self,
        settings: Settings,
        redis_bus: RedisBus,
        storage: PostgresStorage,
        group_name: str = PERSISTER_GROUP_NAME
    ) -> None:
        self.settings = settings
        self.redis_bus = redis_bus
        self.storage = storage
        self.group_name = group_name
        self.is_running = False

    async def handle_timeline_message(
        self,
        message_data: dict[str, Any],
        message_id: str,
        stream_name: str
    ) -> None:
        try:
            entry = TimelineEntry.from_redis_dict(message_data)
        except Exception:
            try:
                entry = TimelineEntry.from_jsonl_dict(message_data)
            except Exception:
                entry = TimelineEntry(**message_data)
        await self.storage.insert_timeline_event(entry)
        await self.redis_bus.acknowledge(stream_name, self.group_name, message_id)

    async def handle_note_message(
        self,
        message_data: dict[str, Any],
        message_id: str,
        stream_name: str
    ) -> None:
        try:
            note = GeneralNoteSegment.from_redis_dict(message_data)
        except Exception:
            try:
                note = GeneralNoteSegment(**message_data)
            except Exception as error:
                logger.error("invalid general note payload: %s", error)
                await self.redis_bus.acknowledge(stream_name, self.group_name, message_id)
                return

        await self.storage.insert_general_note(note)
        definitions_map: dict[str, str] = {}
        for d in getattr(note, "definitions", []):
            if isinstance(d, dict):
                term = str(d.get("term", "")).strip()
                def_text = str(d.get("definition", "")).strip()
                if term and def_text:
                    definitions_map[term.lower()] = def_text

        for tag in note.concept_tags:
            def_text = definitions_map.get(tag.lower())
            await self.storage.insert_concept(
                course_id=self.settings.course_name,
                concept_name=tag,
                definition=def_text,
                related_concepts=[t for t in note.concept_tags if t != tag],
                session_id=note.session_id,
                timestamp_epoch=note.timestamp_epoch
            )
        await self.redis_bus.acknowledge(stream_name, self.group_name, message_id)

    async def consume_stream(
        self,
        stream_name: str,
        handler: Callable[[dict[str, Any], str, str], Any],
        consumer_name: str,
        should_stop: Optional[Callable[[], bool]] = None
    ) -> None:
        await self.redis_bus.ensure_consumer_group(stream_name, self.group_name)

        async def wrapped_handler(data: dict[str, Any], msg_id: str) -> None:
            await handler(data, msg_id, stream_name)

        await self.redis_bus.consume(
            stream_name=stream_name,
            group_name=self.group_name,
            consumer_name=consumer_name,
            handler=wrapped_handler,
            batch_size=10,
            block_ms=100,
            should_stop=should_stop
        )

    async def run_for_session(
        self,
        session_id: str,
        consumer_name: str = "persister_worker",
        should_stop: Optional[Callable[[], bool]] = None
    ) -> None:
        self.is_running = True
        timeline_stream = f"lecturelens:{session_id}:stream:timeline"
        notes_stream = f"lecturelens:{session_id}:stream:notes"

        timeline_task = asyncio.create_task(
            self.consume_stream(
                timeline_stream,
                self.handle_timeline_message,
                f"{consumer_name}_timeline",
                should_stop
            )
        )
        notes_task = asyncio.create_task(
            self.consume_stream(
                notes_stream,
                self.handle_note_message,
                f"{consumer_name}_notes",
                should_stop
            )
        )

        try:
            await asyncio.gather(timeline_task, notes_task)
        finally:
            self.is_running = False
            for task in (timeline_task, notes_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(timeline_task, notes_task, return_exceptions=True)


async def main() -> None:
    settings = Settings()
    if not settings.postgres_url:
        logger.warning("postgres_url not configured, persister exiting")
        return
    namespace = parse_runtime_arguments()
    session_id = namespace.session or settings.session_name or "default_session"
    session_dir = settings.get_session_output_dir(session_id)
    redis_bus = create_redis_bus(settings)
    storage = PostgresStorage(settings.postgres_url, query_limit=getattr(settings, "postgres_query_limit", 1000))
    try:
        await storage.init_schema()
    except Exception as pg_err:
        logger.warning("postgres unavailable for persister (%s), persister exiting", pg_err)
        await storage.close()
        await redis_bus.close()
        return
    persister = PostgresPersister(settings, redis_bus, storage)
    control = AgentControl(session_dir, "persister", stop_flag_name=PERSISTER_STOP_FLAG_NAME)
    control.write_heartbeat("live")
    heartbeat_task = asyncio.create_task(control.heartbeat_loop())
    try:
        await persister.run_for_session(
            session_id=session_id,
            should_stop=control.stop_requested
        )
    finally:
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)
        try:
            control.write_heartbeat("stopped")
        except Exception:
            pass
        await storage.close()
        await redis_bus.close()


if __name__ == "__main__":
    execute_agent_main(main)

