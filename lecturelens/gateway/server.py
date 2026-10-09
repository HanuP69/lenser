import asyncio
import json
from typing import Any, Optional
import websockets
from websockets.server import WebSocketServerProtocol
from lecturelens.config.settings import Settings
from lecturelens.core.redis_bus import RedisBus


class GatewayServer:
    def __init__(
        self,
        settings: Settings,
        redis_bus: RedisBus
    ) -> None:
        self.settings: Settings = settings
        self.redis_bus: RedisBus = redis_bus
        self.is_running: bool = False
        self.server: Optional[Any] = None

    async def handle_connection(self, websocket: WebSocketServerProtocol) -> None:
        client_address = websocket.remote_address
        authenticated = False
        if self.settings.gateway_api_key is None or len(self.settings.gateway_api_key) == 0:
            authenticated = True

        try:
            async for raw_message in websocket:
                if not authenticated:
                    try:
                        auth_data = json.loads(raw_message)
                        token = auth_data.get("api_key", "")
                        if token == self.settings.gateway_api_key:
                            authenticated = True
                            await websocket.send(json.dumps({"status": "authenticated"}))
                            continue
                        else:
                            await websocket.send(json.dumps({"status": "unauthorized"}))
                            await websocket.close()
                            return
                    except Exception:
                        await websocket.close()
                        return

                try:
                    packet = json.loads(raw_message)
                    session_id = packet.get("session_id", "default_session")
                    source_type = packet.get("source_type", "")
                    payload = packet.get("payload", {})

                    if len(source_type) == 0:
                        continue

                    stream_name = f"lecturelens:{session_id}:stream:{source_type}"
                    raw_dict: dict[str, str] = {}
                    for key, val in payload.items():
                        if val is None:
                            raw_dict[key] = ""
                        elif isinstance(val, (dict, list)):
                            raw_dict[key] = json.dumps(val)
                        else:
                            raw_dict[key] = str(val)

                    await self.redis_bus.client.xadd(stream_name, raw_dict)
                    await websocket.send(json.dumps({"status": "acknowledged"}))
                except Exception as processing_error:
                    await websocket.send(json.dumps({
                        "status": "error",
                        "message": str(processing_error)
                    }))
        except websockets.exceptions.ConnectionClosed:
            pass

    async def start(self) -> None:
        self.is_running = True
        self.server = await websockets.serve(
            self.handle_connection,
            self.settings.gateway_host,
            self.settings.gateway_port
        )
        await self.server.wait_closed()

    def stop(self) -> None:
        self.is_running = False
        if self.server is not None:
            self.server.close()


async def main() -> None:
    settings = Settings()
    redis_bus = RedisBus(settings.redis_url)
    gateway = GatewayServer(settings=settings, redis_bus=redis_bus)
    try:
        await gateway.start()
    finally:
        await redis_bus.close()


if __name__ == "__main__":
    asyncio.run(main())
