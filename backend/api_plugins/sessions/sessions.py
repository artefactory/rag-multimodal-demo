import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, Response

from backend.api_plugins.lib.user_management import User
from pydantic import BaseModel


class Message(BaseModel):
    id: str
    timestamp: str | datetime
    session_id: str
    sender: str
    content: str


def session_routes(
    app: FastAPI | APIRouter,
    *,
    authentication: Depends = None,
    dependencies: Sequence[Depends] | None = None,
):
    from backend.database import Database

    with Database() as connection:
        connection.run_script(Path(__file__).parent / "sessions_tables.sql")

    @app.post("/session/new")
    async def chat_new(
        current_user: User = authentication, dependencies=dependencies
    ) -> dict:
        chat_id = str(uuid4())
        timestamp = datetime.utcnow().isoformat()
        user_id = current_user.email if current_user else "unauthenticated"
        with Database() as connection:
            connection.execute(
                "INSERT INTO session (id, timestamp, user_id) VALUES (?, ?, ?)",
                (chat_id, timestamp, user_id),
            )
        return {"session_id": chat_id}

    @app.get("/session/list")
    async def chat_list(
        current_user: User = authentication, dependencies=dependencies
    ) -> list[dict]:
        user_email = current_user.email if current_user else "unauthenticated"
        chats = []
        with Database() as connection:
            result = connection.execute(
                "SELECT id, timestamp FROM session WHERE user_id = ? ORDER BY timestamp DESC",
                (user_email,),
            )
            chats = [{"id": row[0], "timestamp": row[1]} for row in result]
        return chats

    @app.get("/session/{session_id}")
    async def chat(
        session_id: str, current_user: User = authentication, dependencies=dependencies
    ) -> dict:
        messages: list[Message] = []
        with Database() as connection:
            result = connection.execute(
                "SELECT id, timestamp, session_id, message FROM message_history WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,),
            )
            for row in result:
                content = json.loads(row[3])["data"]["content"]
                message_type = json.loads(row[3])["type"]
                message = Message(
                    id=row[0],
                    timestamp=row[1],
                    session_id=row[2],
                    sender=message_type if message_type == "human" else "ai",
                    content=content,
                )
                messages.append(message)
        return {
            "chat_id": session_id,
            "messages": [message.dict() for message in messages],
        }

    @app.get("/session")
    async def session_root(
        current_user: User = authentication, dependencies=dependencies
    ) -> dict:
        return Response("Sessions management routes are enabled.", status_code=200)