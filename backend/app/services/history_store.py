"""Persistent conversations with ownership enforced in every operation."""

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import JSON, Column, ForeignKey, Integer, String, create_engine, delete, select
from sqlalchemy.orm import DeclarativeBase, Session


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id = Column(String(36), primary_key=True)
    email = Column(String(254), nullable=False, unique=True)
    password_hash = Column(String(300), nullable=False)
    created_at = Column(String(40), nullable=False)


class LoginSession(Base):
    __tablename__ = "login_sessions"
    token_hash = Column(String(64), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    expires_at = Column(String(40), nullable=False)


class AuthAttempt(Base):
    __tablename__ = "auth_attempts"
    key = Column(String(64), primary_key=True)
    count = Column(Integer, nullable=False)
    expires_at = Column(String(40), nullable=False)


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(120), nullable=False)
    created_at = Column(String(40), nullable=False)


class Turn(Base):
    __tablename__ = "conversation_turns"
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=False, index=True)
    question = Column(String, nullable=False)
    response = Column(JSON, nullable=False)
    created_at = Column(String(40), nullable=False)


class HistoryStore:
    def __init__(self, url):
        from sqlalchemy.engine import make_url
        if make_url(url).get_backend_name() != "postgresql":
            raise ValueError("This project uses PostgreSQL only. Set DATABASE_URL to a PostgreSQL connection string.")
        self.engine = create_engine(url, pool_pre_ping=True)

    def initialize(self):
        Base.metadata.create_all(self.engine)

    def list_conversations(self, user_id, limit=50, offset=0):
        with Session(self.engine) as session:
            rows = session.scalars(select(Conversation).where(Conversation.user_id == user_id)
                                   .order_by(Conversation.created_at.desc()).limit(limit).offset(offset))
            return [{"id": row.id, "title": row.title, "created_at": row.created_at} for row in rows]

    @staticmethod
    def _owned(session, user_id, conversation_id):
        row = session.scalar(select(Conversation).where(Conversation.id == conversation_id,
                                                        Conversation.user_id == user_id))
        if row is None:
            raise LookupError("Conversation not found")
        return row

    def get_conversation(self, user_id, conversation_id):
        with Session(self.engine) as session:
            row = self._owned(session, user_id, conversation_id)
            turns = session.scalars(select(Turn).where(Turn.conversation_id == row.id).order_by(Turn.id))
            return {"id": row.id, "title": row.title,
                    "turns": [{"question": turn.question, **turn.response} for turn in turns]}

    def assert_owner(self, user_id, conversation_id):
        with Session(self.engine) as session:
            self._owned(session, user_id, conversation_id)

    def save_turn(self, user_id, conversation_id, question, response):
        now = datetime.now(timezone.utc).isoformat()
        with Session(self.engine) as session, session.begin():
            if conversation_id:
                row = self._owned(session, user_id, conversation_id)
            else:
                row = Conversation(id=str(uuid4()), user_id=user_id, title=question[:120], created_at=now)
                session.add(row)
                session.flush()
            session.add(Turn(conversation_id=row.id, question=question, response=response, created_at=now))
            return row.id

    def delete_conversation(self, user_id, conversation_id):
        with Session(self.engine) as session, session.begin():
            row = self._owned(session, user_id, conversation_id)
            session.execute(delete(Turn).where(Turn.conversation_id == row.id))
            session.delete(row)
