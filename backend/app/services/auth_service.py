"""Password hashing and expiring, revocable bearer sessions."""

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.services.history_store import AuthAttempt, LoginSession, User


def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()
    return f"scrypt${salt}${digest}"


def verify_password(password, encoded):
    try:
        _, salt, _ = encoded.split("$")
        return hmac.compare_digest(hash_password(password, salt), encoded)
    except (ValueError, TypeError):
        return False


def token_digest(token):
    return hashlib.sha256(token.encode()).hexdigest()


class AuthService:
    def __init__(self, store, session_hours=24):
        self.engine = store.engine
        self.session_hours = session_hours
        self.dummy_hash = hash_password(secrets.token_urlsafe(32))

    def throttle(self, identifier, limit=10):
        """Persist an atomic per-IP budget for login and registration attempts."""
        key = token_digest(identifier)
        now = datetime.now(timezone.utc)
        with Session(self.engine) as session, session.begin():
            session.execute(delete(AuthAttempt).where(AuthAttempt.expires_at <= now.isoformat()))
            from sqlalchemy.dialects.postgresql import insert
            statement = insert(AuthAttempt).values(key=key, count=1,
                                                   expires_at=(now + timedelta(minutes=15)).isoformat())
            statement = statement.on_conflict_do_update(
                index_elements=[AuthAttempt.key], set_={"count": AuthAttempt.count + 1},
                where=AuthAttempt.count < limit,
            ).returning(AuthAttempt.count)
            if session.scalar(statement) is None:
                raise PermissionError("Too many attempts. Try again in 15 minutes.")

    def register(self, email, password):
        now = datetime.now(timezone.utc).isoformat()
        with Session(self.engine) as session:
            user = User(id=str(uuid4()), email=email, password_hash=hash_password(password), created_at=now)
            session.add(user)
            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                raise ValueError("Unable to create account with these details. Try signing in.") from exc
            return {"id": user.id, "email": user.email}

    def login(self, email, password):
        with Session(self.engine) as session, session.begin():
            user = session.scalar(select(User).where(User.email == email))
            valid = verify_password(password, user.password_hash if user else self.dummy_hash)
            if not user or not valid:
                raise ValueError("Incorrect email or password.")
            now = datetime.now(timezone.utc)
            session.execute(delete(LoginSession).where(LoginSession.expires_at <= now.isoformat()))
            token = secrets.token_urlsafe(32)
            expires = (now + timedelta(hours=self.session_hours)).isoformat()
            session.add(LoginSession(token_hash=token_digest(token), user_id=user.id, expires_at=expires))
            return {"access_token": token, "token_type": "bearer", "expires_at": expires,
                    "user": {"id": user.id, "email": user.email}}

    def authenticate(self, token):
        with Session(self.engine) as session:
            row = session.get(LoginSession, token_digest(token))
            if not row or row.expires_at <= datetime.now(timezone.utc).isoformat():
                raise ValueError("Session expired. Please sign in again.")
            user = session.get(User, row.user_id)
            if not user:
                raise ValueError("Invalid session.")
            return {"id": user.id, "email": user.email}

    def logout(self, token):
        with Session(self.engine) as session, session.begin():
            session.execute(delete(LoginSession).where(LoginSession.token_hash == token_digest(token)))
