# Accounts and saved history

Email/password sign-up and login are handled by FastAPI. Passwords are salted
and hashed using scrypt. Login issues a random bearer token, stores only its
SHA-256 digest, expires it after `SESSION_HOURS` (24 by default), and revokes
it on logout. The frontend keeps the token in its server-side Streamlit session.
After a page reload or session expiry, sign in again to retrieve saved chats.

Every query requires login. The backend derives ownership from the authenticated
session, never from a user ID supplied by the browser. Saved conversations can
be opened or permanently deleted from the sidebar. Starting a new chat retains
previous saved conversations. Chat history is stored for review; it is not
currently included as context for subsequent retrieval/generation.

## Local Docker

Create `.env` if necessary and configure Gemini as before. Then:

```powershell
docker compose up --build -d
```

Open http://localhost:8501 and create an account. Compose starts PostgreSQL with
a persistent `postgres_data` volume. Local PostgreSQL is accessible at
`127.0.0.1:5433`, including from desktop pgAdmin. `docker compose down` retains
the data; `down -v` deletes it.
The local default database password is for development only.

## Direct Python development

Start the database first:

```powershell
docker compose up -d postgres
```

Install the updated `requirements.txt`, then start the backend and frontend
using the existing commands. Direct Python and Docker runs now use the same
local PostgreSQL database. Python connects to `127.0.0.1:5433`; the Docker backend
connects to `postgres:5432`. If you customize `POSTGRES_PASSWORD`, update the
password in the direct Python `DATABASE_URL` too. PostgreSQL is the only
database used for accounts and saved conversations.

## EC2 deployment

The deployment workflow initializes a random hexadecimal `POSTGRES_PASSWORD`
in the **EC2** `.env` if missing, preserving any existing value. It requires an
existing `.env` containing your Gemini settings. To configure it manually:

```bash
python3 -c 'import secrets; print(secrets.token_hex(32))'
```

Copy the output into `POSTGRES_PASSWORD=...`. Keep the value; changing the
environment variable later does not change the password of an existing
PostgreSQL volume. Do not commit it. Production Compose refuses to start
without this setting. Deployment starts PostgreSQL on EC2; it does not create
or connect AWS RDS automatically. Back up the database/volume before relying
on saved history. To use RDS later, replace the Compose backend `DATABASE_URL`
with the RDS connection string and remove the dependency on the local postgres
service. Use TLS for remote PostgreSQL connections.

The current public site uses HTTP. Configure HTTPS before accepting real users'
passwords. Password hashing protects stored credentials, not credentials in
transit. Signup currently has no email verification or password-recovery flow.

## API

- `POST /auth/register`: email and password (4–128 characters)
- `POST /auth/login`: returns token, expiry, and user
- `GET /auth/me`: authenticated user
- `POST /auth/logout`: revokes the current session
- `GET /conversations?limit=50&offset=0`: the user's saved conversations
- `GET /conversations/{id}`: a saved conversation with its turns
- `DELETE /conversations/{id}`: permanently remove an owned conversation
- `POST /query`: question and optional `conversation_id`; returns answer,
  sources, degraded/notice, and `conversation_id`

Send `Authorization: Bearer <access_token>` on protected endpoints. Unknown or
other users' conversation IDs both return 404. Login/signup attempts have a
15-minute database-backed rate limit.

## Redis caching

Start Redis alongside PostgreSQL for direct Python development:

```powershell
docker compose up -d postgres redis
```

The Redis Python client is included in the updated requirements. Restart the
backend after installing dependencies. Python connects to localhost:6379;
Compose overrides `REDIS_URL` to redis:6379. Production Compose runs Redis on
EC2 on its internal network, with no public port. This is not AWS ElastiCache.

Successful answers to the same exact trimmed question are cached separately
for each user for 24 hours by default. They skip retrieval and Gemini on a hit,
but still save a new question/answer record in PostgreSQL. Responses expose
`cached: true` for hits. Failed/degraded responses are not cached. Cache keys
include a fingerprint of index contents, retrieval/generation code, and model
settings. Increment `ANSWER_CACHE_VERSION` for manual invalidation; changes to
retrieval assets require a backend restart. If Redis is unavailable, requests
continue without caching. Login sessions remain in PostgreSQL.

Sample questions live in the frontend only; Redis stores generated answers.
Redis is bounded to 128 MB with least-recently-used eviction and no disk
persistence. Cache data may disappear on restart; accounts/chats are unaffected.

## Verification

```powershell
python -m pytest tests/test_auth_history.py -p no:cacheprovider
python -m pytest tests/test_auth_frontend.py -p no:cacheprovider
```

Set `TEST_DATABASE_URL` to a disposable PostgreSQL database with a name ending
in `_test`. Backend tests recreate its tables; never use your app database.
Without that variable the backend tests are skipped.

These tests use PostgreSQL and a stub answer engine to check
login, hashing, logout, expiry, rate limits, persistence, and cross-user access
without downloading embedding models or calling Gemini.
CI runs backend checks against PostgreSQL and tests the
Streamlit account screens before building/deploying images. Deployment health
checks require `index_loaded: true`, rather than accepting any HTTP 200.
