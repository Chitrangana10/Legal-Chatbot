"""Create an initial PostgreSQL secret without printing it or replacing an existing one."""

from pathlib import Path
import re
import secrets


def prepare(path):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError("EC2 .env is missing; configure GEMINI_API_KEY before deploying")
    content = path.read_text(encoding="utf-8")
    matches = list(re.finditer(r"^POSTGRES_PASSWORD\s*=(.*)$", content, re.MULTILINE))
    if len(matches) > 1:
        raise RuntimeError("Remove duplicate POSTGRES_PASSWORD entries before deploying")
    if matches and matches[0].group(1).strip().strip("\"'"):
        return False
    line = "POSTGRES_PASSWORD=" + secrets.token_hex(32)
    if matches:
        content = content[:matches[0].start()] + line + content[matches[0].end():]
    else:
        content = content.rstrip() + "\n" + line + "\n"
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)
    return True


if __name__ == "__main__":
    created = prepare(Path(__file__).resolve().parents[2] / ".env")
    print("Initialized PostgreSQL password in EC2 .env" if created else "Using existing PostgreSQL password")
