from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from config import DATABASE_PATH


DEFAULT_WASTE_TYPES = (
    ("organic", "Hữu cơ", "Thức ăn thừa, vỏ rau củ quả, bã trà, lá cây.", "Cho vào thùng rác hữu cơ (màu xanh lá). Không lẫn túi nilon hay đồ nhựa. Thích hợp ủ phân vi sinh (compost)."),
    ("plastic", "Tái chế", "Chai nhựa PET, ly nhựa, can nhựa, vỏ bao bì nhựa sạch.", "Tráng sạch nước bẩn, tháo nắp chai riêng, ép bẹp thân chai để tiết kiệm không gian và bỏ vào thùng tái chế (xanh dương)."),
    ("paper", "Tái chế", "Giấy in A4, sách báo cũ, thùng carton, hộp giấy khô ráo.", "Giữ giấy và bìa carton khô ráo, không dính dầu mỡ. Tháo bỏ băng dính, xếp phẳng gọn gàng vào thùng rác tái chế (xanh dương)."),
    ("metal", "Tái chế", "Vỏ lon bia, lon nước ngọt, hộp thiếc, đồ kim loại nhỏ.", "Rửa sạch cặn đồ uống bên trong, dẫm hoặc bóp bẹp nếu có thể, bỏ vào thùng rác tái chế (xanh dương)."),
    ("other", "Còn lại", "Bao bì snack mạ nhôm, túi bóng bẩn, hộp xốp dơ, thủy tinh, pin, quần áo giày dép cũ, rác khó tái chế.", "Buộc kín trong túi rác sinh hoạt và bỏ vào thùng rác còn lại (màu xám)."),
)


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DATABASE_PATH, timeout=15)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    with connection() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user' CHECK(role IN ('user', 'admin')),
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS waste_types (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                category TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                instructions TEXT NOT NULL DEFAULT '',
                is_active INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS instructions (
                id INTEGER PRIMARY KEY,
                waste_type_id INTEGER NOT NULL REFERENCES waste_types(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                UNIQUE(waste_type_id, title)
            );
            CREATE TABLE IF NOT EXISTS detection_history (
                id INTEGER PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                image_path TEXT NOT NULL,
                result_image_path TEXT NOT NULL,
                model_path TEXT NOT NULL,
                image_class_name TEXT,
                image_class_category TEXT,
                image_class_confidence REAL,
                source TEXT NOT NULL DEFAULT 'upload',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS detection_details (
                id INTEGER PRIMARY KEY,
                history_id INTEGER NOT NULL REFERENCES detection_history(id) ON DELETE CASCADE,
                class_name TEXT NOT NULL,
                waste_type_id INTEGER REFERENCES waste_types(id) ON DELETE SET NULL,
                confidence REAL NOT NULL,
                classifier_confidence REAL,
                x1 REAL NOT NULL,
                y1 REAL NOT NULL,
                x2 REAL NOT NULL,
                y2 REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS auth_sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                history_id INTEGER REFERENCES detection_history(id) ON DELETE SET NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        history_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(detection_history)")
        }
        for column_name, column_type in (
            ("image_class_name", "TEXT"),
            ("image_class_category", "TEXT"),
            ("image_class_confidence", "REAL"),
        ):
            if column_name not in history_columns:
                db.execute(
                    f"ALTER TABLE detection_history ADD COLUMN {column_name} {column_type}"
                )
        for name, category, description, instructions in DEFAULT_WASTE_TYPES:
            db.execute(
                """
                INSERT OR IGNORE INTO waste_types(name, category, description, instructions)
                VALUES (?, ?, ?, ?)
                """,
                (name, category, description, instructions),
            )
        db.execute(
            """
            INSERT OR IGNORE INTO instructions(waste_type_id, title, content)
            SELECT id, 'Hướng dẫn phân loại', instructions
            FROM waste_types WHERE instructions != ''
            """
        )
        db.execute(
            "INSERT OR IGNORE INTO app_settings(key, value) VALUES ('detection_model', ?)",
            ("yolo11n.pt",),
        )


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    if len(password) < 8:
        raise ValueError("Mật khẩu phải có ít nhất 8 ký tự.")
    salt_value = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_value), 240_000
    ).hex()
    return digest, salt_value


def create_user(username: str, password: str, role: str = "user") -> int:
    clean_username = username.strip()
    if len(clean_username) < 3 or len(clean_username) > 40:
        raise ValueError("Tên đăng nhập phải dài từ 3 đến 40 ký tự.")
    if role not in {"user", "admin"}:
        raise ValueError("Vai trò không hợp lệ.")
    password_hash, password_salt = hash_password(password)
    with connection() as db:
        cursor = db.execute(
            """
            INSERT INTO users(username, password_hash, password_salt, role)
            VALUES (?, ?, ?, ?)
            """,
            (clean_username, password_hash, password_salt, role),
        )
        return int(cursor.lastrowid)


def authenticate(username: str, password: str) -> dict[str, Any] | None:
    with connection() as db:
        user = db.execute(
            "SELECT * FROM users WHERE username = ? AND is_active = 1",
            (username.strip(),),
        ).fetchone()
    if user is None:
        return None
    expected = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(user["password_salt"]),
        240_000,
    ).hex()
    if not hmac.compare_digest(expected, user["password_hash"]):
        return None
    return {
        "id": int(user["id"]),
        "username": str(user["username"]),
        "role": str(user["role"]),
    }


def get_active_user(user_id: int) -> dict[str, Any] | None:
    with connection() as db:
        user = db.execute(
            """
            SELECT id, username, role
            FROM users
            WHERE id = ? AND is_active = 1
            """,
            (user_id,),
        ).fetchone()
    return dict(user) if user is not None else None


def create_auth_session(user_id: int, token: str) -> None:
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with connection() as db:
        db.execute(
            "INSERT INTO auth_sessions(token_hash, user_id) VALUES (?, ?)",
            (token_hash, user_id),
        )


def get_auth_session_user(token: str) -> dict[str, Any] | None:
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with connection() as db:
        user = db.execute(
            """
            SELECT u.id, u.username, u.role
            FROM auth_sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token_hash = ? AND u.is_active = 1
            """,
            (token_hash,),
        ).fetchone()
    return dict(user) if user is not None else None


def delete_auth_session(token: str) -> None:
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with connection() as db:
        db.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (token_hash,))


def user_count(role: str | None = None) -> int:
    with connection() as db:
        if role:
            row = db.execute("SELECT COUNT(*) AS count FROM users WHERE role = ?", (role,)).fetchone()
        else:
            row = db.execute("SELECT COUNT(*) AS count FROM users").fetchone()
    return int(row["count"])


def list_users() -> list[dict[str, Any]]:
    with connection() as db:
        rows = db.execute(
            "SELECT id, username, role, is_active, created_at FROM users ORDER BY id"
        ).fetchall()
    return [dict(row) for row in rows]


def set_user_active(user_id: int, is_active: bool) -> None:
    with connection() as db:
        db.execute("UPDATE users SET is_active = ? WHERE id = ?", (int(is_active), user_id))


def set_user_role(user_id: int, role: str) -> None:
    if role not in {"user", "admin"}:
        raise ValueError("Vai trò không hợp lệ.")
    with connection() as db:
        db.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))


def delete_user(user_id: int) -> None:
    with connection() as db:
        db.execute("DELETE FROM users WHERE id = ?", (user_id,))


def list_waste_types(active_only: bool = True) -> list[dict[str, Any]]:
    where = "WHERE is_active = 1" if active_only else ""
    with connection() as db:
        rows = db.execute(
            f"SELECT * FROM waste_types {where} ORDER BY category, name"
        ).fetchall()
    return [dict(row) for row in rows]


def save_waste_type(
    name: str,
    category: str,
    description: str,
    instructions: str,
    type_id: int | None = None,
) -> None:
    if not name.strip() or not category.strip():
        raise ValueError("Tên loại rác và nhóm phân loại không được để trống.")
    with connection() as db:
        if type_id is None:
            db.execute(
                """
                INSERT INTO waste_types(name, category, description, instructions)
                VALUES (?, ?, ?, ?)
                """,
                (name.strip(), category.strip(), description.strip(), instructions.strip()),
            )
        else:
            db.execute(
                """
                UPDATE waste_types SET name = ?, category = ?, description = ?,
                    instructions = ? WHERE id = ?
                """,
                (name.strip(), category.strip(), description.strip(), instructions.strip(), type_id),
            )


def set_waste_type_active(type_id: int, is_active: bool) -> None:
    with connection() as db:
        db.execute(
            "UPDATE waste_types SET is_active = ? WHERE id = ?",
            (int(is_active), type_id),
        )


def list_instructions() -> list[dict[str, Any]]:
    with connection() as db:
        rows = db.execute(
            """
            SELECT i.id, i.waste_type_id, w.name AS waste_type,
                   i.title, i.content
            FROM instructions i
            JOIN waste_types w ON w.id = i.waste_type_id
            ORDER BY w.name, i.title
            """
        ).fetchall()
    return [dict(row) for row in rows]


def save_instruction(
    waste_type_id: int,
    title: str,
    content: str,
    instruction_id: int | None = None,
) -> None:
    if not title.strip() or not content.strip():
        raise ValueError("Tiêu đề và nội dung hướng dẫn không được để trống.")
    with connection() as db:
        if instruction_id is None:
            db.execute(
                """
                INSERT INTO instructions(waste_type_id, title, content)
                VALUES (?, ?, ?)
                """,
                (waste_type_id, title.strip(), content.strip()),
            )
        else:
            db.execute(
                """
                UPDATE instructions
                SET waste_type_id = ?, title = ?, content = ?
                WHERE id = ?
                """,
                (waste_type_id, title.strip(), content.strip(), instruction_id),
            )


def delete_instruction(instruction_id: int) -> None:
    with connection() as db:
        db.execute("DELETE FROM instructions WHERE id = ?", (instruction_id,))



def get_setting(key: str, default: str = "") -> str:
    with connection() as db:
        row = db.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
    return str(row["value"]) if row else default




def set_setting(key: str, value: str) -> None:
    with connection() as db:
        db.execute(
            """
            INSERT INTO app_settings(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )


def create_history(
    user_id: int,
    image_path: Path,
    result_image_path: Path,
    model_path: str,
    source: str,
    detections: list[dict[str, Any]],
    image_classification: dict[str, Any] | None = None,
) -> int:
    with connection() as db:
        cursor = db.execute(
            """
            INSERT INTO detection_history
                (user_id, image_path, result_image_path, model_path, source,
                 image_class_name, image_class_category, image_class_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                str(image_path),
                str(result_image_path),
                model_path,
                source,
                image_classification["class_name"] if image_classification else None,
                image_classification["category"] if image_classification else None,
                image_classification["confidence"] if image_classification else None,
            ),
        )
        history_id = int(cursor.lastrowid)
        db.executemany(
            """
            INSERT INTO detection_details
                (history_id, class_name, waste_type_id, confidence,
                 classifier_confidence, x1, y1, x2, y2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    history_id,
                    item["class_name"],
                    item.get("waste_type_id"),
                    item["confidence"],
                    item.get("classifier_confidence"),
                    *item["box"],
                )
                for item in detections
            ],
        )
        return history_id


def history_rows(user_id: int | None = None, limit: int = 500) -> list[dict[str, Any]]:
    query = """
        SELECT h.id, h.user_id, u.username, h.image_path, h.result_image_path,
               h.model_path, h.image_class_name, h.image_class_category,
               h.image_class_confidence, h.source, h.created_at,
               COUNT(d.id) AS object_count,
               GROUP_CONCAT(d.class_name, ', ') AS labels,
               GROUP_CONCAT(DISTINCT w.category) AS categories
        FROM detection_history h
        LEFT JOIN users u ON u.id = h.user_id
        LEFT JOIN detection_details d ON d.history_id = h.id
        LEFT JOIN waste_types w ON w.id = d.waste_type_id
    """
    params: list[Any] = []
    if user_id is not None:
        query += " WHERE h.user_id = ?"
        params.append(user_id)
    query += " GROUP BY h.id ORDER BY h.created_at DESC LIMIT ?"
    params.append(limit)
    with connection() as db:
        rows = db.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def delete_history(history_id: int) -> tuple[Path, Path] | None:
    with connection() as db:
        row = db.execute(
            "SELECT image_path, result_image_path FROM detection_history WHERE id = ?",
            (history_id,),
        ).fetchone()
        if row is None:
            return None
        db.execute("DELETE FROM detection_history WHERE id = ?", (history_id,))
        return Path(row["image_path"]), Path(row["result_image_path"])


def delete_all_history() -> list[tuple[Path, Path]]:
    with connection() as db:
        rows = db.execute(
            "SELECT image_path, result_image_path FROM detection_history"
        ).fetchall()
        db.execute("DELETE FROM detection_history")
    return [(Path(row["image_path"]), Path(row["result_image_path"])) for row in rows]


def statistics(user_id: int | None = None) -> dict[str, Any]:
    where = "WHERE h.user_id = ?" if user_id is not None else ""
    image_user_filter = "AND h.user_id = ?" if user_id is not None else ""
    params = (user_id,) if user_id is not None else ()
    with connection() as db:
        totals = db.execute(
            f"""
            SELECT COUNT(DISTINCT h.id) AS images, COUNT(d.id) AS objects
            FROM detection_history h LEFT JOIN detection_details d ON d.history_id = h.id
            {where}
            """,
            params,
        ).fetchone()
        by_type = db.execute(
            f"""
            SELECT d.class_name AS name, COUNT(*) AS count
            FROM detection_details d JOIN detection_history h ON h.id = d.history_id
            {where}
            GROUP BY d.class_name ORDER BY count DESC
            """,
            params,
        ).fetchall()
        by_category = db.execute(
            f"""
            SELECT classified.name, COUNT(*) AS count
            FROM (
                SELECT COALESCE(NULLIF(TRIM(w.category), ''), 'Chưa ánh xạ') AS name
                FROM detection_details d
                JOIN detection_history h ON h.id = d.history_id
                LEFT JOIN waste_types w ON w.id = d.waste_type_id
                {where}

                UNION ALL

                SELECT COALESCE(
                    NULLIF(TRIM(h.image_class_category), ''), 'Chưa ánh xạ'
                ) AS name
                FROM detection_history h
                WHERE NULLIF(TRIM(h.image_class_name), '') IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1
                      FROM detection_details d
                      WHERE d.history_id = h.id
                  )
                  {image_user_filter}
            ) AS classified
            GROUP BY classified.name
            ORDER BY count DESC
            """,
            (user_id, user_id) if user_id is not None else (),
        ).fetchall()
        by_month = db.execute(
            f"""
            SELECT strftime('%Y-%m', h.created_at) AS month, COUNT(d.id) AS count
            FROM detection_details d
            JOIN detection_history h ON h.id = d.history_id
            {where}
            GROUP BY month ORDER BY month DESC LIMIT 6
            """,
            params,
        ).fetchall()
    return {
        "images": int(totals["images"]),
        "objects": int(totals["objects"]),
        "by_type": [dict(row) for row in by_type],
        "by_category": [dict(row) for row in by_category],
        "by_month": list(reversed([dict(row) for row in by_month])),
    }


def save_feedback(user_id: int, history_id: int, content: str) -> None:
    with connection() as db:
        db.execute(
            "INSERT INTO feedback(user_id, history_id, content) VALUES (?, ?, ?)",
            (user_id, history_id, content.strip()),
        )


def list_feedback() -> list[dict[str, Any]]:
    with connection() as db:
        rows = db.execute(
            """
            SELECT f.id, f.content, f.created_at, u.username, f.history_id
            FROM feedback f LEFT JOIN users u ON u.id = f.user_id
            ORDER BY f.created_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def delete_feedback(feedback_id: int) -> None:
    with connection() as db:
        db.execute("DELETE FROM feedback WHERE id = ?", (feedback_id,))
