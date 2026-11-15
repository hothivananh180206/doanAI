from __future__ import annotations

from datetime import datetime, timedelta
import sqlite3
import secrets
from typing import Any

import streamlit as st

from database.db import (
    authenticate,
    create_auth_session,
    create_user,
    delete_auth_session,
    get_active_user,
    get_auth_session_user,
    user_count,
)


AUTH_COOKIE_NAME = "ecosort_session"


def restore_login_session(cookies: Any) -> dict[str, Any] | None:
    token = cookies.get(AUTH_COOKIE_NAME)
    if not token:
        return None
    user = get_auth_session_user(token)
    if user is None:
        delete_auth_session(token)
        cookies.remove(AUTH_COOKIE_NAME)
        return None
    st.session_state["user"] = user
    return user


def start_login_session(user: dict[str, Any], cookies: Any) -> None:
    token = secrets.token_urlsafe(32)
    create_auth_session(int(user["id"]), token)
    cookies.set(
        AUTH_COOKIE_NAME,
        token,
        expires=datetime.now() + timedelta(days=365),
        same_site="lax",
    )
    st.session_state["user"] = user


def end_login_session(cookies: Any) -> None:
    token = cookies.get(AUTH_COOKIE_NAME)
    if token:
        delete_auth_session(token)
        cookies.remove(AUTH_COOKIE_NAME)
    st.session_state.pop("user", None)


def refresh_session_user() -> dict[str, Any] | None:
    session_user = st.session_state.get("user")
    if session_user is None:
        return None
    if not isinstance(session_user, dict):
        st.session_state.pop("user", None)
        return None
    try:
        user_id = int(session_user["id"])
    except (KeyError, TypeError, ValueError):
        st.session_state.pop("user", None)
        return None

    user = get_active_user(user_id)
    if user is None:
        st.session_state.pop("user", None)
        return None
    st.session_state["user"] = user
    return user


def render_login(cookies: Any) -> None:
    st.caption("♻️ ECOSORT AI · PHÂN LOẠI RÁC THÔNG MINH")
    st.title("Chào mừng bạn!")
    st.write("Cùng AI nhận diện rác và xây dựng thói quen xanh mỗi ngày.")

    intro, form_column = st.columns([1.15, 1], gap="large")
    with intro:
        with st.container(border=True):
            st.subheader("🌿 Bắt đầu từ một hành động nhỏ")
            st.write(
                "Chọn ảnh rác, xem kết quả YOLO, tra cứu cách phân loại "
                "và theo dõi lịch sử trong cùng một nơi."
            )
            st.divider()
            first, second = st.columns(2)
            first.metric("Công nghệ", "YOLO")
            second.metric("Lưu trữ", "SQLite")
            st.caption(
                "Kết quả AI mang tính hỗ trợ; hãy kiểm tra vật liệu thực tế "
                "trước khi xử lý."
            )

    if user_count("admin") == 0:
        with form_column:
            st.subheader("Thiết lập lần đầu")
            st.info("Tạo tài khoản quản trị viên để bắt đầu sử dụng.")
            with st.form("create_admin"):
                username = st.text_input("Tên quản trị viên")
                password = st.text_input("Mật khẩu (ít nhất 8 ký tự)", type="password")
                password_confirm = st.text_input("Nhập lại mật khẩu", type="password")
                submitted = st.form_submit_button(
                    "Tạo tài khoản quản trị",
                    type="primary",
                    use_container_width=True,
                )
        if submitted:
            if password != password_confirm:
                st.error("Hai mật khẩu không khớp.")
            else:
                try:
                    user_id = create_user(username, password, "admin")
                except (ValueError, sqlite3.IntegrityError) as exc:
                    st.error(f"Không tạo được tài khoản: {exc}")
                else:
                    start_login_session(
                        {
                            "id": user_id,
                            "username": username.strip(),
                            "role": "admin",
                        },
                        cookies,
                    )
                    st.session_state["redirect_after_login"] = True
        return

    with form_column:
        login_tab, register_tab = st.tabs(["Đăng nhập", "Tạo tài khoản"])
        with login_tab:
            with st.form("login"):
                username = st.text_input("Tên đăng nhập")
                password = st.text_input("Mật khẩu", type="password")
                submitted = st.form_submit_button(
                    "Đăng nhập", type="primary", use_container_width=True
                )
            if submitted:
                user = authenticate(username, password)
                if user is None:
                    st.error("Tên đăng nhập hoặc mật khẩu không đúng.")
                else:
                    start_login_session(user, cookies)
                    st.session_state["redirect_after_login"] = True
        with register_tab:
            with st.form("register"):
                username = st.text_input("Tên đăng nhập mới")
                password = st.text_input("Mật khẩu mới (ít nhất 8 ký tự)", type="password")
                password_confirm = st.text_input("Nhập lại mật khẩu", type="password")
                submitted = st.form_submit_button(
                    "Tạo tài khoản", type="primary", use_container_width=True
                )
            if submitted:
                if password != password_confirm:
                    st.error("Hai mật khẩu không khớp.")
                else:
                    try:
                        user_id = create_user(username, password)
                    except (ValueError, sqlite3.IntegrityError) as exc:
                        st.error(f"Không tạo được tài khoản: {exc}")
                    else:
                        start_login_session(
                            {
                                "id": user_id,
                                "username": username.strip(),
                                "role": "user",
                            },
                            cookies,
                        )
                        st.session_state["redirect_after_login"] = True
