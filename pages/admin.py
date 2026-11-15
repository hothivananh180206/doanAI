import sqlite3

import streamlit as st

from config import BASE_DIR, DEFAULT_DETECTION_MODEL
from database.db import (
    delete_all_history,
    delete_feedback,
    delete_instruction,
    list_feedback,
    get_setting,
    list_instructions,
    list_users,
    list_waste_types,
    history_rows,
    delete_user,
    create_user,
    save_instruction,
    save_waste_type,
    set_setting,
    set_user_active,
    set_user_role,
    set_waste_type_active,
)


def _show_database_error(action: str, error: Exception) -> None:
    st.error(f"{action} không thành công: {error}")


def _users_tab() -> None:
    st.subheader("Tài khoản và quyền truy cập")
    st.caption(
        "Tạo tài khoản, đổi vai trò hoặc khóa/mở khóa quyền đăng nhập. "
        "Quản trị viên có thể xem và cấu hình toàn hệ thống."
    )
    current_user = st.session_state["user"]
    with st.expander("➕ Thêm người dùng"):
        with st.form("create_user_admin"):
            username = st.text_input("Tên đăng nhập")
            password = st.text_input("Mật khẩu (ít nhất 8 ký tự)", type="password")
            role_labels = {
                "Người dùng": "user",
                "Quản trị viên": "admin",
            }
            role_label = st.selectbox("Quyền truy cập", list(role_labels))
            submitted = st.form_submit_button(
                "Thêm người dùng", type="primary", use_container_width=True
            )
        if submitted:
            try:
                create_user(username, password, role_labels[role_label])
            except (ValueError, sqlite3.IntegrityError) as exc:
                _show_database_error("Thêm tài khoản", exc)
            else:
                st.success("Đã thêm tài khoản.")
                st.rerun()

    users = list_users()
    st.markdown("#### Tài khoản hiện có")
    role_labels = {"user": "Người dùng", "admin": "Quản trị viên"}
    headers = st.columns([0.35, 1.5, 1.15, 0.8, 1, 0.85, 0.55])
    for column, label in zip(
        headers,
        [
            "#",
            "Tên đăng nhập",
            "Vai trò mới",
            "Lưu quyền",
            "Trạng thái",
            "Khóa / mở",
            "Xóa",
        ],
    ):
        column.markdown(f"**{label}**")
    for index, user in enumerate(users, start=1):
        with st.container(border=True):
            (
                number_col,
                name_col,
                role_col,
                save_col,
                status_col,
                lock_col,
                remove_col,
            ) = st.columns(
                [0.35, 1.5, 1.15, 0.8, 1, 0.85, 0.55],
                vertical_alignment="center",
            )
            number_col.write(index)
            name_col.write(f"**{user['username']}**")
            proposed_role = role_col.selectbox(
                "Quyền truy cập",
                ["user", "admin"],
                index=0 if user["role"] == "user" else 1,
                key=f"user_role_{user['id']}",
                format_func=lambda value: role_labels[value],
                label_visibility="collapsed",
            )
            if save_col.button(
                "Lưu",
                key=f"save_role_{user['id']}",
                type="primary",
                use_container_width=True,
            ):
                active_admins = sum(
                    item["role"] == "admin" and item["is_active"] for item in users
                )
                if (
                    int(user["id"]) == int(current_user["id"])
                    and proposed_role != "admin"
                ):
                    st.error("Không thể thay đổi vai trò của tài khoản đang đăng nhập.")
                elif (
                    user["role"] == "admin"
                    and proposed_role != "admin"
                    and user["is_active"]
                    and active_admins <= 1
                ):
                    st.error("Không thể hạ vai trò quản trị viên cuối cùng.")
                else:
                    set_user_role(int(user["id"]), proposed_role)
                    st.rerun()
            status_label = "🟢 Hoạt động" if user["is_active"] else "🔒 Bị khóa"
            status_color = "#e7f8ec" if user["is_active"] else "#fff0ef"
            status_text_color = "#087443" if user["is_active"] else "#c83f43"
            status_col.markdown(
                f"""
                <span style="display:inline-block;padding:.3rem .6rem;border-radius:999px;
                             background:{status_color};color:{status_text_color};font-weight:700">
                  {status_label}
                </span>
                """,
                unsafe_allow_html=True,
            )
            if int(user["id"]) == int(current_user["id"]):
                lock_col.caption("Bạn")
                remove_col.caption("")
            else:
                lock_label = "Khóa" if user["is_active"] else "Mở khóa"
                with lock_col:
                    with st.container(
                        key=(
                            f"eco-danger-lock-{user['id']}"
                            if user["is_active"]
                            else f"eco-green-lock-{user['id']}"
                        )
                    ):
                        lock_clicked = st.button(
                            lock_label,
                            key=f"user_{user['id']}",
                            type="secondary",
                            use_container_width=True,
                        )
                if lock_clicked:
                    if user["role"] == "admin" and user["is_active"]:
                        active_admins = sum(
                            item["role"] == "admin" and item["is_active"]
                            for item in users
                        )
                        if active_admins <= 1:
                            st.error("Không thể khóa quản trị viên cuối cùng.")
                            return
                    set_user_active(int(user["id"]), not bool(user["is_active"]))
                    st.rerun()
                with remove_col:
                    with st.container(key=f"eco-danger-delete-user-{user['id']}"):
                        delete_clicked = st.button(
                            "🗑️",
                            key=f"delete_user_{user['id']}",
                            type="secondary",
                            use_container_width=True,
                        )
                if delete_clicked:
                    if user["role"] == "admin" and user["is_active"]:
                        active_admins = sum(
                            item["role"] == "admin" and item["is_active"]
                            for item in users
                        )
                        if active_admins <= 1:
                            st.error("Không thể xóa quản trị viên cuối cùng.")
                            return
                    delete_user(int(user["id"]))
                    st.rerun()


def _waste_types_tab() -> None:
    st.subheader("Các loại rác và nhóm phân loại")
    st.caption(
        "Tên loại là nhãn mà model nhận diện. Chọn nhóm phù hợp để ứng dụng "
        "hiển thị kết quả và hướng dẫn đúng."
    )
    types = list_waste_types(active_only=False)
    if types:
        selected_id = st.selectbox(
            "Chọn danh mục để sửa",
            options=[int(item["id"]) for item in types],
            format_func=lambda value: next(
                f"{item['name']} · {item['category']}" for item in types if item["id"] == value
            ),
        )
        selected = next(item for item in types if item["id"] == selected_id)
        with st.form("edit_waste_type"):
            name = st.text_input(
                "Tên loại rác / nhãn model",
                value=selected["name"],
                help="Giữ tên lớp của model để kết quả nhận diện khớp với danh mục.",
            )
            category = st.selectbox(
                "Nhóm",
                ["Hữu cơ", "Tái chế", "Còn lại", "Nguy hại", "Chưa ánh xạ"],
                index=(
                    ["Hữu cơ", "Tái chế", "Còn lại", "Nguy hại", "Chưa ánh xạ"].index(
                        selected["category"]
                    )
                    if selected["category"]
                    in ["Hữu cơ", "Tái chế", "Còn lại", "Nguy hại", "Chưa ánh xạ"]
                    else 4
                ),
            )
            description = st.text_area("Mô tả", value=selected["description"])
            instructions = st.text_area(
                "Hướng dẫn xử lý", value=selected["instructions"]
            )
            active = st.checkbox("Đang sử dụng", value=bool(selected["is_active"]))
            save = st.form_submit_button("Lưu danh mục", type="primary")
        if save:
            try:
                save_waste_type(
                    name,
                    category,
                    description,
                    instructions,
                    type_id=selected_id,
                )
                set_waste_type_active(selected_id, active)
            except (ValueError, sqlite3.IntegrityError) as exc:
                _show_database_error("Lưu danh mục", exc)
            else:
                st.success("Đã cập nhật danh mục.")
                st.rerun()

    with st.expander("Thêm loại rác"):
        with st.form("add_waste_type"):
            name = st.text_input("Tên nhãn")
            category = st.selectbox(
                "Nhóm phân loại",
                ["Hữu cơ", "Tái chế", "Còn lại", "Nguy hại", "Chưa ánh xạ"],
                key="new_category",
            )
            description = st.text_area("Mô tả")
            instructions = st.text_area("Hướng dẫn")
            save = st.form_submit_button("Thêm loại", type="primary")
        if save:
            try:
                save_waste_type(name, category, description, instructions)
            except (ValueError, sqlite3.IntegrityError) as exc:
                _show_database_error("Thêm danh mục", exc)
            else:
                st.success("Đã thêm loại rác.")
                st.rerun()


def _instructions_tab() -> None:
    st.subheader("Nội dung hướng dẫn phân loại")
    st.caption(
        "Mỗi hướng dẫn gắn với một loại rác. Chọn hướng dẫn có sẵn để sửa "
        "hoặc chọn “Tạo mới” để thêm."
    )
    types = list_waste_types(active_only=False)
    if not types:
        st.info("Hãy thêm danh mục rác trước.")
        return
    type_by_id = {int(item["id"]): item for item in types}
    existing = list_instructions()
    options: list[int | None] = [None] + [int(item["id"]) for item in existing]
    with st.form("instruction_editor"):
        selected_id = st.selectbox(
            "Hướng dẫn hiện có (chọn mục để sửa)",
            options=options,
            format_func=lambda value: (
                "Tạo mới"
                if value is None
                else next(item["title"] for item in existing if item["id"] == value)
            ),
        )
        selected = next(
            (item for item in existing if item["id"] == selected_id), None
        )
        type_ids = list(type_by_id)
        type_id = st.selectbox(
            "Loại rác",
            options=type_ids,
            index=(
                type_ids.index(int(selected["waste_type_id"]))
                if selected
                else 0
            ),
            format_func=lambda value: type_by_id[value]["name"],
        )
        title = st.text_input(
            "Tiêu đề", value=selected["title"] if selected else ""
        )
        content = st.text_area(
            "Nội dung", value=selected["content"] if selected else ""
        )
        save = st.form_submit_button("Lưu hướng dẫn", type="primary")
    if save:
        try:
            save_instruction(
                int(type_id),
                title,
                content,
                int(selected_id) if selected_id is not None else None,
            )
        except (ValueError, sqlite3.IntegrityError) as exc:
            _show_database_error("Lưu hướng dẫn", exc)
        else:
            st.success("Đã lưu hướng dẫn.")
            st.rerun()

    if existing:
        with st.expander("Xóa hướng dẫn"):
            to_delete = st.selectbox(
                "Chọn hướng dẫn",
                options=[int(item["id"]) for item in existing],
                format_func=lambda value: next(
                    item["title"] for item in existing if item["id"] == value
                ),
                key="delete_instruction_id",
            )
            with st.container(key="eco-danger-delete-instruction"):
                if st.button(
                    "Xóa hướng dẫn đã chọn",
                    type="secondary",
                ):
                    delete_instruction(int(to_delete))
                    st.rerun()


def _history_tab() -> None:
    st.subheader("Lịch sử nhận diện")
    st.caption("Danh sách các lượt nhận diện đã lưu trong hệ thống.")
    rows = history_rows(limit=500)
    if rows:
        st.dataframe(
            [
                {
                    "ID": row["id"],
                    "Người dùng": row["username"] or "đã xóa",
                    "Thời gian": row["created_at"],
                    "Nguồn": row["source"],
                    "Số vật thể": row["object_count"],
                    "Nhãn": (
                        row["image_class_name"]
                        or row["labels"]
                        or "Không phát hiện"
                    ),
                    "Model": row["model_path"],
                }
                for row in rows
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("Chưa có lịch sử nhận diện.")
    feedback = list_feedback()
    st.subheader("Góp ý của người dùng")
    if feedback:
        for item in feedback:
            left, right = st.columns([5, 1])
            left.write(
                f"**{item['username'] or 'đã xóa'}** · {item['created_at']}: "
                f"{item['content']}"
            )
            with right:
                with st.container(key=f"eco-danger-delete-feedback-{item['id']}"):
                    if st.button(
                        "Xóa",
                        key=f"feedback_{item['id']}",
                        type="secondary",
                    ):
                        delete_feedback(int(item["id"]))
                        st.rerun()
    else:
        st.caption("Chưa có góp ý.")
    if rows:
        confirm_delete = st.checkbox(
            "Tôi xác nhận muốn xóa toàn bộ lịch sử nhận diện.",
            key="confirm_delete_all_history",
        )
        with st.container(key="eco-danger-delete-all-history"):
            if st.button(
                "Xóa toàn bộ lịch sử",
                type="secondary",
                disabled=not confirm_delete,
            ):
                paths = delete_all_history()
                for original, result in paths:
                    original.unlink(missing_ok=True)
                    result.unlink(missing_ok=True)
                st.success(f"Đã xóa {len(paths)} lượt nhận diện.")
                st.rerun()


def _models_tab() -> None:
    st.subheader("Cài đặt AI nhận diện")
    st.caption(
        "Chọn model tìm vật thể và model phân loại chất liệu. Model phân loại "
        "cần có tên lớp khớp với tên loại rác trong danh mục."
    )
    with st.expander("Huấn luyện model tìm vật thể theo khung (YOLO detect)"):
        st.write(
            "Dùng ảnh có nhãn bounding box YOLO để tìm nhiều vật thể trong một ảnh, "
            "như chai nhựa và thùng carton. Dataset phân loại ảnh hiện tại không "
            "thể thay thế các nhãn khung này."
        )
        st.caption(
            "Chế độ CPU tiết kiệm RAM: ảnh 320 px, batch 1. Huấn luyện có thể lâu; "
            "đóng ứng dụng nặng trước khi bắt đầu."
        )
        detect_data_yaml = st.text_input(
            "Đường dẫn tệp data.yaml của YOLO detect",
            key="detect_dataset_yaml",
            help=(
                "Tệp phải khai báo train, val và names; mỗi ảnh cần có tệp nhãn "
                "YOLO tương ứng trong thư mục labels."
            ),
        )
        detect_epochs = st.number_input(
            "Số epoch cho detector",
            min_value=1,
            max_value=300,
            value=50,
            key="detect_epochs",
        )
        if st.button("Huấn luyện model phát hiện rác", type="primary"):
            from ai.trainer import train_detector

            try:
                with st.spinner("Đang huấn luyện YOLO detect (GPU nếu có, CPU nếu không)..."):
                    best_model = train_detector(
                        detect_data_yaml,
                        int(detect_epochs),
                        "waste_detect_run",
                    )
                set_setting("detection_model", str(best_model.resolve()))
            except Exception as exc:
                st.error(f"Huấn luyện detector thất bại: {exc}")
            else:
                st.success(f"Đã lưu và bật detector mới: {best_model}")

    detector_model = st.text_input(
        "Model phát hiện vật thể",
        value=get_setting("detection_model", DEFAULT_DETECTION_MODEL),
        help=(
            "Model phát hiện vật thể và vẽ khung. yolo11n.pt mặc định nhận "
            "diện vật thể phổ thông, chưa được huấn luyện riêng cho rác."
        ),
    )
    classifier_model = st.text_input(
        "Model phân loại rác (không bắt buộc)",
        value=get_setting("classification_model", ""),
        help=(
            "Dùng checkpoint classify đã huấn luyện cho các loại rác của dự án. "
            "Tên lớp cần khớp tên loại trong danh mục."
        ),
    )
    candidates = sorted((BASE_DIR / "runs" / "classify").glob("**/weights/best.pt"))
    if candidates:
        selected_path = st.selectbox(
            "Hoặc chọn model đã huấn luyện trong dự án",
            options=[""] + [str(path) for path in candidates],
        )
        if selected_path:
            with st.container(key="eco-blue-model-select"):
                if st.button("Dùng model đã chọn", type="secondary"):
                    classifier_model = selected_path
                    st.session_state["new_classifier_model"] = selected_path
    if "new_classifier_model" in st.session_state:
        classifier_model = st.session_state["new_classifier_model"]
    if st.button("Lưu cấu hình model", type="primary"):
        set_setting("detection_model", detector_model.strip())
        set_setting("classification_model", classifier_model.strip())
        st.success("Đã lưu cấu hình model.")
        st.rerun()
    st.info(
        "Checkpoint phân loại ảnh (classify) không tự tạo bounding box. "
        "Muốn có khung cần checkpoint YOLO task detect. Chỉ nhãn khớp tên loại "
        "rác trong danh mục mới được gán vào nhóm; nhãn YOLO phổ thông không bị "
        "đoán thành loại rác."
    )


def render() -> None:
    st.caption("TRUNG TÂM QUẢN LÝ")
    st.title("⚙️ Quản trị & Cài đặt AI")
    st.write(
        "Quản lý danh mục rác, hướng dẫn xử lý và cài đặt mô hình AI "
        "tại các mục bên dưới."
    )
    types = list_waste_types(active_only=False)
    records = history_rows(limit=500)
    first, second = st.columns(2)
    with first:
        with st.container(border=True):
            st.metric(
                "♻️ Loại rác trong hệ thống",
                sum(bool(item["is_active"]) for item in types),
                help="Số loại rác đang được sử dụng.",
            )
    with second:
        with st.container(border=True):
            st.metric("📷 Tổng lượt nhận diện", len(records))

    types_tab, instructions_tab, history_tab, models_tab = st.tabs(
        [
            "♻️ Danh mục rác",
            "📘 Hướng dẫn xử lý",
            "🧾 Lịch sử & góp ý",
            "🤖 Cài đặt AI",
        ]
    )
    with types_tab:
        _waste_types_tab()
    with instructions_tab:
        _instructions_tab()
    with history_tab:
        _history_tab()
    with models_tab:
        _models_tab()
