from datetime import date
from pathlib import Path

import streamlit as st

from database.db import delete_history, history_rows, list_waste_types


def render() -> None:
    user = st.session_state["user"]
    st.caption("CÁC LƯỢT PHÂN TÍCH ĐÃ ĐƯỢC LƯU")
    st.title("🕒 Lịch sử nhận diện")
    records = history_rows(
        None if user["role"] == "admin" else int(user["id"]),
        limit=5000,
    )
    if not records:
        st.info("Lịch sử sẽ xuất hiện tại đây sau khi bạn nhận diện ảnh đầu tiên.")
        return

    waste_categories = {
        item["category"].strip()
        for item in list_waste_types()
        if item["category"].strip()
    }
    for record in records:
        waste_categories.update(
            category.strip()
            for category in (record["categories"] or "").split(",")
            if category.strip()
        )
        if record["image_class_category"]:
            waste_categories.add(record["image_class_category"].strip())
    category_options = ["Tất cả", *sorted(waste_categories, key=str.casefold)]

    with st.form("history_search"):
        start_col, end_col, category_col, search_col = st.columns(
            [1.2, 1.2, 1.5, 0.8], vertical_alignment="bottom"
        )
        start_date = start_col.date_input("Từ ngày", value=date(2000, 1, 1))
        end_date = end_col.date_input("Đến ngày", value=date.today())
        category_filter = category_col.selectbox("Danh mục rác", category_options)
        submitted = search_col.form_submit_button(
            "🔍 Tìm kiếm", type="primary", use_container_width=True
        )

    if start_date > end_date:
        st.error("Ngày bắt đầu phải trước hoặc bằng ngày kết thúc.")
        return
    records = [
        record
        for record in records
        if start_date.isoformat() <= record["created_at"][:10] <= end_date.isoformat()
    ]
    if category_filter != "Tất cả":
        selected_category = category_filter.casefold()
        records = [
            record
            for record in records
            if selected_category
            in {
                category.casefold()
                for category in (
                    (record["categories"] or "").split(",")
                    + [record["image_class_category"] or ""]
                )
                if category
            }
        ]
    if submitted:
        st.session_state["history_page"] = 0

    st.caption(f"Tìm thấy {len(records)} lượt nhận diện")
    if not records:
        st.info("Không tìm thấy lượt nhận diện phù hợp.")
        return

    page_size = 8
    page_count = (len(records) + page_size - 1) // page_size
    page = min(int(st.session_state.get("history_page", 0)), page_count - 1)
    st.session_state["history_page"] = page
    st.markdown("#### Kết quả tìm kiếm")
    with st.container(key="eco-history-header"):
        header = st.columns([0.35, 1.7, 1.2, 2.8, 0.8])
        for column, label in zip(
            header, ["#", "Thời gian", "Ảnh kết quả", "Kết quả", "Thao tác"]
        ):
            column.markdown(f"**{label}**")

    page_records = records[page * page_size : (page + 1) * page_size]
    for index, record in enumerate(page_records, start=page * page_size + 1):
        with st.container(border=True):
            number, time_col, image_col, result_col, action_col = st.columns(
                [0.35, 1.7, 1.2, 2.8, 0.8], vertical_alignment="center"
            )
            number.write(index)
            time_col.write(record["created_at"])
            image_path = Path(record["result_image_path"])
            if image_path.is_file():
                image_col.image(str(image_path), use_container_width=True)
            else:
                image_col.caption("Không còn ảnh")
            result_col.write(
                record["image_class_name"]
                or record["labels"]
                or "Không phát hiện vật thể"
            )
            result_col.markdown(
                f'<span class="eco-pill">{record["object_count"]} vật thể</span>',
                unsafe_allow_html=True,
            )
            if record["categories"]:
                result_col.caption("Nhóm: " + record["categories"])
            if action_col.button(
                "Xem",
                key=f"view_history_{record['id']}",
                type="primary",
                use_container_width=True,
            ):
                st.session_state["selected_history_id"] = int(record["id"])
                st.rerun()

    if page_count > 1:
        previous, page_label, next_page = st.columns([1, 2, 1])
        with previous:
            with st.container(key="eco-blue-prev"):
                if st.button(
                    "‹ Trang trước",
                    disabled=page == 0,
                    use_container_width=True,
                ):
                    st.session_state["history_page"] = page - 1
                    st.rerun()
        page_label.markdown(
            f"<div style='text-align:center;padding:.5rem'>Trang {page + 1} / {page_count}</div>",
            unsafe_allow_html=True,
        )
        with next_page:
            with st.container(key="eco-blue-next"):
                if st.button(
                    "Trang sau ›",
                    disabled=page >= page_count - 1,
                    use_container_width=True,
                ):
                    st.session_state["history_page"] = page + 1
                    st.rerun()

    selected_id = st.session_state.get("selected_history_id")
    selected = next(
        (record for record in records if record["id"] == selected_id),
        None,
    )
    if selected is not None:
        st.subheader(f"Chi tiết lượt nhận diện #{selected['id']}")
        st.caption(
            f"{selected['created_at']} · "
            f"{'Camera' if selected['source'] == 'camera' else 'Tải ảnh'} · "
            f"Model: {Path(selected['model_path']).name}"
        )
        original_col, result_col = st.columns(2)
        for column, key, label in (
            (original_col, "image_path", "Ảnh gốc"),
            (result_col, "result_image_path", "Ảnh kết quả"),
        ):
            image_path = Path(selected[key])
            if image_path.is_file():
                column.image(str(image_path), caption=label, use_container_width=True)
            else:
                column.warning(f"{label} không còn trên thiết bị.")
        if selected["image_class_confidence"] is not None:
            st.info(
                f"Dự đoán toàn ảnh: {selected['image_class_name']} · "
                f"độ tin cậy model {selected['image_class_confidence']:.1%}"
            )
        with st.container(key=f"eco-danger-history-{selected['id']}"):
            if st.button(
                "🗑️ Xóa lượt nhận diện",
                key=f"delete_selected_{selected['id']}",
                type="secondary",
            ):
                paths = delete_history(int(selected["id"]))
                if paths:
                    for path in paths:
                        path.unlink(missing_ok=True)
                st.session_state.pop("selected_history_id", None)
                st.rerun()
