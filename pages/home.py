import streamlit as st

from database.db import history_rows, list_waste_types, statistics


def render() -> None:
    st.caption("THỨC DẬY MỖI NGÀY, SỐNG XANH HƠN MỘT CHÚT")

    totals = statistics(None)
    waste_types = list_waste_types()
    st.markdown(
        """
        <section class="eco-hero-dashboard">
          <div>
            <div class="eco-hero-kicker">♻️ ECOSORT AI · YOLO</div>
            <h2>Ứng dụng AI phân loại rác thải</h2>
            <p>Vì một môi trường xanh – sạch – đẹp. Nhận diện vật thể, tra cứu
            cách phân loại và chọn cách xử lý phù hợp.</p>
          </div>
          <div class="eco-hero-art" aria-hidden="true">
            <div class="eco-hero-sun"></div>
            <div class="eco-hero-globe">🌍</div>
            <div class="eco-hero-bins">
              <div class="eco-bin eco-bin-green"><span>♻️</span>Hữu cơ</div>
              <div class="eco-bin eco-bin-blue"><span>♻️</span>Tái chế</div>
              <div class="eco-bin eco-bin-orange"><span>♻️</span>Còn lại</div>
              <div class="eco-bin eco-bin-red"><span>♻️</span>Nguy hại</div>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.title("Chào mừng đến với EcoSort AI 👋")
    first, second, third = st.columns(3)
    with first:
        with st.container(border=True):
            st.metric("📷 Tổng lượt nhận diện", totals["images"])
            st.caption("Ảnh đã được phân tích")
    with second:
        with st.container(border=True):
            st.metric("🔎 Vật thể phát hiện", totals["objects"])
            st.caption("Các vật thể AI đã tìm thấy")
    with third:
        with st.container(border=True):
            st.metric("♻️ Nhóm phân loại", len({item["category"] for item in waste_types}))
            st.caption("Hữu cơ · Tái chế · Còn lại · Nguy hại")

    st.subheader("⭐ Hướng dẫn nhanh")
    steps = st.columns(3)
    step_content = (
        ("📷", "1 · Tải ảnh hoặc bật camera", "Chọn ảnh rác rõ nét từ thiết bị."),
        ("🔍", "2 · Nhận diện", "YOLO tìm và đánh dấu vật thể."),
        ("📊", "3 · Xem kết quả", "Đọc gợi ý, lịch sử và thống kê."),
    )
    for column, (icon, title, detail) in zip(steps, step_content):
        with column:
            st.markdown(
                f"""
                <div class="eco-step-card">
                  <strong>{icon} {title}</strong>
                  <span>{detail}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        technology_col, art_col = st.columns([1.5, 1], vertical_alignment="center")
        with technology_col:
            st.subheader("🌟 Giới thiệu công nghệ YOLO")
            st.write(
                "YOLO (You Only Look Once) là thuật toán nhận diện vật thể hiện đại, "
                "giúp phát hiện nhanh vật thể trong ảnh."
            )
            st.link_button(
                "Tìm hiểu thêm →",
                "https://docs.ultralytics.com/",
                type="primary",
            )
        with art_col:
            st.markdown(
                """
                <div style="display:grid;place-items:center;min-height:145px;
                            border-radius:15px;background:linear-gradient(135deg,#e5f4ff,#dff5e8);
                            color:#087443;font-size:2rem;font-weight:850">
                  ✨ YOLO
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="eco-category-card eco-category-organic" style="text-align:center">
          <strong>🌱 Cùng chung tay bảo vệ môi trường!</strong>
          <p>Phân loại đúng từ hôm nay để rác được tái sử dụng và xử lý an toàn.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Nhận diện gần đây")
    records = history_rows(None, limit=5)
    if records:
        for row in records[:3]:
            with st.container(border=True):
                time_col, result_col, count_col = st.columns([2, 4, 1])
                time_col.caption(row["created_at"])
                result_col.write(
                    row["image_class_name"]
                    or row["labels"]
                    or "Chưa phát hiện vật thể"
                )
                count_col.metric("Vật thể", row["object_count"])
    else:
        st.info("Chưa có lượt nhận diện. Tải ảnh đầu tiên để bắt đầu nhé.")
