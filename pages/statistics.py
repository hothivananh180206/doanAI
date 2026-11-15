import plotly.graph_objects as go
import streamlit as st

from database.db import statistics


def render() -> None:
    user = st.session_state["user"]
    stats = statistics(None if user["role"] == "admin" else int(user["id"]))
    st.caption("TỔNG HỢP HOẠT ĐỘNG PHÂN LOẠI")
    st.title("📊 Thống kê tổng quan")

    image_col, object_col = st.columns(2)
    with image_col:
        with st.container(border=True):
            with st.container(key="eco-stat-blue"):
                st.metric("🖼️ Tổng lượt nhận diện", stats["images"])
    with object_col:
        with st.container(border=True):
            with st.container(key="eco-stat-green"):
                st.metric("⬡ Tổng vật thể phát hiện", stats["objects"])

    category_col, month_col = st.columns(2)
    with category_col:
        with st.container(border=True):
            st.markdown("#### Tỷ lệ các loại rác")
            all_categories = stats["by_category"]
            unmapped_count = sum(
                item["count"]
                for item in all_categories
                if str(item["name"]).strip().casefold() == "chưa ánh xạ"
            )
            categories = [
                item
                for item in all_categories
                if str(item["name"]).strip().casefold() != "chưa ánh xạ"
                and item["count"] > 0
            ]
            if categories:
                colors = {
                    "Hữu cơ": "#43b96f",
                    "Tái chế": "#3298e8",
                    "Còn lại": "#f2a323",
                    "Nguy hại": "#ef5964",
                    "Chưa ánh xạ": "#8b9aaa",
                }
                labels = [item["name"] for item in categories]
                values = [item["count"] for item in categories]
                marker_colors = [colors.get(name, "#7d8da0") for name in labels]

                fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=labels,
                            values=values,
                            hole=0.55,
                            marker=dict(colors=marker_colors, line=dict(color="#ffffff", width=2)),
                            textinfo="percent",
                            hoverinfo="label+value+percent",
                            textfont=dict(size=13),
                        )
                    ]
                )
                fig.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=250,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5,
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)
                total = sum(item["count"] for item in categories)
                st.caption(
                    f"Tính trên {total} kết quả đã ghép được danh mục · "
                    + " · ".join(
                        f"{item['name']}: {item['count'] / total:.0%}"
                        for item in categories
                    )
                )
            else:
                st.info(
                    "Chưa có kết quả nào được ghép vào danh mục rác nên chưa thể "
                    "tính tỷ lệ."
                )
            if unmapped_count:
                st.warning(
                    f"{unmapped_count} kết quả chưa xác định được danh mục; "
                    "các kết quả này được tách riêng, không tính vào tỷ lệ."
                )

    with month_col:
        with st.container(border=True):
            st.markdown("#### Số vật thể phát hiện theo tháng")
            months = stats["by_month"]
            if months:
                st.bar_chart(
                    {
                        "Tháng": [
                            f"T{int(item['month'][5:])}/{item['month'][:4]}"
                            for item in months
                        ],
                        "Số vật thể": [item["count"] for item in months],
                    },
                    x="Tháng",
                    y="Số vật thể",
                    color="#3298e8",
                )
            else:
                st.info("Chưa có vật thể được phát hiện theo tháng.")

    if stats["by_type"]:
        with st.container(border=True):
            st.markdown("#### Các nhãn YOLO được nhận diện")
            st.dataframe(
                [
                    {"Nhãn": item["name"], "Số lượng": item["count"]}
                    for item in stats["by_type"]
                ],
                use_container_width=True,
                hide_index=True,
            )
