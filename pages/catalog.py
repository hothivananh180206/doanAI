import html
from typing import Any

import streamlit as st

from ai.detector import VI_LABEL_MAP
from database.db import list_waste_types


def render() -> None:
    st.caption("NHẬN DIỆN NHÓM RÁC · ĐẶC ĐIỂM · CÁCH XỬ LÝ")
    st.title("🗂️ Danh mục rác")
    st.write("Tìm hiểu các nhóm rác phổ biến và cách xử lý phù hợp.")
    waste_types = list_waste_types()
    if not waste_types:
        st.info("Chưa có danh mục rác.")
        return

    category_order = {
        "hữu cơ": ("🌿", "Rác hữu cơ", "organic"),
        "tái chế": ("♻️", "Rác tái chế", "recycle"),
        "còn lại": ("🗑️", "Rác còn lại", "residual"),
        "nguy hại": ("⚠️", "Rác nguy hại", "hazardous"),
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in waste_types:
        grouped.setdefault(item["category"].casefold(), []).append(item)

    ordered_categories = sorted(
        grouped,
        key=lambda category: (
            list(category_order).index(category)
            if category in category_order
            else len(category_order),
            category,
        ),
    )

    for category in ordered_categories:
        icon, title, color_class = category_order.get(
            category, ("♻️", category.title(), "recycle")
        )
        header_html = (
            f'<div class="eco-category-card eco-category-{color_class}" style="margin:1rem 0 0.5rem;padding:1rem 1.2rem;border-radius:14px;">'
            f'<div style="display:flex;align-items:center;justify-content:space-between;">'
            f'<h3 style="margin:0;font-size:1.15rem;color:#173a4b;">{icon} {html.escape(title)}</h3>'
            f'<span class="eco-category-meta" style="font-size:0.85rem;color:#4b5563;">'
            f'DANH MỤC · {html.escape(category.upper())} · {len(grouped[category])} loại'
            f'</span>'
            f'</div>'
            f'</div>'
        )
        st.markdown(header_html, unsafe_allow_html=True)

        for item in grouped[category]:
            raw_name = str(item["name"]).strip().casefold()
            vi_name = VI_LABEL_MAP.get(raw_name, raw_name.replace("_", " ").title())
            description = html.escape(item["description"] or "Chưa có mô tả.")
            instructions = html.escape(item["instructions"] or "")
            details = (
                f'<p style="margin:0.25rem 0 0;color:#047857;font-size:0.9rem;"><strong>✓ Cách xử lý:</strong> {instructions}</p>'
                if instructions
                else ""
            )
            item_html = (
                f'<div class="eco-category-item" style="margin:0.4rem 0;padding:0.75rem 1rem;background:rgba(255,255,255,0.78);border-radius:12px;border:1px solid #e5e7eb;">'
                f'<strong style="color:#173a4b;font-size:1rem;">{html.escape(vi_name)} <span style="font-size:0.82rem;font-weight:normal;color:#6b7280;">({html.escape(raw_name)})</span></strong>'
                f'<p style="margin:0.2rem 0;color:#4b5563;font-size:0.92rem;">{description}</p>'
                f'{details}'
                f'</div>'
            )
            st.markdown(item_html, unsafe_allow_html=True)
