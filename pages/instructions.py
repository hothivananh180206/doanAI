import html
from typing import Any

import streamlit as st

from ai.detector import VI_LABEL_MAP
from database.db import list_instructions, list_waste_types


def render() -> None:
    st.caption("MẸO PHÂN LOẠI RÁC TẠI NGUỒN")
    st.title("📘 Hướng dẫn phân loại")
    st.write("Hướng dẫn phân loại rác chi tiết theo từng nhóm và loại rác phổ biến.")
    waste_types = list_waste_types()
    if not waste_types:
        st.info("Chưa có hướng dẫn phân loại.")
        return

    configured_instructions: dict[str, list[dict[str, Any]]] = {}
    for instruction in list_instructions():
        configured_instructions.setdefault(instruction["waste_type"], []).append(
            instruction
        )

    category_info = {
        "hữu cơ": ("🌿", "Rác hữu cơ", "Cho vào thùng màu xanh lá", "organic"),
        "tái chế": ("♻️", "Rác tái chế", "Cho vào thùng màu xanh dương", "recycle"),
        "còn lại": ("🗑️", "Rác còn lại", "Cho vào thùng màu xám", "residual"),
        "nguy hại": ("⚠️", "Rác nguy hại", "Bỏ vào điểm thu gom riêng", "hazardous"),
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in waste_types:
        grouped.setdefault(item["category"].casefold(), []).append(item)

    ordered_categories = sorted(
        grouped,
        key=lambda category: (
            list(category_info).index(category)
            if category in category_info
            else len(category_info),
            category,
        ),
    )

    for category in ordered_categories:
        icon, title, disposal_hint, color_class = category_info.get(
            category,
            (
                "♻️",
                category.title(),
                "Phân loại theo hướng dẫn tại địa phương",
                "recycle",
            ),
        )

        cards = []
        for item in grouped[category]:
            raw_name = str(item["name"]).strip().casefold()
            vi_name = VI_LABEL_MAP.get(raw_name, raw_name.replace("_", " ").title())
            description = html.escape(item["description"] or "Chưa có mô tả.")
            
            steps = []
            inst = (item["instructions"] or "").strip()
            desc = (item["description"] or "").strip()
            if inst and inst != desc:
                steps.append(html.escape(inst))
            for entry in configured_instructions.get(item["name"], []):
                content = entry.get("content", "").strip()
                if content and content != desc and html.escape(content) not in steps:
                    steps.append(html.escape(content))
            
            if not steps:
                if inst:
                    steps.append(html.escape(inst))
                else:
                    steps.append("Phân loại và xử lý theo đúng nhóm quy định.")
            
            step_list = "".join(f"<li>{step}</li>" for step in steps)
            card_html = (
                f'<div class="eco-category-item" style="margin:0.5rem 0;padding:0.75rem 1rem;background:rgba(255,255,255,0.78);border-radius:12px;">'
                f'<strong style="color:#173a4b;font-size:1.02rem;">{html.escape(vi_name)} <span style="font-size:0.82rem;font-weight:normal;color:#6b7280;">({html.escape(raw_name)})</span></strong>'
                f'<p style="margin:0.25rem 0 0.4rem;color:#4b5563;font-size:0.92rem;">{description}</p>'
                f'<ul style="margin:0;padding-left:1.25rem;color:#355062;font-size:0.9rem;">{step_list}</ul>'
                f'</div>'
            )
            cards.append(card_html)

        cards_combined = "".join(cards)
        section_html = (
            f'<div class="eco-guide-card eco-guide-{color_class}" style="margin:1rem 0;padding:1.2rem;border-radius:16px;">'
            f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
            f'<h3 style="margin:0;font-size:1.2rem;color:#173a4b;">{icon} {html.escape(title)}</h3>'
            f'<span class="eco-pill" style="font-size:0.85rem;">✓ {html.escape(disposal_hint)}</span>'
            f'</div>'
            f'{cards_combined}'
            f'</div>'
        )
        st.markdown(section_html, unsafe_allow_html=True)

    st.success("💡 Hãy phân loại rác đúng cách vì một môi trường xanh hơn!")
