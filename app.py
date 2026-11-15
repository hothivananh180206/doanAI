from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from database.db import init_db
from pages import admin, catalog, history, home, instructions, recognition, statistics


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --eco-green: #079455;
            --eco-green-dark: #087443;
            --eco-mint: #e7f7ee;
            --eco-blue: #2f8fe5;
            --eco-blue-soft: #e9f4ff;
            --eco-orange: #f2a323;
            --eco-orange-soft: #fff4df;
            --eco-red: #e85252;
            --eco-red-soft: #fff0ef;
            --eco-ink: #172b3b;
            --eco-muted: #607387;
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(ellipse at 92% 0%, #e8f8ef 0, transparent 30rem),
                #f5f8fa;
            color: var(--eco-ink);
        }
        [data-testid="stHeader"] { background: transparent; }
        .block-container {
            max-width: 1440px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f0fbf5 0%, #ffffff 55%, #f5f9fd 100%);
            border-right: 1px solid #dce8ee;
        }
        .eco-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 6px 16px;
        }
        .eco-brand-mark {
            width: 48px;
            height: 48px;
            flex: 0 0 48px;
            filter: drop-shadow(0 5px 8px rgba(8, 116, 67, .18));
        }
        .eco-brand-name {
            color: #123d2e;
            font-size: 1.35rem;
            font-weight: 800;
            letter-spacing: -.04em;
            line-height: 1.1;
        }
        .eco-brand-name span { color: #079455; }
        .eco-brand-badge {
            display: inline-block;
            margin-left: 4px;
            padding: 2px 6px;
            border-radius: 6px;
            background: #d9f6e6;
            color: #087443;
            font-size: .62rem;
            letter-spacing: .04em;
            vertical-align: 3px;
        }
        .eco-brand-tagline {
            margin-top: 5px;
            color: #607387;
            font-size: .66rem;
            font-weight: 650;
            letter-spacing: .08em;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: var(--eco-muted);
        }
        [data-testid="stSidebarNav"] {
            padding-top: .5rem;
        }
        [data-testid="stSidebarNav"] a {
            border-radius: 10px;
            margin: 3px 4px;
        }
        [data-testid="stSidebarNav"] a[aria-current="page"] {
            background: #d9f6e6;
            color: #087443;
            font-weight: 700;
        }
        [data-testid="stSidebar"] a {
            border-radius: 10px;
            color: #365448;
            text-decoration: none;
        }
        [data-testid="stSidebar"] a:hover {
            background: #e8f7ee;
            color: #087443;
        }
        [data-testid="stSidebar"] a[aria-current="page"] {
            background: #d9f6e6;
            color: #087443;
            font-weight: 700;
        }
        [data-testid="stSidebar"] [data-testid="stPageLink"] a {
            display: flex;
            align-items: center;
            min-height: 2.45rem;
            padding: .45rem .7rem;
            border-radius: 11px;
            color: #365448;
            font-weight: 600;
        }
        [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {
            background: #e4f7eb;
            color: #087443;
        }
        h1, h2, h3 { color: var(--eco-ink); letter-spacing: -.025em; }
        h1 { font-size: clamp(1.7rem, 3vw, 2.25rem) !important; }
        [data-testid="stCaptionContainer"] { color: var(--eco-muted); }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #dfeaf0;
            border-radius: 17px;
            box-shadow: 0 5px 18px rgba(27, 55, 74, .055);
            background: rgba(255,255,255,.96);
        }
        .eco-hero {
            padding: clamp(1.5rem, 4vw, 3rem);
            border: 1px solid #d5efdf;
            border-radius: 20px;
            background:
                radial-gradient(circle at 88% 18%, rgba(255,255,255,.9) 0 2.5rem, transparent 2.6rem),
                linear-gradient(115deg, #e2f8eb 0%, #e8f7ff 62%, #f5fcf3 100%);
        }
        .eco-hero h2 { margin: 0 0 .45rem; color: #124333; }
        .eco-hero p { margin: 0; color: #426454; max-width: 46rem; }
        .eco-hero-dashboard {
            display: grid;
            grid-template-columns: 1.05fr .95fr;
            gap: 1rem;
            align-items: center;
            min-height: 270px;
            overflow: hidden;
            padding: clamp(1.5rem, 3vw, 2.6rem);
            border: 1px solid #d5ebf5;
            border-radius: 22px;
            background:
                radial-gradient(circle at 85% 20%, rgba(255,255,255,.85) 0 2.2rem, transparent 2.3rem),
                linear-gradient(120deg, #e8f6ff 0%, #eefbf4 62%, #f6fcf5 100%);
        }
        .eco-hero-kicker {
            color: #087443;
            font-size: .82rem;
            font-weight: 800;
            letter-spacing: .08em;
        }
        .eco-hero-dashboard h2 {
            max-width: 34rem;
            margin: .55rem 0;
            color: #153c50;
            font-size: clamp(1.8rem, 3vw, 2.55rem);
            line-height: 1.12;
        }
        .eco-hero-dashboard p {
            max-width: 38rem;
            margin: 0;
            color: #4c6877;
            line-height: 1.65;
        }
        .eco-hero-art {
            position: relative;
            display: flex;
            align-items: end;
            justify-content: center;
            min-height: 220px;
            padding: 1.25rem 1rem .6rem;
            border-radius: 20px;
            background: linear-gradient(180deg, #c8ecff 0%, #e8f8ff 60%, #cdeec9 61%, #9fdb9a 100%);
        }
        .eco-hero-sun {
            position: absolute;
            top: 16px;
            right: 24px;
            width: 46px;
            height: 46px;
            border-radius: 50%;
            background: #ffd76b;
            box-shadow: 0 0 0 10px rgba(255,215,107,.2);
        }
        .eco-hero-globe {
            position: absolute;
            top: 14px;
            left: 20px;
            display: grid;
            place-items: center;
            width: 62px;
            height: 62px;
            border: 4px solid rgba(255,255,255,.8);
            border-radius: 50%;
            background: #fff;
            font-size: 2.1rem;
            box-shadow: 0 5px 15px rgba(29,93,132,.15);
        }
        .eco-hero-bins {
            display: flex;
            align-items: end;
            gap: clamp(.35rem, 1vw, .85rem);
            width: 100%;
            max-width: 420px;
            justify-content: center;
        }
        .eco-bin {
            display: flex;
            flex: 1;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 118px;
            max-width: 86px;
            padding: .5rem .3rem;
            border-radius: 12px 12px 8px 8px;
            color: white;
            font-size: .72rem;
            font-weight: 750;
            text-align: center;
            box-shadow: 0 8px 12px rgba(33,74,67,.18);
        }
        .eco-bin::before {
            width: 112%;
            height: 9px;
            margin-top: -1.05rem;
            margin-bottom: 1rem;
            border-radius: 7px;
            background: rgba(255,255,255,.8);
            content: "";
        }
        .eco-bin span { font-size: 2rem; line-height: 1.1; }
        .eco-bin-green { background: linear-gradient(145deg, #27b96d, #0b8d4e); }
        .eco-bin-blue { background: linear-gradient(145deg, #4ba9f1, #2476c5); }
        .eco-bin-orange { background: linear-gradient(145deg, #ffc04a, #ed8a12); }
        .eco-bin-red { background: linear-gradient(145deg, #ff7774, #de4549); }
        .eco-step-card {
            height: 100%;
            padding: 1rem;
            border: 1px solid #e1edf2;
            border-radius: 15px;
            background: #fff;
        }
        .eco-step-card strong {
            display: block;
            margin-bottom: .35rem;
            color: #173a4b;
        }
        .eco-step-card span { color: #607387; font-size: .9rem; }
        .eco-category-card {
            margin: .8rem 0;
            padding: 1rem 1.1rem;
            border: 1px solid transparent;
            border-radius: 17px;
        }
        .eco-category-card h3 { margin: 0 0 .25rem; font-size: 1.13rem; }
        .eco-category-card p { margin: .15rem 0; color: #355062; }
        .eco-category-symbol {
            float: right;
            margin-left: .8rem;
            font-size: 2rem;
        }
        .eco-category-item {
            margin-top: .55rem;
            padding: .75rem .9rem;
            border: 1px solid rgba(255,255,255,.8);
            border-radius: 12px;
            background: rgba(255,255,255,.72);
        }
        .eco-category-item strong { color: #173a4b; }
        .eco-category-meta { color: #607387; font-size: .82rem; }
        .eco-category-organic { border-color: #ccebd5; background: #eaf8ee; }
        .eco-category-recycle { border-color: #cfe5fb; background: #eaf4ff; }
        .eco-category-residual { border-color: #f8e0b7; background: #fff5e4; }
        .eco-category-hazardous { border-color: #f4cccc; background: #fff0ef; }
        .eco-guide-card {
            margin: .8rem 0;
            padding: 1.1rem 1.2rem;
            border: 1px solid transparent;
            border-radius: 17px;
        }
        .eco-guide-card h3 { margin: 0 0 .7rem; font-size: 1.12rem; }
        .eco-guide-card ul { margin: .25rem 0 0; padding-left: 1.25rem; }
        .eco-guide-card li { margin: .35rem 0; color: #355062; }
        .eco-guide-organic { border-color: #ccebd5; background: #eaf8ee; }
        .eco-guide-recycle { border-color: #cfe5fb; background: #eaf4ff; }
        .eco-guide-residual { border-color: #f8e0b7; background: #fff5e4; }
        .eco-guide-hazardous { border-color: #f4cccc; background: #fff0ef; }
        .eco-pill {
            display: inline-block;
            margin: .15rem .25rem .15rem 0;
            padding: .25rem .6rem;
            border-radius: 999px;
            background: #eef5f8;
            color: #355062;
            font-size: .78rem;
            font-weight: 700;
        }
        .st-key-eco-history-header {
            padding: .7rem .8rem;
            border: 1px solid #dce8f0;
            border-radius: 12px;
            background: #edf5fb;
            color: #25445a;
        }
        .st-key-eco-history-header [data-testid="stMarkdownContainer"] p {
            margin: 0;
        }
        .st-key-eco-stat-blue [data-testid="stMetric"] {
            border-color: #cfe5fb;
            background: linear-gradient(135deg, #e8f4ff, #f3f9ff);
        }
        .st-key-eco-stat-green [data-testid="stMetric"] {
            border-color: #ccebd5;
            background: linear-gradient(135deg, #e9f8ee, #f5fcf6);
        }
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #eef8ff, #f4fbff);
            border: 1px solid #d9eaf4;
            border-radius: 15px;
            padding: 14px 16px;
        }
        [data-testid="stMetricLabel"] { color: #4c6275; }
        [data-testid="stMetricValue"] { color: #087443; }
        [data-testid="stBaseButton-primary"],
        [data-testid="stBaseButton-secondary"],
        [data-testid="stFormSubmitButton"] button {
            border-radius: 10px;
            font-weight: 650;
            transition: transform .15s ease, box-shadow .15s ease;
        }
        [data-testid="stBaseButton-primary"],
        [data-testid="stFormSubmitButton"] button[kind="primary"] {
            background: linear-gradient(100deg, #079455, #0da66a);
            border-color: #079455;
            color: white;
        }
        [data-testid="stBaseButton-secondary"] {
            border-color: #b9dfc8;
            background: #f0faf4;
            color: #087443;
        }
        [data-testid="stBaseButton-primary"]:hover,
        [data-testid="stBaseButton-secondary"]:hover,
        [data-testid="stFormSubmitButton"] button:hover {
            transform: translateY(-1px);
            box-shadow: 0 5px 14px rgba(8, 116, 67, .16);
        }
        [data-testid="stFileUploaderDropzone"] {
            border: 2px dashed #9bd8b2;
            border-radius: 16px;
            background: #f5fff8;
        }
        [data-testid="stSlider"] [role="slider"] {
            background: #0b9a58;
        }
        .st-key-eco-logout [data-testid="stBaseButton-secondary"],
        [class*="st-key-eco-danger"] [data-testid="stBaseButton-secondary"] {
            border-color: #f3c6c4;
            background: #fff0ef;
            color: #c83f43;
        }
        [class*="st-key-eco-blue"] [data-testid="stBaseButton-secondary"] {
            border-color: #c8dff7;
            background: #edf6ff;
            color: #1769ae;
        }
        [class*="st-key-eco-orange"] [data-testid="stBaseButton-secondary"] {
            border-color: #f5d9a8;
            background: #fff6e8;
            color: #a9680a;
        }
        [class*="st-key-eco-green"] [data-testid="stBaseButton-secondary"] {
            border-color: #b9dfc8;
            background: #e9f8ef;
            color: #087443;
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            color: var(--eco-green-dark);
            border-bottom-color: var(--eco-green);
        }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border: 1px solid #e1e9ed;
            border-radius: 12px;
            overflow: hidden;
        }
        @media (max-width: 760px) {
            .block-container { padding: 1rem 1rem 2rem; }
            .eco-hero-dashboard { grid-template-columns: 1fr; }
            .eco-hero-art { min-height: 190px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="EcoSort AI · Phân loại rác",
        page_icon="♻️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.set_option("client.toolbarMode", "minimal")
    init_db()
    _apply_theme()

    # Thiết lập người dùng mặc định (không cần đăng nhập, truy cập toàn bộ chức năng)
    if "user" not in st.session_state or st.session_state["user"] is None:
        st.session_state["user"] = {
            "id": 1,
            "username": "Quản trị viên",
            "role": "admin",
        }

    home_page = st.Page(
        home.render,
        title="Trang chủ",
        icon="🏠",
        url_path="home",
        default=True,
    )
    user_pages = [
        home_page,
        st.Page(
            recognition.render,
            title="Nhận diện rác",
            icon="🔍",
            url_path="recognition",
        ),
        st.Page(
            catalog.render,
            title="Danh mục rác",
            icon="♻️",
            url_path="catalog",
        ),
        st.Page(
            instructions.render,
            title="Hướng dẫn phân loại",
            icon="🌱",
            url_path="instructions",
        ),
        st.Page(
            history.render,
            title="Lịch sử nhận diện",
            icon="🕘",
            url_path="history",
        ),
        st.Page(
            statistics.render,
            title="Thống kê",
            icon="📊",
            url_path="statistics",
        ),
        st.Page(
            admin.render,
            title="Cài đặt mô hình AI",
            icon="⚙️",
            url_path="settings",
        ),
    ]
    page = st.navigation(user_pages)

    with st.sidebar:
        st.markdown(
            """
            <div class="eco-brand">
              <svg class="eco-brand-mark" viewBox="0 0 56 56" role="img"
                   aria-label="Logo EcoSort">
                <defs>
                  <linearGradient id="eco-mark-gradient" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0" stop-color="#12a96c"/>
                    <stop offset="1" stop-color="#087443"/>
                  </linearGradient>
                </defs>
                <rect width="56" height="56" rx="18" fill="url(#eco-mark-gradient)"/>
                <path d="M40 11C27 12 16 19 16 31c0 7 4 11 10 11 12 0 19-13 14-31Z"
                      fill="white"/>
                <path d="M17 43c7-8 13-13 22-19" fill="none" stroke="#15935b"
                      stroke-linecap="round" stroke-width="2.5"/>
                <path d="M12 32a17 17 0 0 1 4-11M11 33l1-5m-1 5 5-1"
                      fill="none" stroke="#d7f8e3" stroke-linecap="round"
                      stroke-linejoin="round" stroke-width="2"/>
              </svg>
              <div>
                <div class="eco-brand-name">Eco<span>Sort</span><b class="eco-brand-badge">AI</b></div>
                <div class="eco-brand-tagline">PHÂN LOẠI RÁC THÔNG MINH</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        for nav_page in user_pages:
            st.page_link(
                nav_page,
                label=nav_page.title,
                icon=nav_page.icon,
                use_container_width=True,
            )
        st.divider()
        st.caption("🌿 EcoSort AI · Đồ án Trí tuệ nhân tạo")

    page.run()


if __name__ == "__main__":
    main()
