"""Dark theme styling configuration and UI animations for Streamlit pages."""

from __future__ import annotations

import streamlit as st


def is_dark_theme() -> bool:
    """Return True permanently as the application is dark-theme only."""
    return True


def apply_theme() -> None:
    """Apply a unified, clinical-grade dark palette with smooth UI animations."""
    bg = "#090d16"
    surface = "#111827"
    surface_hover = "#1e293b"
    text = "#f8fafc"
    muted = "#94a3b8"
    border = "#1e293b"
    border_accent = "#334155"
    input_bg = "#0f172a"
    primary = "#14b8a6"
    primary_light = "#134e4a"
    card_shadow = "0 4px 6px -1px rgba(0, 0, 0, 0.4), 0 2px 4px -2px rgba(0, 0, 0, 0.3)"

    st.markdown(
        f"""<style>
        :root {{
            color-scheme: dark;
            --background-color: {bg};
            --secondary-background-color: {surface};
            --text-color: {text};
            --primary-color: {primary};
        }}

        /* Keyframe Animations */
        @keyframes fadeIn {{
            from {{
                opacity: 0;
                transform: translateY(6px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        @keyframes pulseGlow {{
            0%, 100% {{
                opacity: 1;
                box-shadow: 0 0 6px rgba(20, 184, 166, 0.6);
            }}
            50% {{
                opacity: 0.55;
                box-shadow: 0 0 14px rgba(20, 184, 166, 0.2);
            }}
        }}

        @keyframes shimmer {{
            0% {{
                background-position: -200% 0;
            }}
            100% {{
                background-position: 200% 0;
            }}
        }}

        /* Application Canvas */
        .stApp, [data-testid="stAppViewContainer"] {{
            background-color: {bg} !important;
            color: {text} !important;
        }}

        [data-testid="stHeader"] {{
            background-color: {bg} !important;
        }}

        [data-testid="stSidebar"] {{
            background-color: {surface} !important;
            border-right: 1px solid {border} !important;
        }}

        [data-testid="stMain"] {{
            color: {text};
        }}

        /* Headings & Text */
        h1, h2, h3, h4, h5, h6 {{
            color: {text} !important;
            font-weight: 600;
        }}

        p, span, label, li {{
            color: {text};
        }}

        [data-testid="stMarkdownContainer"] p {{
            color: {text};
            line-height: 1.6;
        }}

        [data-testid="stCaptionContainer"], small, .stCaption {{
            color: {muted} !important;
        }}

        [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {{
            color: {text} !important;
            font-weight: 500 !important;
        }}

        code {{
            background-color: {input_bg} !important;
            color: {primary} !important;
            border: 1px solid {border} !important;
            border-radius: 4px !important;
            padding: 2px 6px !important;
        }}

        /* Animated Pulsing Status Dot */
        .status-dot {{
            display: inline-block;
            width: 8px;
            height: 8px;
            background-color: {primary};
            border-radius: 50%;
            margin-right: 6px;
            animation: pulseGlow 2.2s infinite ease-in-out;
            vertical-align: middle;
        }}

        /* Sidebar Navigation & Links */
        [data-testid="stSidebarNav"] a {{
            color: {text} !important;
            border-radius: 6px;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }}
        [data-testid="stSidebarNav"] a:hover {{
            background-color: {surface_hover} !important;
            transform: translateX(2px);
        }}
        [data-testid="stSidebarNav"] a[aria-current="page"] {{
            background-color: {primary_light} !important;
            color: {primary} !important;
            font-weight: 600 !important;
        }}

        /* Containers, Cards, and Metrics with Hover & Entrance Animations */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: {surface} !important;
            border: 1px solid {border} !important;
            border-radius: 10px !important;
            box-shadow: {card_shadow} !important;
            animation: fadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
            transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1),
                        border-color 0.25s ease,
                        box-shadow 0.25s ease !important;
        }}

        [data-testid="stVerticalBlockBorderWrapper"]:hover {{
            border-color: rgba(20, 184, 166, 0.45) !important;
            box-shadow: 0 10px 24px -4px rgba(0, 0, 0, 0.5), 0 0 16px rgba(20, 184, 166, 0.12) !important;
            transform: translateY(-2px);
        }}

        [data-testid="stVerticalBlockBorderWrapper"] > div {{
            background-color: {surface} !important;
        }}

        [data-testid="stMetric"] {{
            background-color: {surface} !important;
            border: 1px solid {border} !important;
            border-radius: 8px !important;
            padding: 14px 16px !important;
            box-shadow: {card_shadow} !important;
            animation: fadeIn 0.35s ease forwards;
            transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease !important;
        }}

        [data-testid="stMetric"]:hover {{
            border-color: rgba(20, 184, 166, 0.35) !important;
            transform: translateY(-1px);
        }}

        [data-testid="stMetricLabel"] > div {{
            color: {muted} !important;
            font-size: 0.85rem !important;
            font-weight: 500 !important;
        }}

        [data-testid="stMetricValue"] > div {{
            color: {text} !important;
            font-weight: 700 !important;
        }}

        /* Expanders */
        [data-testid="stExpander"] {{
            background-color: {surface} !important;
            border: 1px solid {border} !important;
            border-radius: 8px !important;
            box-shadow: {card_shadow} !important;
            transition: border-color 0.2s ease !important;
        }}

        [data-testid="stExpander"]:hover {{
            border-color: rgba(20, 184, 166, 0.3) !important;
        }}

        [data-testid="stExpander"] details {{
            background-color: {surface} !important;
        }}

        [data-testid="stExpander"] summary {{
            color: {text} !important;
            font-weight: 500 !important;
            transition: color 0.15s ease !important;
        }}

        [data-testid="stExpander"] summary:hover {{
            color: {primary} !important;
        }}

        /* Form Inputs */
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {{
            background-color: {input_bg} !important;
            color: {text} !important;
            border: 1px solid {border_accent} !important;
            border-radius: 6px !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        }}

        [data-testid="stTextInput"] input:focus,
        [data-testid="stNumberInput"] input:focus,
        [data-testid="stTextArea"] textarea:focus,
        [data-testid="stSelectbox"] [data-baseweb="select"] > div:focus {{
            border-color: {primary} !important;
            box-shadow: 0 0 0 1px {primary} !important;
        }}

        /* Selectbox Popovers & Options */
        [data-baseweb="popover"],
        [data-baseweb="menu"],
        ul[role="listbox"] {{
            background-color: {surface} !important;
            color: {text} !important;
            border: 1px solid {border_accent} !important;
            border-radius: 6px !important;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5) !important;
            animation: fadeIn 0.18s ease-out forwards;
        }}

        li[role="option"] {{
            background-color: {surface} !important;
            color: {text} !important;
            transition: background-color 0.15s ease !important;
        }}

        li[role="option"]:hover,
        li[role="option"][aria-selected="true"] {{
            background-color: {surface_hover} !important;
            color: {primary} !important;
        }}

        div[data-baseweb="select"] span {{
            color: {text} !important;
        }}

        div[data-baseweb="select"] svg {{
            fill: {muted} !important;
        }}

        /* Buttons & Page Links with Micro-Interactions */
        [data-testid="stPageLink-NavLink"] {{
            background-color: {surface} !important;
            border: 1px solid {border} !important;
            border-radius: 8px !important;
            color: {text} !important;
            box-shadow: {card_shadow};
            padding: 12px 16px !important;
            text-decoration: none !important;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }}

        [data-testid="stPageLink-NavLink"]:hover {{
            border-color: {primary} !important;
            background-color: {surface_hover} !important;
            box-shadow: 0 6px 20px -2px rgba(20, 184, 166, 0.2) !important;
            transform: translateY(-2px);
        }}

        [data-testid="stPageLink-NavLink"] p,
        [data-testid="stPageLink-NavLink"] span {{
            color: {text} !important;
            font-weight: 500 !important;
        }}

        .stButton > button,
        .stDownloadButton > button {{
            background-color: {surface} !important;
            color: {text} !important;
            border: 1px solid {border_accent} !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            box-shadow: {card_shadow};
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }}

        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            border-color: {primary} !important;
            color: {primary} !important;
            background-color: {surface_hover} !important;
            box-shadow: 0 6px 16px -2px rgba(20, 184, 166, 0.2) !important;
            transform: translateY(-1px);
        }}

        /* File Uploader */
        [data-testid="stFileUploader"] section {{
            background-color: {surface} !important;
            border: 2px dashed {border_accent} !important;
            border-radius: 10px !important;
            color: {text} !important;
            transition: border-color 0.2s ease, background-color 0.2s ease !important;
        }}

        [data-testid="stFileUploader"] section:hover {{
            border-color: {primary} !important;
            background-color: rgba(20, 184, 166, 0.03) !important;
        }}

        [data-testid="stFileUploader"] section button {{
            background-color: {surface_hover} !important;
            color: {text} !important;
            border: 1px solid {border_accent} !important;
            border-radius: 6px !important;
            transition: all 0.2s ease !important;
        }}

        [data-testid="stFileUploader"] section button:hover {{
            border-color: {primary} !important;
            color: {primary} !important;
        }}

        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] span {{
            color: {muted} !important;
        }}

        /* Status & Spinner Widget */
        [data-testid="stStatusWidget"] {{
            background-color: {surface} !important;
            border: 1px solid {border_accent} !important;
            border-radius: 8px !important;
            box-shadow: {card_shadow} !important;
        }}

        /* Dataframe Wrapper */
        [data-testid="stDataFrame"] {{
            background-color: {surface} !important;
            border: 1px solid {border} !important;
            border-radius: 8px !important;
            animation: fadeIn 0.3s ease forwards;
        }}

        /* Divider */
        hr {{
            border-color: {border} !important;
            opacity: 0.8 !important;
            margin: 1.5rem 0 !important;
        }}

        /* Progress Bar Animation */
        [data-testid="stProgressBar"] > div {{
            background-color: {border} !important;
            border-radius: 6px;
            overflow: hidden;
        }}
        [data-testid="stProgressBar"] > div > div {{
            background: linear-gradient(90deg, #0d9488 0%, #14b8a6 50%, #2dd4bf 100%) !important;
            transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }}
        </style>""",
        unsafe_allow_html=True,
    )
