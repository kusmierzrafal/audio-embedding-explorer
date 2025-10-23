from typing import Dict

from src.models.enums.view_names import ViewName

PAGE_TITLE = "Audio Embedding Explorer"

NAVBAR_ICONS: Dict[ViewName, str] = {
    ViewName.HOME: "house",
    ViewName.PAIR_ANALYSIS: "soundwave",
    ViewName.MODEL_COMPARISON: "cpu",
    ViewName.SIMILARITY_RANKING: "list-ol",
    ViewName.PSEUDO_CAPTIONING: "chat-text",
    ViewName.LOCAL_DB: "database",
}

COLOR_BG = "#161B22"
COLOR_TEXT = "#F1F5F9"
COLOR_HOVER = "#1E2530"
COLOR_SELECTED = "#2563EB"
COLOR_WHITE = "#FFFFFF"

BORDER_RADIUS = "8px"
FONT_SIZE_ICON = "20px"
FONT_SIZE_TEXT = "14px"
PADDING_LINK = "6px 8px"
MARGIN_LINK = "4px 0"

NAVBAR_STYLES = {
    "container": {
        "background-color": COLOR_BG,
        "padding": "0 !important",
        "border-radius": BORDER_RADIUS,
    },
    "icon": {
        "color": COLOR_TEXT,
        "font-size": FONT_SIZE_ICON,
    },
    "nav-link": {
        "font-size": FONT_SIZE_TEXT,
        "color": COLOR_TEXT,
        "text-align": "left",
        "margin": MARGIN_LINK,
        "padding": PADDING_LINK,
        "--hover-color": COLOR_HOVER,
    },
    "nav-link-selected": {
        "background-color": COLOR_SELECTED,
        "color": COLOR_WHITE,
        "font-weight": "bold",
        "border-radius": "6px",
    },
}
