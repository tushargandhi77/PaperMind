import html
import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.btw_handler import handle_btw
from backend.paper_loader import load_arxiv, load_document, load_webpage
from backend.rag_graph import build_graph
from backend.vector_store import add_paper, list_papers

st.set_page_config(
    page_title="PaperMind",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# UI THEME (presentation only — works in both light and dark mode)
# Colors are derived from the active Streamlit theme (inherit / translucent
# neutrals), so switching Light/Dark from Settings restyles everything.
# ══════════════════════════════════════════════════════════════════════════════
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

:root {
    --pm-accent: #7c5cff;
    --pm-accent-2: #14b8a6;
    --pm-grad: linear-gradient(135deg, #7c5cff 0%, #4f8cff 55%, #14b8a6 100%);
    --pm-border: rgba(128, 128, 128, 0.22);
    --pm-surface: rgba(128, 128, 128, 0.07);
    --pm-surface-hover: rgba(124, 92, 255, 0.10);
    --pm-shadow: 0 6px 24px rgba(20, 20, 40, 0.10);
    --pm-radius: 14px;
}

html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea {
    font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif !important;
}

/* Soft ambient glow behind the app */
.stApp {
    background-image:
        radial-gradient(ellipse 60% 40% at 15% -5%, rgba(124, 92, 255, 0.14), transparent 60%),
        radial-gradient(ellipse 50% 35% at 100% 0%, rgba(20, 184, 166, 0.10), transparent 60%);
    background-attachment: fixed;
}

footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; padding-bottom: 6rem; max-width: 820px; }

/* ── Hero ───────────────────────────────────────────────────────────────── */
.pm-hero { text-align: center; padding: 0.6rem 0 0.4rem 0; animation: pmRise .7s ease both; }
.pm-hero h1 {
    font-family: 'Fraunces', Georgia, serif !important;
    font-weight: 700; font-size: 3rem; letter-spacing: -0.02em;
    margin: 0; padding: 0; line-height: 1.1;
    background: var(--pm-grad); -webkit-background-clip: text;
    background-clip: text; -webkit-text-fill-color: transparent;
}
.pm-hero p { opacity: .7; font-size: 1.05rem; margin: .5rem 0 0 0; }
.pm-pills { display: flex; gap: .5rem; justify-content: center; flex-wrap: wrap; margin-top: 1.1rem; }
.pm-pill {
    padding: .35rem .85rem; border-radius: 999px; font-size: .82rem; font-weight: 500;
    border: 1px solid var(--pm-border); background: var(--pm-surface);
    transition: transform .18s ease, border-color .18s ease, background .18s ease;
}
.pm-pill:hover { transform: translateY(-2px); border-color: var(--pm-accent); background: var(--pm-surface-hover); }

@keyframes pmRise { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }

/* ── Empty state cards ──────────────────────────────────────────────────── */
.pm-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .9rem; margin: 1.4rem 0 .5rem; }
.pm-card {
    padding: 1.1rem 1rem; border-radius: var(--pm-radius);
    border: 1px solid var(--pm-border); background: var(--pm-surface);
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}
.pm-card:hover { transform: translateY(-4px); box-shadow: var(--pm-shadow); border-color: var(--pm-accent); }
.pm-card .ic { font-size: 1.6rem; }
.pm-card h4 { margin: .5rem 0 .25rem; font-size: 1rem; font-weight: 600; }
.pm-card p { margin: 0; font-size: .85rem; opacity: .7; line-height: 1.45; }
.pm-hint { text-align: center; opacity: .6; font-size: .9rem; margin-top: 1rem; }
@media (max-width: 640px) { .pm-grid { grid-template-columns: 1fr; } .pm-hero h1 { font-size: 2.3rem; } }

/* ── Sidebar ────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] { border-right: 1px solid var(--pm-border); }
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
.pm-brand { display: flex; align-items: center; gap: .6rem; margin: .2rem 0 .8rem; }
.pm-brand .logo {
    width: 38px; height: 38px; border-radius: 11px; display: grid; place-items: center;
    background: var(--pm-grad); font-size: 1.2rem; box-shadow: 0 4px 14px rgba(124, 92, 255, .4);
}
.pm-brand .name { font-family: 'Fraunces', serif; font-weight: 700; font-size: 1.35rem; letter-spacing: -0.01em; }
.pm-brand .tag { font-size: .72rem; opacity: .6; margin-top: -2px; }
.pm-section { font-weight: 600; font-size: .95rem; margin: .4rem 0 .3rem; }
.pm-badge {
    display: inline-block; margin-left: .4rem; padding: .05rem .5rem; border-radius: 999px;
    background: var(--pm-grad); color: #fff; font-size: .72rem; font-weight: 600; vertical-align: middle;
}
.pm-doc {
    display: flex; align-items: center; gap: .5rem; padding: .45rem .65rem; margin: .3rem 0;
    border: 1px solid var(--pm-border); background: var(--pm-surface); border-radius: 10px;
    font-size: .84rem; transition: transform .15s ease, border-color .15s ease;
}
.pm-doc:hover { transform: translateX(3px); border-color: var(--pm-accent); }
.pm-doc span.t { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 11px; border: 1px solid var(--pm-border); background: var(--pm-surface);
    font-weight: 500; transition: all .18s ease; text-align: left;
}
.stButton > button:hover {
    transform: translateY(-1px); border-color: var(--pm-accent);
    background: var(--pm-surface-hover); box-shadow: 0 4px 14px rgba(124, 92, 255, .18);
}
.stButton > button:active { transform: translateY(0); }
.stButton > button:focus-visible { outline: 2px solid var(--pm-accent); outline-offset: 2px; }
.stButton > button[kind="primary"] {
    background: var(--pm-grad); color: #fff; border: none; font-weight: 600;
    box-shadow: 0 4px 16px rgba(124, 92, 255, .35);
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.08); }

/* ── Inputs ─────────────────────────────────────────────────────────────── */
.stTextInput input, .stTextArea textarea {
    border-radius: 11px !important; border: 1px solid var(--pm-border) !important;
    transition: border-color .18s ease, box-shadow .18s ease;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--pm-accent) !important; box-shadow: 0 0 0 3px rgba(124, 92, 255, .22) !important;
}
[data-testid="stFileUploaderDropzone"] {
    border-radius: var(--pm-radius); border: 1.5px dashed var(--pm-border); background: var(--pm-surface);
    transition: border-color .18s ease, background .18s ease;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--pm-accent); background: var(--pm-surface-hover); }

/* ── Chat ───────────────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border: 1px solid var(--pm-border); background: var(--pm-surface);
    border-radius: var(--pm-radius); padding: 1rem 1.1rem; margin-bottom: .8rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: rgba(124, 92, 255, 0.09); border-color: rgba(124, 92, 255, 0.28);
}
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] {
    border-radius: 12px; background: var(--pm-grad) !important; color: #fff;
}
[data-testid="stChatInput"] { border-radius: 16px; border: 1px solid var(--pm-border); box-shadow: var(--pm-shadow); }
[data-testid="stChatInput"]:focus-within { border-color: var(--pm-accent); box-shadow: 0 0 0 3px rgba(124, 92, 255, .22); }

/* ── Expanders, alerts, misc ────────────────────────────────────────────── */
[data-testid="stExpander"] { border: 1px solid var(--pm-border); border-radius: 12px; background: transparent; }
[data-testid="stExpander"] summary { font-weight: 500; font-size: .88rem; }
[data-testid="stAlert"] { border-radius: 12px; }
hr { border-color: var(--pm-border) !important; }
blockquote { border-left: 3px solid var(--pm-accent) !important; border-radius: 4px; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: rgba(128, 128, 128, .35); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: var(--pm-accent); }

@media (prefers-reduced-motion: reduce) {
    * { animation: none !important; transition: none !important; }
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

USER_AVATAR = "🧑‍🎓"
BOT_AVATAR = "🧠"


@st.cache_resource
def get_graph():
    return build_graph()


SESSIONS_FILE = Path("sessions.json")
_rename_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


def load_sessions() -> dict:
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_sessions(sessions_meta: dict) -> None:
    SESSIONS_FILE.write_text(json.dumps(sessions_meta, indent=2), encoding="utf-8")


def _serialize_state(values: dict) -> dict:
    out = {}
    for k, v in values.items():
        if k == "messages":
            out[k] = [
                {
                    "type": type(m).__name__,
                    "content": (
                        m.content[:300]
                        if isinstance(m.content, str)
                        else repr(m.content)[:300]
                    ),
                }
                for m in (v or [])
            ]
        elif k == "retrieved_docs":
            out[k] = [
                {"content": d.page_content[:300], "metadata": d.metadata}
                for d in (v or [])
            ]
        else:
            out[k] = v
    return out


def generate_session_name(first_message: str) -> str:
    try:
        response = _rename_llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Generate a concise 3-5 word title for a research chat session "
                        "based on the user's first message. Return only the title, "
                        "no punctuation at the end, no quotes."
                    ),
                },
                {"role": "user", "content": first_message[:500]},
            ]
        )
        return response.content.strip()
    except Exception:
        return "New Session"


def maybe_rename_session(session_id: str, first_message: str) -> None:
    if st.session_state.sessions_meta.get(session_id, {}).get("is_named"):
        return
    name = generate_session_name(first_message)
    st.session_state.sessions_meta[session_id]["name"] = name
    st.session_state.sessions_meta[session_id]["is_named"] = True
    save_sessions(st.session_state.sessions_meta)


def create_session() -> str:
    sid = str(uuid.uuid4())
    st.session_state.sessions_meta[sid] = {
        "id": sid,
        "name": "New Session",
        "created_at": datetime.now().isoformat(),
        "is_named": False,
    }
    save_sessions(st.session_state.sessions_meta)
    st.session_state.chats[sid] = []
    st.session_state.turns[sid] = 0
    return sid


def load_session_chats(session_id: str) -> list[dict]:
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        if not state or not state.values:
            return []
        chats = []
        turn = 0
        for msg in state.values.get("messages", []):
            type_name = type(msg).__name__
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if type_name == "HumanMessage":
                chats.append({"role": "user", "content": content})
            elif type_name in ("AIMessage", "AIMessageChunk"):
                turn += 1
                chats.append({"role": "assistant", "content": content, "turn": turn, "graph_state": {}})
        return chats
    except Exception:
        return []


def switch_session(session_id: str) -> None:
    st.session_state.active_session_id = session_id
    if session_id not in st.session_state.chats:
        st.session_state.chats[session_id] = load_session_chats(session_id)
    if session_id not in st.session_state.turns:
        turn_count = sum(1 for m in st.session_state.chats[session_id] if m["role"] == "assistant")
        st.session_state.turns[session_id] = turn_count


graph = get_graph()

# ── Bootstrap ──────────────────────────────────────────────────────────────────
if "sessions_meta" not in st.session_state:
    st.session_state.sessions_meta = load_sessions()
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "turns" not in st.session_state:
    st.session_state.turns = {}
if "active_session_id" not in st.session_state:
    if st.session_state.sessions_meta:
        latest = max(
            st.session_state.sessions_meta.values(),
            key=lambda s: s["created_at"],
        )
        switch_session(latest["id"])
    else:
        sid = create_session()
        st.session_state.active_session_id = sid

active_sid = st.session_state.active_session_id

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="pm-brand">
            <div class="logo">🧠</div>
            <div>
                <div class="name">PaperMind</div>
                <div class="tag">Your research copilot</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("＋  New Chat", use_container_width=True, type="primary"):
        new_sid = create_session()
        st.session_state.active_session_id = new_sid
        active_sid = new_sid
        st.rerun()
    st.divider()
    st.markdown('<div class="pm-section">💬 Sessions</div>', unsafe_allow_html=True)

    sorted_sessions = sorted(
        st.session_state.sessions_meta.values(),
        key=lambda s: s["created_at"],
        reverse=True,
    )
    for session in sorted_sessions:
        sid = session["id"]
        is_active = sid == st.session_state.active_session_id
        btn_type = "primary" if is_active else "secondary"
        if st.button(
            session["name"],
            key=f"sess_{sid}",
            use_container_width=True,
            type=btn_type,
        ):
            if not is_active:
                switch_session(sid)
                st.rerun()

    st.divider()
    st.markdown('<div class="pm-section">📄 Documents</div>', unsafe_allow_html=True)

    # ── Section 1: File upload ─────────────────────────────────────────────────
    st.markdown("**Upload Files**")
    uploaded_files = st.file_uploader(
        "PDF, TXT, or Markdown",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
        key=f"uploader_{active_sid}",
        label_visibility="collapsed",
    )
    if st.button("Add Files", use_container_width=True, key="btn_add_files"):
        if uploaded_files:
            processed_key = f"processed_files_{active_sid}"
            if processed_key not in st.session_state:
                st.session_state[processed_key] = set()
            with st.spinner("Processing files…"):
                for f in uploaded_files:
                    if f.name in st.session_state[processed_key]:
                        st.info(f"Already loaded: {f.name}")
                        continue
                    suffix = Path(f.name).suffix
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(f.read())
                            tmp_path = tmp.name
                        docs = load_document(tmp_path)
                        for doc in docs:
                            doc.metadata["title"] = Path(f.name).stem
                        add_paper(docs, active_sid)
                        st.session_state[processed_key].add(f.name)
                        st.success(f"Added: {f.name}")
                    except Exception as e:
                        st.error(f"Failed: {f.name} — {e}")
                    finally:
                        if tmp_path:
                            Path(tmp_path).unlink(missing_ok=True)
            st.rerun()
        else:
            st.warning("No files selected.")

    # ── Section 2: Web URL loader ──────────────────────────────────────────────
    st.markdown("**Web Pages**")
    url_input = st.text_area(
        "URLs (one per line)",
        key=f"url_area_{active_sid}",
        height=80,
        label_visibility="collapsed",
        placeholder="https://example.com/paper",
    )
    if st.button("Load URLs", use_container_width=True, key="btn_load_urls"):
        urls = [u.strip() for u in url_input.splitlines() if u.strip()]
        if urls:
            with st.spinner("Loading web pages…"):
                for url in urls:
                    try:
                        docs = load_webpage(url)
                        add_paper(docs, active_sid)
                        st.success(f"Loaded: {url[:60]}")
                    except Exception as e:
                        st.error(f"Failed: {url[:60]} — {e}")
            st.rerun()
        else:
            st.warning("Enter at least one URL.")

    # ── Section 3: ArXiv loader ────────────────────────────────────────────────
    st.markdown("**ArXiv Papers**")
    arxiv_title = st.text_input(
        "Paper title or ArXiv ID",
        key=f"arxiv_input_{active_sid}",
        label_visibility="collapsed",
        placeholder="1706.03762  or  Attention Is All You Need",
    )
    if st.button("Load ArXiv Paper", use_container_width=True, key="btn_load_arxiv"):
        if arxiv_title.strip():
            with st.spinner("Loading from ArXiv…"):
                try:
                    docs = load_arxiv(arxiv_title.strip())
                    add_paper(docs, active_sid)
                    loaded_title = docs[0].metadata.get("title") if docs else arxiv_title.strip()
                    st.success(f"Loaded: {loaded_title}")
                except Exception as e:
                    st.error(f"Failed: {e}")
            st.rerun()
        else:
            st.warning("Enter a paper title or ArXiv ID.")

    # ── Loaded Documents list ──────────────────────────────────────────────────
    st.divider()
    try:
        doc_titles = list_papers(active_sid)
    except Exception:
        doc_titles = None
    count_badge = (
        f'<span class="pm-badge">{len(doc_titles)}</span>' if doc_titles else ""
    )
    st.markdown(
        f'<div class="pm-section">📚 Loaded Documents{count_badge}</div>',
        unsafe_allow_html=True,
    )
    if doc_titles is None:
        st.caption("Could not load document list — try refreshing.")
    elif doc_titles:
        for title in doc_titles:
            st.markdown(
                f'<div class="pm-doc">📄<span class="t" title="{html.escape(str(title))}">'
                f"{html.escape(str(title))}</span></div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No documents loaded yet.")

# ── Page header ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="pm-hero">
        <h1>PaperMind</h1>
        <p>Your research paper assistant</p>
        <div class="pm-pills">
            <span class="pm-pill">🔍 Ask your papers</span>
            <span class="pm-pill">✅ Verify claims</span>
            <span class="pm-pill">🌐 Search the web</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Empty state ────────────────────────────────────────────────────────────────
if not st.session_state.chats.get(active_sid, []):
    st.markdown(
        """
        <div class="pm-grid">
            <div class="pm-card"><div class="ic">🔍</div><h4>Ask questions</h4>
                <p>Get grounded answers from the papers, pages and files you upload.</p></div>
            <div class="pm-card"><div class="ic">✅</div><h4>Verify claims</h4>
                <p>Check a statement against recent literature and superseding work.</p></div>
            <div class="pm-card"><div class="ic">🌐</div><h4>Search the web</h4>
                <p>Pull in the latest findings when your documents aren't enough.</p></div>
        </div>
        <div class="pm-hint">Upload documents in the sidebar, then start chatting below.
        Tip: prefix a question with <code>/btw</code> for a quick side question.</div>
        """,
        unsafe_allow_html=True,
    )
st.divider()

# ── Chat display ───────────────────────────────────────────────────────────────
for msg in st.session_state.chats.get(active_sid, []):
    avatar = USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            with st.expander(f"📊 Graph state · turn {msg['turn']}", expanded=False):
                st.json(msg["graph_state"])

# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about your papers, verify a claim, or search the web…"):
    is_btw = prompt.strip().lower().startswith("/btw")

    if is_btw:
        query = prompt.strip()[4:].strip()

        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)
            st.caption("Side channel — not saved to session history.")

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            if not query:
                st.markdown("Please add a question after `/btw`, e.g. `/btw What is attention?`")
            else:
                placeholder = st.empty()
                response_text = ""
                for chunk in handle_btw(query):
                    response_text += chunk
                    placeholder.markdown(response_text + "▌")
                placeholder.markdown(response_text)
            st.caption("Side channel — not saved to session history.")

    else:
        if active_sid not in st.session_state.chats:
            st.session_state.chats[active_sid] = []
        if active_sid not in st.session_state.turns:
            st.session_state.turns[active_sid] = 0

        is_first_message = len(st.session_state.chats[active_sid]) == 0

        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)
        st.session_state.chats[active_sid].append({"role": "user", "content": prompt})
        st.session_state.turns[active_sid] += 1
        current_turn = st.session_state.turns[active_sid]

        if is_first_message:
            maybe_rename_session(active_sid, prompt)

        input_state = {
            "messages": [HumanMessage(content=prompt)],
            "session_id": active_sid,
            "query": prompt,
            "route": None,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "claim_verdict": None,
            "claim_source": None,
            "superseding_papers": [],
            "answer": None,
            "is_relevant": None,
            "rewrite_count": 0,
        }
        config = {"configurable": {"thread_id": active_sid}}

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            placeholder = st.empty()
            response_text = ""

            for chunk, metadata in graph.stream(input_state, config, stream_mode="messages"):
                if (
                    metadata.get("langgraph_node") == "generate_answer"
                    and hasattr(chunk, "content")
                    and chunk.content
                ):
                    response_text += chunk.content
                    placeholder.markdown(response_text + "▌")

            if not response_text:
                final_values = graph.get_state(config).values
                response_text = final_values.get("answer") or "No response generated."

            placeholder.markdown(response_text)

            final_values = graph.get_state(config).values
            state_snapshot = _serialize_state(final_values)

            with st.expander(f"📊 Graph state · turn {current_turn}", expanded=False):
                st.json(state_snapshot)

        st.session_state.chats[active_sid].append(
            {
                "role": "assistant",
                "content": response_text,
                "graph_state": state_snapshot,
                "turn": current_turn,
            }
        )

        if is_first_message:
            st.rerun()