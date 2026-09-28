import os
import streamlit as st
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# Model config (sab free)
# ----------------------------
# Writer   -> NVIDIA NIM par gpt-oss-20b (open-weight, free API key)
# Reviewer -> Google Gemini Flash-Lite (AI Studio free tier)
WRITER_MODEL = os.getenv("WRITER_MODEL", "openai/gpt-oss-20b")
WRITER_BASE_URL = "https://integrate.api.nvidia.com/v1"

REVIEWER_MODEL = os.getenv("REVIEWER_MODEL", "gemini-3.5-flash-lite")  # exact ID AI Studio me check karo
REVIEWER_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

MAX_ATTEMPTS = 5
REQUEST_TIMEOUT = 40  # seconds per LLM call
MAX_RETRIES = 1

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="LinkedIn Post Generator",
    page_icon="✍️",
    layout="centered",
)


# ----------------------------
# Step 1 - Tools & LLMs (cached, sirf ek baar load hote hain)
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_resources():
    search_tool = TavilySearch(max_results=3)
    tools = [search_tool]

    writer_llm = ChatOpenAI(
        model=WRITER_MODEL,
        base_url=WRITER_BASE_URL,
        api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=0.7,
        timeout=REQUEST_TIMEOUT,
        max_retries=MAX_RETRIES,
    )
    writer_llm_with_tools = writer_llm.bind_tools(tools)

    reviewer_llm = ChatOpenAI(
        model=REVIEWER_MODEL,
        base_url=REVIEWER_BASE_URL,
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1,
        timeout=REQUEST_TIMEOUT,
        max_retries=MAX_RETRIES,
    )

    return tools, writer_llm_with_tools, reviewer_llm


tools, writer_llm_with_tools, reviewer_llm = load_resources()


# ----------------------------
# Helper
# ----------------------------
def to_text(content) -> str:
    """Kuch models content ko list of blocks me dete hain, use plain string bana do."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


# ----------------------------
# Step 2 - State
# ----------------------------
class State(TypedDict):
    topic: str
    messages: Annotated[list, add_messages]
    draft: str
    review_feedback: str
    is_approved: bool
    attempt: int


# ----------------------------
# Step 3 - Nodes
# ----------------------------
WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Your job is to write "
    "engaging, professional LinkedIn posts about the given topic. "
    "If the topic requires up-to-date information, statistics, or "
    "current trends, use the web search tool to gather fresh context "
    "before writing. If you have already received feedback on a "
    "previous draft, carefully address every point in the new draft. "
    "Rules for good LinkedIn posts: strong hook in the first line, "
    "1 clear takeaway, easy to skim (short paragraphs), around "
    "150-200 words, ends with a question or call-to-action to invite "
    "engagement. Do not use hashtags. "
    "Output ONLY the final post text, with no preamble or explanation."
)


def writer_node(state: State) -> dict:
    """Post likhta (ya dobara likhta) hai. Zarurat ho to Tavily se search karta hai."""
    # Tool call se wapas aaye: same attempt continue karo
    if state["messages"] and getattr(state["messages"][-1], "type", "") == "tool":
        response = writer_llm_with_tools.invoke(
            [("system", WRITER_SYSTEM_PROMPT)] + state["messages"]
        )
        return {"messages": [response]}

    attempt = state.get("attempt", 0) + 1
    topic = state["topic"]
    previous_feedback = state["review_feedback"]

    if attempt == 1:
        user_message = (
            f'Write a LinkedIn post about this topic: "{topic}". '
            "If you need current info, search the web first."
        )
    else:
        user_message = (
            f'Your previous draft on "{topic}" was rejected.\n\n'
            f"Reviewer feedback:\n{previous_feedback}\n\n"
            "Write a new, improved post that fixes every issue mentioned. "
            "Do not repeat the same mistakes."
        )

    messages = [("system", WRITER_SYSTEM_PROMPT), ("human", user_message)]
    response = writer_llm_with_tools.invoke(messages)

    return {"messages": [("human", user_message), response], "attempt": attempt}


tool_node = ToolNode(tools)


def extract_draft_node(state: State) -> dict:
    """Writer ka final text draft ke roop me nikalta hai."""
    last_message = state["messages"][-1]
    return {"draft": to_text(last_message.content).strip()}


REVIEWER_SYSTEM_PROMPT = (
    "You are a strict LinkedIn content reviewer. You judge whether a "
    "post is publish-ready. Evaluate against these criteria:\n"
    "1. Strong hook in the first line\n"
    "2. One clear, valuable takeaway\n"
    "3. Easy to skim - uses short paragraphs\n"
    "4. Roughly 150-200 words\n"
    "5. Ends with an engaging question or CTA\n"
    "6. Professional but human tone (not corporate-robotic)\n"
    "7. No hashtags\n\n"
    "Respond in exactly this format:\n"
    "VERDICT: APPROVED or REJECTED\n"
    "FEEDBACK: <one short paragraph explaining why>\n\n"
    "Be strict but fair. Approve only if the post genuinely meets all "
    "criteria. Reject if even one criterion is clearly missing."
)


def reviewer_node(state: State) -> dict:
    """Draft review karta hai: approve ya feedback ke saath reject."""
    draft = state["draft"]

    prompt = f"Review this LinkedIn post draft:\n\n{draft}\n\nGive your review."
    response = reviewer_llm.invoke(
        [("system", REVIEWER_SYSTEM_PROMPT), ("human", prompt)]
    )
    review_text = to_text(response.content).strip()

    # Sirf VERDICT wali line dekho
    verdict_line = ""
    for line in review_text.splitlines():
        if line.strip().upper().startswith("VERDICT"):
            verdict_line = line.upper()
            break
    is_approved = "APPROVED" in verdict_line and "REJECTED" not in verdict_line

    if "FEEDBACK:" in review_text:
        feedback = review_text.split("FEEDBACK:", 1)[1].strip()
    else:
        feedback = review_text

    return {"review_feedback": feedback, "is_approved": is_approved}


# ----------------------------
# Step 4 - Router functions
# ----------------------------
def should_use_tool(state: State):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "extract_draft"


def should_stop_looping(state: State):
    if state["is_approved"]:
        return END
    if state["attempt"] >= MAX_ATTEMPTS:
        return END
    return "writer"


# ----------------------------
# Step 5 - Graph build (cached)
# ----------------------------
@st.cache_resource(show_spinner=False)
def build_graph():
    graph = StateGraph(State)

    graph.add_node("writer", writer_node)
    graph.add_node("tools", tool_node)
    graph.add_node("extract_draft", extract_draft_node)
    graph.add_node("reviewer", reviewer_node)

    graph.add_edge(START, "writer")
    graph.add_conditional_edges("writer", should_use_tool)
    graph.add_edge("tools", "writer")
    graph.add_edge("extract_draft", "reviewer")
    graph.add_conditional_edges("reviewer", should_stop_looping)

    return graph.compile()


app = build_graph()

# ----------------------------
# Step 6 - Streamlit UI
# ----------------------------
st.markdown(
    """
<style>
.main-header { text-align: center; padding: 1rem 0 0.5rem 0; }
.main-header h1 { font-size: 2.2rem; margin-bottom: 0.2rem; }
.main-header p { color: #888; font-size: 0.95rem; }
.post-card {
    border: 1px solid #333;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    background-color: #111318;
    white-space: pre-wrap;
    line-height: 1.5;
}
.status-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 10px;
}
.badge-approved { background-color: #1f4a2e; color: #86efac; }
.badge-maxed { background-color: #4a3110; color: #fcd34d; }
.badge-unreviewed { background-color: #3a1f1f; color: #fca5a5; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="main-header">
    <h1>✍️ LinkedIn Post Generator</h1>
    <p>AI writes a draft, reviews it, and iterates until it's publish-ready</p>
</div>
""",
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ About")
    st.caption("This tool runs a writer → reviewer loop:")
    st.caption("✍️ Writer drafts the post (searches the web if needed)")
    st.caption("🔎 Reviewer checks it against 7 quality criteria")
    st.caption(f"🔁 Loops with feedback until approved (max {MAX_ATTEMPTS} attempts)")
    st.markdown("---")
    st.caption(f"**Writer:** `{WRITER_MODEL}`")
    st.caption(f"**Reviewer:** `{REVIEWER_MODEL}`")
    st.markdown("---")
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# --- Session state ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Input ---
topic = st.text_input(
    "What topic do you want a LinkedIn post about?",
    placeholder="e.g. why AI agents are the next big shift in SaaS",
)
generate = st.button("🚀 Generate Post", use_container_width=True, type="primary")

if generate:
    if not topic.strip():
        st.warning("Please enter a topic first.")
    else:
        # Missing keys ka jaldi pata chale
        missing = [
            k for k in ("NVIDIA_API_KEY", "GEMINI_API_KEY", "TAVILY_API_KEY")
            if not os.getenv(k)
        ]
        if missing:
            st.error(f"Missing API key(s): {', '.join(missing)}. Add them in .env or Streamlit Secrets.")
            st.stop()

        result = {
            "draft": "",
            "review_feedback": "",
            "is_approved": False,
            "attempt": 0,
        }
        initial_state = {"topic": topic.strip(), "messages": [], **result}

        status = st.status("Starting...", expanded=True)
        failed_error = None

        try:
            for step in app.stream(
                initial_state,
                {"recursion_limit": 40},
                stream_mode="updates",
            ):
                for node, update in step.items():
                    if node == "writer":
                        attempt_no = update.get("attempt")
                        if attempt_no:
                            status.write(f"✍️ Writer drafting (attempt {attempt_no})...")
                        else:
                            status.write("✍️ Writer continuing with search results...")
                    elif node == "tools":
                        status.write("🔍 Web search done")
                    elif node == "extract_draft":
                        status.write("📝 Draft ready, sending to reviewer...")
                    elif node == "reviewer":
                        verdict = "APPROVED ✅" if update.get("is_approved") else "REJECTED ❌"
                        status.write(f"🔎 Reviewer: {verdict}")

                    for k in ("draft", "review_feedback", "is_approved", "attempt"):
                        if update and k in update:
                            result[k] = update[k]

            status.update(label="Done", state="complete", expanded=False)
        except Exception as e:
            failed_error = e
            status.update(label="Stopped due to an error", state="error", expanded=True)
            st.error(f"Something went wrong: {e}")

        # Draft mil gaya ho to (review fail hone par bhi) dikhao
        if result["draft"]:
            st.session_state.history.insert(
                0,
                {
                    "topic": topic.strip(),
                    "draft": result["draft"],
                    "attempt": result["attempt"],
                    "approved": result["is_approved"],
                    "unreviewed": failed_error is not None,
                    "feedback": result["review_feedback"]
                    or (str(failed_error) if failed_error else ""),
                },
            )

# --- Results ---
for i, item in enumerate(st.session_state.history):
    st.markdown("---")
    if item["approved"]:
        st.markdown(
            '<span class="status-badge badge-approved">✅ APPROVED</span>',
            unsafe_allow_html=True,
        )
    elif item.get("unreviewed"):
        st.markdown(
            '<span class="status-badge badge-unreviewed">⚠️ STOPPED BY ERROR (draft may be unreviewed)</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge badge-maxed">⚠️ MAX ATTEMPTS REACHED</span>',
            unsafe_allow_html=True,
        )

    st.caption(f"**Topic:** {item['topic']}  ·  **Attempts:** {item['attempt']}")
    st.markdown(f'<div class="post-card">{item["draft"]}</div>', unsafe_allow_html=True)

    if not item["approved"]:
        with st.expander("Last reviewer feedback / error"):
            st.write(item["feedback"])

    st.download_button(
        "📋 Download as .txt",
        data=item["draft"],
        file_name="linkedin_post.txt",
        mime="text/plain",
        key=f"download_{len(st.session_state.history) - i}",
    )
