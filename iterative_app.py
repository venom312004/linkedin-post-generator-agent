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
# Page config
# ----------------------------
st.set_page_config(
    page_title="LinkedIn Post Generator",
    page_icon="✍️",
    layout="centered",
)

# ----------------------------
# Step 1 - Tools & LLMs (cached so they load only once)
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_resources():
    search_tool = TavilySearch(max_result=3)
    tools = [search_tool]

    writer_llm = ChatOpenAI(model="openai/gpt-oss-20b", base_url="https://integrate.api.nvidia.com/v1", api_key=os.getenv("NVIDIA_API_KEY"), temperature=0.7, timeout=30)
    writer_llm_with_tools = writer_llm.bind_tools(tools)

    reviewer_llm = ChatOpenAI(model="qwen/qwen3.8-27b:free", base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"), temperature=0.1, timeout=30)

    return tools, writer_llm_with_tools, reviewer_llm


tools, writer_llm_with_tools, reviewer_llm = load_resources()


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
writer_system_prompt = (
    "You are an expert LinkedIn content writer. Your job is to write "
    "engaging, professional LinkedIn posts about the given topic. "
    "If the topic requires up-to-date information, statistics, or "
    "current trends, use the web search tool to gather fresh context "
    "before writing. If you have already received feedback on a "
    "previous draft, carefully address every point in the new draft. "
    "Rules for good LinkedIn posts: strong hook in the first line, "
    "1 clear takeaway, easy to skim (short paragraphs), around "
    "150–200 words, ends with a question or call-to-action to invite "
    "engagement. Do not use hashtags."
)


def writer_node(state: State) -> dict:
    """Writes (or rewrites) the LinkedIn post. Can call Tavily to search first."""
    # coming back from a tool call: continue the SAME attempt using the search results
    if state['messages'] and getattr(state['messages'][-1], 'type', '') == 'tool':
        response = writer_llm_with_tools.invoke([("system", writer_system_prompt)] + state['messages'])
        return {"messages": [response]}

    attempt = state.get("attempt", 0) + 1
    topic = state['topic']
    previous_feedback = state['review_feedback']

    if attempt == 1:
        user_message = (
            f"Write a LinkedIn post on this {topic}"
            f"if you need current info search the web first"
        )
    else:
        user_message = (
            f"your previous draft on '{topic}' was rejected"
            f"Here is the reviewer's feedback \n\n {previous_feedback}\n\n"
            f"Write a new ,improved that fixes every issue mentioned"
            f"do not repeat the same mistake"
        )
    messages = [("system", writer_system_prompt), ("human", user_message)]
    response = writer_llm_with_tools.invoke(messages)

    return {"messages": [("human", user_message), response],
            "attempt": attempt}


tool_node = ToolNode(tools)


def extract_draft_node(state: State) -> dict:
    """After the writer finishes tool calls, pulls the final text out as the draft."""
    last_message = state['messages'][-1]
    draft = last_message.content
    return {"draft": draft}


REVIEWER_SYSTEM_PROMPT = (
    "You are a strict LinkedIn content reviewer. You judge whether a "
    "post is publish-ready. Evaluate against these criteria:\n"
    "1. Strong hook in the first line\n"
    "2. One clear, valuable takeaway\n"
    "3. Easy to skim — uses short paragraphs\n"
    "4. Roughly 150-200 words\n"
    "5. Ends with an engaging question or CTA\n"
    "6. Professional but human tone (not corporate-robotic)\n"
    "7. No hashtags\n\n"
    "Respond in exactly this format:\n"
    "VERDICT: APPROVED or REJECTED\n"
    "FEEDBACK: <one short paragraph explaining why>\n\n"
    "Be strict but fair. Approve only if the post genuinely meets all "
    "criteria. Reject if even one criterion is clearly missing.")


def reviewer_node(state: State) -> dict:
    """Reviews the draft and decides: approve or reject with feedback."""
    draft = state['draft']

    prompt = (
        f"Review this LinkedIn post draft : \n"
        f"{draft}\n"
        f"give your reviews"
    )
    response = reviewer_llm.invoke(
        [("system", REVIEWER_SYSTEM_PROMPT), ("human", prompt)]
    )
    review_text = response.content.strip()

    is_approved = "APPROVED" in review_text.upper().split("FEEDBACK")[0]

    if "FEEDBACK:" in review_text:
        feedback = review_text.split("FEEDBACK:", 1)[1].strip()
    else:
        feedback = review_text

    return {
        "review_feedback": feedback,
        "is_approved": is_approved
    }


# ----------------------------
# Step 4 - Router functions
# ----------------------------
def should_use_tool(state: State):
    last_message = state['messages'][-1]

    if getattr(last_message, 'tool_calls', None):
        return "tools"
    return "extract_draft"


def should_stop_looping(state: State):
    if state['is_approved']:
        return END
    if state['attempt'] >= 5:
        return END
    return "writer"


# ----------------------------
# Step 5 - Build the graph (cached)
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

# --- Custom CSS ---
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 1rem 0 0.5rem 0;
}
.main-header h1 {
    font-size: 2.2rem;
    margin-bottom: 0.2rem;
}
.main-header p {
    color: #888;
    font-size: 0.95rem;
}
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
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>✍️ LinkedIn Post Generator</h1>
    <p>AI writes a draft, reviews it, and iterates until it's publish-ready</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ About")
    st.caption("This tool runs a writer → reviewer loop:")
    st.caption("✍️ Writer drafts the post (searches the web if needed)")
    st.caption("🔎 Reviewer checks it against 7 quality criteria")
    st.caption("🔁 Loops back with feedback until approved (max 5 attempts)")
    st.markdown("---")
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# --- Session state init ---
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"topic", "draft", "attempt", "approved"}

# --- Topic input ---
topic = st.text_input("What topic do you want a LinkedIn post about?", placeholder="e.g. why AI agents are the next big shift in SaaS")
generate = st.button("🚀 Generate Post", use_container_width=True, type="primary")

if generate:
    if not topic.strip():
        st.warning("Please enter a topic first.")
    else:
        initial_state = {
            "topic": topic.strip(),
            "messages": [],
            "draft": "",
            "review_feedback": "",
            "is_approved": False,
            "attempt": 0,
        }

        with st.spinner("Writing, reviewing, and iterating..."):
            final_state = app.invoke(initial_state)

        st.session_state.history.insert(0, {
            "topic": topic.strip(),
            "draft": final_state["draft"],
            "attempt": final_state["attempt"],
            "approved": final_state["is_approved"],
            "feedback": final_state["review_feedback"],
        })

# --- Render results ---
for item in st.session_state.history:
    st.markdown("---")
    if item["approved"]:
        st.markdown(
            '<span class="status-badge badge-approved">✅ APPROVED</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-badge badge-maxed">⚠️ MAX ATTEMPTS REACHED</span>',
            unsafe_allow_html=True
        )

    st.caption(f"**Topic:** {item['topic']}  ·  **Attempts:** {item['attempt']}")
    st.markdown(f'<div class="post-card">{item["draft"]}</div>', unsafe_allow_html=True)

    if not item["approved"]:
        with st.expander("Last reviewer feedback"):
            st.write(item["feedback"])

    st.download_button(
        "📋 Copy as .txt",
        data=item["draft"],
        file_name="linkedin_post.txt",
        mime="text/plain",
        key=f"download_{item['topic']}_{item['attempt']}_{item['approved']}"
    )
