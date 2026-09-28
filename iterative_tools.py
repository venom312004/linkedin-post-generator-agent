import os
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
REVIEWER_MODEL = os.getenv("REVIEWER_MODEL", "gemini-3.5-flash-lite")  # exact ID AI Studio me check karo
MAX_ATTEMPTS = 5

# ----------------------------
# Tools
# ----------------------------
search_tool = TavilySearch(max_results=3)
tools = [search_tool]

# ----------------------------
# LLMs
# ----------------------------
writer_llm = ChatOpenAI(
    model=WRITER_MODEL,
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.7,
    timeout=60,
    max_retries=3,
)
writer_llm_with_tools = writer_llm.bind_tools(tools)

reviewer_llm = ChatOpenAI(
    model=REVIEWER_MODEL,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1,
    timeout=60,
    max_retries=3,  # free tier me 429 aaye to retry
)


def to_text(content) -> str:
    """Content list of blocks ho to plain string bana do."""
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
# State
# ----------------------------
class State(TypedDict):
    topic: str
    messages: Annotated[list, add_messages]
    draft: str
    review_feedback: str
    is_approved: bool
    attempt: int


# ----------------------------
# Nodes
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
    draft = to_text(last_message.content).strip()
    print(f"\n\n--- Generated post (attempt {state['attempt']}) ---\n{draft}\n")
    return {"draft": draft}


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

    print(f"[Verdict: {'APPROVED' if is_approved else 'REJECTED'}]")
    print(f"[Feedback: {feedback}]")

    return {"review_feedback": feedback, "is_approved": is_approved}


# ----------------------------
# Routers
# ----------------------------
def should_use_tool(state: State):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "extract_draft"


def should_stop_looping(state: State):
    if state["is_approved"]:
        print("Post has been approved.\n")
        return END
    if state["attempt"] >= MAX_ATTEMPTS:
        print("Reached max attempts.")
        return END
    return "writer"


# ----------------------------
# Build graph
# ----------------------------
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

app = graph.compile()

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("Welcome to the LinkedIn Post Generator")
    print("=" * 55)
    print("\nThis tool drafts a LinkedIn post, reviews it itself,")
    print("and iterates until it's publish-ready.")
    print("=" * 55)

    topic = input("\nWhat topic do you want a LinkedIn post about?\n> ").strip()

    if not topic:
        print("\nNo topic given. Exiting.")
    else:
        print("\nStarting generation...\n")

        initial_state = {
            "topic": topic,
            "messages": [],
            "draft": "",
            "review_feedback": "",
            "is_approved": False,
            "attempt": 0,
        }

        try:
            final_state = app.invoke(initial_state, {"recursion_limit": 50})

            print("\n" + "=" * 55)
            print("FINAL LINKEDIN POST")
            print("=" * 55)
            print(final_state["draft"])
            print("=" * 55)
            print(f"Total attempts: {final_state['attempt']}")
            print(f"Approved: {final_state['is_approved']}")
        except Exception as e:
            print(f"\nSomething went wrong: {e}")
