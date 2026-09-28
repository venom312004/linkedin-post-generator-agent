# ✍️ LinkedIn Post Generator

An autonomous **writer → reviewer agent loop** that drafts LinkedIn posts, critiques its own work, and keeps rewriting until the post is genuinely publish-ready — or gives up after 5 honest attempts.

Built with **LangGraph**, powered by **NVIDIA NIM / Nemotron 3 Ultra** (writer) + **OpenRouter** (reviewer), with live web search via **Tavily**. Ships as both a **Streamlit app** and a **CLI script**.


🔗 **Live Demo:** [linkedin-post-generator-agent-pranjal-003.streamlit.app](https://linkedin-post-generator-agent-pranjal-003.streamlit.app/)

---

## 🧠 How it works

The whole thing is a small state graph:

```
        START
          │
          ▼
      ┌────────┐        needs search       ┌────────┐
      │ writer │ ─────────────────────────▶ │ tools  │
      └────────┘ ◀───────────────────────── └────────┘
          │  no tool calls
          ▼
   ┌───────────────┐
   │ extract_draft │
   └───────────────┘
          │
          ▼
     ┌──────────┐    rejected & attempts < 5
     │ reviewer │ ─────────────────────────▶ back to writer
     └──────────┘
          │  approved  OR  attempt >= 5
          ▼
         END
```

1. **Writer** (`nvidia/nemotron-3-ultra-550b-a55b` via NVIDIA NIM) drafts a post from the topic. If the topic needs current facts or stats, it calls the **Tavily search tool** before writing.
2. **Reviewer** (`nvidia/nemotron-3-ultra-550b-a55b:free` via OpenRouter) grades the draft against 7 strict criteria and returns `APPROVED` or `REJECTED` with feedback.
3. If rejected, the **feedback is fed straight back to the writer**, which rewrites the post addressing every point.
4. This loop repeats until the post is approved **or** 5 attempts are used up, whichever comes first.

### The 7 review criteria
| # | Criterion |
|---|-----------|
| 1 | Strong hook in the first line |
| 2 | One clear, valuable takeaway |
| 3 | Easy to skim (short paragraphs) |
| 4 | ~150–200 words |
| 5 | Ends with an engaging question or CTA |
| 6 | Professional but human tone |
| 7 | No hashtags |

---

## 🚀 Features

- **Self-correcting agent loop** — no human in the loop needed, the model reviews itself
- **Live web search** grounding for topics that need current data (Tavily)
- **Separate writer and reviewer agents** — the writer and reviewer run as independent LLM calls on different providers (NVIDIA NIM and OpenRouter), each with its own prompt, so the reviewer judges every draft strictly against its own criteria
- **Hard attempt cap (5)** to avoid infinite loops and runaway API costs
- **Two interfaces**:
  - `iterate_app.py` — a polished Streamlit UI with post history, approval badges, and one-click `.txt` download
  - `iterate_loop.py` — a lightweight CLI for quick terminal use

---

## 🛠️ Tech stack

| Layer | Tool |
|---|---|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| Writer LLM | NVIDIA NIM (`nvidia/nemotron-3-ultra-550b-a55b`) via `langchain-openai` (OpenAI-compatible endpoint) |
| Reviewer LLM | Qwen (qwen/qwen3.8-27b:free) via OpenRouter and langchain-openai |
| Web search | [Tavily](https://tavily.com/) via `langchain-tavily` |
| UI | [Streamlit](https://streamlit.io/) |
| Env management | `python-dotenv` |

---

## 📦 Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>

python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

pip install streamlit langgraph langchain-openai langchain-tavily python-dotenv
```

### Environment variables

Create a `.env` file in the project root:

```env
NVIDIA_API_KEY=your_nvidia_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
TAVILY_API_KEY=your_tavily_api_key
```

> Get keys from [NVIDIA Build](https://build.nvidia.com/), [OpenRouter](https://openrouter.ai/settings/keys), and [Tavily](https://app.tavily.com/).

---

## ▶️ Usage

### Streamlit app

```bash
streamlit run iterate_app.py
```

Enter a topic, hit **🚀 Generate Post**, and watch the writer/reviewer loop run. Every generated post is kept in a session history with an **APPROVED** or **MAX ATTEMPTS REACHED** badge, the reviewer's last feedback, and a download button.

### CLI script

```bash
python iterate_loop.py
```

You'll be prompted for a topic in the terminal; the final approved (or best-effort) post prints once the loop finishes, along with the attempt count and approval status.

---

## 📁 Project structure

```
.
├── iterate_app.py     # Streamlit UI version
├── iterate_loop.py     # CLI version
├── .env                # API keys (not committed)
└── README.md
```

---

## ⚠️ Notes & limitations

- Max **5 attempts** per topic — if the reviewer keeps rejecting, generation stops and returns the last draft with feedback attached.
- The reviewer's verdict parsing looks for the literal string `APPROVED` before the `FEEDBACK:` marker in its response — reviewer prompt formatting should stay consistent for this to parse reliably.
- Web search is optional and only triggered when the writer model decides the topic needs current info.

---

## 📄 License

Add your license of choice here (MIT recommended for open-source projects).
