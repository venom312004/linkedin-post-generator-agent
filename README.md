# ✍️ LinkedIn Post Generator

An autonomous **writer → reviewer agent loop** that drafts LinkedIn posts, critiques its own work, and keeps rewriting until the post is genuinely publish-ready — or gives up after 5 honest attempts.

Built with **LangGraph**, powered by **NVIDIA NIM / GPT-OSS 20B** (writer) + **Gemini Flash-Lite via Google AI Studio** (reviewer), with live web search via **Tavily**. Every model used runs on a free tier. Ships as both a **Streamlit app** and a **CLI script**.


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

1. **Writer** (`openai/gpt-oss-20b` via NVIDIA NIM) drafts a post from the topic. If the topic needs current facts or stats, it calls the **Tavily search tool** before writing.
2. **Reviewer** (`gemini-3.5-flash-lite` via Google AI Studio) grades the draft against 7 strict criteria and returns `APPROVED` or `REJECTED` with feedback.
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
- **Two-model setup** — a creative writer model (GPT-OSS 20B) and a separate, strict reviewer model (Gemini), so the same LLM isn't grading its own homework
- **100% free-tier friendly** — NVIDIA NIM, Google AI Studio and Tavily all offer free plans
- **Swappable models** — change `WRITER_MODEL` / `REVIEWER_MODEL` in `.env` without touching the code
- **Hard attempt cap (5)** to avoid infinite loops and runaway API usage
- **Two interfaces**:
  - `iterate_app.py` — a polished Streamlit UI with post history, approval badges, and one-click `.txt` download
  - `iterate_loop.py` — a lightweight CLI for quick terminal use

---

## 🛠️ Tech stack

| Layer | Tool |
|---|---|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| Writer LLM | NVIDIA NIM (`openai/gpt-oss-20b`) via `langchain-openai` (OpenAI-compatible endpoint) |
| Reviewer LLM | Gemini Flash-Lite (`gemini-3.5-flash-lite`) via Google AI Studio's OpenAI-compatible endpoint and `langchain-openai` |
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
GEMINI_API_KEY=your_google_ai_studio_key
TAVILY_API_KEY=your_tavily_api_key

# Optional: override the default models
# WRITER_MODEL=openai/gpt-oss-20b
# REVIEWER_MODEL=gemini-3.5-flash-lite
```

> Get keys from [NVIDIA Build](https://build.nvidia.com/), [Google AI Studio](https://aistudio.google.com/apikey), and [Tavily](https://app.tavily.com/).
>
> Check the exact Gemini model ID in AI Studio's model list; if it differs, set `REVIEWER_MODEL` accordingly.

### Deploying on Streamlit Community Cloud

Add the same three keys (`NVIDIA_API_KEY`, `GEMINI_API_KEY`, `TAVILY_API_KEY`) under **App settings → Secrets**. Remove the old `OPENROUTER_API_KEY` if you no longer use it.

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
├── iterate_loop.py    # CLI version
├── .env               # API keys (not committed)
└── README.md
```

---

## ⚠️ Notes & limitations

- Max **5 attempts** per topic — if the reviewer keeps rejecting, generation stops and returns the last draft with feedback attached.
- The Gemini free tier has daily request limits (Flash-Lite is far more generous than regular Flash), and free-tier inputs may be used by Google to improve its products. Avoid pasting confidential content.
- The reviewer's verdict parsing reads the `VERDICT:` line of the response, so the reviewer prompt format should stay consistent for this to parse reliably.
- Web search is optional and only triggered when the writer model decides the topic needs current info.

---

## 📄 License

Add your license of choice here (MIT recommended for open-source projects).
