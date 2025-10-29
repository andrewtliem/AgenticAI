# Agentic AI Customer Service Agent

This project implements an advanced agentic workflow as a ready-to-run Streamlit application. The agent behaves like a customer service specialist for an eyewear shop: it plans which tools to call, executes them in sequence, reflects on the tool outputs, and delivers a polished reply to the user.

## Project structure

- `streamlit_app.py` – Streamlit UI and Gemini-powered agent workflow.
- `customers.json`, `products.json`, `invoices.json` – TinyDB JSON tables persisted on disk.
- `requirements.txt` – Python dependencies for the app.
- `.gitignore` – Development-time exclusions (virtual envs, `.env`, caches, etc.).

> Tip: keep your `.env` file alongside `streamlit_app.py` so the helper can automatically load `GOOGLE_API_KEY`.

## Features

- **Multi-tool planning loop** powered by Gemini 2.5 Flash. The agent can string together several tool calls per request.
- **Rich toolbelt** covering the full customer workflow: catalog lookup, stock checks, budget filtering, credit balance, invoice history, warranty validation, and simulated purchases that debit credits and generate invoices.
- **Reflection pass** so the final answer is grounded in tool outputs without revealing implementation details.
- **Persistent state** using TinyDB; any purchases or credit updates are written straight into the JSON tables so they survive app restarts.
- **Streamlit chat UI** with sidebar snapshots of products, customers, and recent invoices, plus an expandable “Agent trace” debugging panel.

## Prerequisites

- Python 3.10 or newer.
- A Google API key with access to the Gemini models.
- (Optional) Virtual environment manager such as `venv`, `conda`, or `pipenv`.

## Setup

1. **Clone** the repository and navigate into the project folder:
   ```bash
   git clone <your-repo-url>
   cd agenticAI/CustomerServiceAgent
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure secrets** by creating a `.env` file next to `streamlit_app.py`:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

5. *(Optional)* Delete any of the JSON tables if you want to reseed them with the tutorial defaults on the next launch.

## Running the app

```bash
streamlit run streamlit_app.py
```

Open the URL that Streamlit prints (typically `http://localhost:8501`). Use the chat input to ask about products, warranty status, customer credits, or to place a new order.

When Gemini is available, the agent will:
1. Inspect prior conversation context.
2. Decide which tool(s) to call next and emit a JSON plan.
3. Execute the tools directly in Python, capturing their results.
4. Reflect on the accumulated context to compose a conversational response.

If the Gemini SDK or API key is missing, the UI gracefully surfaces instructions instead of attempting to run the agent.

## Tooling reference

| Tool | Purpose |
| ---- | ------- |
| `get_product_description(product_name)` | Returns price, stock count, and marketing blurb for a specific product. |
| `check_stock(product_name)` | Confirms whether an item is in stock and how many units remain. |
| `recommend_product(budget)` | Lists all in-stock products priced at or below the specified budget. |
| `list_available_products()` | Summarizes every product currently available. |
| `check_customer_credits(customer_name)` | Shows how many store credits a customer has left. |
| `find_customer_invoices(customer_name)` | Retrieves purchase history with invoice numbers, dates, and warranty length. |
| `check_warranty(invoice_id)` | Calculates whether an invoice is still covered under warranty based on purchase date and term. |
| `buy_product(customer_name, product_name)` | Simulates a purchase: validates credits and stock, decrements both, and appends a new invoice record. |

All tool calls read and write to the TinyDB JSON tables in this folder, so the Streamlit session always works with the same persisted data.

## Troubleshooting

- **`GOOGLE_API_KEY` errors** – verify the key is present in `.env` and that you restarted Streamlit after editing the file.
- **Model quota issues** – Gemini rate limits propagate as exceptions in the Streamlit log; try again after a short delay or switch to a different key with quota.
- **Stale data** – delete `customers.json`, `products.json`, or `invoices.json` to reset the data stores; the app will reseed them on the next run.
- **Trace debugging** – expand the “Agent trace” panel inside the chat to inspect the raw tool plans and results.

## Roadmap ideas

- Add authentication for staff-only workflows.
- Introduce analytics dashboards summarizing sales and credit usage.
- Swap TinyDB for a production-grade store (PostgreSQL, Firestore, etc.).
- Extend the agent with upsell recommendations, repair scheduling, or returns processing.

---

Enjoy exploring the full agentic workflow, and feel free to iterate on the tools or UI to match your own customer service scenarios!
