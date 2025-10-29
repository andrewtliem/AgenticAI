"""
Streamlit UI showcasing the Agentic customer service workflow from the notebook.
Run with: streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List
from html import escape
from datetime import datetime
import calendar

import streamlit as st
from tinydb import Query, TinyDB

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - handled at runtime in UI
    genai = None


BASE_DIR = Path(__file__).resolve().parent

PRODUCTS_DB = BASE_DIR / "products.json"
DEFAULT_PRODUCTS = [
    {
        "name": "Smart Glasses X1",
        "price": 120,
        "stock": 8,
        "description": "AR-powered glasses with touch control and Bluetooth audio.",
    },
    {
        "name": "Reading Glasses Classic",
        "price": 40,
        "stock": 20,
        "description": "Lightweight reading glasses with anti-glare coating.",
    },
    {
        "name": "Blue Light Shield Pro",
        "price": 95,
        "stock": 12,
        "description": "Blue light filtering glasses with ergonomic frame.",
    },
]

CUSTOMERS_DB = BASE_DIR / "customers.json"
DEFAULT_CUSTOMERS = [
    {"id": 1, "name": "John Doe", "credits": 300},
    {"id": 2, "name": "Sarah Lee", "credits": 150},
    {"id": 3, "name": "David Kim", "credits": 500},
]

INVOICES_DB = BASE_DIR / "invoices.json"
DEFAULT_INVOICES = [
    {
        "invoice_id": 1,
        "customer_id": 1,
        "product_name": "Smart Glasses X1",
        "price": 120,
        "purchase_date": "2025-01-15",
        "warranty_months": 12,
    },
    {
        "invoice_id": 2,
        "customer_id": 2,
        "product_name": "Reading Glasses Classic",
        "price": 40,
        "purchase_date": "2025-02-10",
        "warranty_months": 6,
    },
    {
        "invoice_id": 3,
        "customer_id": 1,
        "product_name": "Sun Glasses Retro",
        "price": 60,
        "purchase_date": "2024-11-20",
        "warranty_months": 3,
    },
]


def load_env_file(path: Path | str = BASE_DIR / ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def init_tinydb(path: Path, defaults: List[Dict[str, Any]]) -> TinyDB:
    db = TinyDB(path)
    if len(db) == 0 and defaults:
        db.insert_multiple(defaults)
    return db


@lru_cache(maxsize=1)
def get_db() -> TinyDB:
    return init_tinydb(PRODUCTS_DB, DEFAULT_PRODUCTS)


def get_all_products() -> List[Dict[str, Any]]:
    return get_db().all()


def find_product(product_name: str) -> Dict[str, Any] | None:
    if not product_name:
        return None
    lookup = product_name.strip().casefold()
    Product = Query()
    result = get_db().get(Product.name.test(lambda v: v.casefold() == lookup))
    return result


@lru_cache(maxsize=1)
def get_customer_db() -> TinyDB:
    return init_tinydb(CUSTOMERS_DB, DEFAULT_CUSTOMERS)


def get_all_customers() -> List[Dict[str, Any]]:
    return get_customer_db().all()


def get_customer_by_name(customer_name: str) -> Dict[str, Any] | None:
    if not customer_name:
        return None
    lookup = customer_name.strip().casefold()
    Customer = Query()
    return get_customer_db().get(Customer.name.test(lambda v: v.casefold() == lookup))


@lru_cache(maxsize=1)
def get_invoice_db() -> TinyDB:
    return init_tinydb(INVOICES_DB, DEFAULT_INVOICES)


def get_all_invoices() -> List[Dict[str, Any]]:
    records = get_invoice_db().all()
    return sorted(records, key=lambda item: item.get("invoice_id", 0), reverse=True)


def recommend_product(budget: float | int | str) -> str:
    try:
        budget_value = float(budget)
    except (TypeError, ValueError):
        return "⚠️ Please provide a numeric budget."

    candidates = [
        p
        for p in get_all_products()
        if p["price"] <= budget_value and p["stock"] > 0
    ]

    if not candidates:
        return f"😔 Sorry, no products are available under ${budget_value:.2f}."

    message = f"🎯 Products within your ${budget_value:.2f} budget:\n"
    for product in candidates:
        message += (
            f"- {product['name']} (${product['price']}) — "
            f"{product['stock']} in stock\n"
        )
    return message.strip()


def check_stock(product_name: str) -> str:
    product = find_product(product_name)
    if not product:
        return f"⚠️ {product_name} is not in the catalog."
    stock = product["stock"]
    if stock > 0:
        return f"✅ {product['name']} is in stock with {stock} units available."
    return f"❌ {product['name']} is currently out of stock."


def get_product_description(product_name: str) -> str:
    product = find_product(product_name)
    if not product:
        return f"⚠️ {product_name} is not in the catalog."
    return (
        f"ℹ️ {product['name']} costs ${product['price']} and we have "
        f"{product['stock']} in stock. Description: {product['description']}"
    )


def list_available_products() -> str:
    available = [p for p in get_all_products() if p["stock"] > 0]
    if not available:
        return "⚠️ No products are available at the moment."
    message = "🟢 Available products:\n"
    for product in available:
        message += (
            f"- {product['name']} (${product['price']}) — "
            f"{product['stock']} in stock\n"
        )
    return message.strip()


def check_customer_credits(customer_name: str) -> str:
    customer = get_customer_by_name(customer_name)
    if not customer:
        return f"❌ Customer '{customer_name}' not found."
    credits = customer["credits"]
    return f"💳 {customer['name']} has {credits} store credits available."


def find_customer_invoices(customer_name: str) -> str:
    customer = get_customer_by_name(customer_name)
    if not customer:
        return f"❌ Customer '{customer_name}' not found."

    Invoice = Query()
    invoices = get_invoice_db().search(Invoice.customer_id == customer["id"])
    if not invoices:
        return f"🧾 No invoices found for {customer['name']}."

    message = [f"🧾 Purchase history for {customer['name']}:"]
    for inv in sorted(invoices, key=lambda item: item.get("invoice_id", 0)):
        message.append(
            f"- Invoice #{inv['invoice_id']} | {inv['product_name']} | "
            f"${inv['price']} | Purchased: {inv['purchase_date']} | "
            f"Warranty: {inv['warranty_months']} months"
        )
    return "\n".join(message)


def add_months(dt: datetime, months: int) -> datetime:
    year = dt.year + (dt.month - 1 + months) // 12
    month = (dt.month - 1 + months) % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return datetime(year, month, day)


def check_warranty(invoice_id: int) -> str:
    Invoice = Query()
    record = get_invoice_db().search(Invoice.invoice_id == invoice_id)
    if not record:
        return f"❌ Invoice #{invoice_id} not found."

    invoice = record[0]
    purchase_date = datetime.strptime(invoice["purchase_date"], "%Y-%m-%d")
    warranty_end = add_months(purchase_date, invoice["warranty_months"])
    today = datetime.today()

    if today <= warranty_end:
        return (
            f"✅ Invoice #{invoice_id} is under warranty until {warranty_end.date()}."
        )
    return f"⚠️ Warranty expired on {warranty_end.date()} for Invoice #{invoice_id}."


def buy_product(customer_name: str, product_name: str) -> Dict[str, Any]:
    customer = get_customer_by_name(customer_name)
    if not customer:
        return {
            "status": "error",
            "message": f"Customer '{customer_name}' not found.",
        }

    product = find_product(product_name)
    if not product:
        return {
            "status": "error",
            "message": f"Product '{product_name}' not found.",
        }

    if product["stock"] <= 0:
        return {
            "status": "error",
            "message": f"Product '{product['name']}' is out of stock.",
        }

    if customer["credits"] < product["price"]:
        return {
            "status": "error",
            "message": (
                f"Insufficient credits. {customer['name']} has {customer['credits']} "
                f"credits, but {product['name']} costs {product['price']}."
            ),
        }

    new_credits = customer["credits"] - product["price"]
    Customer = Query()
    get_customer_db().update({"credits": new_credits}, Customer.id == customer["id"])

    Product = Query()
    get_db().update(
        {"stock": product["stock"] - 1},
        Product.name.test(lambda v: v.casefold() == product["name"].casefold()),
    )

    invoice_db = get_invoice_db()
    existing_ids = [item.get("invoice_id", 0) for item in invoice_db.all()]
    next_invoice_id = max(existing_ids, default=0) + 1
    invoice_record = {
        "invoice_id": next_invoice_id,
        "customer_id": customer["id"],
        "product_name": product["name"],
        "price": product["price"],
        "purchase_date": datetime.today().strftime("%Y-%m-%d"),
        "warranty_months": 12,
    }
    invoice_db.insert(invoice_record)

    return {
        "status": "success",
        "message": "Purchase completed successfully.",
        "invoice_id": next_invoice_id,
        "product": product["name"],
        "price": product["price"],
        "remaining_credits": new_credits,
    }


TOOL_EXECUTORS = {
    "get_product_description": get_product_description,
    "check_stock": check_stock,
    "recommend_product": recommend_product,
    "list_available_products": list_available_products,
    "check_customer_credits": check_customer_credits,
    "find_customer_invoices": find_customer_invoices,
    "check_warranty": check_warranty,
    "buy_product": buy_product,
}


def parse_json_response(raw_text: str) -> Dict[str, Any]:
    clean = re.sub(r"```json|```", "", raw_text).strip()
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in model output:\n{raw_text[:200]}")
    return json.loads(match.group(0))


def format_conversation(messages: List[Dict[str, str]]) -> str:
    if not messages:
        return ""
    role_labels = {"user": "User", "assistant": "Assistant"}
    lines = []
    for message in messages:
        role = role_labels.get(message.get("role"), "Other")
        content = (message.get("content") or "").strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def load_model():
    if genai is None:
        return None, "Install google-generativeai to enable the agent."

    load_env_file()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None, "Set the GOOGLE_API_KEY environment variable to run the agent."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    return model, None


def agentic_customer_service(
    user_prompt: str, model, conversation_context: str = ""
) -> Dict[str, Any]:  # pragma: no cover - relies on external API
    tools_description = """
You have access to these tools:
1. get_product_description(product_name: str)
2. check_stock(product_name: str)
3. recommend_product(budget: float)
4. list_available_products()
5. check_customer_credits(customer_name: str)
6. find_customer_invoices(customer_name: str)
7. check_warranty(invoice_id: int)
8. buy_product(customer_name: str, product_name: str)  # returns a JSON receipt

Each step, return JSON:
{"actions": [
    {"tool": "tool_name", "args": {...}},
    ...
]}
You may call one or more tools depending on user need.
"""

    decisions: List[Dict[str, Any]] = []
    history: List[Dict[str, Any]] = []

    for step in range(1, 4):
        step_prompt = f"""
You are an AI assistant for an eyewear shop.
{tools_description}

Tool results so far:
{json.dumps(history, indent=2)}

User request: {user_prompt}
Previous conversation transcript:
{conversation_context or "None"}

Decide what tools to use next (if any), or return "done" if ready to answer.
Output strictly in JSON.
"""

        response = model.generate_content(step_prompt)
        text = response.text.strip()
        decisions.append({"step": step, "model_decision": text})

        if text.lower() == "done":
            break

        try:
            decision_data = parse_json_response(text)
            actions = decision_data.get("actions", [])
        except Exception as exc:
            decisions[-1]["error"] = f"Failed to parse JSON: {exc}"
            break

        if not actions:
            break

        for action in actions:
            tool_name = action.get("tool")
            args = action.get("args", {})

            executor = TOOL_EXECUTORS.get(tool_name)
            if not executor:
                result = f"Unknown tool '{tool_name}'"
            else:
                try:
                    if args:
                        result = executor(**args)
                    else:
                        result = executor()
                except Exception as exc:  # pragma: no cover - defensive
                    result = f"Error while executing {tool_name}: {exc}"

            history.append({"tool": tool_name, "args": args, "result": result})

    reflection_prompt = f"""
The user asked: {user_prompt}
You have this context from tools:
{json.dumps(history, indent=2)}
Previous conversation transcript:
{conversation_context or "None"}

Write a final, polite, customer-friendly summary.
Don't mention tools or steps. Just the helpful final answer.
"""

    final_reply = model.generate_content(reflection_prompt)
    return {
        "final": final_reply.text,
        "history": history,
        "decisions": decisions,
    }


def render_plaintext(content: str) -> None:
    """Render chat text without markdown interpretation."""
    safe = escape(content).replace("\n", "<br/>")
    st.markdown(f"<p>{safe}</p>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title="Eyewear Shop Agent",
        page_icon="🕶️",
        layout="wide",
    )
    st.title("🧠 Agentic Customer Service Demo")
    st.write(
        "Ask our AI assistant anything about the eyewear catalog. "
        "This full workflow agent inspects inventory, looks up customers, "
        "verifies warranties, and can even complete purchases before reflecting on its answer."
    )

    products = get_all_products()
    customers = get_all_customers()
    invoices = get_all_invoices()
    with st.sidebar:
        st.header("🛍️ Catalog")
        for product in products:
            st.markdown(
                f"**{product['name']}** — ${product['price']} &nbsp; "
                f"({product['stock']} in stock)\n\n"
                f"{product['description']}"
            )
        st.header("👥 Customers")
        for customer in customers:
            st.markdown(
                f"**{customer['name']}** — {customer['credits']} credits remaining"
            )
        st.header("🧾 Recent invoices")
        customer_lookup = {c["id"]: c["name"] for c in customers}
        for invoice in invoices[:5]:
            customer_name = customer_lookup.get(
                invoice["customer_id"], f"Customer #{invoice['customer_id']}"
            )
            st.markdown(
                f"**Invoice #{invoice['invoice_id']}** — {invoice['product_name']} "
                f"for {customer_name} on {invoice['purchase_date']} "
                f"(Warranty: {invoice['warranty_months']} months)"
            )
        st.divider()
        st.caption(
            "The agent can describe items, check stock, recommend within a budget, "
            "list availability, look up customer credits, review invoices, verify warranty coverage, "
            "and create new purchases using store credits."
        )

    model, model_error = load_model()
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            render_plaintext(message["content"])

    if prompt := st.chat_input("What can I help you find today?"):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            render_plaintext(prompt)

        if model_error:
            response_text = model_error
            debug_payload: Dict[str, Any] = {}
        else:
            transcript = format_conversation(st.session_state["messages"])
            result = agentic_customer_service(prompt, model, transcript)
            response_text = result.get("final", "I couldn't generate a response.")
            debug_payload = {
                "decisions": result.get("decisions", []),
                "tool_results": result.get("history", []),
            }

        st.session_state["messages"].append(
            {"role": "assistant", "content": response_text}
        )
        with st.chat_message("assistant"):
            render_plaintext(response_text)
            if debug_payload:
                with st.expander("Agent trace"):
                    st.json(debug_payload)

    if model_error:
        st.info(model_error)


if __name__ == "__main__":
    main()
