import sys
from pathlib import Path

import streamlit as st
from langchain.memory import ConversationBufferMemory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_providers import PROVIDER_ORDER, PROVIDERS

from utils import qa_agent


def _reset_pdf_session():
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer",
    )
    st.session_state.pop("chat_history", None)


st.set_page_config(page_title="PDF 智能问答", layout="wide")
st.title("📃 智能 PDF 问答工具")

if "memory" not in st.session_state:
    _reset_pdf_session()

with st.sidebar:
    st.subheader("模型服务")
    provider_labels = {pid: PROVIDERS[pid]["label"] for pid in PROVIDER_ORDER}
    provider_id = st.radio(
        "服务商",
        options=list(PROVIDER_ORDER),
        format_func=lambda x: provider_labels[x],
        key="pdf_llm_provider",
    )
    cfg = PROVIDERS[provider_id]
    if st.session_state.get("_pdf_provider_saved") != provider_id:
        st.session_state["_pdf_provider_saved"] = provider_id
        _reset_pdf_session()

    api_key = (st.text_input("API Key", type="password", help="DeepSeek / 通义百炼 / Kimi 控制台密钥") or "").strip()
    st.markdown(f"[如何获取密钥]({cfg['key_doc_url']})")

    with st.expander("高级：自定义模型 ID（可选）"):
        chat_model = st.text_input(
            "对话模型",
            value=cfg["chat_model"],
            key=f"pdf_chat_model_{provider_id}",
        )
        if cfg["embedding_model"]:
            emb_model = st.text_input(
                "向量模型（PDF 检索）",
                value=cfg["embedding_model"],
                key=f"pdf_emb_model_{provider_id}",
            )
        else:
            emb_model = None

if not cfg["embedding_model"]:
    st.info(
        "当前服务商未配置云端向量模型，PDF 将使用 **BM25 / 全文合并** 检索；"
        "超长文档可能截断。需要向量检索可选 **通义千问（百炼）**。"
    )

uploaded_file = st.file_uploader("⬆️ 上传 PDF 文件", type="pdf")
question = st.text_input(
    "💬 针对 PDF 文件内容的提问",
    disabled=not uploaded_file,
)

if uploaded_file and question and not api_key:
    st.info("请先填写侧栏 API Key。")

if uploaded_file and question and api_key:
    try:
        with st.spinner("🤖️ AI 正在思考中 >>>"):
            response = qa_agent(
                api_key,
                st.session_state["memory"],
                uploaded_file,
                question,
                base_url=cfg["base_url"],
                chat_model=chat_model,
                embedding_model=emb_model if cfg["embedding_model"] else None,
                embedding_dimensions=cfg["embedding_dimensions"]
                if cfg["embedding_model"]
                else None,
            )
    except ValueError as e:
        st.error(str(e))
    except ImportError as e:
        hint = str(e).lower()
        if "pypdf" in hint:
            st.error("缺少 **pypdf**：请执行 `pip install pypdf` 后重启应用。")
        elif "faiss" in hint:
            st.error("缺少 **faiss-cpu**（百炼向量检索需要）：`pip install faiss-cpu`，或改用 DeepSeek/Kimi。")
        elif "dashscope" in hint:
            st.error("百炼向量需要 **dashscope**：`pip install dashscope`")
        else:
            st.error(f"依赖导入失败：{e}")
    except Exception as e:
        st.error(f"调用失败（{type(e).__name__}）：{e}")
    else:
        st.write("### AI 回答")
        st.write(response["answer"])
        st.session_state["chat_history"] = response["chat_history"]

if "chat_history" in st.session_state:
    with st.expander("🕐 历史消息列表"):
        for i in range(0, len(st.session_state["chat_history"]), 2):
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i + 1]
            st.write(human_message.content)
            st.write(ai_message.content)
            if i < len(st.session_state["chat_history"]) - 2:
                st.divider()
