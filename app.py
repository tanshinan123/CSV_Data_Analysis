"""统一入口：CSV 数据分析与 PDF 智能问答，通过 Tab 切换。"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from langchain.memory import ConversationBufferMemory

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_providers import PROVIDER_ORDER, PROVIDERS


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_csv_utils = None
_csv_utils_error: str | None = None
_pdf_utils = None
_pdf_utils_error: str | None = None

try:
    _csv_utils = _load_module("csv_data_utils", ROOT / "CSV_Data_Analysis" / "utils.py")
except Exception as e:  # 允许单侧依赖缺失，避免整页无法启动
    _csv_utils_error = f"{type(e).__name__}: {e}"

try:
    _pdf_utils = _load_module("pdf_qa_utils", ROOT / "PDF_QA_Tool" / "utils.py")
except Exception as e:
    _pdf_utils_error = f"{type(e).__name__}: {e}"


def create_chat(input_data, chat_type):
    df_data = pd.DataFrame(input_data["data"], columns=input_data["columns"])
    df_data.set_index(input_data["columns"][0], inplace=True)
    if chat_type == "bar":
        st.bar_chart(df_data)
    elif chat_type == "line":
        st.line_chart(df_data)
    elif chat_type == "scatter":
        st.scatter_chart(df_data)


def _reset_pdf_session():
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer",
    )
    st.session_state.pop("chat_history", None)


st.set_page_config(page_title="数据分析工具集", layout="wide")
st.title("📊 数据分析工具集")

if "memory" not in st.session_state:
    _reset_pdf_session()

with st.sidebar:
    st.subheader("模型服务")
    provider_labels = {pid: PROVIDERS[pid]["label"] for pid in PROVIDER_ORDER}
    provider_id = st.radio(
        "服务商",
        options=list(PROVIDER_ORDER),
        format_func=lambda x: provider_labels[x],
        key="llm_provider",
    )
    cfg = PROVIDERS[provider_id]
    if st.session_state.get("_provider_id_saved") != provider_id:
        st.session_state["_provider_id_saved"] = provider_id
        _reset_pdf_session()

    api_key = (st.text_input("API Key", type="password", help="对应所选服务商控制台中的密钥") or "").strip()
    st.markdown(f"[如何获取密钥]({cfg['key_doc_url']})")

    with st.expander("高级：自定义模型名（可选）"):
        chat_model = st.text_input(
            "对话模型 ID",
            value=cfg["chat_model"],
            key=f"chat_model_{provider_id}",
        )
        if cfg["embedding_model"]:
            emb_model = st.text_input(
                "向量模型 ID（仅 PDF 向量检索）",
                value=cfg["embedding_model"],
                key=f"emb_model_{provider_id}",
            )
        else:
            emb_model = None

    if _csv_utils is None or _pdf_utils is None:
        st.warning(
            "部分 Tab 因依赖未就绪不可用，请按对应 Tab 内提示安装缺失包，"
            "或在项目根目录使用合并依赖的虚拟环境运行。"
        )

tab_csv, tab_pdf = st.tabs(["📈 CSV 数据分析", "📃 PDF 智能问答"])

with tab_csv:
    if _csv_utils is None:
        st.error("CSV 分析模块未能加载。")
        st.code(_csv_utils_error or "未知错误", language="text")
        st.info(
            "若使用 PDF 项目的虚拟环境，请补充安装 CSV 侧依赖，例如：\n"
            "`pip install langchain-experimental`"
        )
    st.caption("上传 CSV 后提问，支持文字回答、表格与条形图 / 折线图 / 散点图。")
    data = st.file_uploader("上传数据文件（CSV 格式）", type="csv", key="csv_uploader")
    if data:
        st.session_state["df"] = pd.read_csv(data)
        with st.expander("原始数据"):
            st.dataframe(st.session_state["df"])
    query = st.text_area(
        "请在上传表格后，针对表格数据提出问题，可以满足数据提取、数据可视化需求。"
        "⚠️ 当前版本仅支持散点图、折线图、条形图",
        key="csv_query",
    )
    button = st.button("AI 作答", key="csv_submit")
    if button and not api_key:
        st.info("请先填写侧栏 API Key。")
    if button and "df" not in st.session_state:
        st.info("记得上传数据文件啦～")
    if button and api_key and "df" in st.session_state and _csv_utils is not None:
        try:
            with st.spinner("🤖️ AI 正在思考中..."):
                response_dict = _csv_utils.dataframe_agent(
                    api_key,
                    st.session_state["df"],
                    query,
                    base_url=cfg["base_url"],
                    chat_model=chat_model,
                )
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"调用失败（{type(e).__name__}）：{e}")
        else:
            if "answer" in response_dict:
                st.write(response_dict["answer"])
            if "table" in response_dict:
                st.table(
                    pd.DataFrame(
                        response_dict["table"]["data"],
                        columns=response_dict["table"]["columns"],
                    )
                )
            if "bar" in response_dict:
                create_chat(response_dict["bar"], "bar")
            if "line" in response_dict:
                create_chat(response_dict["line"], "line")
            if "scatter" in response_dict:
                create_chat(response_dict["scatter"], "scatter")

with tab_pdf:
    if _pdf_utils is None:
        st.error("PDF 问答模块未能加载。")
        st.code(_pdf_utils_error or "未知错误", language="text")
        st.info("请在该环境中安装 `PDF_QA_Tool/requirements.txt` 中的依赖。")
    if not cfg["embedding_model"]:
        st.info(
            "当前服务商未配置云端向量模型，PDF 将使用 **BM25 关键词检索**（可安装 "
            "`rank-bm25` 提升效果）；超长文档可能截断。完整向量检索请选 **通义千问（百炼）**。"
        )
    st.caption("上传 PDF 后针对文档内容连续问答。")
    uploaded_file = st.file_uploader("⬆️ 上传 PDF 文件", type="pdf", key="pdf_uploader")
    question = st.text_input(
        "💬 针对 PDF 文件内容的提问",
        disabled=not uploaded_file,
        key="pdf_question",
    )
    if uploaded_file and question and not api_key:
        st.info("请先填写侧栏 API Key。")
    if uploaded_file and question and api_key and _pdf_utils is not None:
        try:
            with st.spinner("🤖️ AI 正在思考中 >>>"):
                response = _pdf_utils.qa_agent(
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
                st.error(
                    "缺少 **pypdf**（解析 PDF 必需）。请在运行 Streamlit 的同一 Python 环境中执行：\n\n"
                    "`python -m pip install pypdf`\n\n"
                    "然后**完全重启** Streamlit。"
                )
            elif "faiss" in hint:
                st.error(
                    "缺少 **faiss-cpu**（使用「通义千问（百炼）」做 PDF 向量检索时必需）。请执行：\n\n"
                    "`python -m pip install faiss-cpu`\n\n"
                    "或暂时改用 **DeepSeek / Kimi**（关键词检索，不依赖 FAISS）。"
                )
            elif "dashscope" in hint:
                st.error(
                    "使用「通义千问（百炼）」向量模型需要 **dashscope** SDK。请执行：\n\n"
                    "`python -m pip install dashscope`\n\n"
                    "安装后重启 Streamlit。"
                )
            else:
                st.error(f"依赖导入失败（ImportError）：{e}")
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
