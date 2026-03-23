import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_providers import PROVIDER_ORDER, PROVIDERS

from utils import dataframe_agent


def create_chat(input_data, chat_type):
    df_data = pd.DataFrame(input_data["data"], columns=input_data["columns"])
    df_data.set_index(input_data["columns"][0], inplace=True)
    if chat_type == "bar":
        st.bar_chart(df_data)
    elif chat_type == "line":
        st.line_chart(df_data)
    elif chat_type == "scatter":
        st.scatter_chart(df_data)


st.set_page_config(page_title="CSV 数据分析", layout="wide")
st.title("📈 CSV 数据分析工具")

with st.sidebar:
    st.subheader("模型服务")
    provider_labels = {pid: PROVIDERS[pid]["label"] for pid in PROVIDER_ORDER}
    provider_id = st.radio(
        "服务商",
        options=list(PROVIDER_ORDER),
        format_func=lambda x: provider_labels[x],
        key="csv_llm_provider",
    )
    cfg = PROVIDERS[provider_id]

    api_key = (st.text_input("API Key", type="password", help="DeepSeek / 通义百炼 / Kimi 控制台密钥") or "").strip()
    st.markdown(f"[如何获取密钥]({cfg['key_doc_url']})")

    with st.expander("高级：自定义对话模型 ID（可选）"):
        chat_model = st.text_input(
            "对话模型",
            value=cfg["chat_model"],
            key=f"csv_chat_model_{provider_id}",
        )

data = st.file_uploader("上传数据文件（CSV 格式）", type="csv")

if data:
    st.session_state["df"] = pd.read_csv(data)
    with st.expander("原始数据"):
        st.dataframe(st.session_state["df"])

query = st.text_area(
    "请在上传表格后，针对表格数据提出问题，可以满足数据提取、数据可视化需求。"
    "⚠️ 当前版本仅支持散点图、折线图、条形图"
)

button = st.button("AI 作答")

if button and not api_key:
    st.info("请先填写侧栏 API Key。")
if button and "df" not in st.session_state:
    st.info("记得上传数据文件啦～")
if button and api_key and "df" in st.session_state:
    try:
        with st.spinner("🤖️ AI 正在思考中..."):
            response_dict = dataframe_agent(
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
