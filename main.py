import pandas as pd
import streamlit as st
from utils import dataframe_agent

# 创建函数用于处理用户的数据可视化需求
# 根据键名调用不同的图表类型
def create_chat(input_data,chat_type):
    df_data = pd.DataFrame(input_data["data"],columns=input_data["columns"])
    df_data.set_index(input_data["columns"][0],inplace=True)
    if chat_type == "bar":
        st.bar_chart(df_data)
    elif chat_type == "line":
        st.line_chart(df_data)
    elif chat_type == "scatter":
        st.scatter_chart(df_data)

# 创建网站的标题和侧边栏
st.title("📈 CSV数据分析职能工具")

with st.sidebar:
    openai_api_key = st.text_input("请输入OpenAI API密钥",type="password")
    st.markdown("[获取OpenAI API key](https://platform.openai.com/account/api-keys)")

# 创建上传CSV文件的入口
data = st.file_uploader("上传数据文件(CSV格式)",type="csv")

# 若用户上传了CSV文件，调用pandas的read_csv方法将CSV读取为数据帧并存储在会话状态中
if data:
    st.session_state["df"] = pd.read_csv(data)
    # 展示上传文件的原始数据
    with st.expander("原始数据"):
        st.dataframe(st.session_state["df"])

# 创建问题的输入文本框
query = st.text_area("请在上传表格后，针对表格数据提出问题，可以满足数据提取，数据可视化需求。⚠️当前版本仅支持散点图、折线图、条形图")

button = st.button("AI作答")

# 判断AI生成内容的条件是否均满足，包括：密钥、文件、问题
if button and not openai_api_key:
    st.info("记得输入OpenAI API密钥啦～")
if button and "df" not in st.session_state:
    st.info("记得上传数据文件啦～")
if button and openai_api_key and "df" in st.session_state:
    with st.spinner("🤖️AI正在思考中..."):
        response_dict = dataframe_agent(openai_api_key,st.session_state["df"],query)
        # 后端接口返回给前端的内容需要作判断
        # 当用户的提问需要返回的是文字描述相关的回答
        if "answer" in response_dict:
            st.write(response_dict["answer"])
        # 当用户的提问需要返回的是数据提取(表格)相关的回答
        # streamlit的table函数接收数据帧作为参数，调用DataFrame对返回的结果作类型转换
        # 提取返回的数据帧中data所对应的值，并将columns对应的值赋值给列名
        if "table" in response_dict:
            st.table(pd.DataFrame(response_dict["table"]["data"],
                                  columns=response_dict["table"]["columns"]))
        if "bar" in response_dict:
            create_chat(response_dict["bar"],"bar")
        if "line" in response_dict:
            create_chat(response_dict["line"],"line")
        if "scatter" in response_dict:
            create_chat(response_dict["scatter"],"scatter")