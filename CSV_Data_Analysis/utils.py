import json
import re

import pandas
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# 针对用户提出的问题,AI需要返回不同的结果：字符串、表格、图表
# 将agent返回的结果设置为字典,根据字典的不同键判断下一步操作
# answer键：展示字符串;table键：表格;bar键：条形图;line键：折线图;scatter键：散点图
# 其中column表示图表的数据轴,data表示图表的数值
PROMPT_TEMPLATE = """
你是一位数据分析助手，可以结合用户的请求回应相应的数据格式，包括字符串，表格及图表，具体如下：

1. 如果用户提出的问题可以通过文字描述回答时，请按照该格式回答：
    {"answer":"<答案>"}
    例如：
    {"answer":"最高的房子单价是'13300000'"}

2. 如果用户需要一个数据表格，请按照该格式回答：
    {"table":{"columns":["column1","column2",...],"data":[[value1,value2,...],[value1,value2,...],...]}}

3. 如果用户的请求适合返回条形图，请按照该格式回答：
    {"bar":{"columns":["A","B","C",...],"data":[11,22,33,...]}}

4. 如果用户的请求适合返回折线图，请按照该格式回答： 
    {"line":{"columns":["A","B","C",...],"data":[11,22,33,...]}}

5. 如果用户的请求适合散点图，请按照该格式回答：
    {"scatter":{columns:["A","B","C",...],"data""[11,22,33,...]}}

请注意：当前我们只支持以上三类图表"bar","line","scatter"。

请将所有的输出作为JSON字符串返回，请注意将"columns"列表和数据列表中的所有字符串均用双引号包围。
例如：{"columns":["price","bathrooms","furnishingstatus"],"data":[[13300000,2,"furnished"],[10850000,3,"semi-furnished"]]}     

**重要**：完成全部工具调用后，**最终回复必须且只能是一个合法 JSON 对象**（不要用 ``` 代码块包裹，不要追加解释文字）。若因数据或问题无法完成，请输出：
{"answer":"简要说明原因"}

你要处理的用户请求如下：    
"""


def _parse_json_from_agent_output(raw: str) -> dict:
    """Agent 有时会在 JSON 外套 Markdown 或夹杂说明，这里尽量抽取 JSON。"""
    if not raw or not str(raw).strip():
        raise ValueError("模型未返回任何内容，请简化问题后重试。")

    text = str(raw).strip()
    low = text.lower()
    if "agent stopped due to iteration limit" in low or "agent stopped due to time limit" in low:
        raise ValueError(
            "分析步数或时间已用尽，模型没能给出最终结果。"
            "请尝试：① 把问题拆成更小的步骤；② 减少一次提问中的任务量；③ 确认表格不要过大。"
        )

    def _loads(s: str) -> dict:
        return json.loads(s)

    try:
        return _loads(text)
    except json.JSONDecodeError:
        pass

    # ```json ... ``` 或 ``` ... ```
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        inner = fence.group(1).strip()
        try:
            return _loads(inner)
        except json.JSONDecodeError:
            pass

    # 从首个 { 到最后一个 } 截取
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return _loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "模型返回内容无法解析为 JSON，请换种问法或稍后重试。"
        f"（原始片段：{text[:400]!r}…）"
    )


# 定义函数，用于接收用户对于上传的CVS图表提出的数据分析相关的问题
# 定义参数：用户的API密钥，文件，用户问题
# 创建agent执行器：传入模型、提示词模版、agent及其他相关参数
def dataframe_agent(
    openai_api_key,
    dataframe,
    query,
    *,
    base_url: str,
    chat_model: str,
):
    model = ChatOpenAI(
        model=chat_model,
        openai_api_key=openai_api_key,
        temperature=0,
        base_url=base_url,
        timeout=120,
        max_retries=2,
    )
    agent = create_pandas_dataframe_agent(
        llm=model,
        df=dataframe,
        max_iterations=45,
        max_execution_time=300,
        early_stopping_method="generate",
        agent_executor_kwargs={"handle_parsing_errors": True},
        verbose=True,
    )
    prompt = PROMPT_TEMPLATE + query
    response = agent.invoke({"input": prompt})
    raw = response.get("output", "")
    return _parse_json_from_agent_output(raw)

# 使用pandas的read_csv方法获取临时文件的路径
# df = pandas.read_csv("house_price.csv")
# print(dataframe_agent(os.getenv("OPENAI_API_KEY"),df,"数据中浴室最多的数量是多少？"))