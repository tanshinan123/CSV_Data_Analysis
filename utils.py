import os
import pandas
import json
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

你要处理的用户请求如下：    
"""

# 定义函数，用于接收用户对于上传的CVS图表提出的数据分析相关的问题
# 定义参数：用户的API密钥，文件，用户问题
# 创建agent执行器：传入模型、提示词模版、agent及其他相关参数
def dataframe_agent(openai_api_key,dataframe,query):
    model = ChatOpenAI(
        model="gpt-4-turbo",
        openai_api_key=openai_api_key,
        temperature=0,
        base_url="https://api.aigc369.com/v1")
    agent = create_pandas_dataframe_agent(
        llm=model,
        df=dataframe,
        agent_executor_kwargs={"handle_parsing_errors":True},
        verbose=True
    )
    # 组合系统要求和用户要求
    prompt = PROMPT_TEMPLATE + query

    # 调用agent的invoke方法，并将返回的结果保存在response中
    response = agent.invoke({"input":prompt})

    # 传入需要解析的json字符串
    response_dict = json.loads(response["output"])
    return response_dict

# 使用pandas的read_csv方法获取临时文件的路径
# df = pandas.read_csv("house_price.csv")
# print(dataframe_agent(os.getenv("OPENAI_API_KEY"),df,"数据中浴室最多的数量是多少？"))