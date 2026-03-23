"""侧栏可选的大模型服务商（OpenAI 兼容接口）默认配置。"""

PROVIDERS: dict[str, dict] = {
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "chat_model": "deepseek-chat",
        "embedding_model": None,
        "embedding_dimensions": None,
        "key_doc_url": "https://platform.deepseek.com/api_keys",
    },
    "bailian": {
        "label": "通义千问（百炼）",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "chat_model": "qwen-plus",
        "embedding_model": "text-embedding-v3",
        "embedding_dimensions": 1024,
        "key_doc_url": "https://help.aliyun.com/zh/model-studio/get-api-key",
    },
    "kimi": {
        "label": "Kimi（Moonshot）",
        "base_url": "https://api.moonshot.cn/v1",
        "chat_model": "moonshot-v1-8k",
        "embedding_model": None,
        "embedding_dimensions": None,
        "key_doc_url": "https://platform.moonshot.cn/console/api-keys",
    },
}

PROVIDER_ORDER = ("deepseek", "bailian", "kimi")
