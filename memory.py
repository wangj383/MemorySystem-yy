from openai import OpenAI
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

## 核心功能
# - sentence embedding
#     - function: 把输入的str 做embedding，存起来
#     - 输入：str
#     - 输出：None
# - query
#     - function：基于cos-sim 来做搜索
#     - 输入
#         - str: 问题
#         - top-k: 返回多少条
#     - 输出: list(str)

# ## 实现步骤
# ### 设计
# - 1. 用什么工具/package (embedding 模型的接入 - OpenAI SDK/本地部署)
#     - 存储 - list(vector)
# - 2. Class MemorySystem
#     - 1. insert(str): embedding
#     - 2. query(str, int)
            # 1. "isabelle" --> embedding vector
            # 2. find embedding vector from saved embedding list
            # 3. cos-sim --> top-k
client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="sk-crovezvxrwxcfrhfwabfsonmxdjlzydivybwrvldnlhswdrg",
        base_url="https://api.siliconflow.cn/v1",
    )
class MemorySystem:
    def __init__(self):
        self.memory = []
        self.embeddings = {}

    def insert(self, str):
        # 1. str --> list[float]
        # 2. 存起来

        self.embeddings[str] = []

        completion = client.embeddings.create(
            model='Qwen/Qwen3-Embedding-8B',
            input=str,
            dimensions=64, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
            encoding_format="float"
        )
        # print(completion)
        self.embeddings[str] = completion.data[0].embedding
        print(self.embeddings[str])

        print(self.embeddings)
        return None

    def query(self, str, top_k):
        completion = client.embeddings.create(
            model='Qwen/Qwen3-Embedding-8B',
            input=str,
            dimensions=64, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
            encoding_format="float"
        )
        query_str_embedd = completion.data[0].embedding
        # 2. find embedding vector from saved embedding list
        cos_sim_list = []
        embedding_list = []
        for embedding in self.embeddings:
            cos_sim = cosine_similarity([query_str_embedd], [self.embeddings[embedding]])
            cos_sim_list.append(cos_sim)
            embedding_list.append(embedding)
        cos_sim_list.sort(reverse=True)
        top_k_cos_sim_list = cos_sim_list[:top_k]
        print(top_k_cos_sim_list)
        # 3. cos-sim --> top-k

        # return the original text
        return [embedding_list[cos_sim_list.index(cos_sim)] for cos_sim in top_k_cos_sim_list]  
       
        

    # def embedding_change(self, str):
    #     return self.embeddings[str]
        

if __name__ == "__main__":
    memory_system = MemorySystem()
    memory_system.insert("my name is isabelle")
    memory_system.insert("my father's name is john")
    memory_system.insert("my mother's name is jane")
    memory_system.insert("my sister's name is lisa")
    memory_system.insert("my brother's name is jim")
    memory_system.insert("my sister's name is lisa")
    print(memory_system.query("what is my age?", 3))