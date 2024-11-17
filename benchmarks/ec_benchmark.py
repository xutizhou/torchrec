
# 导入 EmbeddingCollection
from torchrec.modules.fused_embedding_modules import FusedEmbeddingCollection
import torchrec
import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
# 定义 embedding 配置
embedding_configs = [
    torchrec.EmbeddingConfig(
        name="table1",
        embedding_dim=16,
        num_embeddings=1000,
        feature_names=["feature1"]
    ),
    torchrec.EmbeddingConfig(
        name="table2",
        embedding_dim=16,
        num_embeddings=2000,
        feature_names=["feature2"]
    ),
]
device = torch.device("cuda")
# 初始化 EmbeddingCollection
fused_embedding_collection = FusedEmbeddingCollection(tables=embedding_configs,
        optimizer_type=torch.optim.SGD,
        optimizer_kwargs={"lr": 0.02},
        device=device,)
# 示例输入
# 示例 sparse_features
sparse_features = {
    "feature1": torch.tensor([1, 2, 3, 4, 5, 6]),
    "feature2": torch.tensor([10, 20, 30, 40, 50, 60])
}

# 转换为 KeyedJaggedTensor
keyed_jagged_tensor = KeyedJaggedTensor.from_offsets_sync(
    keys=list(sparse_features.keys()),
    values=torch.cat(list(sparse_features.values())),
    offsets=torch.tensor([0, 3, 6, 9, 12])
).to(device)

# 前向传播
output = fused_embedding_collection(keyed_jagged_tensor)
for key, value in output.items():
    print(f"{key}: {value}")                         