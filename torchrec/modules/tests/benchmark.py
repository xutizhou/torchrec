import unittest
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Type

import hypothesis.strategies as st

import torch
import torch.fx
import torchrec
from fbgemm_gpu.split_table_batched_embeddings_ops_training import EmbeddingLocation
from hypothesis import given, settings
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.modules.fused_embedding_modules import (
    fuse_embedding_optimizer,
    FusedEmbeddingBagCollection,
    FusedEmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torch.utils.data import Dataset, DataLoader
import random
from torch.profiler import profile, record_function, ProfilerActivity

devices: List[torch.device] = [torch.device("cpu")]
if torch.cuda.device_count() > 1:
    devices.append(torch.device("cuda"))

class CustomDataset():
    def __init__(self, num_steps, hash_size, batch_size, seq_len, device):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.hash_size = hash_size
        self.device = device
        self.min_ids_per_features = 0
        self.ids_per_features = seq_len
        self.keys = ["feature_0"]
        self.data = self._generate_data()
    def _generate_data(self):
        data = []
        for _ in range(self.num_steps):
            values = []
            lengths = []
            hash_size = self.hash_size
            length = torch.full((self.batch_size,), self.ids_per_features)
            value = torch.randint(
                0, hash_size, (int(length.sum()),)
            )
            lengths.append(length)
            values.append(value)
            sparse_features = KeyedJaggedTensor.from_lengths_sync(
                keys=self.keys,
                values=torch.cat(values),
                lengths=torch.cat(lengths),
            )
            data.append(sparse_features)
        return data
    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        return self.data[idx]


def test_fc_forward() -> None:
    optimizer_type_and_kwargs = (torch.optim.SGD, {"lr": 0.1})
    device = torch.device("cuda")
    optimizer_type, optimizer_kwargs = optimizer_type_and_kwargs
    print(f"device: {device}")
    print(f"optimizer_type: {optimizer_type}")
    print(f"optimizer_kwargs: {optimizer_kwargs}")
    hash_size = 80000000
    embedding_dim = 128
    batch_size = 2048
    seq_len = 4096
    num_epochs = 1
    dataset_size = 8000000000
    num_steps = dataset_size // (batch_size * seq_len)
    print(f"hash_size: {hash_size}")
    print(f'hash_size GB: {hash_size * embedding_dim * 4 / 1024 / 1024 / 1024}')
    print(f"embedding_dim: {embedding_dim}")
    print(f"batch_size: {batch_size}")
    print(f"seq_len: {seq_len}")
    print(f"dataset_size: {dataset_size}")
    print(f"fetched embedding size GB: {dataset_size * embedding_dim * 4 / 1024 / 1024 / 1024}")
    print(f"num_epochs: {num_epochs}")
    print(f"num_steps: {num_steps}")
    
    embedding_configs = [
        EmbeddingConfig(
            num_embeddings=hash_size,
            embedding_dim=embedding_dim,
            name="table_0",
            feature_names=["feature_0"],
        ),
    ]

    fused_ec = FusedEmbeddingCollection(
        tables=embedding_configs,
        optimizer_type=optimizer_type,
        optimizer_kwargs=optimizer_kwargs,
        device=device,
    )

    ec = EmbeddingCollection(tables=embedding_configs, device=device)

    #        0       1        2  <-- batch
    # "f1"   [] [0]    [0,1]
    # "f2"   [1]    [0,1]    []
    #  ^
    # feature

    opt = optimizer_type(ec.parameters(), **optimizer_kwargs)

    import time

    # 创建数据集和数据加载器
    dataset = CustomDataset(num_steps, hash_size, batch_size=batch_size, seq_len=seq_len, device=device)
    #get dataset size
    cnt = 0
    for step in range(num_steps):
        features = dataset.__getitem__(step)
        cnt += features.values().shape[0]
    print(f"dataset size={cnt}")
            
    start_time = time.perf_counter()
    # 迭代数据加载器

    for epoch in range(num_epochs):
        for step in range(num_steps):
            # torch.cuda.nvtx.range_push("FEC Dataloader Pass")
            features = dataset.__getitem__(step)
            features = features.to(device)
            # torch.cuda.nvtx.range_pop() 
            # torch.cuda.nvtx.range_push("FEC Forward Pass")
            fused_embeddings = fused_ec(features)
            # torch.cuda.nvtx.range_pop() 
            # fused_vals = []
            # for _name, jt in fused_embeddings.items():
            #     fused_vals.append(jt.values())
            # # torch.cuda.nvtx.range_push("FEC Backward + Gradient Pass")
            # torch.cat(fused_vals).sum().backward()
            # torch.cuda.nvtx.range_pop() 
    end_time = time.perf_counter()
    fused_ec_time = end_time - start_time
    print(f"fused ec Time: {fused_ec_time}")

test_fc_forward()