#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import datetime

import unittest
from typing import Dict, List, Optional

import hypothesis.strategies as st
import torch
import torch.nn as nn
from hypothesis import given, settings, Verbosity
from torch.distributed._tensor.api import DTensor
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torchrec import distributed as trec_dist
from torchrec.distributed.embedding import (
    EmbeddingCollectionSharder,
    ShardedEmbeddingCollection,
)

from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.sharding_plan import (
    column_wise,
    construct_module_sharding_plan,
    row_wise,
    table_wise,
)
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import ShardedTensor, ShardingEnv, ShardingPlan
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection

from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from torchrec.test_utils import skip_if_asan_class

hash_size = 80000000
embedding_dim = 128
batch_size = 2048
seq_len = 64
num_epochs = 1
dataset_size = 800000000
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
def print_gpu_memory_usage():
    allocated = torch.cuda.memory_allocated() / 1024**2  # 转换为 MB
    reserved = torch.cuda.memory_reserved() / 1024**2  # 转换为 MB
    print(f"Allocated memory: {allocated:.2f} MB")
    print(f"Reserved memory: {reserved:.2f} MB")
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
            length = torch.full((self.batch_size,), self.ids_per_features, device=self.device)
            value = torch.randint(
                0, hash_size, (int(length.sum()),), device=self.device
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
    
def _test_sharding(  # noqa C901
    tables: List[EmbeddingConfig],
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[KeyedJaggedTensor],
    backend: str,
    local_size: Optional[int] = None,
    use_apply_optimizer_in_backward: bool = False,
    use_index_dedup: bool = False,
) -> None:
    trec_dist.comm_ops.set_gradient_division(False)
    dataset = CustomDataset(num_steps=num_steps, hash_size=hash_size, batch_size=batch_size, seq_len=seq_len, device=torch.device("cuda"))
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        print(f"###########ctx.device: {ctx.device}")  
        sharder = EmbeddingCollectionSharder(use_index_dedup=use_index_dedup)
        kjt_input_per_rank = [kjt.to(ctx.device) for kjt in kjt_input_per_rank] 
        # print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        unsharded_model = EmbeddingCollection(
            tables=tables,
            device=torch.device("meta"),
            need_indices=True,
        )     
        # print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))    
        # syncs model across ranks
        torch.manual_seed(0)
        for param in unsharded_model.parameters():
            nn.init.uniform_(param, -1, 1)
        torch.manual_seed(0)
        # print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


        module_sharding_plan = construct_module_sharding_plan(
            unsharded_model,
            per_param_sharding={
                "table_0": row_wise(),
            },
            local_size=local_size,
            world_size=world_size,
            device_type=ctx.device.type,
            # pyre-ignore
            sharder=sharder,
        )

        sharded_model = _shard_modules(
            module=unsharded_model,
            plan=ShardingPlan({"": module_sharding_plan}),
            # pyre-fixme[6]: For 1st argument expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            env=ShardingEnv.from_process_group(ctx.pg),
            # pyre-fixme[6]: For 4th argument expected
            #  `Optional[List[ModuleSharder[Module]]]` but got
            #  `List[EmbeddingCollectionSharder]`.
            sharders=[sharder],
            device=ctx.device,
        )

        # sharded model
        # each rank gets a subbatch
        # sharded_model_pred_jts_dict: Dict[str, JaggedTensor] = sharded_model(
        #     kjt_input_per_rank[ctx.rank]
        # )

        # length = torch.full((64,), 1024)
        # value = torch.randint(
        #     0, 40, (int(length.sum()),)
        # )            
        # indices = KeyedJaggedTensor.from_lengths_sync(
        #         keys=["feature_0"],
        #         values=value,
        #         lengths=length,
        #     ).to(ctx.device)
        # sharded_model_pred_jts_dict = sharded_model(indices)
        # print(sharded_model_pred_jts_dict['feature_0'].values())

        # Check memory usage on each GPU
        # print(f"GPU {ctx.rank}: {torch.cuda.memory_allocated(ctx.rank) / 1e9:.2f} GB allocated")  

        # for fqn in sharded_model.state_dict():
        #     sharded_state = sharded_model.state_dict()[fqn]

        #     metadata = sharded_state.metadata()
        #     print(f"Global ShardedTensor Metadata: {metadata}")      

        import time
        train_start_time = time.perf_counter()
        for step in range(num_steps):
            # torch.cuda.nvtx.range_push("FEC Dataloader Pass")
            features = dataset.__getitem__(step)
            features = features.to(ctx.device)
            # torch.cuda.nvtx.range_pop() 
            # torch.cuda.nvtx.range_push("FEC Forward Pass")
            fused_embeddings = sharded_model(features)  
            # torch.cuda.nvtx.range_pop() 
            # print(f"embeddings are {fused_embeddings['feature_0'].values()}")  
        train_end_time = time.perf_counter()
        train_time = train_end_time - train_start_time
        if ctx.rank == 0:
            print(
                "[%s] [TRAIN_TIME] train time is %.2f seconds"
                % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), train_time)
            )
            print(
                "[EPOCH_TIME] %.2f seconds."
                % (train_time / num_epochs,)
            )
            
@skip_if_asan_class
class ShardedEmbeddingCollectionParallelTest(MultiProcessTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)

    def test_sharding_ebc(
        self,

    ) -> None:

        WORLD_SIZE = 8

        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=embedding_dim,
                num_embeddings=hash_size,
            ),
        ]

        # Rank 0
        #             instance 0   instance 1  instance 2
        # "feature_0"   [0, 1]       None        [2]
        # "feature_1"   [0, 1]       None        [2]

        # Rank 1

        #             instance 0   instance 1  instance 2
        # "feature_0"   [3, 2]       [1,2]       [0,1,2,3]
        # "feature_1"   [2, 3]       None        [2]

        kjt_input_per_rank = [  # noqa
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0"],
                values=torch.LongTensor([0, 1, 2, 0, 1, 2]),
                lengths=torch.LongTensor([2, 0, 1, 2, 0, 1]),
            ),
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0"],
                values=torch.LongTensor([3, 2, 1, 2, 0, 1, 2, 3, 2, 3, 2]),
                lengths=torch.LongTensor([2, 2, 4, 2, 0, 1]),
            ),
            
            ]
        self._run_multi_process_test(
            callable=_test_sharding,
            world_size=WORLD_SIZE,
            tables=embedding_config,
            kjt_input_per_rank=kjt_input_per_rank,
            backend="nccl",
            use_apply_optimizer_in_backward=False,
            use_index_dedup=False,
        )
