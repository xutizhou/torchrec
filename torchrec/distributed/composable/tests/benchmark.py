#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import time
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
import resource



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
    # 创建数据集和数据加载器
    dataset = CustomDataset(num_steps, hash_size, batch_size=batch_size, seq_len=seq_len, device=torch.device("cuda"))
    #get dataset size
    cnt = 0
    for step in range(num_steps):
        features = dataset.__getitem__(step)
        cnt += features.values().shape[0]
    print(f"dataset size={cnt}")

    trec_dist.comm_ops.set_gradient_division(False)
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        print(f"###########ctx.device: {ctx.device}")
        sharder = EmbeddingCollectionSharder(use_index_dedup=use_index_dedup)
        kjt_input_per_rank = [kjt.to(ctx.device) for kjt in kjt_input_per_rank]

        unsharded_model = EmbeddingCollection(
            tables=tables,
            device=ctx.device,
            need_indices=True,
        )

        # syncs model across ranks
        torch.manual_seed(0)
        for param in unsharded_model.parameters():
            nn.init.uniform_(param, -1, 1)
        torch.manual_seed(0)

        if use_apply_optimizer_in_backward:
            apply_optimizer_in_backward(
                torch.optim.SGD,
                unsharded_model.embeddings.parameters(),
                {"lr": 1.0},
            )
        else:
            unsharded_model_optimizer = torch.optim.SGD(
                unsharded_model.parameters(), lr=1.0
            )

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

        if not use_apply_optimizer_in_backward:
            sharded_model_optimizer = torch.optim.SGD(
                sharded_model.parameters(), lr=1.0
            )

        assert isinstance(sharded_model, ShardedEmbeddingCollection)

        if not use_apply_optimizer_in_backward:
            unsharded_model_optimizer.zero_grad()
            sharded_model_optimizer.zero_grad()

        unsharded_model_pred_jt_dict = []
        for unsharded_rank in range(ctx.world_size):
            # simulate the unsharded model run on the entire batch
            unsharded_model_pred_jt_dict.append(
                unsharded_model(kjt_input_per_rank[unsharded_rank])
            )

        # sharded model
        # each rank gets a subbatch
        sharded_model_pred_jts_dict: Dict[str, JaggedTensor] = sharded_model(
            kjt_input_per_rank[ctx.rank]
        )

        unsharded_model_pred_jt_dict_this_rank: Dict[str, JaggedTensor] = (
            unsharded_model_pred_jt_dict[ctx.rank]
        )

        embedding_names = unsharded_model_pred_jt_dict_this_rank.keys()
        assert set(unsharded_model_pred_jt_dict_this_rank.keys()) == set(
            sharded_model_pred_jts_dict.keys()
        )

        for embedding_name in embedding_names:
            unsharded_jt = unsharded_model_pred_jt_dict_this_rank[embedding_name]
            sharded_jt = sharded_model_pred_jts_dict[embedding_name]
            # print(f"unsharded_jt.values(), sharded_jt.values(): {unsharded_jt.values()}, {sharded_jt.values()}")
            torch.testing.assert_close(unsharded_jt.values(), sharded_jt.values())
            torch.testing.assert_close(unsharded_jt.lengths(), sharded_jt.lengths())
            torch.testing.assert_close(unsharded_jt.offsets(), sharded_jt.offsets())
            torch.testing.assert_close(
                unsharded_jt.weights_or_none(), sharded_jt.weights_or_none()
            )
        # start_time = time.perf_counter()
        # # 迭代数据加载器
        print("####start training")
        indices = KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0"],
                values=torch.LongTensor([0, 1, 2, 0, 1, 2]),
                lengths=torch.LongTensor([2, 0, 1, 2, 0, 1]),
            ).to(ctx.device)
        embeddings = sharded_model(indices)
        print(f"#########embeddings:{embeddings['feature_0'].values()}")
        # for epoch in range(num_epochs):
        #     for step in range(num_steps):
        #         print(f"epoch:{epoch}, step:{step}")
        #         # torch.cuda.nvtx.range_push("FEC Dataloader Pass")
        #         features = dataset.__getitem__(step)
        #         features = features.to(ctx.device)
        #         print(f"features:{features.values()}")
        #         # torch.cuda.nvtx.range_pop() 
        #         # torch.cuda.nvtx.range_push("FEC Forward Pass")
        #         embeddings = sharded_model(features)
        #         print(f"embeddings:{embeddings['feature_0'].values()}")
                 
        # end_time = time.perf_counter()
        # ec_time = end_time - start_time
        # print(f"fused ec Time: {ec_time}")


@skip_if_asan_class
class ShardedEmbeddingCollectionParallelTest(MultiProcessTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    # pyre-ignore
    @given(
        use_apply_optimizer_in_backward=st.booleans(),
        use_index_dedup=st.booleans(),
    )
    def test_sharding_ebc(
        self,
        use_apply_optimizer_in_backward: bool,
        use_index_dedup: bool,
    ) -> None:

        WORLD_SIZE = 2

        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=128,
                num_embeddings=8000000,
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
            use_apply_optimizer_in_backward=use_apply_optimizer_in_backward,
            use_index_dedup=use_index_dedup,
        )
