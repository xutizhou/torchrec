    def test_optimizer_fusion(
        self,
        optimizer_type_and_kwargs: Tuple[Type[torch.optim.Optimizer], Dict[str, Any]],
        device: torch.device,
    ) -> None:
        optimizer_type, optimizer_kwargs = optimizer_type_and_kwargs
        print(f"device: {device}")
        print(f"optimizer_type: {optimizer_type}")
        print(f"optimizer_kwargs: {optimizer_kwargs}")
        hash_size = 5000000
        embedding_dim = 128
        batch_size = 64
        # 定义数据集参数
        num_epochs = 100
        num_steps = 10     
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
        dataset = CustomDataset(num_steps, hash_size, batch_size=batch_size, device=device)
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
                features = dataset.__getitem__(step)
                features = features.to(device)
                opt.zero_grad()
                sequence_embeddings = ec(features)
                vals = []
                for _name, jt in sequence_embeddings.items():
                    vals.extend(jt.to_dense())
                torch.cat(vals).sum().backward()
                opt.step()

        end_time = time.perf_counter()
        ec_time = end_time - start_time
        print(f"ec Time: {ec_time}")

        start_time = time.perf_counter()
        # 迭代数据加载器
        for epoch in range(num_epochs):
            for step in range(num_steps):
                features = dataset.__getitem__(step)
                features = features.to(device)
                fused_embeddings = fused_ec(features)
                fused_vals = []
                for _name, jt in fused_embeddings.items():
                    fused_vals.extend(jt.to_dense())
                torch.cat(fused_vals).sum().backward()

        end_time = time.perf_counter()
        fused_ec_time = end_time - start_time
        print(f"fused ec Time: {fused_ec_time}")
        print(f"speedup:{ec_time/fused_ec_time}")

