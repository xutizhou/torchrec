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
            EmbeddingBagConfig(
                num_embeddings=hash_size,
                embedding_dim=embedding_dim,
                name="table_0",
                feature_names=["feature_0"],
            ),
        ]

        fused_ec = FusedEmbeddingBagCollection(
            tables=embedding_configs,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
            device=device,
        )

        ec = EmbeddingBagCollection(tables=embedding_configs, device=device)

        #        0       1        2  <-- batch
        # "f1"   [] [0]    [0,1]
        # "f2"   [1]    [0,1]    []
        #  ^
        # feature

        opt = optimizer_type(ec.parameters(), **optimizer_kwargs)


        import time

        # 创建数据集和数据加载器
        dataset = CustomDataset(num_steps,hash_size,batch_size=batch_size, device=device)
        dataset = ebc_benchmarks_utils.get_random_dataset(
            batch_size=64,
            num_batches=10,
            num_dense_features=1024,
            embedding_bag_configs=embedding_configs,
        )
        cnt=0
        for data in dataset:
            cnt+=data.sparse_features.values().shape[0]
        print(f"dataset size: {cnt}")
        start_time = time.perf_counter()
        # 迭代数据加载器
        for epoch in range(num_epochs):
            for data in dataset:
                features = data.sparse_features
                features = features.to(device)
                opt.zero_grad()
                sequence_embeddings = ec(features)
                vals = []
                for _name, param in sequence_embeddings.to_dict().items():
                    vals.append(param)
                torch.cat(vals, dim=1).sum().backward()            
                opt.step()

        end_time = time.perf_counter()
        ec_time = end_time - start_time
        print(f"ec Time: {ec_time}")
        
        start_time = time.perf_counter()
        # 迭代数据加载器
        for epoch in range(num_epochs):
            for data in dataset:
                features = data.sparse_features
                features = features.to(device)
                fused_embeddings = fused_ec(features)
                fused_vals = []
                for _name, param in fused_embeddings.to_dict().items():
                    fused_vals.append(param)
                torch.cat(fused_vals, dim=1).sum().backward() 

        end_time = time.perf_counter()
        fused_ec_time = end_time - start_time
        print(f"fused ec Time: {fused_ec_time}")
        print(f"speedup:{ec_time/fused_ec_time}")
