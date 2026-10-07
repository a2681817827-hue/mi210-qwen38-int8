# MI210 Qwen3.8-27B INT8 + DFlash2

单张 AMD MI210 的 INT8 优化和复现项目。目标为 48K 输入、单请求生成速度
110–120 tok/s。当前最佳性能候选约 **103 tok/s**，目标尚未达到。

保留 GPTQ INT8 W8A8 GS128 主模型和 DFlash2 草稿、两侧 G128 INT8 KV、
BF16 残差计算、FP32 Mamba 状态和 GPU 解码图。当前本地模型为
`Qwen3.8-27B-PTQR-R10S60` + `dflash2-ptqr-r1`。

[完整测量记录](docs/recipes/mi210_measured_20261007.md) 包含算子组合、镜像 ID、
参数、接受率、数值检查范围和失败实验结论。
[源码快照与原始计时数组](docs/recipes/mi210-artifacts/20261007/) 便于核对结果。

## 启动已测量的本地版本

该项目复用主机上已优化的 Docker 镜像和编译缓存；Dockerfile 是本地镜像
的参数覆盖层。当前不提供可从公共注册表直接拉取的完整镜像。

```bash
export MI210_API_KEY='your-local-api-key'
bash scripts/start-best-measured.sh --dry-run
bash scripts/start-best-measured.sh
```

默认 API 监听 `127.0.0.1:8080`，主模型目录从 `/data/models` 只读挂载。
环境变量 `MODEL_ROOT`、`MI210_IMAGE`、`MI210_COMPILE_CACHE`、`GPU_NODES` 可
指定已有资源。运行前用 `docker image inspect` 检查镜像 ID 是否与记录一致。

## 当前组合参数

| 参数 | 值 |
|---|---|
| TP / 测量并发 | 1 / 1 |
| DFlash2 推测长度 | 13 |
| Query tile / KV tile / 分区数 | 5 / 64 / 32 |
| INT8 LM head / native GDN / INT8 QK / tuned CK | 全部开启 |
| KV | 主模型与草稿均 `int8_block_g128` |
| Mamba 状态 | FP32 |
| 最大上下文 / 批处理 tokens | 131072 / 2048 |
| 解码图 | FULL_DECODE_ONLY，C8 捕获容量 |

## 验证和记录方式

性能使用同一份 48081-token 输入、512-token 输出、五轮中位数，生成速度
排除首 token 等待时间，同时记录端到端速度和 TTFT。新组合先通过真实
形状数值及图回放检查，再测试答案、视觉、并发和长上下文。

当前性能候选已通过 52 条受约束答案检查和视觉 smoke，复测的 52 条多样化
输出逐位一致。相对旧 query10 的完整下游质量验证仍未完成，因此不能
把它标为正式生产配置。算子微测加速也不能直接当作模型 tok/s。

失败容器和镜像在保存结果后清理；模型权重、最好的镜像及复现缓存保留。
后续成功组合会更新记录和启动脚本，标明真实达标状态。

## 来源

框架基于 [curvedinf/int8-vllm](https://github.com/curvedinf/int8-vllm)，参考
[davetha 的 MI210 记录](https://github.com/davetha/mi210-llm-stack)、
[aiter-cdna2](https://github.com/davetha/aiter-cdna2)、
[mi210-vllm](https://github.com/davetha/mi210-vllm)，并持续审查 AMD 官方算子
实现。具体 commit 与许可见源码快照中的 `sources.json`。未包含模型权重
或登录凭据。
