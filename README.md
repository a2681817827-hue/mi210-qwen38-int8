# MI210 Qwen3.8-27B INT8 + DFlash2

已暂停进一步优化，保存可恢复的单 MI210 配置。本轮完整回归通过的配置在
48,081 输入、512 输出、C1 请求下，排除 TTFT 的五轮生成中位数为
**102.56 token/s**。历史最佳107.73；120/110目标尚未达到。

## 保存的镜像和容器

- 镜像：`int8-vllm-mi210:saved-container-20261008`
- 完整ID：`sha256:23fe3b8ed7cb2dca13cee345a233970b0cc0b165a267bc6d75c3ec4fdc55dbf7`
- 源容器：`mi210-qkpv-screen-20261008-control-after`，已停止并保留
- 新服务容器默认名：`mi210-serving`

52/52受约束答案、文字/视觉smoke、16/16并发答案、120K三位置检索及生成
通过；前后控制52条多样输出完全一致，记录的top20粗化KL为0。
这些检查不是完整下游准确率评估；不同温度、负载和软件版本会改变速度。

## 启动

先导入已备份镜像，恢复模型和编译缓存，再执行：

```bash
export MI210_API_KEY='your-new-api-key'
bash scripts/start-mi210-saved-20261008.sh --dry-run
bash scripts/start-mi210-saved-20261008.sh
```

默认127.0.0.1:8080，TP1、服务容量8但性能口径C1；DFlash NS13；
query/KV tile=5/64、flash splits32、draft splits16；INT8 head、Native GDN、
INT8-QK、调优CK和草稿窗口开启；两侧G128 INT8 KV、BF16残差、FP32 Mamba；
上下文131072、batched tokens2048、FULL_DECODE_ONLY图。新QK/PV、归一化
选型策略、量化alias和扩展CK均关闭。启动脚本包含完整Docker参数。

[结果、参数和备份恢复详解](docs/recipes/mi210_saved_20261008.md)。
[完整原始数据下载](https://github.com/a2681817827-hue/mi210-qwen38-int8/releases/tag/results-20261008)
包含脱敏原始测量、失败记录、算子门禁、分析、操作源码和校验和。
镜像、模型权重和编译缓存不放入Git；保存在本机完整备份中。

## 重装后恢复

将完整备份先复制到外接盘/NAS/另一台主机。只有Docker镜像不包含挂载的
模型和缓存，只有container export也不包含挂载卷。

```bash
# 在重装前，目标目录必须新建；本次完整备份约64GiB，建议使用128GB以上外部存储
bash scripts/backup-mi210.sh /mnt/backup/mi210-new --all-images
# 重装后恢复Docker、AMD驱动及/dev/kfd，再运行备份内的恢复脚本
bash /mnt/backup/mi210-new/handoff/restore-mi210.sh /mnt/backup/mi210-new
```

本机已保存目录见完整报告；重装后镜像ID和模型内容校验，容器由命令重建，
容器ID无需与原来相同。不要省略已验证的编译缓存。

## 本轮未选用方案

显式归一化控制106.85 tok/s，固定归一化+QK/PV为101.46；二者52条多样
输出及概率与原控制完全一致。未固定归一化的新编译候选90.82，仅5/52
多样输出一致。比较发现自动调优改变RMSNorm求和分块；显式选型恢复输出。
QK/PV算子约1.4倍加速没有转化成整模型提升。split8/16/64全部超过原
0.0005数值门槛而拒绝；split32变化深度图重放逐位一致。
可选实验源码保留，默认不开启，不宣称已达到120或110。

## 来源和许可

框架来自 [curvedinf/int8-vllm](https://github.com/curvedinf/int8-vllm)，参考
[davetha/aiter-cdna2](https://github.com/davetha/aiter-cdna2)、
[davetha/mi210-llm-stack](https://github.com/davetha/mi210-llm-stack)、
[davetha/mi210-vllm](https://github.com/davetha/mi210-vllm)、
[jebchowDCGPU/vllm-mi210](https://github.com/jebchowDCGPU/vllm-mi210)。
固定提交和各算子许可见`docker/mi210/sources.json`、`operators/`及NOTICE。
Dockerfile仅复用已保存本地镜像，不是可从公共Ubuntu全量构建的配方。
