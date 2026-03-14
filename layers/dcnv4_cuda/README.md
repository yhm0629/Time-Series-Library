# DCNv4 1D CUDA扩展

基于性能分析的CUDA优化实现，针对时间序列预测中的可变性卷积进行深度优化。

## 性能分析结果

基于对现有DCNv3_1D实现的性能分析，识别出以下主要瓶颈：

| 规模 | 平均时间 | 估计FLOPs | 估计性能 | GPU内存 |
|------|----------|-----------|----------|---------|
| 小规模 (2,64,100) | 1.33 ms | 3.99 M | 3.00 GFLOPS | 8.73 MB |
| 中等规模 (8,128,256) | 1.34 ms | 148.59 M | 111.18 GFLOPS | 19.56 MB |
| 大规模 (16,256,512) | 6.98 ms | 2261.61 M | 324.15 GFLOPS | 97.84 MB |

### 主要性能瓶颈
1. **grid_sample**: 离散内存访问，缓存不友好
2. **多次reshape/permute**: 额外内存拷贝
3. **Python层面坐标计算**: CPU-GPU同步可能
4. **多个独立操作**: 缺乏算子融合

## 架构设计

### 目录结构
```
layers/dcnv4_cuda/
├── __init__.py              # 包初始化
├── dcnv4_1d_cuda.py         # Python包装器
├── setup.py                 # 构建配置
├── cuda/                    # CUDA代码
│   ├── dcnv4_1d_kernel.cuh  # 内核头文件
│   ├── dcnv4_1d_kernel.cu   # 内核实现
│   └── dcnv4_1d_ops.cpp     # C++包装器
└── tests/                   # 测试代码
    └── test_correctness.py  # 正确性测试
```

### 分阶段优化计划

#### 阶段1：基础CUDA实现（1-2周）
- 实现基础采样内核（替换grid_sample）
- 优化内存布局（NCHW连续存储）
- 目标：2-3倍性能提升

#### 阶段2：内存访问优化（2-3周）
- Shared Memory分块加载
- 访存合并优化
- 寄存器优化
- 目标：5-10倍性能提升

#### 阶段3：算子融合（3-4周）
- 全融合内核：offset预测 + 采样 + 聚合
- 在线softmax计算
- 中间结果寄存器存储
- 目标：10-20倍性能提升

## 安装与使用

### 环境要求
- CUDA Toolkit >= 11.7
- PyTorch >= 2.0.0
- ninja >= 1.10.0
- NVIDIA GPU (Compute Capability >= 6.1)

### 安装步骤

1. **设置CUDA环境变量**（Windows）：
```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
```

2. **安装构建工具**：
```bash
pip install ninja
```

3. **编译CUDA扩展**：
```bash
cd layers/dcnv4_cuda
python setup.py build_ext --inplace
```

4. **安装到当前环境**：
```bash
pip install -e .
```

### 基本使用

```python
import torch
from layers.dcnv4_cuda import DCNv4_1D_CUDA

# 创建CUDA优化层
dcn = DCNv4_1D_CUDA(
    channels=64,
    kernel_size=3,
    group=4,
    use_cuda=True  # 启用CUDA优化
).cuda()

# 前向传播
x = torch.randn(2, 64, 100).cuda()
output = dcn(x)
print(f"输出形状: {output.shape}")
```

### 兼容性使用

```python
# 自动回退到PyTorch实现
dcn = DCNv4_1D_CUDA(
    channels=64,
    kernel_size=3,
    group=4,
    use_cuda=True  # 如果CUDA不可用，自动使用PyTorch实现
)

# 在CPU上运行
x_cpu = torch.randn(2, 64, 100)
output_cpu = dcn(x_cpu)  # 使用PyTorch实现

# 在GPU上运行
x_gpu = torch.randn(2, 64, 100).cuda()
output_gpu = dcn(x_gpu)  # 使用CUDA实现（如果可用）
```

## API文档

### `DCNv4_1D_CUDA` 类

```python
class DCNv4_1D_CUDA(
    channels: int = 64,
    kernel_size: int = 3,
    stride: int = 1,
    pad: int = 1,
    dilation: int = 1,
    group: int = 4,
    offset_scale: float = 1.0,
    act_layer: str = 'GELU',
    norm_layer: str = 'LN',
    center_feature_scale: bool = False,
    remove_center: bool = False,
    use_cuda: bool = True
)
```

**参数**:
- `channels`: 输入输出通道数
- `kernel_size`: 卷积核大小
- `group`: 分组数
- `offset_scale`: 偏移量缩放因子
- `use_cuda`: 是否使用CUDA优化（如果不可用则自动回退）

**方法**:
- `forward(x: torch.Tensor) -> torch.Tensor`: 前向传播

### 工具函数

```python
from layers.dcnv4_cuda import create_dcnv4_1d

# 快速创建层
dcn = create_dcnv4_1d(
    channels=64,
    kernel_size=3,
    use_cuda=True
)
```

## 测试

### 运行测试套件

```bash
cd layers/dcnv4_cuda/tests
python test_correctness.py
```

### 测试内容
1. **初始化测试**: 验证参数初始化和形状匹配
2. **CPU前向传播测试**: 验证CPU模式下的数值正确性
3. **GPU前向传播测试**: 验证GPU模式下的数值正确性
4. **性能测试**: 对比CUDA和PyTorch实现的性能
5. **内存使用测试**: 验证内存使用情况

## 性能优化技巧

### 1. 选择合适的阶段
```python
# 根据需求选择优化阶段
# 阶段1：基础优化（兼容性好）
# 阶段2：内存优化（性能好）
# 阶段3：全融合（极致性能）

# 在dcnv4_1d_cuda.py中修改：
# launch_dcnv4_kernel(..., stage=2)  # 使用阶段2优化
```

### 2. 调整线程块配置
```cpp
// 在dcnv4_1d_kernel.cuh中调整：
constexpr int MAX_BLOCK_SIZE = 256;  // 根据GPU调整
constexpr int SHARED_MEM_SIZE = 48 * 1024;  // 根据GPU调整
```

### 3. 使用混合精度
```python
# 启用FP16支持（需要GPU支持）
with torch.cuda.amp.autocast():
    output = dcn(x.half())
```

## 故障排除

### 常见问题

1. **CUDA扩展编译失败**
   - 检查CUDA Toolkit是否安装
   - 检查环境变量`CUDA_PATH`是否设置正确
   - 检查PyTorch CUDA版本是否匹配

2. **运行时CUDA错误**
   - 检查GPU内存是否足够
   - 检查输入张量是否在GPU上
   - 检查CUDA驱动版本

3. **性能不达预期**
   - 检查是否真正使用了CUDA内核（查看警告信息）
   - 调整线程块配置
   - 使用性能分析工具（nsight）

### 调试信息

启用调试输出：
```python
import warnings
warnings.simplefilter('always')  # 显示所有警告
```

## 开发指南

### 添加新优化

1. **在内核头文件中声明新内核** (`dcnv4_1d_kernel.cuh`)
2. **在内核实现文件中实现新内核** (`dcnv4_1d_kernel.cu`)
3. **在C++包装器中添加接口** (`dcnv4_1d_ops.cpp`)
4. **在Python包装器中添加调用** (`dcnv4_1d_cuda.py`)
5. **添加测试用例** (`tests/`)

### 性能分析

使用Nsight Systems进行性能分析：
```bash
nsys profile --stats=true python test_performance.py
```

## 路线图

### v0.1.0 (当前)
- [x] 基础CUDA内核框架
- [x] 兼容性包装器
- [x] 基本测试套件

### v0.2.0 (计划中)
- [ ] 内存访问优化（Shared Memory）
- [ ] 访存合并优化
- [ ] 性能基准测试

### v0.3.0 (计划中)
- [ ] 算子融合
- [ ] 在线softmax
- [ ] 混合精度支持

### v1.0.0 (计划中)
- [ ] 生产就绪
- [ ] 完整文档
- [ ] 性能优化指南

## 贡献指南

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 许可证

本项目基于MIT许可证 - 查看 [LICENSE](../LICENSE) 文件了解详情。

## 致谢

- 基于 [InternImage](https://github.com/OpenGVLab/InternImage) 的DCNv4设计
- 受 [FlashAttention](https://github.com/Dao-AILab/flash-attention) 的内存优化启发
- 感谢所有贡献者和用户

## 联系方式

如有问题或建议，请：
- 提交 [Issue](https://github.com/yourusername/time-series-library/issues)
- 发送邮件至 your-email@example.com

---
*最后更新: 2026年2月15日*
*基于实际性能分析数据开发*
