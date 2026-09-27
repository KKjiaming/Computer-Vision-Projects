# 深智城-举手表决算法SDK

## SDK功能支持

1.  人体关键点检测，举手计数

## Release Note：

#### 1.2.0
1. 改进算法pipeline，增加检测模型，优化人员跟踪和举手计数逻辑
2. 模型用Tensorrt FP16推理

#### 1.1.0
1. 增加时间段内举手人数统计接口

#### 1.0.0

- 新增功能:
1. 封装人体关键点检测模型，模型加密


## 开始项目

#### 代码结构

```bash
.
├── cmake                              # 配置第三方库
├── CMakeLists.txt                     # 项目CMakeList
├── README.md                          # 项目说明文档
├── thirdParty                         # 项目第三方依赖库
├── tutorials                          # 项目测试demo
└── HandsUpDetectModule                # SDK主要代码
    ├── CMakeLists.txt                 # 用来编译SDK动态
    ├── include                        # SDK接口头文件
    ├── internal                       # 算法层代码，由接口层调
    ├── lib
    │   └── libHandsUpDetectModule.so  # SDK动态连链接库
    └── src                            # 接口层代码，功能接口的实现
```

#### 第三方依赖库
```bash
thirdParty
├── smdnnmodel
├── InferModule
├── ONNXRuntime    
├── TensorRT         
└── OpenCV       
```

#### 编译demo
```bash
mkdir build
cd build
cmake ..
make -j
```

#### 运行demo
```bash
├── models                   # 模型
├── test_data                # 测试视频
└── HandsUpDetectCppSDK      # SDK

cd bin
export LD_LIBRARY_PATH=../thirdParty/ONNXRuntime/lib/:../thirdParty/OpenCV/lib:../thirdParty/TensorRT/lib:../thirdParty/smdnnmodel/lib/:../thirdParty/InferModule/lib/
./demo
```

#### demo示例
<div align="left">
    <img src="tutorials/example.jpg" width="960" alt="logo"/>
</div>


