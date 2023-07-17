# RingRWKV
修复Transformer官方库中RWKV的适配问题，支持RWKV所有系列模型在转换后，通过RingRWKV库，与其他transfomer模型一样简单方便地部署和微调。<br>


RWKV-4-World的Hugface格式，因新版World的tokenizer较之前Raven\Pile版本有较大变化，因而需要进行新版HF适配 ringrwkv兼容了原生rwkv库和transformers的rwkv库，同时新添入World版本的配置及代码（支持1.5B，3B，7B等全部参数类别），并修复了原HF的RWKV在 Forward RWKVOutput时的细微问题，主要是引入和明确last_hidden_state。<br>


RingRWKV 理论上支持包括Raven\Pile\PilePlus\World\World-CHN等全系列RWKV模型，在转换为HF格式后的使用，但不排除需要进一步适配的可能<br>

RWKV的Raven模型转为HF格式并全量微调：https://github.com/StarRing2022/HF-For-RWKVRaven-Alpaca<br>
RWKV的World模型转为HF格式并全量+增量微调：https://github.com/StarRing2022/HF-For-RWKVWorld-LoraAlpaca<br>

一个基于RWKV-4-World-1.5B的使用范例：<br>

HF开源地址：https://huggingface.co/StarRing2022/RWKV-4-World-1.5B
