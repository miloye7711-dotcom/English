# English Adaptive Trainer（英语听说自适应训练闭环）

一个开源的英语听说训练系统，实现 **摸底测水平 → 分级选材跟读 → 音素级纠音 → 阶段测试 → 自动升级/回炉** 的完整闭环。

参考项目：Echo-Loop（训练闭环）、Enjoy/everyone-can-use-english（素材与转写）、Echoic（音素级评分）、LinguaPal（听写评分）。

## 闭环流程

```
DIAGNOSE(摸底定位 CEFR 等级与弱项)
   → TRAIN(按弱项分级跟读、错词重练、间隔复习)
   → 练满 15 个单元自动触发 EVALUATE(同级全新题)
   → 听力≥85% 且 口语≥80% 自动升级；任一破降级线自动回炉
```

## 功能

- **听力听写评分**：词级 WER（Word Error Rate），逐词对照原文，标出写错的词。
- **口语跟读评分**：ASR 转写与原文对比，近似发音分（MVP 用 Whisper 词级置信度；生产升级路径为 wav2vec2 强制对齐 + 音素级后验）。
- **自适应分级**：CEFR A1–C1，按听力/口语双指标决定升级、降级或继续。
- **Web 界面**：浏览器内置 TTS 朗读句子，无需额外音频文件；录音、播放、错词高亮、进度展示。
- **核心逻辑零依赖可测**：评分与状态机不依赖外部模型，单测可直接跑。

## 快速开始

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
# 浏览器打开 http://127.0.0.1:8000
```

无需 GPU 即可运行完整闭环（口语评分用演示转写输入）。接 Whisper 真实录音转写时取消 `requirements.txt` 中 `faster-whisper` 的注释（RTX 5080 16GB 可本地跑 large-v3）。

## 测试

```bash
# 单测（评分与分级逻辑）
set PYTHONPATH=. && python tests\test_scorer.py

# 端到端闭环（摸底→训练→评估→升级/降级）
set PYTHONPATH=. && python tests\e2e_loop.py
```

## 项目结构

```
English/
├── app/
│   ├── main.py            # FastAPI 入口与 API
│   ├── scorer.py          # 词级 WER、听力/口语评分
│   ├── leveling.py        # CEFR 等级、升降级阈值
│   ├── state_machine.py   # DIAGNOSE/TRAIN/EVALUATE 闭环
│   ├── asr.py             # Whisper ASR 封装（可选）
│   └── data/sentences.json  # 分级句库（A2/B1/B2）
├── static/index.html      # Web 前端
├── tests/
│   ├── test_scorer.py      # 单元测试
│   └── e2e_loop.py         # 端到端闭环测试
├── requirements.txt
└── README.md
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | Web 界面 |
| GET | `/api/state` | 当前等级/阶段/弱项/进度 |
| GET | `/api/next` | 取下一道题（听力或口语） |
| POST | `/api/answer` | 提交答案（听写文本或跟读转写），返回评分与状态 |
| POST | `/api/reset` | 重置进度 |

## 评分阈值（可调）

| 指标 | 升级线 | 降级线 |
|---|---|---|
| 听力词正确率 | ≥ 0.85 | < 0.65 |
| 口语发音分 | ≥ 0.80 | < 0.60 |

## 后续路线

- 接入 wav2vec2 强制对齐，做音素级错音高亮（/θ/ 误读为 /s/）。
- 接入 Piper TTS 批量生成分级音频，替换浏览器 TTS。
- 间隔重复（SRS）错词队列。
- 扩充 CEFR-J 分级句库（A1–C1 约 6000 句）。
- 用 speechocean762 数据集校准评分阈值。

## License

MIT
