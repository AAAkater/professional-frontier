# Seq2Seq 与注意力机制解析

## 1 Seq2Seq 模型

### 1.1基本概念与应用场景

​Seq2Seq（Sequence to Sequence）作为通用的 Encoder - Decoder 框架，在自然语言处理及多模态任务中应用广泛 ，涵盖机器翻译（如将中文文本转化为英文文本 ）、文本摘要（提炼长文本核心内容 ）、聊天机器人（生成对话回复 ）、阅读理解（依据文本回答问题 ）、语音识别（将语音信号转为文字 ）、图像描述生成（为图像生成文字描述 ）、图像问答（结合图像内容回答问题 ）等场景，成为跨序列转换任务的基础架构。

​在机器翻译场景中，源语言句子作为输入序列，目标语言句子是输出序列。比如将 “我喜欢中国美食” 翻译成英文，Seq2Seq 模型需把中文序列转化为英文序列 “ I like Chinese cuisine”。文本摘要里，长文章是输入序列，提炼出的简短摘要为输出序列。如一篇关于科技发展的长文，模型可能生成 “近期科技在人工智能与量子计算领域取得显著进展” 这样的摘要。

![img](https://p3-flow-imagex-sign.byteimg.com/ocean-cloud-tos/image_skill/25afe0a2-87af-4d0c-ac82-a3d983a8bcd5_1750522429955504480_origin~tplv-a9rns2rl98-image-qvalue.jpeg?rk3s=6823e3d0&x-expires=1782058430&x-signature=e4OPvoUAYCSATjgG2qt9vz4SCQc%3D)

### 1.2模型起源与发展

​2014 年，Bengio 等 与 Bowman 等 先后提出基于 RNN/LSTM 的 Seq2Seq 模型用于机器翻译，开启了该框架在 NLP 领域的应用先河。

#### 1.2.1 RNN Encoder-Decoder（Bengio 等提出）

​模型包含两个 RNN，分别承担 Encoder（编码器）与 Decoder（解码器）职责，流程如下：

- **编码阶段**：对于输入序列 `x = [x₁, x₂, ..., x_T]`，RNN 编码器按顺序读取每个 `x_t` 并更新隐含层状态。当读取到序列结束符号 `EOS` 时，最终的隐含层状态 `c` 被生成，它代表整个输入序列的语义编码，是对输入信息的压缩表示。

- **解码阶段**：RNN 解码器负责生成输出序列 `y = [y₁, y₂, ..., y_T]`。基于隐含层状态 `h⟨t⟩` 预测下一个输出 `y_t`，其中 `t` 时刻的隐含层状态 `h⟨t⟩` 和输出 `y_t` 依赖上一时刻的输出 `y_{t-1}` 以及输入序列的语义编码 `c`。以句子 “苹果是红色的” 为例，当 RNN 编码器读取 “苹果” 时，会结合前一时刻隐含层状态（初始时刻为零向量）与 “苹果” 对应的词向量，通过激活函数（如 tanh）计算更新隐含层状态。接着读取 “是” 时，用上一时刻更新后的隐含层状态与 “是” 的词向量再次计算更新隐含层状态，依此类推。当读取到序列结束符号EOS，最终的隐含层`c`状态被生成，它代表整个输入序列的语义编码，是对输入信息的压缩表示，这个`c`包含了 “苹果是红色的” 这句话整体语义信息。

- 具体而言：

  - ```
    t时刻解码器隐含层状态计算公式为：
    ```

    ```markdown
    h⟨t⟩ = f(h⟨t-1⟩, y⟨t-1⟩, c)
    ```

  - 输出的概率分布为：

    ```markdown
    P(y_t | y⟨t-1⟩, y⟨t-2⟩, ..., y₁, c) = g(h⟨t⟩, y⟨t-1⟩, c)
    ```

  通过循环迭代生成目标序列，如图 1 所示。

  ![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvMTUzNTcxLzE1NjQzNzg5NTQ3MzgtNDA4MDU2ODUtMDVhMS00N2EzLTk2MGQtZDM3MDEzOThiMjA0LnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=356&originHeight=356&originWidth=1080&size=0&status=done&width=720#align=left&display=inline&height=356&originHeight=356&originWidth=1080&search=&status=done&width=1080)
*图 1：Seq2Seq 模型中 RNN Encoder-Decoder 结构示意*

#### 1.2.2 基于多层 LSTM 的 Seq2Seq（Bowman 等提出）

​该模型采用多层 LSTM 替换传统 RNN ，将输入序列映射为固定维数向量（类似上述语义编码 c ），再由另一个深层 LSTM 从该向量解码出目标序列 。多层 LSTM 借助多层网络结构，能捕捉更复杂的序列特征，提升对输入信息的表征能力，在处理长文本、复杂语义任务时表现更优。例如在处理法律条文类长文本翻译时，普通 RNN 可能因长距离依赖问题丢失关键语义，但多层 LSTM 凭借其门控机制（输入门、遗忘门、输出门），能更好地记住前面出现的重要信息，并在后续解码中有效利用。其流程可参考图 2 所示的序列转换逻辑（输入经编码、解码生成输出 ），输入序列进入多层 LSTM 编码器，每层 LSTM 对序列特征进行层层提取与转换，最终得到一个语义更为丰富、准确的固定维数向量，解码器的多层 LSTM 基于此向量与已生成输出逐步解码出目标序列。流程参考图 2 的序列转换逻辑。

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4ubmxhcmsuY29tL3l1cXVlLzAvMjAxOS9wbmcvMTUzNTcxLzE1NjQzNzg5NTQ4MzktNDkyYmViYTUtMzg1Ny00YWNkLTllNWUtOGY5MzUwYzBkZDVkLnBuZw?x-oss-process=image/format,png#align=left&display=inline&height=187&originHeight=187&originWidth=1080&size=0&status=done&width=720#align=left&display=inline&height=187&originHeight=187&originWidth=1080&search=&status=done&width=1080)
*图 2：Seq2Seq 模型中输入输出序列转换示意（以多层 LSTM 为例）*

## 2 注意力机制（Attention-based Model）

### 2.1机制提出背景与核心解决问题

​传统 Encoder - Decoder 框架存在局限性：编码器需将输入序列全部编码为固定维度向量 c，当输入过长或信息量庞大时，与当前任务无关信息会被强制纳入编码，干扰语义表达，无法实现 “选择性编码” 。例如在翻译 “在遥远的东方，有一个历史悠久、文化灿烂的国家，它拥有壮丽的山河、丰富的美食以及勤劳智慧的人民，这个国家就是中国” 这样长句子时，传统框架会将所有信息编码进 c，但在翻译某些部分（如 “美食” 相关内容）时，遥远东方等无关信息会干扰翻译准确性。注意力机制正是为解决此痛点诞生，它让模型聚焦关键信息，增强有效特征提取与利用，最早在机器翻译等基于 Encoder - Decoder 框架的 NLP 场景应用，后拓展至情感分析、句对关系判别等任务。

### 2.2注意力机制原理与类比理解

#### 2.2.1 类比人类视觉注意力

​类比人类识别他人时，眼睛会聚焦于脸部（关键信息），弱化身体其他区域。在语言模型中，通过对输入文本的每个单词赋予不同权重，让模型聚焦关键信息 —— 关键单词获更高权重，突出其对语义决策的影响。

#### 2.2.2 抽象计算逻辑

​对于输入 `Input`，存在 query 向量（当前解码的 “关注点”）、key-value 向量集合（输入序列各位置的 “特征 - 价值”）。通过计算 query 与 key 的关系（如点积、余弦相似度），为每个 value 分配权重，加权求和得到输出 `Output`，公式表示为：

```markdown
Output = ∑(Attention-Weightᵢ × Valueᵢ)
```

其中 `Attention-Weightᵢ` 是依据 query 与 `Keyᵢ` 计算出的权重，实现对关键信息的聚焦。

### 3 注意力机制的拓展应用

- 注意力机制最初为解决 Seq2Seq 问题设计，后经研究者拓展，应用到更多 NLP 任务：

  - **情感分析**：如关注 aspect 的情感分析模型 ATAE LSTM ，通过注意力聚焦与特定 aspect 相关的文本词汇，精准判断情感倾向（如评价 “手机拍照效果好” 中，聚焦 “拍照效果” 相关描述判断情感 ）。模型会计算 “拍照效果” 相关词汇与 query 向量（代表对拍照效果情感判断关注点）权重，若描述为 “清晰、色彩还原度高” 等正面词汇权重高，则判断为正面情感。

  - **句对关系判别**：像分析句对关系的 ABCNN 模型，利用注意力挖掘两个句子间关键关联信息，辅助判断句对是否同义、蕴含、矛盾等关系 。例如句对 “今天天气晴朗” 与 “今日阳光明媚”，注意力机制聚焦两句话中相似语义词汇（如 “今天” 与 “今日” 、“晴朗” 与 “阳光明媚”），计算权重，若相似词汇权重高，倾向判断为同义关系。

通过 Seq2Seq 框架实现序列到序列的基础转换，再结合注意力机制突破传统框架信息编码瓶颈，二者协同推动了自然语言处理等领域众多任务的发展与性能提升，为构建更智能、精准的语言与多模态模型奠定基础 。