## PI：江纬 Wei Jiang
## 当前研究课题：多模态内容感知、生成及个性化时尚创意应用场景研究
## 理论框架依托：认知计算广告 认知计算时尚
## 核心相关工作
### 1. 中国认知计算广告

### 2. 认知计算时尚

## 子课题
### 子课题及论文题目（拟） - x - Fashion Hallucination is all you need
* 中文标题：注意力机制在AIGC在认知计算广告内容创意中的应用场景研究（需结合经典传播学注意力等理论）
* 状态：可作为社会科学研究论文

### 子课题及论文题目（拟） - x - Fashion Alignment is All you need
* 

### 子课题及论文题目（拟） - x - Fashion Attention is all you need
* 中文标题：注意力机制在AIGC在认知计算广告内容创意中的应用场景研究（需结合经典传播学注意力等理论）
* 状态：可作为社会科学研究论文

### 子课题及论文题目（拟） - x - Fashion Bias is All you need
* 

![](./img/sub.topic.3.png)
### 子课题及论文题目（拟） - x - “创意垂类模型"在垂直领域的应用研究
#### 摘要 Abstract
* This paper examines the realm of created models, which have the ability to generate text, images, and videos in parallel. While these models hold great promise, they also come with limitations. One significant limitation is their limited capability in expressing creativity. However, this constraint can potentially trigger inner model training and iteration. The research delves into the architecture, functionality, and applications of these created models, analyzing both their advantages and drawbacks. It explores ways to overcome the limitations and enhance the creative expression capabilities of these models. By understanding the challenges and opportunities associated with created models, this paper aims to contribute to the development and improvement of content generation and creative expression in a multi-modal world.

#### 关键词 Keywords
* 

#### 作者 Authors
* 

#### 同行评审
* 段淳林（deadline: 2024 fall）

#### 组织 Organization
* South China University of Technology

#### 简介 Introduction
* 

#### 相关工作 Related Work
* Ten years ago, language models took their first steps with Google's n-gram paper[1]. This approach employed a vast amount of web data to model language, considering n-grams like two-gram or three-gram. Following this, the word2vec paper[2] came into the picture. It modeled the world as a higher-dimensional vector space, where entities with similarity would exhibit a close cosine similarity score. This marked the beginning of a long journey in the field of language models[3]. The language model then entered a new era with the Transformer paper titled "Attention is All You Need"[4]. Google researchers proposed an encoder-decoder architecture or later variants for encoding and decoding text. This model demonstrated the remarkable ability to generate surprisingly coherent conversations, laying the foundation for subsequent works such as ChatGPT[5] in 2022. ChatGPT's emergence surprised and captivated the world at scale. In addition to these, leading industry models like Google's Gemini have been developed. In China, works like Doubao[6] also deserve mention. Doubao showcases advanced Chinese language processing capabilities and has made significant contributions to the field.
* In addition to the advancements in language models, there have been remarkable developments in other generation created models. For instance, Stable Diffusion[7] has made significant contributions in the field of image generation. It employs innovative techniques to create high-quality images with great detail and realism. Another notable model is Flux[8] by the Black Forest Lab, which also excels in generating images. Moreover, in the realm of video generation, models utilizing improved architectures like diffusion-transformer(short for DiT) have emerged. These models have the potential to revolutionize the way videos are created, offering new possibilities for creative expression and content production. By combining the power of transformers and diffusion processes, they can generate videos with enhanced visual quality and coherence[9]. These developments in image and video generation models complement the progress in language models, opening up new frontiers in the field of created models and expanding the boundaries of what can be achieved in content generation.
* Another interesting aspect of created models is the emergence of those that focus on generating music snippets or music demos[10]. These models have the potential to inspire musicians and composers, providing new musical ideas and elements. They can contribute to the creation of unique and engaging musical compositions. In addition, there are created models that generate code[11]. These models can produce code snippets or even complete programs that can be executed in environments like Python. This has significant implications for software development, as it can assist developers in generating code effectively and efficiently, leading to faster development cycles and much more.

#### 现有典型垂类模型及所存问题 Current Creative Models and Problems
* 

#### 所提出关于“垂类模型”的创意设计新范式 New Design Paradigm for Creative Models
* 

#### 实验原型系统及应用场景 Real-world Applications and Prototypes
* 

#### 结论 Conclusion
* Created models have made remarkable progress in various domains, including text generation, image and video production, code generation, and music creation. Each of these areas has seen the emergence of powerful models that offer unique capabilities and opportunities for creative expression. TODO: However, as these models focus on different domains, there is a need for a unified interface and application to use them coherently and efficiently.

#### 参考资料 References
1. n-gram
2. word2vec
3. classic textbook in the field of natural language processing
4. Attention is All You Need
5. ChatGPT official website
6. Doubao official website
7. Stable Diffusion official website
8. Flux official website
9. Sora official website
10. Suno official website
11. Gemini Code Assist official website

#### 附加信息
* paper 3 - model.md
* 关键词及着眼点：垂类模型
* 写作思路：使用RAG技术使生文通用模型足够垂类，使用微调技术使生图通用模型足够垂类。
* 基于认知计算广告理论，着眼于于垂类大模型RAG（生文模型）、微调（生图、生视频模型）等技术，系统阐明如何高效挖掘新知识，使通用知识大模型插上垂类翅膀，并提及提示词工程及智能词元（token）生成，即大模型是通过什么架构去处理多模态的输入及输出的。
* 创新点1：多模态混合专家模型 multi-modal mixed of expert model. The arch contains the broker model and a set of worker models such as text-to-image model, text-to-video and text-to-music model etc.
* 创新点2：on the side of the broker model, add long context & memory support for it.
* 创新点3：on the side of the worker models, add the ability to learn from the broker model's feedback.
* 创新点4：TODO: maybe ADD support for edge & cloud model as well?
* 案例1：如何使用“通用生文大模型 + RAG“做带有风格的时尚内容创作（文本）
* 案例2：如何使用“通用生图大模型 + 微调”做带有风格的时尚内容创作（图片）
* 案例3：如何使用“通用生视频大模型 + 微调”做带有风格的时尚内容创作（视频）
* 所投期刊：新闻与传播如 TODO: 未确定

### 子课题及论文题目（拟） - x - “创意智能体”在垂类场景中的应用研究
![](./img/sub.topic.4.png)
#### Abstract
* This paper begins by clarifying the concept that an intelligent agent is distinct from large language models or large vision models. An intelligent agent can be metaphorically described as having a "brain" and "body" and it is designed to perform specific tasks. The control of an intelligent agent often relies on large language models or large visual models. The paper then presents different types of intelligent agents, including those for cost reduction and efficiency improvement and those for content creation. The focus of this paper is on content creation. An in-depth exploration of these content creation intelligent agents are provided, while specific case studies are yet to be determined.
#### 摘要
* 

#### 状态
* 构思中
* 社会科学论文

#### 关键词 Keywords
* AI Agent; LLM; Large Vison Model; workflow; creative models; scene;

#### 作者 Authors
* 

#### 同行评审
* 全宇晖（deadline: 2024 fall）

#### 组织 Organization
* South China University of Technology
* 华南理工大学

#### 理论构建
* 创新点：
* 创意智能体人物设定：计算机科学教授帮助学生优化算法
* 智能体间社交：真人及类人在一起协作和交流

#### 技术路径（含实验原型系统及应用场景）
* 

#### 相关工作 Related Work
##### history
* The history of AI agents traces back to early works that laid the foundation for this field. One of the most important papers is "A Logical Calculus of the Ideas Immanent in Nervous Activity" by Warren McCulloch and Walter Pitts[1], published in the 1940s. This paper introduced the concept of neural networks and their potential for computing, inspiring subsequent research in AI. In the 1950s, Alan Turing's "Computing Machinery and Intelligence"[2] proposed the Turing test, a milestone in evaluating machine intelligence. This work set the stage for the development of intelligent agents that could interact with humans. Moving forward, "Deep Reinforcement Learning from Human Preferences" by Jan Leike et al.[3] in recent years has made a significant impact. It demonstrated how agents can learn from human preferences and adapt their behavior in complex environments, advancing the capabilities of AI agents in decision-making.
##### AI Agent: code generation & exection
* One paper that focuses on agents that can not only output text but also execute code is "Program Synthesis with Large Language Models" by Mark Chen et al[4]. This paper explores the use of large language models for program synthesis, where the agent can generate code based on natural language descriptions. It shows how language models can be trained to understand programming tasks and generate executable code, blurring the line between text generation and code execution. Another relevant paper could be "Code Generation with Transformer Models" by Yuxin Wang et al[5]. This paper investigates the use of transformer models for code generation, enabling agents to produce code snippets or complete programs. It highlights the potential of combining natural language understanding and code generation capabilities in intelligent agents.
##### AI Agent: RL learning & interaction with the env
* Another aspect of related work focuses on agents that not only have a large language model as a "brain" but also possess a "body." These agents are designed to exhibit dynamic movement and motivation, and have the capability to interact with the outside world. In the realm of reinforcement learning, papers such as "Deep Reinforcement Learning for Robotic Manipulation" by Sergey Levine et al.[6] explore how agents can learn complex motor skills and interact with the physical environment through reinforcement learning. This work shows how agents can be trained to perform tasks such as grasping objects and navigating in three-dimensional spaces. Another relevant paper could be "Reinforcement Learning for Interactive Agents" by Peter Stone et al[7]. This paper examines the use of reinforcement learning for agents that interact with humans or other agents in dynamic environments. It highlights the challenges and opportunities of designing agents that can adapt and respond to changes in the environment and the actions of others.
##### AI Agent: the creative way
* In the domain of AI agents for creativity, there are several notable papers. For example, "Generative Adversarial Networks for Artistic Creation" by Ian Goodfellow et al. showcases how generative adversarial networks (GANs) can be used to generate artistic works[8]. These agents have demonstrated some degree of imagination in creating images that can be shared and appreciated. Another paper could be "Creativity in Neural Network Art Generation" by [Author's Name] [9]. This work explores the creative capabilities of neural networks in generating art and highlights the potential for such agents to push the boundaries of artistic expression. While these papers show great promise in the area of AI-driven creativity, it is important to note that there is still much work to be done. There are challenges in ensuring that the generated art is truly original and not simply mimicking existing styles[10]. Additionally, there is a need to further explore how these agents can collaborate with human artists and enhance the creative process rather than replace it[11].

#### 现有典型智能体及所存问题 Current Intelligent Agents and Problems
* 

#### 规范与治理
* 

#### 结论 Conclusion
* 

#### 参考资料 References
1. foundation work
2. turing test
3. RL concept
4. code generation protype
5. transformer based model
6. robotic manipulation
7. interaction in dynamic env
8. GAN for artistic creation
9. Creativity in art work generation
10. original art work or not
11. agent collaboration

#### 附加信息
* paper 4 - scene.md
* 关键词及着眼点：“创意型智能体”、“垂直落地场景”
* 写作思路：在开篇给出智能体的定义及行动流程，即环境交互 -> 感知信息 -> 思考 -> 采取行动。由客户需求倒推如何设计落地场景及相关智能体，并且在设计与实现智能体过程中要非常重视智能体的价值对齐问题，写作过程中需强调人机协作，强调目前智能体能达到一定的智能推理能力，但尚不够。
* 基于认知计算广告理论
* 进一步提出在不同垂直场景及细分需求下的技术选型，以满足客户需求。
* 创新点1：不强调 使用智能体以降本增效，强调 使用智能体以提升内容创作及灵感获取能力
* 创新点2：强调 提示词工程、知识库、分数的“融合微调”核心技术（案例：如何在实际工程中价值对齐一个时尚买手智能体的音容相貌乃至部分时尚大脑职能）
* 难点：多智能体如何协作与感知？如何有效从个体智能到群体智能？
* TODO: 案例1：如何设计一个智能体，她的目标是提升广告投放效果？
* TODO: 案例2：如何设计一个智能体，她的目标是输出源源不断的创意及灵感？
* TODO: 案例3：如何设计一个时尚买手智能体，她的目标是对人群或个体时尚进行评价，评分并作搭配推荐
* TODO: 案例4：华为HAS 2024 AIGC 宣传视频
* TODO: 案例5：新增 playground.com 为设计师而设计，自由设计任何东西
* TODO: 案例6：新增 civiti.com 开源生成式人工智能之家
* TODO: 实验设计：如何设计一个智能体，做落地应用场景相匹配决策（决策如智能排期）。如此广告在此刻是否生成，此广告是否投放在某个用户的折扣圈中？
* 论文中不会出现1：把智能体上升到具身智能，并且引出机器人。原因：目前业界机器人主要用于降本增效，和内容创意关系不大。
* 所投会议：计算机科学如 SIGIR, CIKM et. al.

### 子课题及论文题目（拟） - x - Improved Methods for Model Pruning 改进的模型剪枝方法
![](./img/sub.topic.8.v2.png)
#### Abstract
Model pruning is presented as a performance optimization technique for large language and vision models. This technique aims to identify and remove neurons, connections unlikely to lead to the contribution during the machine generation phase. Our goal is to obtain a much smaller and faster foundational model that can quickly generate AIGC content almost as good as those of the unpruned models. Through careful analysis of the weights, bias, activations and other potential indicators, an improved algorithm based on new indicators have been proposed and evaluated. Emprical results show that our proposed algorithm achieves significant improvement in different pruning ranges over previous STOAs.
#### 摘要
模型剪枝是一种用于大语言和视觉模型的性能优化技术。该技术旨在识别并移除在机器生成阶段不太可能产生贡献的神经元和连接。我们的目标是获得一个更小、更快速的基础模型，能够以接近未剪枝模型的水平快速生成AIGC内容，这些内容包括文本、图片、视频及代码。通过对权重、偏置、激活函数和其他潜在指标的深入分析，我们提出并评估了一种基于新指标的改进算法。实验结果表明，在不同剪枝范围内，我们提出的算法相比之前的最新技术(SOTA)取得显著改进效果。
#### 状态
* Under Review and will submit to ICLR 2025 main conference
* 自然科学论文

### 子课题及论文题目（拟） - x - Improved Methods for Model Tiering
![](./img/sub.topic.10.png)
* TODO: 结合MoE

### 子课题及论文题目（拟） - x - AIGC创意灵感生成平台及商业闭环研究（拟）
![](./img/sub.topic.11.png)
![](./img/aigc.image.gen.platform.png)
* TODO: 在上图“代表性AI生成平台”基础上提供C端用户半自主参与下的AIGC服装设计机理图。同时结合机理图提供对于用户半自主参与下的AIGC服装设计过程的英文描述及参考文献。时间节点为：10月8日。联系人：张博士后
* 状态：作为社会科学论文提交

### 子课题及论文题目（拟） - x - AI Agents will significantly hurt the job market for programmers！(meaning losing jobs and being replaced quickly)
* 更新日期：20240922
* 中文标题：AI智能体会极大伤害程序员就业市场！（他们会没有工作且被迅速替代）
* 状态：作为社会科学研究论文提交

### 子课题及论文题目（拟） - x - “创意类思维”在垂类场景中的应用研究
![](./img/sub.topic.5.png)

### 子课题及论文题目（拟） - x - “创意智能体社交”在垂类场景中的应用研究
![](./img/sub.topic.6.png)

## 核心相关工作
### 1. 计算广告
（写作思路：根据计算广告历史及技术演进表意）
计算广告是利用计算机技术对广告投放进行优化、定制和智能化,以提高广告的精准投放和效果的一种广告模式【1】。它是互联网广告发展的重要形式,成为各大互联网公司的核心收入来源【2】。计算广告经历了以下3个主要发展阶段【6】。（1）1.0 时代: 程序化广告,基于简单数学模型,以文字广告为主,通过竞价排名等机制进行广告投放,如Google Ads、百度广告等基于搜索业务的投放平台；（2）2.0 时代: 智能推荐广告,对用户画像和广告特征进行更精细的分析,在广告智能投放和效果付费方面有长足进步,如Meta Ads、微信朋友圈、抖音等基于社交的智能广告投放平台；（3）3.0 时代: 认知计算广告,以AI技术为基础,以AIGC为逻辑,以认知科学为理论,重点解决广告创意生成等问题,利用多模态垂类模型生成文字、图片、视频等广告内容。而以AIGC内容创作作为核心的相关行业领导者在业界则尚未出现。市场预估在垂类领域如认知计算时尚、认知计算广告等垂直领域将会出现市值万亿的独角兽高科技公司。

### 2. 认知计算广告知识体系
TODO: 补充 阿诺德 认知-情感-意动 模型
#### 大中华地区
认知计算广告的理论体系诞生于2024，由段淳林及其研究团队首先提出，在【10】中有对其理论体系发展脉络的系统性论述。其中，类思维，智能体社交及价值对齐是三个核心营销概念。类思维【】是指智能体通过分类、归纳等认知过程,对事物进行抽象和概括,从而达到对复杂环境的理解和预测。这种类概念思维是智能体社交的基础,帮助智能体快速感知外界,并做出相应的反应。智能体社交【】指智能体之间基于类思维进行信息交互和行为协调的过程。不同的智能体可以通过交流与协作,共享知识和经验,实现目标的达成。这种社交互动是智能系统发展的关键。价值对齐【】是指智能体内部price和reward等价值系统,以及智能体之间的目标价值趋同。当智能体内部价值系统和外部环境的价值目标达成一致时,就实现了价值对齐。这是智能体最终实现高度自主和协同的基础。

理论外延在众多学者的共同努力下得到较好扩展。在【7】中，段淳林及魏方等在此核心理论下进一步率先提出基于认知计算广告的研究范式。在【8】中，段淳林及陆昊琪等在此核心理论下进一步率先提出基于认知计算广告的生产范式。在【9】中，段淳林及蒲源等在此核心理论下通过量化实验方法，融合深度神经网络等AIGC技术，使用领域数据训练了一个广告传播力预测模型，并使用B站大数据进行了可行性分析及原型验证。实验表明，此广告传播力预测模型能准确预测B站上某条广告或视频的传播力和影响力。
#### 其他地区
TODO: 待补充

### 3. 基于认知计算广告的多模态内容感知、生成及个性化创意应用研究
如【11】中，段淳林及江纬等基于认知计算广告理论，进一步提出了分布式协同工作流、基于节点的编排、低代码等核心技术的落地应用，并以时尚行业设计师工作流为例，加以验证。在【12】中，段淳林及江纬等基于认知计算广告理论，进一步系统阐明在实际生产应用中，如何综合使用关系型、非关系型，对象及新型向量数据库以构筑垂类AIGC知识库。核心待解决问题是如何对知识进行收集、整理、加工及存储，以使得更好的搜索、推荐及数据挖掘。在【13】中，段淳林及江纬等基于认知计算广告理论，着眼于于垂类大模型的RAG、微调等算法技术，系统阐明如何高效和挖掘新知识，并与所选基模型形成互补。作者进一步阐明了如何在垂类行业如时尚中去灵活使用不同的大模型、RAG及微调等方法，以满足既定评价指标。在【14】中，段淳林及江纬等基于认知计算广告理论，进一步提出在不同垂直场景及细分需求下的技术选型，以满足客户需求。

## 参考文献
### 论文
1. 段淳林,任静等.智能广告的程序化创意及其RECM模式研究[J].新闻大学,2020(2):17-31 + 119-120【已见刊 领域：计算广告】
2. 段淳林,杨恒等.数据、模型与决策:计算广告的发展与流变[J].新闻大学,2018(1):128-136 + 154【已见刊 领域：计算广告】
3. 段淳林,宋成等.用户需求、算法推荐与场景匹配:智能广告的理论逻辑与实践思考[J].现代传播.2020,42(8):199-128【已见刊 领域：计算广告】
4. 段淳林,崔钰婷等.计算广告学科建设持续创新能力的影响研究——组织学习与知识共享的链式中介效应分析[J].现代传播.2024年第三期【已见刊 领域：计算广告】
5. 段淳林,周学琴等.虚拟主播的形象行为相似性对消费者品牌信任的影响研究——基于临场感的中介效应[J].【已见刊 领域：计算广告】
6. 段淳林,魏方等.（关于认知计算广告研究范式）.2024【在投中 领域：认知计算广告】
7. 段淳林.AI智能体与知识库生成：计算广告知识生产的演进逻辑.2024【在投中 领域：认知计算广告】
8. 段淳林,蒲源等.（关于认知计算广告及使用B站数据所做的实验及模型）.2024【在投中 领域：认知计算广告】
9.  段淳林等.（关于认知计算广告蓝皮书）.华南理工大学.2024【撰写中 领域：认知计算广告】
10. TODO: 段淳林,江纬等.AIGC创意工作流.workflow【撰写中 领域：认知计算广告】
11. TODO: 段淳林,江纬等.AIGC创意知识库.kb【撰写中 领域：认知计算广告】
12. TODO: 段淳林,江纬等.AIGC创意模型.model【撰写中 领域：认知计算广告】
13. TODO: 段淳林,江纬等.AIGC创意智能体.agent【撰写中 领域：认知计算广告】
14. TODO: 段淳林,江纬等.AIGC创意类思维.think【撰写中 领域：认知计算广告】
15. TODO: 段淳林,江纬等.AIGC创意智能体社交.communicate【撰写中 领域：认知计算广告】
16. TODO: 段淳林,江纬等.AIGC创意价值对齐.alignment【撰写中 领域：认知计算广告】

### 教科书
1. 段淳林.计算广告学导论[M].武汉:华中科技大学出版社,2022
2. 刘鹏,王超.计算广告:互联网商业变现的市场与技术[M].第3版.北京:人民邮电出版社,2022
3. 段淳林.整合品牌传播:从IMC到IBC理论建构[M].北京:人民出版社,2020
4. 陈刚等.创意传播管理:数字时代的营销革命[M].北京:机械工业出版社,2012
5. BISHOP C M. Pattern Recognition and Machine Learning. Springer[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2006, 16(4):049901
6. TODO: ARENSW.当代广告学[M].丁俊杰,程坪,译.北京:人民邮电出版社,2005

### 相关学术研讨会
1. 生成式AI与跨学科融合：机遇、挑战与应对.华南理工大学.20240613

### Cool Research Problems
1. given a pre-trained dense LLMs, how to obtain a effective sparse LLM? For example, the sparsification of linear layers.
2. given a pre-trained dense NOT LLMs but Large Vision Models, how to prune?
3. 
