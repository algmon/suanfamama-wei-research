![](./img/central.topic.png)

## PI：江纬 Wei Jiang
## 当前研究课题：多模态内容感知、生成及个性化创意应用场景研究
## 当前理论框架依据：认知计算广告
## 子课题
![](./img/sub.topic.1.png)
### （初阶）子课题及论文题目（拟） - 1 - “创意工作流”在垂直领域的应用研究
#### 摘要
* 本文着重探讨 AIGC 内容创意工作流在垂直领域的应用。AIGC 工作流包含多个分步节点，各节点有明确的输入与输出。在内容创意中，从创意激发到独特内容输出及优化调整，呈现丰富流程。当前业界多为单一工作流，而本文研究分布式协同工作流问题，旨在解决不同节点间的协同难题。提出可利用先进的通信技术与智能分配机制，促进各节点高效协作。此工作流对内容创意具有深远价值，广泛应用于多模态内容感知生成及个性化创意应用场景，为推动垂直领域的内容创新提供新路径。

#### 关键词
* AIGC 多模态内容 感知 生成 个性化创意应用 算法 场景 工作流

#### 作者
* （相关研究团队成员如段淳林、江纬及陆昊琪等）

#### 同行评审
* 陈刚（deadline: 2024 fall）

#### 组织
* 华南理工大学

#### 简介
* 在当今数字化时代，AIGC（人工智能生成内容）正逐渐在各个垂直领域展现出巨大的潜力。其中，内容创意工作流的不断发展与变革为认知计算广告和认知计算时尚等领域带来了新的机遇。
* 单一工作流通常是指传统的内容创作方式，由个人或小团队独立完成从创意构思到最终成品的整个过程。例如，在广告领域，一位文案撰写人员独自负责广告文案的创作，可能经过多次修改后提交给上级审核。在时尚领域，设计师凭借个人灵感和经验进行服装设计，然后制作出样品。这种工作流的优势在于创作者对整个作品有较高的掌控度，但也存在效率低下、创意局限性大等问题。
* 分布式协同工作流则与单一工作流有着显著区别。分布式协同工作流借助 AIGC 技术，将内容创作过程分解为多个环节，并由不同的参与者在不同的地点协同完成。在认知计算广告中，AIGC 可以快速生成多个广告创意方案，然后由策划人员、设计师、营销人员等共同对这些方案进行评估和优化。在认知计算时尚领域，AIGC 可以根据时尚趋势和用户需求生成设计草图，设计师在此基础上进行进一步的创作，同时市场人员可以利用 AIGC 生成的营销内容进行推广。这种工作流能够充分发挥各参与者的专业优势，提高工作效率，同时也能激发更多的创意灵感。
##### 现存问题与挑战
* ![](./img/linkai-workflow.png)
* linkai workflow【16】
* 现有工作流通常包含基础节点和应用，能够进行流程的运行、保存以及接入等操作。在节点设置方面，可以涉及大模型，如意图分支、知识库等，其中也可能包括搜索商品知识库以及文案润色等插件的应用。同时，也存在分支节点，这些节点往往是预先由真人专家预先编排好的，用于处理不同的情况和任务。例如在商品导购等场景中，通过搜索商品知识库、利用插件进行销售数据查询等操作，实现特定的功能。然而，现有的工作流尚存一定局限性。
![](./img/google-pathways-distributed-workflow.png)
* google distributed workflow【17】
* Google 的这一架构呈现出分布式的设计特点。该架构主要用于语言模型和视觉模型的训练，其中涉及 “A” 和 “B” 两个元素进行正向和反向的传递过程，采用了正向反向传播算法。同时，架构中存在调度器，能够调度不同的算法算力，可被视为分布式工作流的典型例子。然而，美中不足的是，在创意内容创作领域，这种分布式协同工作流的应用还比较少见，目前未见先例。虽然 Google 在算力方面有一定的心得，但在创意内容创作的分布式协同方面仍有很大的探索空间。
* 创新点及难点在于（1）如何设计分布式协同工作流及（2）分布式协同工作流在垂直领域的应用。

#### 相关工作
* TODO: 新增 计算广告相关文献【】【】【】【】【】
* TODO: 新增 中国认知计算广告知识体系【】【】【】【】【】
* TODO: 新增 基于认知计算广告的多模态内容感知、生成及个性化创意应用研究【】【】【】【】【】
* TODO: 重新组织技术逻辑推荐相关。随着人工智能技术的不断发展，AIGC 在内容创意领域展现出巨大潜力。ChatGPT【1】 以 Transformer【2】 为基础架构，开创了大模型语言智能对话的先河，为内容感知提供了新的思路。从 Google【3】 到 OpenAI 的发展历程，标志着语言模型的重大突破。内容感知从单一维度逐渐向多模态发展，Gemini 在此方面表现出色。它能够综合处理多种模态的信息，为用户提供更加丰富的体验。在生成方面，encoder-decoder 架构【】被广泛应用。不仅在文本生成中发挥重要作用，还在图片生成领域取得显著成果。例如，Stable Diffusion【4】 在图片生成方面具有很高的质量和效率。
* 从技术推演路径来看，2013 年 AlexNet【6】 的出现将深度学习带入人工智能领域，首先在图像领域取得突破，随后在文字处理方面也获得成功。此后，不断有新的技术突破，多模态框架的整合【7】使得不同类型的数据能够更好地融合。Transformer 架构的出现更是为语言处理带来了革新，随后  Diffusion-Transformer 架构以及像 Sora【8】 这样能够生成视频的技术不断涌现。在算力方面，随着硬件设备的不断升级，计算能力得到大幅提升，为大规模模型的训练和应用【9】提供了支持。算法的创新也不断推动着 AIGC 的发展，数据的丰富和质量的提高则为模型的训练提供了坚实的基础【10】。
* 目前业界的工作流主要有两种类型【11】，一种是以降本增效为目标的工作流【12】，另一种则是聚焦于内容创意的工作流【13】。然而，现有的工作流大多为单一工作流，在协同性和灵活性方面存在一定的局限性【14】。本文的创新点在于提出了分布式协同工作流，旨在解决传统单一工作流的问题，更好地满足垂直领域中内容创意的需求。通过节点编排和低代码等技术【15】，实现各环节的高效协同，提高内容创意的质量和效率。

#### 我们提出的理论 - 创意设计工作流新范式
1. TODO: 分布式协同
2. TODO: 聚焦创意灵感生成
3. TODO: 高度智能化
4. TODO: 人机协同【18】
5. TODO: 环保与可持续【19】
6. TODO: 智能任务调度
7. TODO: 智能计算资源分配
8. TODO: 智能节点编排
9. TODO: 低代码
10. TODO: 新增 理论图

#### 我们提出的相匹配技术
1. TODO: 技术1 待讨论
2. TODO: 技术2 待讨论
3. TODO: 技术3 待讨论
4. TODO: 新增 技术框架图

#### 实验原型系统及落地应用情况
* TODO: 待设计案例：时尚行业设计师分布式协作工作流 - 设计折扣券【20】
##### 典型案例
![](./img/gimc-design-workflow.png)
* GIMC design workflow
* 省广集团的灵犀案例为个性化创意应用场景研究提供了宝贵的实践经验【5】。该案例展示了如何利用 AIGC 技术实现个性化的创意内容，为企业带来了良好的经济效益和社会效益。此外，还有其他相关研究和实践也为 AIGC 内容创意工作流在垂直领域的应用提供了参考。这些工作共同推动了 AIGC 技术的发展和应用，为未来的创新提供了坚实的基础。其核心工作流程如下：
1. AI文案 TODO:
2. MJ图像 TODO:
3. 扩图+优化 TODO:
4. 大师图像 TODO:
5. 高清放大 TODO:
6. 扣除+替换 TODO:

#### 结论
* 本文深入研究了 AIGC 内容创意工作流在垂直领域的应用，创新性地提出了分布式协同工作流。该工作流基于节点编排和低代码等核心技术，充分发挥其优势，有效解决了传统单一工作流的局限性。在实际应用中，分布式协同工作流展现出了良好的落地效果。它能够提高内容创意的效率和质量，满足垂直领域多样化的需求。同时，通过各节点的协同合作，实现了多模态内容的感知生成和个性化创意的应用。未来，随着技术的不断发展，AIGC 内容创意工作流将在更多垂直领域得到广泛应用，为推动行业的创新与发展发挥更大的作用。
* 我们同时基于认知计算广告理论内核及外延，提出在垂直领域应用下的“工作流”创意设计新范式，并以时尚行业设计师工作流为例，设计原型实验并加在实际生产系统进行小范围验证。

#### 参考资料
1. 
2. 
3. 
4. 
5. 
6. 
7. 
8. 
9. 
10. 

#### 附加信息
* paper 1 - workflow.md
* 关键词：工作流
* 写作思路：从基本工作流过渡到分布式协同工作流
* 创新点及难点：（1）如何设计分布式协同工作流及（2）分布式协同工作流在垂直领域的应用
* 候选案例1：如何使用 linkai 构建营销工作流？
* 候选案例2：省广集团是如何有效使用工作流及协同功能以降本增效？
* 候选案例3：在广告创意垂类中如何解决分布式协同问题？
* 候选案例4：智能营销洞察
  * 步骤1：对数据进行分析洞察
  * 步骤2：获得消费者反馈，丰富顾客画像
  * 步骤3：优化营销运营策略
  * 步骤1-3 不断迭代
* 候选案例5：生成式AI的多模态内容生成
  * 线性前置步骤：创意确定
    * 并行步骤: 文本
    * 并行步骤: 图片
    * 并行步骤: 声音
    * 并行步骤: 视频
    * 并行步骤: 代码
  * 线性后置步骤：人工审核
* TODO:使用数字签名是否能有效防止工作流被篡改？
* 所投期刊《新闻界》

![](./img/sub.topic.2.png)
### （初阶）子课题及论文题目（拟） - 2 - “创意知识库”在垂直领域的应用研究
#### 摘要 Abstract
* This paper begins by examining the traditional single modality knowledge base and then progresses to explore the development and significance of a multimodal knowledge base in the realm of AI-generated content (AIGC). As the field of AIGC expands, the need for a knowledge base that can handle multiple modalities such as text, image, video, and other objects becomes crucial. The research focuses on the construction, organization, and application of this multimodal knowledge base specifically within the context of AIGC, rather than in the areas of increasing productivity or streamlining workflows. By analyzing the transition from single modality to multimodal knowledge bases, this paper provides insights into the future directions of AIGC and its potential impact on various industries.
#### 关键词 Keywords
* AIGC multimodal perception generation personalization creativity application knowledgebase

#### 作者 Authors
* (related research team)

#### 同行评审
* 吴小坤（deadline: 2024 fall）

#### 组织 Organization
* South China University of Technology

#### 简介 Introduction
* 随着信息技术的快速发展，知识库在各个垂直领域的应用日益广泛。然而，传统的知识库大多是以单一模态为主，主要依赖文本、图表等结构化或半结构化数据的形式来存储和管理知识。随着现代需求的不断变化和信息量的爆炸式增长，传统的单一模态知识库逐渐暴露出其局限性，无法全面、高效地支持复杂的创意需求。在垂直领域中，诸如广告、医疗、教育等行业对创意的要求不断提高，而这些领域不仅需要传统的文字、数据等信息，还需要利用图像、声音、视频等多模态数据来丰富知识库的表现力。因此，多模态知识库应运而生，成为解决这一问题的重要途径。
* 单一模态知识库：单一模态知识库最早期的形态是以文本为主要存储和组织方式的知识管理系统。在这种体系中，知识以纯文本、结构化数据或简单图表的形式呈现，系统通过关键字检索和规则匹配来进行信息的管理与获取。由于其形式简单、开发相对容易，单一模态知识库被广泛应用于早期的行业知识管理，如法律、学术研究、技术文档等领域。然而，随着数据的复杂性增加，特别是随着互联网和移动设备的普及，知识获取的多样化需求日益增加，单一模态知识库的不足也逐渐显现出来。其局限性体现在以下几个方面：（1）信息表达的局限性：单一模态知识库往往只能处理文字、数据等少数形式的信息，对于包含图像、视频、声音等丰富信息的多媒体资源支持不足。（2）检索效率低下：传统的文本检索方式难以处理多维度的信息需求，用户在需要获取复杂内容时往往需要依赖手动筛选，降低了信息获取的效率。（3）创意支持不足：特别是在广告、设计等对创意要求较高的行业，单一模态的文字和数据无法充分支持创意工作者的需求，限制了创意的多元表达与激发。
* 多模态知识库：为了解决这些问题，多模态知识库逐渐被提出和发展。多模态知识库结合了文本、图像、音频、视频等多种信息模态，通过智能化技术（如自然语言处理、计算机视觉、深度学习等）将不同类型的数据统一整合并组织起来，从而为用户提供更加丰富、全面的信息资源。在多模态知识库中，各种模态的信息相互补充。例如，在广告创意领域，文案和视觉素材的结合至关重要，多模态知识库不仅可以存储广告文案，还能提供对应的图片、音频甚至视频建议。这种综合性的信息组织方式极大提升了创意生产的效率和质量。多模态知识库的核心优势包括：（1）丰富的信息表达：多模态知识库不仅能处理文字，还能整合图像、视频、音频等多种形式的信息，使得知识的存储和检索更加多样化。（2）提高检索精准度：借助人工智能和深度学习技术，用户可以通过文字描述找到图像，或通过图像搜索相似的多模态资源，极大提高了信息检索的效率和准确性。（3）增强创意支持：通过多模态数据的结合，创意工作者可以获得更多灵感来源，从而推动创意的进一步发展。
* 现存问题与挑战：尽管多模态知识库带来了诸多优势和创新，然而在应用过程中也面临着一系列的问题和挑战：（1）数据整合难度大：多模态数据的异质性使得不同模态的数据之间难以进行有效的整合和关联。不同模态的处理方式和存储格式各不相同，导致数据间的交互和转换成为技术上的难点。（2）计算资源要求高：处理多模态数据需要大量的计算资源，尤其是在涉及图像和视频的分析时，系统的计算和存储负担显著增加。这对系统的可扩展性和成本控制提出了更高的要求。（3）跨模态理解不完善：尽管目前有许多针对多模态数据处理的技术，但在不同模态之间进行深度关联和语义理解仍然是一大挑战。如何让系统真正理解文字与图像、视频等信息之间的关联，依然需要更多的技术突破。（4）数据隐私和安全问题：多模态知识库中可能涉及大量的个人隐私信息，特别是在医疗、广告等领域，如何确保数据的隐私和安全，是其在实际应用中的重要挑战。
* 论文后续章节我们排列如下：我们在简介部分提出了问题与挑战，在第二部分论述相关工作，在第三部分论述理论构建，在第四部分论述技术推演，在第五部分论述典型落地案例及实验原型（跨学科优势），我们在第六部分进行总结。
* 本论文的贡献主要体现在理论及技术创新点上。理论创新点在于构建了一个结合多模态信息的知识库理论框架，特别针对创意驱动型行业的特殊需求进行优化。技术创新点则体现在（1）多模态（2）快速存储（3）快速信息挖掘等方面的技术创新与突破。这些理论及技术创新不仅有助于提升创意工作的效率，还能够为知识库的未来发展提供新的技术方向。

#### 相关工作 Related Work
* The history of databases traces back to the efforts of IBM researchers who first developed relational databases[1]. These databases consist of tables with primary keys and reference keys, and support operations like join, select, and where in SQL query languages. With the growth of giant applications like web search[2] & recommendation[3], the need to handle unstructured text became more prominent in recent decades. This led to the emergence of non-relational databases like MongoDB[4] and others. These databases allow for storing strings of variable lengths, providing a certain degree of flexibility. In most of the distributed system architectures[5], Redis[6] is usually servered as a caching database. Being an in-memory database, it offers fast retrieval speed but has the known limitations such as cache miss. If a cache miss occurs, a query to the backend database is sometimes required, which can be time-consuming though more precise. The era of web search engines[7] brought about the use of data structures like inverted indexes[8], which support faster retrieval and better quality for web searches[9]. New developments in databases include vector databases[10] that find important applications[11] in large language models and other related retrieval techniques[12]. The focus of this paper is on how to orchestrate[13] those databases and provide a unified interface[14] for upper-level applications[15]. 

#### 我们提出的理论（理论构建）
* 

#### 我们提出技术支撑（即技术逻辑及推演）
* 

#### 实验原型系统及落地应用情况（实际应用）
* 

#### 结论 Conclusion
* 

#### 参考资料 References
1. RMDBs
2. Google Search
3. Amazon Recommendations
4. MongoDB and other NoSQL databases
5. Distributed systems such as Hadoop and Spark
6. Redis
7. New York Times reporting on web search
8. Inverted indexes tutorial from University of Melbourne
9. Web search engines like Google, Bing, and Yahoo
10. Vector databases such as TODO:
11. RAG (short for retrieval augmented generation)
12. consin similarity and bm25 ranking algorithms
13. Orchestration of databases

#### 附加信息
* paper 2 - kb.md
* 关键词及着眼点：知识库
* 写作思路：从基本单一知识库到多模态AIGC创意知识库
* 基于认知计算广告理论，进一步系统阐明在实际生产应用中，如何综合使用关系型、非关系型，对象及新型向量数据库以构筑垂类AIGC知识库供下游应用调用。
* 创新点1：多模态AIGC创意知识库
* 创新点2：AIGC创意知识库是如何把 (1) 元认知知识 (2) 概念性知识 (3) 程序性知识及 (4)技能性知识整理、归纳并融合在一起的
* 创新点3：多模态知识库半自动生成法的重要作用（保证入库数据质量）
* 案例1：如何使用dify平台构建多模态AIGC创意知识库
* 案例2：如何使用coze平台构建多模态AIGC创意知识库
* 所投期刊：新闻与传播如 TODO: 未确定

![](./img/sub.topic.3.png)
### （中阶）子课题及论文题目（拟） - 3 - “创意垂类模型"在垂直领域的应用研究
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

### （中阶）子课题及论文题目（拟） - 4 - “创意智能体”在垂类场景中的应用研究
![](./img/sub.topic.4.png)
#### 摘要 Abstract
* This paper begins by clarifying the concept that an intelligent agent is distinct from large language models or large vision models. An intelligent agent can be metaphorically described as having a "brain" and "body" and it is designed to perform specific tasks. The control of an intelligent agent often relies on large language models or large visual models. The paper then presents different types of intelligent agents, including those for cost reduction and efficiency improvement and those for content creation. The focus of this paper is on content creation. An in-depth exploration of these content creation intelligent agents are provided, while specific case studies are yet to be determined.
* TODO: Needs more details.

#### 关键词 Keywords
* 

#### 作者 Authors
* 

#### 同行评审
* 全宇晖（deadline: 2024 fall）

#### 组织 Organization
* 

#### 简介 Introduction
* 

#### 相关工作 Related Work
* TODO:

#### 现有典型智能体及所存问题 Current Intelligent Agents and Problems
* 

#### 所提出关于“智能体”的创意设计新范式 New Design Paradigm for Intelligent Agents
* 

#### 实验原型系统及应用场景 Real-world Applications and Prototypes
* 

#### 结论 Conclusion
* 

#### 参考资料 References
1. 
2. 
3. 
4. 
5. 
6. 
7. 
8. 
9. 
10. 

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

### （高阶）子课题及论文题目（拟） - 5 - “创意类思维”在垂类场景中的应用研究
![](./img/sub.topic.5.png)

### （高阶）子课题及论文题目（拟） - 6 - “创意智能体社交”在垂类场景中的应用研究
![](./img/sub.topic.6.png)

### （高阶）子课题及论文题目（拟） - 7 - “创意价值对齐”在垂类场景中的应用研究
![](./img/sub.topic.7.png)

### （中阶）子课题及论文题目（拟） - 8 - Improved Methods for Static Neural Network Pruning
![](./img/sub.topic.8.png)
#### Abstract
Static neural network pruning is presented as a performance optimization technique for large language models and large vision models. The approach aims to identify and remove neurons unlikely to lead to expected results for typical user queries. The goal is to obtain a smaller large language model that can quickly return results almost as good as those of the unpruned model.

First, through careful analysis of embedded knowledge and user queries, an initial mathematical model based on certain probabilities obtained from the environment is developed to improve on previous results for pruned large language model size, achieving additional improvement in some cases. A simple method for generating queries into the large language model in the absence of real-life queries is also devised, which is useful for modeling top results.

Second, this paper explores and compares to previously proposed approaches that perform pruning based on other methods.
#### Keywords
LLM pruning, LVM pruning, quantization, dense & sparse models
#### Open Source Repo
* https://github.com/algmon/mama-prune
#### 目标投稿学术会议
##### 候选
* ICLR 2025, Full Paper Submission Deadline: Oct 02 2024
* Plan: Submit the Model Pruning paper for publication
##### 候选
* SIGIR 2025, July 13-18, Padova, Italy. Full Paper Submission Deadline: TODO: around Jan, 2025.
* https://sigir2025.dei.unipd.it/
* 会议调性：强调量化分析及数值方法，如量化指标设计，性能论文及质量论文均可，需要与IR有一定关联
##### 候选
* IEEE Big Data 2025, IEEE
* 会议调性：偏向于大数据，计算模型及人工智能等，会议包容性较强
* https://dblp.uni-trier.de/db/conf/bigdataconf/
##### 候选
* NeurIPS 2025. Full Paper Submission Deadline: TODO:
* http://dblp.uni-trier.de/db/conf/nips/
* NeurIPS 2024 Author notification: Sep 25, 2024
##### 候选
* ACL 2025. Full Paper Submission Deadline: TODO:
##### 候选
* CVPR 2025. Full Paper Submission Deadline: Nov 15, 2024.
##### 候选
* ICCV 2025. Full Paper Submission Deadline: TODO:
##### 候选
* ICML 2025. Full Paper Submission Deadline: TODO:
##### 候选（已过期）
* IEEE Big Data 2024, December 15-18, Washington DC, USA. Full Paper Submission Deadline: Sept 8, 2024.
* https://www3.cs.stonybrook.edu/~ieeebigdata2024/ImportantDates.html
##### 候选（已过期）
* COLM 2024, October 7-9. Full Paper Submission Deadline: March 29, 2024.
* https://colmweb.org/dates.html
##### 候选（已过期）
* WSDM 2025, March 10-14. Full Paper Submission Deadline: Aug 14, 2024.
* https://www.wsdm-conference.org/2025/call-for-papers/
##### 候选（已过期）
* AAAI 2025, AAAI Conference on Artificial Intelligence, AAAI. Full Paper Submission Deadline: August 15, 2024.
* https://aaai.org/aaai-24-conference/save-the-date-aaai-25/
* http://dblp.uni-trier.de/db/conf/aaai/
#### 目标投稿学术期刊
##### 候选1
* JMLR, Journal of Machine Learning Research, MIT Press, http://dblp.uni-trier.de/db/journals/jmlr/
* No specific deadline

#### 1. Introduction
* (General Intro) Large language models and large vision models are facing significant performance challenges due to the massive data size and query loads they need to support. These models, along with other related systems, crawl, analyze, and incorporate billions of web pages, videos, and multimodal data into their network architectures such as transformer. One crucial cost factor is the query processing per user, which must scale with both data size and query load. As a result, large language models devote substantial hardware and energy resources to this task. There has been extensive research on improving query processing performance, including work on various caching techniques, retrieval information systems, and high-performance knowledge representation. A large category of optimization techniques commonly referred to as static pruning or dynamic pruning has emerged in the context of query processing for large language models and large vision models. This paper aims to explore and contribute to the understanding and improvement of these pruning techniques to enhance the performance and efficiency of large language and vision models.
* (More Specific Intro) In this paper, our attention is directed towards a particular optimization technique known as static network pruning. In essence, the approach involves conducting a suitable analysis of the knowledge representation, document collections, and query distribution. The objective is to determine those entries or neurons that are highly likely to yield top user query results for typical queries. Subsequently, any other neurons that are unlikely to contribute to user inputs are removed from the neural network. The aim is to obtain a significantly smaller neural network with a reduced amount of parameters. This pruned neural network can achieve almost the same quality of results as the unpruned one while requiring much less memory and GPU resources. Consequently, it leads to faster query processing over a shorter neural network with optimized layers.
* (Give an example) Consider a leading large language model provider today. There are 4 billion documents incorporated into its knowledge base, with an average of 300 words per document, resulting in a total of approximately 10^15 tokens. The engine receives 5 billion queries per day, with each query represented by around 10^11 terms to convey the user's intention for interaction with the large language model. This implies that nearly 117 billion tokens in the knowledge representation could potentially lead to expected output tokens. However, in reality, far fewer tokens actually result in an output within a month. Considering the repetition of queries and postings, more than 99.5% of all routing and neuronal activation and triggers do not yield a single result output from the decoder within a month. Although we cannot reliably identify the 0.5% of active neurons that contribute to the result precisely for the next month, we might hope to identify a large subset of the optimized neurons that contains most of the important information and knowledge representation as the full neural network on common measures of effectiveness.
* (Small Set of Closed Related Previous Work) Previous work on static pruning for large language models and large vision models has primarily focused on approaches such as retaining layers above a global impact threshold or keeping high-scoring neurons in each layer. For example [][][][]. These efforts have yielded promising results, but there is room for further improvement. The goal of this paper is to build on this existing work and develop a methodology that combines different ideas to achieve a better balance between neural network size and result quality as measured by standard retrieval or information generation quality metrics. Given the feature-rich environment, pruning is considered as a prediction problem where suitable statistical techniques or deep learning methods such as language modeling and machine learning are employed to determine which neuron/sets of neurons/layers to keep.
* (Paper Organization) The rest of this paper is structured in a systematic manner. In the following section, background information on knowledge representation, neural networks, and pruning will be provided. Additionally, related work in this field will be discussed to situate our research within the broader context. Section 3 will summarize our key contributions, highlighting the novelty and significance of our approach. Section 4 will delve into the technical details of our proposed approach, providing a comprehensive explanation of the methodology and algorithms employed. The results presented in our study will be explained in section 5, including an analysis of their implications and performance. Finally, section 6 will offer concluding remarks, summarizing the main findings of the paper and suggesting potential directions for future research.
#### 2. Background and Related Work
In this section, we first provide some background on neural network architectures, ranked queries, pruning, and early termination techniques. We then discuss previous work related to static pruning in the context of large language and vision models. For additional details on general neural network architectures, we refer to [][].
##### A. Background
1. Nerual Network Architectures
2. User Input/Queries & Ranking & Content Generation
3. Pruning
4. Early Termination & Dropout Technique
##### B. Related Work
1. There are three typical network pruning algorithms. (1) The magnitude pruning algorithm[18, 19]: Simplest approach: Prunes weights based purely on their absolute magnitude. Threshold-based: A global threshold is determined based on the desired sparsity ratio. Weights below this threshold are set to zero. Unstructured: Can prune individual weights anywhere in the matrix, potentially leading to irregular sparsity patterns that might not be hardware-friendly. Fast but less accurate: Generally the fastest method, but might remove important connections, leading to a larger accuracy drop compared to more sophisticated methods. (2) The WANDA(Weights and Activations) pruning algorithm[3]: Importance-aware: Considers both weight magnitudes and activation statistics to estimate weight importance. Calibration phase: Requires a calibration step where the model processes a small dataset to collect activation data. Row-wise scaling: Normalizes weight magnitudes within each row based on activation statistics, making the pruning less sensitive to weight scale variations across neurons. Unstructured or structured: Can be applied in an unstructured manner (pruning individual weights) or a structured manner (pruning within blocks of weights). Improved accuracy: Often achieves better accuracy-sparsity trade-offs compared to magnitude pruning. (3) The SparseGPT pruning algorithm[5]: Gradient-based: Leverages gradient information during pruning to identify less important connections. Iterative pruning: Prunes the model iteratively, gradually increasing sparsity while minimizing accuracy loss. Block-sparse structure: Encourages a block-sparse structure, which can be more hardware-efficient for some architectures and libraries. Computationally intensive: Can be more computationally expensive than magnitude or WANDA pruning due to the iterative nature and gradient calculations. State-of-the-art results: Often achieves very high sparsity levels with minimal accuracy degradation, making it suitable for compressing large language models. In summary, Magnitude pruning is the simplest and fastest but might be less accurate. WANDA improves upon magnitude pruning by considering activation information, potentially leading to better accuracy. SparseGPT is a more advanced method that uses gradient information and iterative pruning to achieve high sparsity with minimal accuracy loss, but it comes with higher computational cost.
5. Using Query Traces
6. complementary approach: quantization
7. Comparison to our work
#### 3. Our Contributions
In this paper, we study LLM & LVM static pruning that attempt to achieve a good trade-off between network size and generation quality. Our main contributions are as follows:
1. We describe an approach that ...
2. We describe several algorithms that ...
3. We perform an experimental evaluation on two datasets that ...
4. We compare our methods with ...
#### 4. Our Proposed Pruning Algorithms
##### Key Considerations when designing the pruning algorithms
1. Target Sparsity Level: What percentage of weights do we aim to prune? Higher sparsity can lead to greater compression and speedups but might sacrifice more accuracy.
2. Quality-Size Trade-off: Finding the right balance between model size & speed and quality is crucial. Some algorithms prioritize accuracy (SparseGPT), while others are more aggressive in pursuing sparsity (magnitude).
3. Pruning Criterion: How do you determine which connections to prune? Options may include: Magnitude (simplest), Activation statistics (WANDA), Gradient information (SparseGPT) and Sensitivity analysis
4. Structured vs. Unstructured Pruning: Unstructured: Prune individual weights anywhere, potentially leading to irregular sparsity patterns that might not be hardware-friendly. Structured: Prune in blocks (e.g., 2:4, 4:8), which can be more efficient for some hardware and libraries.
5. Pruning Schedule: When and how do we prune? One-shot pruning: Prune once at the beginning or after training. Gradual pruning: Incrementally prune over multiple training epochs. Iterative pruning: Prune, fine-tune, and repeat.
6. Calibration Data: Some algorithms (like WANDA) require a small calibration dataset to collect activation statistics before pruning. The choice of this data can impact pruning effectiveness.
7. Hardware Awareness: Consider the target hardware (CPUs, GPUs, specialized accelerators) and design pruning strategies that align with hardware constraints for optimal efficiency.
8. (Preferred) Layer-Wise Sparsity: Allow different layers to have varying sparsity levels based on their sensitivity. Not all layers contribute equally to a model's performance.
9. Regularization and Stability: Pruning can sometimes lead to instability during training. Techniques like weight decay or gradual pruning can help mitigate this.
##### 1. Movement Pruning
* Core Idea: Instead of directly removing weights, movement pruning identifies unimportant weights and "moves" their values to other more significant connections. This helps preserve the overall information flow within the network.
* Potential Benefits: Can achieve higher sparsity levels with less accuracy degradation compared to traditional pruning methods.
* Example: The "Rigging the Lottery: Making All Tickets Winners" paper introduces a similar movement pruning technique.
* Prelimary Results: NOT as good as expected. For detail, see the related tables and figures.
##### 2. Bias pruning
* Core Idea: set all bias in neruons to 0 layer by layer, to see if it can improve the performance. Mostly for sanity check to understand more about the effectiveness of the bias.
* Prelimary Results: TODO: Not Yet Implemented.
##### 3. WBA pruning (weights, bias and activations)
* Core Idea: Build upon on the Wanda alg, add one dimension 'bias' into the consideration of the algorithm.
* Prelimary Results: TODO: Not Yet Implemented.
##### 4. Flow Pruning
* Core Idea: Flow Pruning focuses on pruning connections early in the training process by analyzing the sensitivity of the loss function to weight perturbations.
* Potential Benefits: Can be particularly effective for finding important connections and achieving high sparsity even before full training.
* Considerations: Might require more computation during the initial pruning phase.
* Prelimary Results: TODO: Not Yet Implemented.
##### 5. Variational Pruning
* Core Idea: Applies Bayesian principles to pruning by treating weights as random variables and pruning connections with low signal-to-noise ratios.
* Potential Benefits: Provides a more principled way to handle uncertainty in weight importance and can lead to more robust pruning.
* Considerations: Often more computationally expensive than deterministic pruning methods.
* Prelimary Results: TODO: Not Yet Implemented.
##### 6. Reinforcement Learning-Based Pruning
* Core Idea: Treats pruning as a sequential decision-making problem and uses reinforcement learning agents to learn optimal pruning policies.
* Potential Benefits: Can potentially discover more complex and adaptive pruning strategies.
* Considerations: Can be challenging to design effective reward functions and training procedures.
* Prelimary Results: TODO: Not Yet Implemented.
##### 7. Combining Pruning with Other Techniques
* Knowledge Distillation: After pruning a large model (teacher), use it to train a smaller, sparser model (student) to recover lost accuracy.
* Quantization: Combine pruning with weight quantization to further reduce model size and improve inference speed.
* Prelimary Results: TODO: Not Yet Implemented.
#### 5. (Preliminary) Experimental Results
(TODO: Thoroughly evaluate the pruned model's performance on relevant tasks and datasets to ensure it meets your accuracy and efficiency requirements.)

| Pruned Level | Wanda | SparseGPT | Magnitude | Movement |
|-----------|-----------------|-----------------|-----------------|-----------------|
| 0.01      | NA              |NA               |NA               | 5.677           |
| 0.05      | NA              |NA               |NA               | 5.714           |
| 0.10      | 5.696           |5.696            |5.806            | 5.806           |
| 0.20      | 5.817           |5.799            |6.020            | 6.020           |
| 0.30      | 5.999           |5.963            |6.669            | 6.668           |
| 0.40      | 6.387           |6.311            |8.601            | 8.5943          |
| 0.50      | 7.257           |7.234            |17.285           | 17.247          |
| 0.60      | 10.691          |10.442           |559.987          | 554.727         |
| 0.70      | 84.905          |27.214           |48414.551        | 51841.121       |
| 0.80      | 5782.432        |182.463          |132175.578       | 135494.797      |
| 0.90      | 19676.668       |3198.101         |317879.250       | 301472.500      |
| 0.95      | 28309.178       |4088.413         |273552.281       | 273629.750      |
| 0.99      | 108234.484      |16869.203        |222543.047       | 214966.484      |
* Table 1: Perplexity on pruned llama-7B models from Human Domain Experts.

| Pruned Level | Magnitude(weights) | Opposite Magnitude(-weights) |
|-----------|------------------|------------------|
| 0.01      | NA               | 24377.635        |
| 0.05      | NA               | 25804.920        |
| 0.10      | 5.806            | 104948.891       |
| 0.20      | 6.020            | 352772.500       |
| 0.30      | 6.669            | 335747.406       |
| 0.40      | 8.601            | 260632.641       |
| 0.50      | 17.285           | 227413.484       |
| 0.60      | 559.987          | 185086.078       |
| 0.70      | 48414.551        | 273153.688       |
| 0.80      | 132175.578       | 188488.000       |
| 0.90      | 317879.250       | 185304.016       |
| 0.95      | 273552.281       | NA               |
| 0.99      | 222543.047       | NA               |
* Table 2: The effectiveness of the weights as a major pruning measure

| Pruned Level | Magnitude(bias) | Opposite Magnitude(-bias) |
|-----------|------------------|------------------|
| 0.01      | | |
| 0.05      | | |
| 0.10      | | |
| 0.20      | | |
| 0.30      | | |
| 0.40      | | |
| 0.50      | | |
| 0.60      | | |
| 0.70      | | |
| 0.80      | | |
| 0.90      | | |
| 0.95      | | |
| 0.99      | | |
* Table 3: (TODO: The effectiveness of the bias as a major pruning measure)

| 算法编号 | 核心算法思想 | 运行状态 | 创意应用场景 | 创意灵感分数 | 降本增效分数 | 结论及归因 | 是否进行算法性能分析 |
|----------|----------|----------|----------|----------|----------|----------|----------|
| 01 | 梯度敏感剪枝 | 程序运行出错 | 代码生成 | 计算中 | 计算中 | 需由算法专家进一步参与定义算法逻辑 | 否 |
| 02 | L1范数剪枝 | 程序运行良好 | 代码生成 | 计算中 | 计算中 | 量化指标混沌指数高，不具实际使用价值 | 是 |
| 03 | 结构化剪枝 | 程序运行良好 | 代码生成 | 计算中 | 计算中 | 量化指标混沌指数高，不具实际使用价值 | 是 |
| 04 | K-means聚类剪枝 | 程序运行出错 | 代码生成 | 计算中 | 计算中 | 需运行环境提供更大GPU算力支持 | 否 |
| 05 | 随机剪枝-a | 程序运行良好 | 代码生成 | 计算中 | 计算中 | 效果仅和实验基线持平，需更有效算法设计 | 否 |
| 06 | Random Pattern Pruning-b | 程序运行良好 | 代码生成 | 计算中 | 计算中 | 效果仅和实验基线持平，需更有效算法设计 | 是 |
| 07 | Variational Dropout Pruning | 程序运行出错 | 代码生成 | 计算中 | 计算中 | 错误原因：算法无考虑不同矩阵维度的不同| 否 |
| 08 | Gradient-Based Pruning | 程序运行出错 | 代码生成 | 计算中 | 计算中 | 错误原因：算法无考虑边界 | 否 |
| 09 | Elastic Weight Consolidation Pruning | 程序运行出错 | 代码生成 | 计算中 | 计算中 | 错误原因：定位中| 否 |
| 10 | Dynamic Pruning with Reinforcement Learning | 程序运行出错 | 代码生成 | 计算中 | 计算中 | 错误原因：定位中 | 否 |
* Table 4: 一次性代码生成及有效性测试（o1模型）
* 由以上表格可知
* 1. o1模型在创意应用场景“核心算法生成”中难以做到一次生成有效算法，尽管我们已在实验中清楚明晰上下文及所涉及知识领域；
* 2. 我们针对每一个所设计算法新增“创意灵感分数”及“降本增效分数”。由于直到截稿前算法初始数据仍在收集中，我们将稍后把此部分重要数据支撑以补充材料的形式整理并提交；
* 3. 初步实验表明，2024年9月12日对外发布的o1模型，在宣传及传播中所强调的“慢思考”、“卓越数理逻辑推理”及“编程能力”并没有在我们的创新应用场景中展现出能被科学指标显著量化的过人之处；
* 4. 我们的未来工作可集中于以下几部分：（1）挖掘算法无法一次生成并成功运行的原因（2）AIGC生成算法与真人算法工程师所设计算法效能横向对比（3）从评估生成式人工智能“生代码”到更全面的评估如“生文”、“生图”及“生视频”上的综合表现（4）新增gpt-4, gemini pro等模型在垂类的横向对比等；

| Pruned Level | aigc_technique2 | aigc_technique3 | aigc_technique6 |
|----------|----------|----------|----------|
| 0.50 | 193740.406 | 266826.094 | 294350.188   |
| 0.60 | 110879.422 | 244139.875 | 138577.469   |
| 0.70 | 174815.859 | 453267.031 | 171725.375   |
| 0.80 | 287734.844 | 570346.750 | 186493.797   |
| 0.90 | 157028.844 | 384411.375 | 298142.469   |
| 0.95 | 90220.781  | 455298.469 | 187259.063   |
| 0.99 | 991519.125 | 206585.391 | 70452.703    |
* Table 5: Perplexity on pruned llama-7B models from AIGC Domain Experts(model: o1).

| Pruned Level | Perplexity | "University is" | 中文翻译（非模型原始生成内容） | 与真人价值对齐 |
|----------|----------|----------|----------|----------|
| 0.00 | * | University is a great place to learn about the world. | 大学是一个向世界学习的好地方。 | 是 |
| 0.50 | 19.191 | University is a great place to start a new year. | 大学是一个开始新的一年的好地方。 | 否 |
| 0.60 | 23.205 | University is a great place to start. | 大学是一个开始的好地方。 | 否 |
| 0.70 | 44.246 | University is a good place to get a good place to get a good place to get a good | 大学是一个好地方好地方好 | 否 |
| 0.80 | 364.304 | University is a lot lot lot lot lot lot lot lot lot lot lot lot lot lot lot lot | 大学是许多许多许多许多许多许多许多许多许多许多许多许多许多许多许多许多 | 否 |
| 0.90 | 3772.829 | University is. | 大学是. | 否 |
| 0.95 | 8892.167 | University is is is is is is is is is is is is is is is is is is | 大学是.................. | 否 |
| 0.99 | 22548.809 | University is is is is is is is is is is is is,,,,,, | 大学是是是是是是是是是是是是,,,,,, | 否 |
* Table 6: Example 1: the effect of pruned model(OPT-1.3B) for downstream text generation application
* 纵然Perplexity（混沌程度）是衡量一个语言模型有序性的重要学术界通用量化指标之一。模型的实际输出及与真人价值对齐在实际生产环境中是是分重要。故一个详细的A/B模型效果测试往往是大模型基座公司如OpenAI必须完成的部署步骤。
* 我们在这里希望给读者一个直观感受，即不同剪枝比例下模型输出质量的明显变化趋势，以彰显剪枝在实际生产中的重要应用价值。通过结合模型下游应用如文本生成（text generation），我们可以直观感受语言模型经剪枝后展现出的“不合理性”。
* 以上Table是一个初始实验结果，以"University is"作为词元序列开端，顺序生成，我们可以观察到：
* 1. OPT-1.3B模型在十亿参数量级时表现出一定程度的智能和价值对齐，但在剪枝程度大于0.5时生成的词元序列人类却难以理解，如“大学是是是是是是是是是是是是,,,,,,”的生成序列则另我们不知所云；
* 2. 十亿参数量的模型参数量过小，针对百亿，千亿参数量及万亿参数量的模型剪枝效果实验是我们的未来工作；
* 3. 我们需进一步把剪枝算法应用于不同体系架构的大模型中，如Transformer架构模型，Diffusion架构模型等。

* Table 7: (TODO: Running Time for each pruning algs.)

* Table 8: (TODO: End-End Unpruned & Pruned Model Evaluation)
![](./prune.fig1.v3.png)
* 由以上Fig初始实验结果，我们可知：
* 1. 随着剪枝程度的加深，从剪枝50%的神经元到剪枝95%的神经元，语言模型的内在混沌指数（Perplexity）呈现指数级别的上升。这并不理想，我们的目标是希望设计一种算法，使其Perplexity指数在高百分比剪枝的情况下，混沌指数只有线性轻微上升。
* 2. 三种主流剪枝算法横向对比中，在低百分比剪枝，即当Pruned_Level<=0.5时，三种算法表现不相伯仲。在高百分比剪枝，即当Pruned_Level > 0.6时，SparseGPT算法表现比其余两种算法有明显优势。这可能因为以下原因：（1）SparseGPT's Pruning Strategy: SparseGPT likely employs a more sophisticated pruning strategy compared to Wanda and Magnitude. It might be selectively removing less important connections in the model, even at high pruning levels. (2) Wanda and Magnitude's Sensitivity: Wanda and Magnitude might be more sensitive to high pruning levels. (3) Dataset Characteristics: The dataset used for evaluation plays a crucial role. SparseGPT's advantage might be more pronounced on certain types of data. (4) Hyperparameter Tuning: The performance of pruning methods is sensitive to hyperparameters. SparseGPT might be benefiting from better hyperparameter optimization for this specific scenario.
* 3. 对于7B参数级别的LLM，我们相信，随着其内部混沌指数上升，模型向外输出的文本质量会呈现下降趋势，性能会有一定幅度提升。我们将在未来汇报被剪枝模型向外输出文本质量的实验结果。
* 4. 后续在有足够算力支撑下，我们会陆续汇报在十亿，百亿及千亿规模参数量下LLM经剪枝算法后的性能与质量trade-off，并为进一步探寻MoE混合专家架构(the tiering problem)做前置实验分析准备。
* 5. 我们后续将同时汇报在不同语言大模型的混沌指数横向对比分析，如主语言为中文的智谱清言、主语言为英文的llama及阿拉伯文为主的语言模型等。
* 6. 我们尝试了Movement Pruning剪枝方法，量化实验表明和Magnitude方法在Perplexity量化评估指标上相差不大。此算法背后的核心设计思想是：为保证单个神经元保有足够信息流，需把目标权重在剪枝前移到其他同元连接上。
![](./prune.fig2.v1.png)
* Here are some of our observations from the above figure:
* 1. Y-Axis (Perplexity) Range: Perplexity measures how well a model predicts sample data, and lower values generally indicate better performance.
* 2. X-Axis (Pruned Level): The x-axis indicates the level of pruning, ranging from 0 to 1, with 0 being no pruning and 1 being full pruning.
* 3. Magnitude (Green): The green dashed line represents the perplexity for the "Magnitude" approach. Perplexity remains relatively low for lower levels of pruning (e.g., around 5.8 to 8.6 for pruning levels of 0.0 to 0.4). There is a significant jump in perplexity from pruning level 0.6 onwards, reaching 48,414.6 at pruning level 0.8 and further increasing to over 300,000 by pruning level 1.0, indicating that higher pruning severely worsens model performance.
* 4. Opposite Magnitude (Blue): The solid blue line represents the perplexity for the "Opposite Magnitude" approach. Perplexity starts at a higher value compared to the "Magnitude" approach and remains consistently high across all pruning levels. The perplexity peaks at around pruning level 0.2, reaching over 350,000, but then drops slightly for higher pruning levels, fluctuating between 180,000 and 300,000 as pruning increases beyond 0.4.
* 5. Key Takeaway: the feature weights (also called magnitude) is important.
![](./prune.fig3.v2.png)
* 在10个由o1模型经一次性代码生成的AIGC剪枝算法集合中，我们选择其中能“一次生成通过测试并能稳定运行在实验环境”的3个AIGC算法（即算法编号为2、3及6），和人工研究者所设计的剪枝算法集作横向性能对比，并汇报其Perplexity量化指标，初始实验结果十分有趣（见上图），我们的初步洞察如下：
* 1. 由o1模型生成的AIGC算法（图中标记为Machine）在不同百分比的剪枝中，混沌程度相对较高。我们推断的原因是：AIGC算法无法抓住在特定垂类场景如大模型剪枝下的关键逻辑影响因子及信号，如神经元权重（weight），偏置（bias）及外部激活（activations）；
* 2. 真人算法从业者所设计的算法在中百分比剪枝中，如剪枝范围在0.5-0.9时，算法在量化指标Perplexity上优势明显；
* 3. 无论是由o1模型生成的AIGC算法集合，还是由真人算法从业者设计的算法，在高百分比剪枝中，如剪枝范围在0.9-1.0时，无论是机器还是真人算法，混沌值都很高（尽管我们尚未定义什么是“不能在实际使用的语言模型”）。机器在这个范围内经剪枝后的语言模型混沌程度均值达到315657.624，真人设计的算法效果则稍好，但混沌程度均值也达到惊人的149544.114；
* 4. 大模型剪枝算法研究是深远且具有重要现实研究意义，我们一方面能从代码生成的角度探寻业界通用大模型（如o1）在创意灵感、推理及代码生成上的表现；另一方面，研究剪枝算法具有普适性意义。我们相信，随着模型参数量的不断增大，能进行性能优化及剪枝的空间将变得具体，且能被社会科学及自然科学研究方法所捕捉和感知。这在一定程度上，为世界各地的研究团队及个人开启了生成式人工智能研究新篇章。
* 我们的未来工作包含：如何在认知计算广告知识体系下对多模态内容感知、生成及个性化创意应用场景进行更细化的研究。在这里，我们设定的创意应用场景为特定垂类下的“代码生成”。
#### 6. Conclusions
* In this paper, we have introduced several novel algorithms for static pruning in large language models and large vision models. Through comparison with query wheel and query covering approaches, our methodology, which attempts to estimate the likelihood of neurons resulting in top results based on diverse neuron features, collections, and query statistics, has demonstrated measurable improvement over prior work as evidenced by our experimental results.
* For future work, we plan several extensions. This includes conducting experiments with other language models that may potentially achieve even better pruning performances. We also aim to optimize our approach further, such as exploring hybrid methods. Additionally, we plan to study the tradeoff between model size and query cost under different cost models and for actual query processing algorithms. This research holds promise for enhancing the efficiency and performance of large language and vision models through more effective static pruning techniques.
#### References
1. Improved Methods for Static Index Pruning. W. Jiang, J. Rodriguez, and T. Suel, IEEE International Conference on Big Data, December 2016. http://engineering.nyu.edu/~suel/papers/prune.pdf
2. Exploring Size-Speed Trade-Offs in Static Index Pruning. J. Rodriguez and T. Suel, IEEE International Conference on Big Data, December 2018. http://engineering.nyu.edu/~suel/papers/prunetrade-bd18.pdf
3. A Simple and Effective Pruning Approach for Large Language Models. Mingjie Sun, Zhuang Liu, Anna Bair, J. Zico Kolter, ICLR Poster, 2024. https://openreview.net/forum?id=PxoFut3dWW
4. Rethinking the Value of Network Pruning. Zhuang Liu, Mingjie Sun, Tinghui Zhou, Gao Huang, Trevor Darrell, ICLR, 2019.
5. SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot. Elias Frantar, Dan Alistarh, ICML Oral, 2023.
6. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han, MLSys, 2024 (Best Paper Award).
7. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. Dettmers et al, NeurIPS 2022.
8. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. Xiao et al, ICML 2023.
9. Movement pruning: Adaptive sparsity by fine-tuning. Sanh et al, 2020.
10. Platon: Pruning large transformer models with upper confidence bound of weight importance. Zhang et al, 2022.
11. Pruning Pre-trained Language Models with Principled Importance and Self-regularization. Ren et al, 2023.
12. LLM-Pruner: On the Structural Pruning of Large Language Models. Ma et al, NeurIPS 2023.
13. The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. 2019.
14. Rigging the Lottery: Making All Tickets Winners. Evci et al, ICML 2020.
15. Molchanov, Pavlo, et al. "Importance estimation for neural network pruning." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.
16. LeCun, Yann, John Denker, and Sara Solla. "Optimal brain damage." Advances in neural information processing systems 2 (1989).
17. SNIP: Single-shot Network Pruning based on Connection Sensitivity. Lee et al, ICLR 2019.
18. Song Han, Jeff Pool, John Tran, and William J Dally. Learning both weights and connections for efficient neural networks. In NeurIPS, 2015.
19. Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov. A Simple Way to Prevent Neural Networks from Overfitting. JMLR, 2014.
20. Michael Zhu, Suyog Gupta. To Prune, or Not to Prune: Exploring the Efficacy of Pruning for Model Compression. ICLR (Workshop) 2018.

### （中阶）子课题及论文题目（拟） - 9 - Improved Methods for Dynamic Neural Network Pruning
![](./img/sub.topic.9.png)

### （高阶）子课题及论文题目（拟） - 10 - Improved Methods for  Neural Network Tiering
![](./img/sub.topic.10.png)

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
