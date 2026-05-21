# ST4001 毕业论文中文审阅版草稿

> 说明：本文件是为了方便中文阅读、理解和审查论文逻辑而生成的审阅版，不是最终提交版。最终提交仍建议使用英文 Word 模板稿，并按 ZJUI Appendix VI 模板排版。

题目暂定：基于 Transformer 的低剂量 CT 图像去噪方法与残差噪声分析  
英文题目暂定：Transformer-Based Low-Dose CT Image Denoising with CTRestormer and Residual Noise Analysis

## 一、是否还需要继续填充

需要继续填充。当前 `v4_standardized` 已经把论文的核心框架搭起来了，也已经加入了比较有价值的 residual noise analysis，但是如果目标是“优秀毕业论文”，目前仍有几个明显短板：

1. 正文体量仍偏短。当前英文稿大约 5600 个英文词，虽然图表较多，但距离稳定满足“正文不少于 20 页”以及优秀论文的充分论证还不够保险。
2. 文献综述还需要扩展。现在已经满足至少 10 条 references 的硬性要求，但 Chapter 2 还需要更系统地说明 LDCT 去噪、CNN、Transformer、self-supervised denoising 和 residual analysis 的关系。
3. 方法章节还需要更像“可复现论文”。需要进一步写清楚数据预处理、归一化、patch 训练、tiled inference、loss、checkpoint 选择、指标计算方式。
4. 贡献部分方向是对的，但还可以再强化。论文应该明确突出：我们不仅复现模型，还完成了一个可运行的 LDCT 去噪 pipeline，提出/适配了 CTRestormer，并加入 residual noise map 分析来解释模型行为。
5. residual analysis 目前主要是代表性切片 L506_88 的分析。论文里可以使用它作为亮点，但需要诚实说明它还不是全测试集统计证明。优秀版最好继续把 residual metrics 跑完整个 L506。

因此我的判断是：**继续填充是必要的**。但填充不应该只是机械加字，而应该围绕“贡献、方法可复现、实验可解释、图表规范”四个方向补。

## 二、中文摘要审阅版

低剂量 CT 可以降低患者在医学影像检查中的辐射暴露，但剂量降低会引入更强的量子噪声和条纹状伪影，从而影响组织边界、低对比度结构以及潜在病灶的观察。为了改善低剂量 CT 图像质量，本文围绕深度学习低剂量 CT 去噪任务，构建了一个从数据准备、模型训练、推理评估到残差分析的完整实验流程。

本文使用 AAPM-Mayo 低剂量 CT 数据，采用低剂量图像到常规剂量图像的配对监督学习设置。实验中选取 RED-CNN 作为卷积神经网络基线，选取 CTformer 作为 Transformer 类低剂量 CT 去噪基线，并进一步适配 Restormer 风格的图像复原网络，构建 CTRestormer 模型。为了在本地有限 GPU 资源下处理 512 × 512 的完整 CT 切片，本文采用 patch-based training、混合精度训练、梯度累积以及 tiled inference 等工程策略。

在 held-out patient L506 上，低剂量输入图像的 PSNR、SSIM 和 RMSE 分别为 29.2489、0.8759 和 14.2416。训练后的模型均显著改善图像质量，其中 CTRestormer 21500 iteration checkpoint 达到 PSNR 32.6738、SSIM 0.9096、RMSE 9.4842，表现接近或略优于 RED-CNN，并优于早期 CTRestormer checkpoints。除了常规指标，本文还进一步引入 residual noise analysis，将真实近似噪声 `input - target` 与模型移除残差 `input - prediction` 进行比较，并从残差相关性、边缘泄露、残差强度、直方图和频谱角度分析不同模型的去噪行为。该分析表明，单纯依赖 PSNR/SSIM 不足以判断模型是否存在过度平滑风险，残差图可以帮助观察模型移除的是噪声纹理还是潜在结构信息。

本文的主要贡献包括：第一，完成了一个可复现的低剂量 CT 去噪实验 pipeline；第二，面向 CT 去噪任务适配了 Restormer 风格的 CTRestormer 模型，并在有限硬件条件下实现完整切片推理；第三，在常规指标之外加入 residual noise map 分析，为模型是否依赖过度平滑取胜提供了更细粒度的解释依据。

关键词：低剂量 CT；图像去噪；Transformer；CTRestormer；残差噪声分析；医学图像复原

## 三、推荐论文结构

最终英文论文建议按照以下结构组织：

1. Chapter 1 Introduction
2. Chapter 2 Literature Review
3. Chapter 3 Methodology
4. Chapter 4 Experimental Setup
5. Chapter 5 Experimental Results
6. Chapter 6 Residual Noise Analysis and Discussion
7. Chapter 7 Conclusion and Future Work
8. References
9. Appendix

这版结构的关键是：**把 residual noise analysis 从附加实验提升成论文亮点之一**。这样论文就不只是“训练几个模型比指标”，而是有更明确的研究问题：模型到底移除了什么？它是否只是把图像抹平？

## 四、Chapter 1 Introduction 中文审阅稿

### 1.1 研究背景

CT 是临床诊断中非常重要的医学影像技术，它能够提供人体内部组织的横断面结构信息，并广泛用于肺部、腹部、血管、肿瘤和创伤等场景。然而，CT 成像依赖 X 射线，因此患者会受到一定辐射剂量。降低 CT 辐射剂量是医学影像研究和临床应用中的重要目标，尤其对于需要多次随访检查的患者、儿童患者以及筛查场景具有现实意义。

低剂量 CT 的主要问题是图像噪声和伪影增加。由于投影数据中的光子数减少，重建图像往往出现更明显的颗粒噪声、条纹状伪影以及局部结构模糊。这些退化会降低图像视觉质量，也可能影响低对比度病灶、器官边界和细小结构的观察。因此，如何在降低辐射剂量的同时保持图像可诊断性，是低剂量 CT 研究中的核心问题。

传统去噪方法通常依赖滤波、迭代重建或先验模型。虽然这些方法具有一定解释性，但往往需要复杂参数调节，并且在强噪声情况下可能出现细节损失。近年来，深度学习方法在医学图像复原中表现突出。卷积神经网络能够从配对数据中学习低剂量图像到常规剂量图像的映射，而 Transformer 类模型进一步引入长距离依赖建模能力，为复杂噪声分布和全局结构恢复提供了新的可能。

### 1.2 研究问题

本文关注的问题不是简单地问“哪个模型的 PSNR 更高”，而是进一步追问：

1. 在相同数据设置下，CNN、CT-specific Transformer 和 Restormer-style Transformer 在低剂量 CT 去噪任务中表现如何？
2. CTRestormer 是否能够在有限硬件条件下有效处理完整 512 × 512 CT 切片？
3. 不同模型移除的 residual 是否主要对应低剂量噪声，还是包含较多解剖结构边缘？
4. residual noise map 是否能帮助解释模型是否存在过度平滑风险？

### 1.3 本文贡献

本文贡献建议在英文终稿中明确写成 numbered list，并放在 Chapter 1 前半部分。中文理解如下：

**贡献一：构建完整低剂量 CT 去噪实验 pipeline。**  
本文从 AAPM-Mayo 配对 CT 数据出发，完成数据整理、patient-level split、patch-based training、checkpoint 保存、完整切片推理、指标评估、图像可视化和结果汇总，形成了一个可复现的实验流程。

**贡献二：适配 CTRestormer 到低剂量 CT 去噪任务。**  
本文在 Restormer 风格图像复原思想基础上，构建用于低剂量 CT 去噪的 CTRestormer。该模型利用层级式 encoder-decoder、overlap patch embedding、Transformer restoration blocks 和 residual restoration 策略，在有限 GPU 显存下通过 patch training 与 tiled inference 完成完整 CT 切片处理。

**贡献三：与 RED-CNN 和 CTformer 进行公平比较。**  
RED-CNN 代表经典 CNN-based LDCT denoising 方法，CTformer 代表面向 CT 去噪的 Transformer 方法。通过将 CTRestormer 与二者比较，论文能够说明 proposed model 在不同模型家族中的位置，而不是只展示单一模型结果。

**贡献四：引入 residual noise analysis。**  
本文不仅报告 PSNR、SSIM、RMSE，还进一步分析 `input - target`、`input - prediction` 和 `prediction - target`。通过 residual correlation、edge leakage、residual histogram 和 power spectrum，可以观察模型移除的信号是否接近噪声，或者是否包含较多结构边缘。

**贡献五：记录有限硬件下的实践经验。**  
论文记录了 patch-based training、AMP、gradient accumulation 和 tiled inference 等工程策略。这些内容对本科毕业论文很重要，因为它体现了实际完成系统的能力，而不只是理论描述。

### 1.4 章节安排

Chapter 2 综述低剂量 CT 去噪、CNN 方法、Transformer 图像复原和 self-supervised denoising 的相关研究。Chapter 3 介绍数据、模型、CTRestormer 结构和 residual analysis 方法。Chapter 4 描述实验设置、训练配置和评价指标。Chapter 5 给出定量和定性结果。Chapter 6 重点讨论 residual noise analysis，并分析模型是否可能通过过度平滑获得较好指标。Chapter 7 总结论文贡献并提出未来工作。

## 五、Chapter 2 Literature Review 中文审阅稿

### 2.1 低剂量 CT 去噪

低剂量 CT 的目标是在降低 X 射线剂量的同时尽可能保持图像质量。剂量降低通常会导致投影数据噪声增加，从而在重建图像中表现为随机噪声、条纹伪影和局部纹理退化。去噪方法大致可以分为 reconstruction-domain 方法、image-domain 后处理方法以及基于深度学习的端到端复原方法。

本文采用 image-domain denoising 设置，即输入低剂量 CT 图像，输出接近常规剂量 CT 的图像。这种设置便于利用 paired LDCT/NDCT 数据进行监督学习，也便于将不同神经网络模型放在统一评价协议下比较。

### 2.2 CNN-based 方法

CNN 是医学图像去噪中非常重要的一类方法。RED-CNN 使用 residual encoder-decoder 结构进行低剂量 CT 去噪，是该领域常见基线。CNN 的优势是局部特征提取能力强、训练稳定、推理成本相对可控。对于 CT 图像中的局部噪声和边缘结构，卷积核能够学习到较有效的局部映射。

但 CNN 的局限也很明显。卷积操作天然偏局部，如果要捕捉长距离结构关系，需要更深的网络、更大的 receptive field 或额外的多尺度设计。此外，CNN 模型有时会倾向于平滑输出，从而提高全局指标但损失细节。因此，在论文中使用 RED-CNN 作为 baseline 是合理的：它既是强基线，也能帮助比较 Transformer 类模型是否在结构保持或 residual 行为上有不同表现。

### 2.3 Transformer-based 图像复原

Transformer 最初用于序列建模，后来被引入视觉任务。相比 CNN，Transformer 的优势在于 attention 机制可以建模更远距离的信息交互。对于 CT 图像，这一点可能有帮助，因为噪声分布和解剖结构并不完全局限于小窗口区域。

CTformer 是针对低剂量 CT 去噪设计的 Transformer 类模型，说明 Transformer 在 CT 去噪任务中具有实际潜力。另一方面，Restormer 是图像复原领域的重要模型，它通过更高效的 attention 和层级式结构处理高分辨率图像。本文的 CTRestormer 正是在这一方向上进行任务适配：将 Restormer-style restoration 思想用于低剂量 CT，并结合本地硬件限制设计训练和推理流程。

### 2.4 Self-supervised 和 Noise2Noise 思路

Noise2Noise 说明，在某些条件下，即使没有干净 target，也可以从 noisy-to-noisy 的训练中学习到去噪映射。Noise2Void 和 Noise2Self 进一步发展了 blind/self-supervised denoising 思路。这类方法对于医学影像很有意义，因为真实 paired clean target 有时难以获得。

在本文中，Noise2Noise 不是主要已完成实验，而是一个可以扩展的未来方向。当前论文主要使用 paired supervised setting，但 residual analysis 可以自然连接到 self-supervised 扩展：如果未来没有 NDCT target，可以考虑利用多次扫描、邻近切片、合成噪声或盲点网络来设计训练目标。

### 2.5 文献缺口与本文定位

现有研究通常重点报告 PSNR、SSIM、RMSE 等指标，但这些指标无法完全说明模型是否保留了解剖结构。例如，一个模型可能通过更强平滑获得较高 PSNR，却同时削弱边缘和细小结构。对于医学图像，这种风险尤其需要注意。

本文的定位是：在常规指标比较之外，加入 residual noise analysis 来观察模型移除了什么。通过比较 `input - target` 与 `input - prediction`，可以判断模型移除的 residual 是否与真实低剂量噪声相似。通过 edge leakage，可以进一步估计移除 residual 中是否包含较强结构边缘。这个角度是本文区别于普通复现实验的重要亮点。

## 六、Chapter 3 Methodology 中文审阅稿

### 3.1 问题定义

本文将低剂量 CT 去噪定义为配对图像复原问题。给定低剂量 CT 输入图像 \(x\)，目标是学习一个模型 \(f_\theta\)，输出预测图像 \(\hat{y} = f_\theta(x)\)，使其尽可能接近常规剂量 CT target \(y\)。模型训练目标通常是最小化 \(\hat{y}\) 与 \(y\) 之间的像素误差或混合损失。

评价时使用 PSNR、SSIM 和 RMSE 衡量整体图像质量。PSNR 和 SSIM 越高越好，RMSE 越低越好。但这些指标不直接告诉我们模型移除的内容是什么，因此本文进一步定义 residual analysis。

### 3.2 数据集与预处理

实验使用 AAPM-Mayo 低剂量 CT 数据中的 3mm B30 paired slices。数据包含多个 patient 的低剂量和常规剂量 CT 图像。本文当前使用的 patient IDs 包括 L067、L096、L109、L143、L192、L286、L291、L310、L333 和 L506，其中 L506 被作为 held-out test patient。

采用 patient-level split 是必要的，因为如果同一 patient 的不同切片同时出现在训练和测试中，模型评估可能产生数据泄漏，导致结果过于乐观。使用 L506 作为独立测试 patient，可以更合理地评估模型对未见 patient 的泛化能力。

图像大小为 512 × 512。训练阶段为了降低显存压力，使用 patch-based training；推理阶段为了得到完整切片输出，使用 tiled inference。对于 CT 图像显示，建议在论文所有 qualitative figures 中统一 CT window，并在 caption 中说明 window 或 residual scale。

### 3.3 Baseline Models

**RED-CNN** 是 CNN-based LDCT denoising 的经典方法。它通过 encoder-decoder 和 residual connection 学习低剂量图像到常规剂量图像的映射。本文使用 RED-CNN 作为强基线，用于比较 CTRestormer 是否能达到或超过经典 CNN 方法。

**CTformer** 是 Transformer-based LDCT denoising 方法，代表面向 CT 任务设计的 Transformer baseline。使用 CTformer 可以帮助判断 CTRestormer 与已有 CT-specific Transformer 的相对表现。

### 3.4 Proposed CTRestormer

CTRestormer 是本文重点适配的模型。它基于 Restormer-style restoration framework，用于低剂量 CT 去噪任务。模型核心思想包括：

1. 使用 overlap patch embedding 保留局部连续性。
2. 使用层级式 encoder-decoder 建模多尺度特征。
3. 使用 Transformer restoration blocks 捕捉长距离依赖。
4. 使用 residual restoration 思路，让模型学习从低剂量图像到常规剂量图像的校正。
5. 使用 tiled inference 处理完整 512 × 512 CT 切片。

这个模型的价值不只是结构本身，也包括实际可运行性。Transformer 类模型在高分辨率图像上往往显存开销较大，因此 patch training、gradient accumulation 和 tiled inference 是论文中必须写清楚的工程贡献。

### 3.5 Residual Noise Analysis 方法

本文 residual analysis 使用三个核心量：

1. 真实近似噪声：`input - target`
2. 模型移除残差：`input - prediction`
3. 预测误差：`prediction - target`

如果一个模型真的在去除低剂量噪声，那么 `input - prediction` 应该与 `input - target` 在纹理分布上有一定相似性。但如果 `input - prediction` 中包含大量器官边界或结构轮廓，则说明模型可能在去除噪声的同时削弱了解剖结构。

因此本文使用以下指标辅助分析：

1. Residual correlation：衡量模型移除残差与真实近似噪声之间的相关性。
2. Removed residual standard deviation：衡量模型移除信号强度。
3. Edge leakage：衡量移除 residual 中结构边缘成分的比例或强度。
4. Residual histogram：观察残差值分布。
5. Power spectrum：观察残差在不同频率上的分布。

这部分分析可以回答你之前问的那句话：“residual noise analysis 证明它不是靠过度平滑取胜”是什么意思。更准确地说，它不是绝对证明，而是提供证据：如果模型移除 residual 与真实噪声相似，且 edge leakage 不高，那么它更可能是在去噪；如果移除 residual 中边缘很多，就要警惕它可能也抹掉了解剖结构。

## 七、Chapter 4 Experimental Setup 中文审阅稿

### 4.1 数据划分

| 项目 | 设置 | 说明 |
|---|---|---|
| 数据集 | AAPM-Mayo 3mm B30 paired LDCT/NDCT | 低剂量到常规剂量监督去噪 |
| 图像大小 | 512 × 512 | 完整 CT 切片 |
| Patient IDs | L067, L096, L109, L143, L192, L286, L291, L310, L333, L506 | 多 patient 数据 |
| 测试 patient | L506 | held-out patient-level test |
| 任务 | LDCT → NDCT | image-domain denoising |

### 4.2 模型配置

| 模型 | 类型 | 论文中的作用 |
|---|---|---|
| Low-dose input | 未处理输入 | 作为下限 baseline |
| RED-CNN | CNN denoiser | 经典强基线 |
| CTformer | Transformer CT denoiser | CT-specific Transformer baseline |
| CTRestormer | Restormer-style Transformer | 本文适配的主要模型 |

### 4.3 评价指标

| 指标 | 趋势 | 含义 |
|---|---|---|
| PSNR | 越高越好 | 衡量像素误差对应的峰值信噪比 |
| SSIM | 越高越好 | 衡量结构相似性 |
| RMSE | 越低越好 | 衡量预测与 target 的均方根误差 |
| Residual correlation | 需结合解释 | 衡量移除 residual 与真实近似噪声的一致性 |
| Edge leakage | 通常越低越安全 | 衡量移除 residual 中可能包含的结构边缘 |

注意：PSNR/SSIM/RMSE 是主结果指标，residual metrics 是解释模型行为的辅助指标。论文中不能把 residual metrics 说成临床安全性的直接证明，只能说它们提供了模型是否过度平滑的分析线索。

## 八、Chapter 5 Experimental Results 中文审阅稿

### 5.1 定量结果

当前主要结果如下：

| 方法 | PSNR ↑ | SSIM ↑ | RMSE ↓ |
|---|---:|---:|---:|
| Low-dose input | 29.2489 | 0.8759 | 14.2416 |
| CTRestormer 11500 | 32.1294 | 0.8997 | 10.0689 |
| CTRestormer 16500 | 32.3584 | 0.9049 | 9.8145 |
| CTRestormer 18000 | 32.3534 | 0.9038 | 9.8134 |
| CTformer | 32.3852 | 0.9026 | 9.8366 |
| RED-CNN | 32.6656 | 0.9067 | 9.4867 |
| CTRestormer 21500 | 32.6738 | 0.9096 | 9.4842 |

从结果看，所有深度学习模型都明显优于 low-dose input，说明监督去噪有效。CTRestormer 随训练推进整体性能提升，21500 iteration checkpoint 达到最优或接近最优表现。与 RED-CNN 相比，CTRestormer 21500 的 PSNR 和 RMSE 非常接近，并且 SSIM 略高。与 CTformer 相比，CTRestormer 21500 在三个指标上均有提升。

这组结果可以支撑一个比较稳健的结论：**CTRestormer 在当前实验设置下是有效的，并且能够达到强 CNN baseline 的水平，同时优于当前 CTformer baseline。**

### 5.2 Checkpoint 分析

CTRestormer 的不同 checkpoint 显示出训练过程中的性能变化。11500 iteration 时已经显著优于 low-dose input，但与后续 checkpoint 相比仍有差距。16500 和 18000 的 PSNR/RMSE 接近，说明中期训练进入较稳定阶段。21500 checkpoint 进一步提升 SSIM，并达到最佳 RMSE。

这部分对论文很有价值，因为它说明模型不是只汇报一个偶然结果，而是观察了训练进程。英文终稿中建议加入一句：checkpoint comparison helps verify that the reported CTRestormer performance is consistent with training progression rather than an isolated random checkpoint.

### 5.3 定性结果

代表性切片 L506_88 的 qualitative comparison 应该重点观察三点：

1. 低剂量输入是否有明显噪声纹理。
2. 模型输出是否更接近 normal-dose target。
3. 输出是否出现过度平滑、边缘变钝或结构丢失。

论文 caption 中建议写清楚：slice ID、模型 checkpoint、CT window。图像本身要保证相同窗口、相同大小、相同裁剪区域，否则比较不公平。

## 九、Chapter 6 Residual Noise Analysis and Discussion 中文审阅稿

### 6.1 为什么需要 residual analysis

PSNR、SSIM、RMSE 可以告诉我们 prediction 与 target 的整体接近程度，但不能告诉我们模型到底移除了什么。如果模型输出看起来更平滑，指标可能变好，但它也可能把一些真实结构细节一起去掉。对于医学图像，这个问题很关键，因为临床影像不仅需要视觉干净，也需要保留病灶、组织边界和细微结构。

Residual noise map 的核心思想是：不要只看 denoised image，还要看模型从输入中减掉了什么。

如果 `input - prediction` 主要是随机噪声状纹理，并且与 `input - target` 接近，则说明模型更可能是在去噪。如果 `input - prediction` 中出现明显器官边缘、骨结构边界或组织轮廓，则说明模型可能也移除了结构信息。

### 6.2 当前 residual 结果

当前 L506_88 的 residual metrics 如下：

| 模型 | Residual correlation | Edge leakage | Removed std | Removed MAE | Error MAE |
|---|---:|---:|---:|---:|---:|
| CTformer | 0.7259 | 0.2506 | 9.8406 | 4.3490 | 4.1493 |
| CTRestormer | 0.7300 | 0.3699 | 11.3863 | 5.2081 | 4.1636 |
| RED-CNN | 0.7358 | 0.3683 | 11.7800 | 5.3967 | 4.1823 |

可以这样理解：

1. 三个模型的 residual correlation 都在 0.72-0.74 左右，说明它们移除的 residual 与 `input - target` 有较强相关性。
2. CTformer 的 edge leakage 较低，说明它的去噪更保守，移除的结构边缘相对少。
3. CTRestormer 和 RED-CNN 的 removed std 更高，说明它们移除的 residual 强度更大，去噪更积极。
4. CTRestormer 的 edge leakage 接近 RED-CNN，说明它虽然指标优秀，但也需要关注是否有更强结构抑制倾向。
5. 从 Error MAE 看，三者在该切片上差距不大，说明 residual behavior 提供了不同于普通误差指标的信息。

### 6.3 如何写“它不是靠过度平滑取胜”

论文中不要写成绝对句：

错误表达：Residual noise analysis proves that the model is not over-smoothing.

建议表达：

Residual noise analysis provides additional evidence that the model does not rely solely on uniform smoothing. The removed residual shows substantial correlation with the estimated LDCT noise, while the prediction error map remains limited. However, because edge leakage is not zero and the current analysis is based on a representative slice, the conclusion should be interpreted as model-behavior evidence rather than clinical proof.

中文意思是：residual analysis 可以提供支持证据，但不能单独构成严格证明。它能说明模型移除的东西比较像低剂量噪声，而不是完全随机地把图像抹平；但如果 edge leakage 较高，仍需要谨慎。

### 6.4 标准化图表怎么读

当前已经生成的标准化图组建议这样理解：

1. `L506_88_prediction_comparison_standard.png`  
   用来看输入、target 和各模型 prediction 的整体视觉效果。

2. `L506_88_removed_residuals_standard.png`  
   用来看每个模型到底从 low-dose input 中减去了什么。重点观察 residual 是否呈随机噪声纹理，还是出现明显结构轮廓。

3. `L506_88_prediction_errors_standard.png`  
   用来看 prediction 与 target 之间还剩哪些误差。误差集中在边缘区域，说明模型在结构位置仍有偏差。

4. `L506_88_residual_metrics_bar_standard.png`  
   用统计柱状图把 residual correlation 和 edge leakage 可视化，帮助读者快速比较模型行为差异。

### 6.5 本章结论

Residual analysis 给论文增加了一个重要解释层。定量指标说明 CTRestormer 21500 表现优秀，而 residual analysis 进一步说明它移除的 residual 与真实近似噪声有较强相关性。但它也显示 CTRestormer 的 edge leakage 高于 CTformer，接近 RED-CNN。因此论文应采取平衡表述：CTRestormer 在整体指标上表现强，但 residual behavior 说明仍需要进一步评估结构保持能力。

## 十、Chapter 7 Conclusion and Future Work 中文审阅稿

### 7.1 结论

本文围绕低剂量 CT 图像去噪任务，构建并评估了多个深度学习模型。实验表明，与原始 low-dose input 相比，RED-CNN、CTformer 和 CTRestormer 均显著提升 PSNR、SSIM 并降低 RMSE。CTRestormer 21500 checkpoint 在 held-out patient L506 上达到 PSNR 32.6738、SSIM 0.9096、RMSE 9.4842，说明 Restormer-style Transformer 在低剂量 CT 去噪任务中具有有效性。

除了常规图像质量指标，本文引入 residual noise analysis 来解释模型行为。通过比较 `input - target` 与 `input - prediction`，论文分析了不同模型移除 residual 的强度、相关性和边缘泄露。该分析显示，CTformer 的 residual removal 更保守，而 CTRestormer 与 RED-CNN 移除更强 residual，并在指标上取得更好表现。这说明 residual map 可以帮助理解模型是否可能通过过度平滑获得指标优势。

总体而言，本文的价值不只是训练出一个较好模型，而是完成了一个较完整的低剂量 CT 去噪研究流程：包括数据、模型、训练、推理、指标、可视化和 residual 解释分析。

### 7.2 局限性

当前论文仍有以下局限：

1. residual visual analysis 主要基于 L506_88 代表性切片，还没有扩展到全部 L506 测试切片。
2. 当前实验主要使用 PSNR、SSIM、RMSE，缺少 ROI-based 或 task-based clinical evaluation。
3. CTRestormer 的结构消融实验还不充分，例如 attention block、patch size、loss function 等因素没有逐一比较。
4. 论文尚未完成 Noise2Noise 或 self-supervised denoising 的正式实验，目前这部分更适合作为 future work。
5. 由于本地硬件限制，训练规模和超参数搜索范围有限。

### 7.3 未来工作

下一步最值得做的扩展包括：

1. 将 residual metrics 扩展到 L506 全部切片，报告 mean ± std。
2. 增加 ROI-based evaluation，例如 soft tissue、bone boundary、low-contrast region。
3. 增加 Noise2Noise 或 pseudo-Noise2Noise 实验，探索没有 clean target 时的去噪能力。
4. 增加 ablation study，例如去掉 tiled inference、改变 patch size、比较不同 loss。
5. 增加效率分析，包括参数量、推理时间、显存占用。

## 十一、当前英文终稿继续填充建议

我建议下一轮重点做三件事：

1. **扩写 Chapter 2 Literature Review。**  
   目标是至少增加 1500-2500 个英文词。不要只堆 references，要写清楚研究脉络：传统 LDCT denoising → CNN → Transformer → self-supervised → residual/interpretability gap。

2. **扩写 Chapter 3 Methodology。**  
   目标是至少增加 1500-2000 个英文词。重点写数据、预处理、模型结构、训练策略、tiled inference、residual metrics 公式。

3. **扩展 residual analysis 到全 L506。**  
   如果时间允许，跑全测试 patient 的 residual metrics。这样 Chapter 6 就不只是 case study，而可以写成 test-patient-level analysis。

## 十二、你审查时重点看什么

你读这版中文稿时，建议重点检查：

1. 贡献是否符合你实际做过的事情。
2. CTRestormer 的描述是否与代码实现一致。
3. 数据集、patient split、checkpoint 数字是否准确。
4. residual analysis 的解释是否你能接受。
5. 未来工作里 Noise2Noise 是否要写成“已完成实验”还是“未来扩展”。目前我建议写成未来扩展，除非我们真的跑完。

我的建议是：英文正式版继续写，但中文审阅版先帮你把逻辑捋顺。只要你确认这版中文逻辑没问题，我就可以继续把对应内容扩写进英文 Word 终稿。
