# 基于 Transformer 的低剂量 CT 图像去噪：AAPM-Mayo 数据上的比较研究

## 摘要

低剂量计算机断层成像（Low-Dose Computed Tomography, LDCT）能够降低患者接受的电离辐射剂量，因此在临床筛查、随访和重复检查中具有重要意义。然而，降低 X 射线剂量通常会增加图像中的量子噪声和条纹伪影，使软组织结构、边界细节和低对比度病灶更难观察。本文将低剂量 CT 去噪视为一个人工智能图像复原问题，重点讨论近年来 Transformer 图像复原模型相对于传统卷积编码器-解码器模型的优势与局限。

本文综述了 U-Net、RED-CNN、SwinIR、Restormer 和 CTformer 等代表性方法，并基于本地已完成的 AAPM-Mayo 低剂量 CT 数据实验进行比较分析。实验采用配对的 quarter-dose 与 full-dose CT 切片，使用峰值信噪比（PSNR）、结构相似性（SSIM）和均方根误差（RMSE）作为评价指标。在本地测试集 L506 患者上，原始低剂量输入的 PSNR 为 29.2489 dB，SSIM 为 0.8759，RMSE 为 14.2416。去噪模型均显著提升图像质量：CTRestormer 在 11500 次迭代时达到 32.1294 dB PSNR 和 0.8997 SSIM；CTformer 达到 32.3852 dB PSNR 和 0.9026 SSIM；RED-CNN 达到 32.6656 dB PSNR 和 0.9067 SSIM。

这些结果表明，Transformer 模型确实能够有效用于低剂量 CT 去噪，但在训练资源有限、数据规模较小或模型尚未充分收敛时，经典 CNN 残差模型仍然具有很强竞争力。医学图像去噪不能只追求数值指标，还必须关注解剖结构保持、细节不被过度平滑、训练成本以及临床可靠性。

关键词：低剂量 CT；医学图像去噪；图像复原；Transformer；RED-CNN；CTformer；Restormer

## 1. 引言

CT 是现代医学中最重要的影像技术之一，能够提供人体内部结构的横断面信息，在肿瘤筛查、急诊诊断、术前评估和随访检查中都十分常见。但是 CT 使用电离辐射，过高或反复的辐射暴露会带来潜在健康风险。因此，降低 CT 扫描剂量一直是医学影像领域的重要研究方向。

降低剂量最直接的方法是减少管电流或改变扫描协议，但这会使投影数据中的光子数量减少，从而导致重建图像中出现更强的噪声和伪影。低剂量 CT 图像常表现为颗粒状噪声、低对比度结构模糊以及局部条纹伪影。这些退化会影响放射科医生对细微结构的判断，尤其是在软组织、血管、低对比度病灶或边界不清的区域。

因此，低剂量 CT 去噪可以被看作一个典型的图像复原问题：给定一张低剂量 CT 图像，算法需要估计出一张尽可能接近常规剂量或全剂量 CT 的图像，同时保留真实解剖结构。传统去噪方法依赖手工设计的先验，例如局部平滑、非局部均值、稀疏表示或迭代重建。这些方法有一定效果，但往往需要复杂调参，并且难以建模复杂的空间变化噪声。

深度学习改变了这一问题的求解方式。通过配对的低剂量和全剂量 CT 数据，神经网络可以直接学习从噪声图像到干净图像的映射。卷积神经网络（CNN）长期以来是医学图像复原的主流方法，其中 U-Net 和 RED-CNN 是代表性结构。近年来，Transformer 在自然语言处理和计算机视觉中取得成功，其自注意力机制能够建模长距离依赖，因此也被用于图像复原任务，例如 SwinIR、Restormer 和 CTformer。

本文研究的核心问题是：在低剂量 CT 去噪任务中，Transformer 模型相比卷积残差模型是否具有明显优势？本文不简单假设 Transformer 一定更好，而是结合本地实验结果进行分析。实验表明，Transformer 模型能够有效去噪，但 RED-CNN 这样的经典 CNN baseline 在充分训练后仍然非常强。

## 2. 相关工作

### 2.1 AAPM-Mayo 低剂量 CT 数据集

AAPM-Mayo Low Dose CT Grand Challenge 是低剂量 CT 去噪研究中常用的数据集。该数据集提供 normal-dose/full-dose CT 与模拟 quarter-dose CT 的配对数据，使监督学习成为可能。模型可以以 quarter-dose 图像作为输入，以 full-dose 图像作为目标，学习从低剂量图像到高质量图像的映射。

本文使用本地已预处理好的 AAPM-Mayo 风格数据，保存为 `.npy` 格式的二维 CT 切片。数据目录为 `npy_img_3mm_B30`，包含 2378 个 input slice，来自 10 个患者：L067、L096、L109、L143、L192、L286、L291、L310、L333 和 L506。实验中 L506 被作为测试患者。使用患者级划分很重要，因为如果训练集和测试集包含同一患者的相邻切片，模型性能可能被高估。

### 2.2 CNN 去噪方法

U-Net 最初用于生物医学图像分割，但其编码器-解码器结构和 skip connection 对图像复原也非常有用。编码器逐步提取多尺度语义特征，解码器恢复空间分辨率，而 skip connection 可以把高分辨率细节从编码端传递到解码端。这种结构非常适合像素级任务。

RED-CNN 是专门针对低剂量 CT 去噪提出的残差编码器-解码器卷积网络。它使用卷积层和反卷积层组成对称结构，并通过残差连接保留细节。低剂量 CT 图像本身已经包含大部分真实解剖结构，因此模型不需要从零生成整张图像，而应主要学习噪声和伪影的校正项。残差学习能够降低优化难度，也能减少模型对图像结构的破坏。

RED-CNN 的优势在于结构相对简洁、训练稳定、计算成本较低，而且卷积本身非常适合局部纹理和噪声建模。其局限在于普通卷积的感受野有限，若要建模远距离依赖，需要更深网络、多尺度结构或膨胀卷积。

### 2.3 Transformer 图像复原方法

Transformer 的核心是自注意力机制。与卷积主要关注局部邻域不同，自注意力可以在更大范围内建立像素或 token 之间的关系。这对于低剂量 CT 去噪有潜在优势，因为判断一个局部纹理是噪声还是真实结构，往往需要结合更大范围的解剖上下文。

但是，标准全局自注意力在高分辨率图像上的计算成本很高。SwinIR 通过窗口注意力和 shifted window 机制降低复杂度，使 Transformer 能够应用于图像超分辨率、去噪和压缩伪影去除等复原任务。Restormer 则进一步面向高分辨率图像复原设计，引入 MDTA 和 GDFN 模块，在保留高效注意力建模的同时结合深度卷积的局部归纳偏置。

CTformer 是专门用于低剂量 CT 去噪的 Transformer 模型。它采用 convolution-free Token2Token dilated vision Transformer 结构，通过 Token2Token 和 dilated attention 提取局部与全局信息。相比通用自然图像复原模型，CTformer 更直接面向医学 CT 去噪任务。

本文中的 CTRestormer 是基于 Restormer 思想设计的实验模型。它包含 overlap patch embedding、多尺度编码器-解码器、MDTA 风格注意力、GDFN 风格前馈网络和残差输出。由于本地 GPU 为 6GB RTX 3060 Laptop GPU，模型规模经过压缩，因此 CTRestormer 当前结果应被视为资源受限条件下的实验结果。

## 3. 技术方法

### 3.1 问题定义

设低剂量 CT 切片为 \(x\)，对应全剂量 CT 切片为 \(y\)。监督式去噪模型学习一个映射：

\[
\hat{y}=f_{\theta}(x)
\]

其中 \(\theta\) 为模型参数，\(\hat{y}\) 是模型预测的去噪图像。训练目标是使 \(\hat{y}\) 尽可能接近 \(y\)。常见损失函数包括均方误差：

\[
\mathcal{L}_{MSE}=\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_{i}-y_{i})^2
\]

也可以使用 Charbonnier loss、SSIM loss 或混合损失。CTRestormer 实验中使用了 hybrid loss，即结合 Charbonnier 与 SSIM 相关项，以同时优化像素误差和结构相似性。

由于低剂量图像已经包含大部分真实解剖信息，很多模型采用残差学习：

\[
\hat{y}=x+r_{\theta}(x)
\]

其中 \(r_{\theta}(x)\) 表示模型预测的校正项。这样模型主要学习如何去除噪声和伪影，而不是重新合成整张图像。

### 3.2 RED-CNN

RED-CNN 使用卷积编码器-解码器结构。编码器提取局部特征，解码器恢复图像分辨率，残差连接帮助保留细节并改善梯度传播。卷积模型对局部噪声非常有效，因为 CT 图像中的很多噪声模式具有局部统计规律。

RED-CNN 的主要优点是训练稳定、推理高效、参数量相对可控。在小规模医学数据集上，强局部归纳偏置反而是一种优势。其不足是难以直接建模远距离依赖，对全局结构一致性的处理能力弱于注意力模型。

### 3.3 SwinIR、Restormer 与 CTformer

SwinIR 把图像划分为窗口，在窗口内计算自注意力，并通过 shifted window 在不同窗口之间交换信息。这种方法兼顾了计算效率和较大范围上下文建模。

Restormer 面向高分辨率图像复原，使用 MDTA 和 GDFN。MDTA 使注意力计算更适合高分辨率特征图，GDFN 使用门控机制控制信息流，并通过深度卷积保留局部空间信息。Restormer 的思想非常适合 CT 去噪，因为 CT 图像既需要局部边缘保持，也需要更大范围的结构一致性。

CTformer 则直接面向 LDCT 去噪任务。它使用 Token2Token 模块逐步构造 token 表示，并通过 dilated attention 扩大感受野。对于 CT 图像而言，骨骼边界、软组织纹理和低对比度结构都可能需要局部与全局信息共同判断，因此 CTformer 的设计具有明确动机。

## 4. 实验设置

### 4.1 数据与划分

实验使用本地 `npy_img_3mm_B30` 数据集。该数据集包含 2378 个二维配对切片，每个样本包含一个 quarter-dose input 和一个 full-dose target，图像大小为 \(512 \times 512\)。患者 ID 包括 L067、L096、L109、L143、L192、L286、L291、L310、L333 和 L506。

本地复现实验中，L506 作为 held-out test patient。使用患者级测试可以更真实地评估模型泛化能力，因为同一患者相邻切片之间高度相关。

### 4.2 模型与训练

本文报告的模型包括：

| 方法 | 类型 | 说明 |
|---|---|---|
| Low-dose input | 原始输入 | 未经过深度学习去噪 |
| RED-CNN | CNN baseline | 残差编码器-解码器卷积网络 |
| CTformer | Transformer baseline | Token2Token dilated vision Transformer |
| CTRestormer | Transformer experiment | Restormer 风格 CT 去噪模型 |

CTformer 使用已有本地复现实验结果，训练目录为 `runs/train_100ep_bs4`。RED-CNN 结果来自单独 baseline 测试。CTRestormer 使用 11500 iteration checkpoint；由于该训练仅完成原计划约 5.3%，因此结果不能代表完全收敛性能。

### 4.3 评价指标

本文使用 PSNR、SSIM 和 RMSE：

- PSNR 越高越好，反映像素误差对应的信噪比。
- SSIM 越高越好，反映结构相似性。
- RMSE 越低越好，反映预测图像与 full-dose target 的平均误差幅度。

这些指标能够反映图像复原质量，但并不能完全代表临床价值。医学图像还需要关注病灶可见性、边缘保持、细节保真度和是否产生伪结构。

## 5. 实验结果

### 5.1 定量结果

本地测试结果如下：

| Method | PSNR ↑ | SSIM ↑ | RMSE ↓ |
|---|---:|---:|---:|
| Low-dose input | 29.2489 | 0.8759 | 14.2416 |
| CTRestormer, 11500 iters | 32.1294 | 0.8997 | 10.0689 |
| CTformer | 32.3852 | 0.9026 | 9.8366 |
| RED-CNN | 32.6656 | 0.9067 | 9.4867 |

所有去噪模型都显著优于原始低剂量输入。CTformer 相比 low-dose input 提升约 3.14 dB PSNR，RMSE 从 14.2416 降至 9.8366。CTRestormer 也将 PSNR 提升至 32.1294 dB，说明 Restormer 风格模型确实学习到了有效去噪映射。

当前结果中 RED-CNN 最好，CTformer 接近 RED-CNN，CTRestormer 略低。这一结果说明 Transformer 并不会自动优于 CNN。模型是否占优取决于训练充分程度、模型规模、损失函数、数据量和硬件约束。在小数据和有限训练条件下，CNN 的局部归纳偏置仍然非常有价值。

### 5.2 训练趋势

CTformer 的训练 loss 曲线显示模型逐渐收敛到较低 loss 区域，说明复现实验是有效的。loss 曲线可以证明模型学到了稳定映射，但不能单独证明临床质量。因此，本文同时使用测试指标和可视化结果进行分析。

### 5.3 可视化结果

L506_88 的 qualitative panel 展示了 quarter-dose、模型输出和 full-dose target。原始 quarter-dose 图像在软组织区域存在明显颗粒状噪声，而模型输出更平滑，整体接近 full-dose target。单张切片上的指标也显示，去噪结果相比原始输入有明显提升。

不过，去噪图像在某些区域也可能比 full-dose target 更平滑。这是监督去噪中的常见问题：如果损失函数过度强调平均像素误差，模型可能会把高频结构也当作噪声去除。在医学影像中，这一点尤其重要，因为细小结构可能具有诊断意义。

## 6. 讨论

第一，实验表明低剂量 CT 去噪可以通过配对数据有效学习。所有深度学习模型都提高了 PSNR、SSIM 并降低 RMSE，说明 quarter-dose 图像中仍然保留了足够的解剖信息，神经网络能够学习从噪声图像到更清晰图像的映射。

第二，Transformer 模型有效，但不必然压倒 CNN。CTformer 结果接近 RED-CNN，说明注意力机制对 LDCT 去噪有价值；但 RED-CNN 仍然获得最佳本地指标，说明经典卷积残差模型在该任务上仍然强大。CT 图像噪声具有大量局部模式，卷积滤波器天然适合这类问题。

第三，CTRestormer 的结果应结合训练进度理解。它在 11500 iteration 时已经显著优于原始输入，但训练只完成计划的一小部分。继续训练可能提升性能，但也可能受到模型规模、数据量和过平滑问题限制。因此，未来应进行 checkpoint-level comparison，例如比较 11500、20000、30000 或更高 iteration 的 PSNR/SSIM/RMSE。

第四，医学图像去噪不能只依赖 PSNR 和 SSIM。模型可能通过平滑获得更高 PSNR，但同时损失细节。更严格的研究应加入 radiologist evaluation、lesion detectability、noise power spectrum、edge preservation 或下游诊断任务评价。对于临床应用，最重要的问题不是图像是否“看起来干净”，而是是否保留真实病理信息且不会产生误导性伪影。

最后，数据规模也是限制因素。本地数据虽然有 2378 个切片，但只来自 10 个患者。相邻切片高度相关，因此真实多样性有限。Transformer 通常更依赖大规模数据和长时间训练，而 CNN 在小数据条件下可能更稳定。这也是 RED-CNN 在当前实验中表现很强的原因之一。

## 7. 结论

本文将低剂量 CT 去噪作为一个人工智能图像复原任务，综述了 U-Net、RED-CNN、SwinIR、Restormer 和 CTformer 等代表性方法，并基于本地 AAPM-Mayo 数据实验进行了比较分析。

实验结果表明，深度学习模型能够显著改善 quarter-dose CT 图像质量。RED-CNN 在当前本地实验中取得最佳指标，CTformer 与其接近，CTRestormer 在早期训练阶段也表现出有效去噪能力，但尚未超过 baseline。总体来看，Transformer 模型具有建模全局上下文的潜力，但其优势依赖充分训练、合理模型规模和严谨评价。

未来工作可以继续训练 CTRestormer，增加更多 patient-level validation，比较不同 checkpoint 的性能，并加入临床相关评价指标。对于医学图像去噪，最终目标不是简单获得更平滑的图像，而是在降低噪声的同时最大限度保留真实解剖和病理信息。

## 参考文献

[1] American Association of Physicists in Medicine, “Low Dose CT Grand Challenge,” 2016. https://www.aapm.org/GrandChallenge/LowDoseCT/

[2] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” MICCAI, 2015.

[3] H. Chen et al., “Low-dose CT with a residual encoder-decoder convolutional neural network,” IEEE Transactions on Medical Imaging, 2017.

[4] J. Liang et al., “SwinIR: Image restoration using Swin Transformer,” ICCV Workshops, 2021.

[5] S. W. Zamir et al., “Restormer: Efficient Transformer for high-resolution image restoration,” CVPR, 2022.

[6] Z. Wang et al., “Uformer: A general U-shaped Transformer for image restoration,” CVPR, 2022.

[7] D. Wang et al., “CTformer: Convolution-free Token2Token dilated vision Transformer for low-dose CT denoising,” Physics in Medicine & Biology, 2023.

