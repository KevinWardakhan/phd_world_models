

# Di[M]O: Distilling Masked Diffusion Models into One-step Generator

Yuanzhi Zhu<sup>1</sup> Xi Wang<sup>1</sup> Stéphane Lathuilière<sup>2</sup> Vicky Kalogeiton<sup>1</sup>

<sup>1</sup>LIX, École Polytechnique, CNRS, IPP <sup>2</sup>Inria, Univ. Grenoble Alpes, CNRS, LJK  
<https://yuanzhi-zhu.github.io/DiMO/>

![Figure 1: A line graph and a grid of generated images. The graph plots HPSv2 [112] Score (y-axis, 23 to 29) against Generation Steps (x-axis, 12 to 50). It compares various models: DMs (orange circles), One-step DMs (red triangles), MDM (green diamonds), and One-step MDM (blue stars). Di[M]O (ours) is highlighted as a One-step MDM. The grid shows generated images from the teacher model Meisssonc [4] (64, 16, 4 steps) and the proposed One-step Generator, demonstrating competitive performance at fewer steps.](3ad4d1734817fe2ac05fe8d4da4f68e8_img.jpg)

| Model               | Generation Steps | HPSv2 [112] Score |
|---------------------|------------------|-------------------|
| Meisssonc (teacher) | 64               | ~28.5             |
| Meisssonc (teacher) | 16               | ~27.5             |
| Meisssonc (teacher) | 4                | ~26.5             |
| Di[M]O (ours)       | 12               | ~25.5             |
| Di[M]O (ours)       | 16               | ~27.0             |
| Di[M]O (ours)       | 25               | ~28.0             |
| Di[M]O (ours)       | 32               | ~28.5             |
| Di[M]O (ours)       | 48               | ~28.8             |
| Di[M]O (ours)       | 50               | ~29.0             |

Figure 1: A line graph and a grid of generated images. The graph plots HPSv2 [112] Score (y-axis, 23 to 29) against Generation Steps (x-axis, 12 to 50). It compares various models: DMs (orange circles), One-step DMs (red triangles), MDM (green diamonds), and One-step MDM (blue stars). Di[M]O (ours) is highlighted as a One-step MDM. The grid shows generated images from the teacher model Meisssonc [4] (64, 16, 4 steps) and the proposed One-step Generator, demonstrating competitive performance at fewer steps.

Figure 1. Unlike **continuous Diffusion Models (DMs)** that have been successfully distilled into **one-step DM generators** with performances competitive with the teacher, distilling **Masked Diffusion Models (MDM)** into one-step generator remains a challenge. In this paper, we propose the **first one-step distillation method: Di[M]O for MDM**, e.g., from the recent text-to-image masked diffusion model Meisssonc [4]. We demonstrate that our Di[M]O can successfully distill the teacher model into a one-step generator, achieving competitive performance both quantitatively and qualitatively, while the teacher model’s performance deteriorates rapidly with reduced generation steps, i.e., comparing our one-step results (large, bottom images) to 4-step teacher outputs (right corner of each image).

## Abstract

*Masked Diffusion Models (MDMs) have emerged as a powerful generative modeling technique. Despite their remarkable results, they typically suffer from slow inference with several steps. In this paper, we propose Di[M]O, a novel approach that distills masked diffusion models into a one-step generator. Di[M]O addresses two key challenges: (1) the intractability of using intermediate-step information for one-step generation, which we solve through token-level distribution matching that optimizes model output logits by an ‘on-policy framework’ with the help of an auxiliary model; and (2) the lack of entropy in the initial distribution, which we address through a token initialization strategy that injects randomness while maintaining similarity to teacher training distribution. We show Di[M]O’s effectiveness on both class-conditional and text-conditional image generation, impressively achieving performance competitive to multi-step teacher outputs while drastically reducing inference time. To our knowledge, we are the first to successfully achieve one-step distillation of masked diffusion models and the first to apply discrete distillation to text-to-image generation, opening new paths for efficient generative modeling.*

## 1. Introduction

Recently, Masked Diffusion Models (MDMs) [3, 11, 40, 96] have emerged as a prominent framework in generative modeling, demonstrating strong performance in a broad spectrum of tasks. Unlike auto-regressive models that generate data sequentially, MDMs enable faster discrete data generation. Moreover, compared to continuous diffusion models [38], MDMs are particularly advantageous for visual modeling within multimodal frameworks, as they leverage a unified vocabulary to seamlessly integrate visual and textual content. Their versatility has enabled successful applications in image generation [4, 11, 12, 40, 43, 74], text generation [20, 27, 56, 67, 68], video generation [24, 104], multi-modal modeling [39, 50, 108, 113], protein design [10, 28], audio synthesis [85], and motion generation [15, 75]. In addition, several algorithms have been proposed to guide the generation process in MDMs [49, 69, 80, 92, 97], further enhancing their flexibility and alignment with user-defined objectives.

Despite these advantages, current MDMs suffer from

low inference speed, i.e. high number of sampling steps required during generation [23]. Besides accelerated sampling techniques [8, 9, 47, 72, 81, 122] that yield good performances but tend to drastically deteriorate the generation with fewer inference steps, distillation techniques address this efficiency bottleneck. They transfer knowledge from a well-performing but slow teacher model to a student model capable of achieving comparable results with significantly fewer steps (even one). For this reason, distillation has been widely adopted for continuous diffusion models, typically with consistency-based [57, 100, 102] or distillation matching approaches [60, 115, 119]. These effectively reduce inference time while maintaining generation quality. However, they cannot be directly applied to masked diffusion due to fundamental differences in their mathematical formulation, i.e. the absence of a Probability Flow Ordinary Differential Equation (PF-ODE) and the inability to re-parameterize MDM outputs as denoising score functions [20, 56, 63]. Furthermore, recent work [99] points out that multi-token prediction in MDMs is inherently challenging due to their naïve conditional independence assumption.

To overcome this, the community is just starting to explore distillation alternatives for masked diffusion [20, 31]. [20, 31] are the only works attempting this. Hayakawa *et al.* [31] propose viewing MDMs as mixture models, progressively distilling dimension correlations into a few-step student. Meanwhile, Deschenaux *et al.* [20] collect the teacher’s multi-step output log probability as a target for the student to learn with fewer steps. Despite their improved efficiency, these methods have critical limitations: both require a multi-round distillation process, which is computationally expensive and prone to error accumulation when further reducing student steps.

Instead, in this work, we directly address **one-step distillation for masked diffusion models**. While achieving this, we identified two main challenges. First, there is the difficulty of leveraging the intermediate steps information for one-step generator<sup>1</sup>. To distill the teacher’s whole multi-step information into a *one-step* model, neither can we compute the loss between the teacher’s intermediate outputs and intractable student intermediate ones, nor can we update the one-step generator using the loss gradients due to the non-differentiable sampling operation. Second, the lack of entropy in the initial distribution used during one-step generation [31]. While continuous diffusion models typically start with a Gaussian distribution that naturally incorporates randomness, MDM begins with a sequence of identical mask tokens. This lack of entropy can lead to mode collapse in one-step distillation, making effective distribution matching more difficult.

Therefore, in this work, we propose **Distilling [M]asked**

<sup>1</sup>Without explicit distinction, we use the terms “one-step generator” and “student model” interchangeably when referring to our distilled model.

diffusion models into a **One-step generator**, coined Di[M]O, trained to transform an initial input distribution into outputs that closely resemble the teacher’s multi-step generation distribution. Inspired by *data-free* on-policy distillation [1, 2, 5, 35, 37, 84, 93], Di[M]O matches the conditional *token-level* distribution along all intermediate states to ensure the overall generation distribution matching. Our first technical contribution is that Di[M]O approximates the training objective using pseudo-intermediate states combined with an auxiliary model that surrogates the gradient. Our second technical contribution addresses the insufficient entropy in the initial state of one-step generation. Di[M]O employs a token initialization strategy that injects randomness while maintaining similarity to the input sequences used to train the teacher, ensuring robust distillation while effectively avoiding mode collapse.

Using image generation as a representative example, we conduct experiments on class-conditional generation with MaskGit [11], and on text-conditional image generation with Meissson [4] as the teacher model. Our results show that for both tasks, Di[M]O can efficiently distill the teacher model into a one-step generator while impressively achieving performance competitive with the multi-step outputs of the teachers. Our findings also contain training objective with generalized divergence, ablation studies discussing our design choices, limitations and future directions.

Our contributions can be summarized as: (1) We are the first to successfully achieve one-step distillation of masked diffusion models, paving the path for future work. (2) To address this, we propose Di[M]O, a token-level distillation method that enables the one-step distillation of masked diffusion models, with proposed efficient token initialization. (3) Our findings show that Di[M]O successfully reaches performance close to that of MDM teachers, while greatly enhances the sampling efficiency, verified by both class-conditional and text-conditional image generation tasks; note that, Di[M]O is the first to experiment with discrete *distillation* for text-to-image generation.

## 2. Related Works

**Acceleration of Masked Diffusion Models.** Improving the efficiency of MDMs has been an active area of research, with various methods proposed to accelerate inference while maintaining generation quality [8, 9, 47, 72, 81, 122]. For instance, Park *et al.* [72] minimize compounding decoding error to improve the allocation of discrete sampling timesteps, achieving efficiency gains without additional computational overhead. Ren *et al.* [81] introduce a higher-order solver that allows for larger step sizes while reducing numerical errors, enabling faster generation with minimal quality loss. Other works have explored alternative strategies to enhance efficiency. Hayakawa *et al.* [31] propose a distillation approach in which a few-step stu-

dent is trained using a combination of distillation loss and consistency loss to improve sampling efficiency. However, their method relies on either simulating the teacher’s trajectory or utilizing real data to obtain intermediate states for training the student, leading to a mismatch between training and inference. Similar to Progressive Distillation [87], Deschenaux *et al.* [20] distill a student model to approximate the teacher’s multi-step output distribution via divergence minimization. However, this approach requires multiple inferences of the teacher model during student training. Additionally, since the teacher’s output tokens are independent, the distillation step size is constrained, necessitating a multi-round distillation process similar to [87]. Furthermore, all these approaches still require multiple sampling steps and suffer from significant performance degradation when the number of steps is further reduced. This limitation highlights the need for further research into efficient one-step generation strategies, which is the task we propose and tackle with our proposed Di[M]O.

**Continuous Diffusion Distillation.** The goal of diffusion distillation is to reduce sampling steps, and can be generally divided into through two primary approaches [127]: regression-based and distribution-based methods [121]. Regression-based techniques [29, 54, 59, 64, 87, 102, 117, 128] train student models via regression objectives derived from the PF-ODE of the teacher model. It is challenging to apply these techniques for MDMs due to their missing of explicit PF-ODE as sampling trajectory. Distribution-based methods [18, 46, 60, 61, 66, 115, 118, 119, 123–125] instead directly match the student’s output distribution to the teacher’s multi-step sampling distribution, often leveraging adversarial or divergence-minimization strategies. Similar to prior Variational Score Distillation (VSD) approaches [18, 60, 66, 89, 110, 114, 118, 119], our Di[M]O approach employs an auxiliary model  $\psi$  to approximate the intermediate distributions of an one-step generator. However, our method differs in three key aspects. First, it is specifically designed for masked diffusion and does not rely on the score function used in continuous diffusion models. Second, while VSD-based methods minimize the divergence of *denoising distributions* [32, 120] over the full space of latent images, we instead optimize the divergence of *token-level conditional distributions*. Third, our framework generalizes beyond the reverse Kullback–Leibler (KL) divergence employed in these prior works, supporting arbitrary  $f$ -divergences [17] for distribution matching, effectively mitigating the mode-seeking bias.

## 3. Preliminary: Masked Diffusion Models

Masked Diffusion Models (MDMs) [11, 40, 96] rely on a pre-trained discrete image tokenizer, such as VQ-VAE [22, 106]. Given an image of shape  $3 \times H \times W$ , the tokenizer

![Figure 2: Reverse process of MDM. The diagram shows a 3x3 grid of tokens x_t with some masked (grey). An arrow labeled 'MDM' and 'Predict Logits independently' points to a set of logits z_phi^i. These logits are used to sample new tokens x_phi^i. An arrow labeled 'p = (r_t - r_s) / r_t' points to a new grid x_s, where the masked tokens from x_t are replaced by the sampled tokens x_phi^i.](163688ca8da9787f5d48edd68d8cc75b_img.jpg)

Figure 2: Reverse process of MDM. The diagram shows a 3x3 grid of tokens x\_t with some masked (grey). An arrow labeled 'MDM' and 'Predict Logits independently' points to a set of logits z\_phi^i. These logits are used to sample new tokens x\_phi^i. An arrow labeled 'p = (r\_t - r\_s) / r\_t' points to a new grid x\_s, where the masked tokens from x\_t are replaced by the sampled tokens x\_phi^i.

Figure 2. **Reverse process of MDM.** With a masked sequence  $x_t$  as input, the MDM *independently* output logits  $z_\phi^i$  at each masked position  $i$ , which are then used to sample the new tokens  $x_\phi^i$  as the model prediction. In the next state  $x_s$ , we use  $x_\phi^i$  to replace each masked token in  $x_t$  with probability  $(r_t - r_s)/r_t$ .

encodes it into a sequence  $x_0 = \{x_\theta^i\}_{i=1}^L$  of discrete tokens, where  $x_\theta^i \in [V]$  is the token at position  $i$ . Here,  $[V] = \{0, 1, \dots, V-1\}$  represents the set of token indices, with  $V$  the vocabulary size. The sequence length is given by  $L = hw$ , where  $h$  and  $w$  correspond to the height and width of the 2D token grid produced by the VQ-VAE encoder. Similar to continuous diffusion models [38, 98, 101], MDMs generate discrete data by learning to reverse an absorbing-state forward diffusion process [3].

**Forward Process.** The forward process gradually corrupts the original token sequence  $x_0$  into a fully masked sequence  $x_1$ . For any timestep  $t \in [0, 1]$ , this yields intermediate masked token sequences  $x_t$ , composed of mixed image and mask tokens at a mask ratio  $r_t$ <sup>2</sup>. The transition from  $x_0$  to  $x_t$  follows the probability [71, 86, 96]:

$$q_{t|0}(x_t|x_0) = \prod_{i=0}^{L-1} \text{Cat}(x_t^i; (1 - r_t)\delta(x_0^i) + r_t\delta([M])), \quad (1)$$

where  $i \in \{0, 1, \dots, L-1\}$  denotes the token position,  $[M]$  specifies the mask token,  $\delta(\cdot)$  is the Dirac function, and  $\text{Cat}(x^i; p)$  refers to the categorical distribution over  $V+1$  categories in  $[V] \cup \{[M]\}$ . Essentially, tokens in  $x_0$  are *independently* masked with probability  $r_t$  at timestep  $t$ .

**Parameterized Reverse Process.** The reverse process reconstructs the original image tokens by iteratively replacing masked tokens with predicted ones, starting from a fully masked sequence  $x_1$ . This process is guided by a model  $\phi$ , which predicts a categorical distribution over each masked token at position  $i$  given  $x_t$ :

$$p_\phi(x_0^i|x_t) := \text{softmax}(z_\phi^i(x_t)/\tau), \quad (2)$$

<sup>2</sup>In masked diffusion, each token in a sequence can only be either  $[M]$  token or image token, and different timesteps influence only the proportion of these two token types in the sequence (following the mask ratio  $r_t$ ), e.g. in half timesteps ( $r_t = 0.5$ ), half the tokens will be masked.

where  $z_\phi^i \in \mathbb{R}^V$  are the independently predicted logits, and  $\tau$  controls sampling diversity. As shown in Fig. 2, each predicted image token  $x_\phi^i$  is *independently* sampled from the token-level predicted distribution  $p_\phi(x_0^i|x_t)$  and is used to transition from timestep  $t$  to  $s$  ( $0 \leq s < t \leq 1$ ). During this transition, unmasked tokens remain unchanged, whereas each masked token  $x_t^i$  is updated to  $x_\phi^i$  with probability  $(r_t - r_s)/r_t$  or stay masked with probability  $r_s/r_t$ . The model  $\phi$  operates independently across token positions, enabling parallel token prediction, hence efficiently refining  $x_1$  into  $x_0$  in a progressive manner.

**MDM Training Objective.** The model  $\phi$  is trained to reconstruct the original token sequence  $x_0$  from their masked counterparts  $x_t$  by minimizing the negative log-likelihood across all mask ratios  $r_t$ :

$$\mathcal{L}_{\text{MDM}} = \mathbb{E}_{x_0,t} \left[ \left( \mathbb{E}_{q_{t|0}} [-\log p_{0|t}(x_0|x_t, \phi)] \right) \right], \quad (3)$$

with  $x_0 \sim p_0(x_0)$  and  $t \sim \mathcal{U}[0, 1]$ . This loss is computed as cross-entropy per masked token and is the same objective used in Masked Image Modeling (MIM) approaches [6, 40] or masked generative models [25, 42], such as MaskGit [11] and Meissonic [4], where the models also take condition  $c$  (class label or text prompt) as input.

## 4. Method

We introduce **Distilling [M]DM into One-step generator (Di[M]O)**, a novel method for distilling a multi-step MDM teacher model  $\phi$  into a one-step generator  $\theta$ . An overview of Di[M]O is illustrated in Fig. 3. The first step is to define the initial distillation objective, which aims to align the generation distribution of the one-step generator and the multi-step teacher in Sec. 4.1. In Sec. 4.2, we introduce an approximation strategy that formulates an effective gradient to update the generator. In Sec. 4.3, we propose initialization strategies to address the low-entropy issue of the initial masked state in MDMs. By combining these components, our method enables efficient and high-fidelity discrete data generation, bridging the gap between multi-step MDMs and one-step inference models.

### 4.1. One-step On-policy Distillation

Our goal is to train the student model to transform an initial random input distribution,  $p_{\text{init}}$ , into outputs that closely resemble the teacher’s multi-step generation distribution,  $p_\phi$  (note that we discuss the specific design of  $p_{\text{init}}$  in Section Sec. 4.3.). Taking inspiration from recent works on on-policy distillation [1, 2, 5, 35, 37, 84, 93], we propose to first sample intermediate state  $\tilde{x}_t$  derived from the student’s own predictions and then match the conditional predicted distributions of teacher  $p_\phi(x_0|\tilde{x}_t)$  and student  $p_\theta(x_0|\tilde{x}_t)$ . We denote this procedure as  $D(p_\phi||p_\theta)(\tilde{x}_t) := D(p_\phi(x_0|\tilde{x}_t)||p_\theta(x_0|\tilde{x}_t))$ , where  $D$  is the divergence. By

enforcing this match for all possible intermediate states  $\tilde{x}_t$ , we effectively ensure that the overall generation distribution of the student model aligns with that of the teacher. The overall distillation objective can be formulated as follows:

$$\mathcal{L}_{\text{Di[M]O}}(\theta) := \mathbb{E}_{x_{\text{init}},t} \left[ w(t) \left( \mathbb{E}_{q_{t|0}} [D(p_\phi||p_\theta)(\tilde{x}_t)] \right) \right], \quad (4)$$

where  $t \sim \mathcal{U}[0, 1]$  and  $w(t)$  is a weighting function. Since the generator  $\theta$  can now only take  $x_{\text{init}}$  as input and directly produces  $x_0$ , we apply the forward diffusion process to get  $\tilde{x}_t \sim q_{t|0}(\tilde{x}_t|x_\theta(x_{\text{init}}))$  as the pseudo-intermediate state from its output  $x_\theta(x_{\text{init}})$ . This divergence is calculated across all possible  $x_{\text{init}}$  and  $t$  to cover every possible intermediate state.

**Token-level Distribution Matching.** The conditional divergence in Eq. (4) can be further decomposed and formulated as the averaged divergence of each masked tokens given  $\tilde{x}_t$ , defined as:

$$D(p_\phi||p_\theta)(\tilde{x}_t) := \frac{1}{L_M} \sum_{\substack{i=1 \\ \tilde{x}_t^i=[M]}}^L D(p_\phi(x_0^i|\tilde{x}_t)||p_\theta(x_0^i|\tilde{x}_t)), \quad (5)$$

where  $L_M$  is the number of the mask tokens in  $\tilde{x}_t$ . Unlike in continuous diffusion model distillation [60, 66, 119], the divergence is computed over the fully generated latents of the models, our proposed Di[M]O instead calculates divergence at the *token level* with the decomposition in Eq. (5).

### 4.2. Approximation of Loss Gradient

Directly optimizing Eq. (4) is not feasible because the exact form of  $p_\theta(x_0^i|\tilde{x}_t)$  is unknown. Similar to prior work [41, 77, 125], we seek to approximate the following gradient of training objective (see Appendix A.1):

$$\nabla_\theta \mathcal{L}_{\text{Di[M]O}} = \mathbb{E}_{x_{\text{init}},t} \left[ w(t) \left( \mathbb{E}_{q_{t|0}} \left[ \nabla_{z_\theta} D(p_\phi||p_\theta)(\tilde{x}_t) \frac{dz_\theta(\tilde{x}_t)}{d\theta} \right] \right) \right], \quad (6)$$

where the logits  $z_\theta$  is the direct output of student model. In this way, we can view the loss gradient as the product of two terms: the gradient of the divergence with respect to model logits (in orange), and the Jacobian of the model output with respect to student parameters  $\theta$  (in blue). By approximating these two terms separately, we can address the intractability of directly optimizing the original objective:

1. **Approximation of  $p_\theta(x_0^i|\tilde{x}_t)$ :** Given that the highlighted divergence gradient term in Eq. (6) can be viewed as a functional of the teacher output  $p_\phi(x_0^i|\tilde{x}_t)$  and the unknown student output  $p_\theta(x_0^i|\tilde{x}_t)$ , we adopt the strategy from [60, 110, 119] and introduce an auxiliary model to approximate the unknown  $p_\theta(x_0^i|\tilde{x}_t)$ . The auxiliary model  $\psi$  is trained as a  $x_0$ -predictor using data generated by the one-step generator and is optimized via the MDM training loss in Eq. (3) to have  $p_\psi(x_0^i|\tilde{x}_t) \approx p_\theta(x_0^i|\tilde{x}_t)$ .

![Figure 3: Di[M]O Pipeline diagram. The diagram shows the flow from Token Initialization Strategy to the One-step Generator, then to Logits, Tokens, Masked Tokens, and finally to the Teacher and Auxiliary Model. It includes a legend for initialization strategies and a detailed view of the generator's internal process.](f9a14fbfecbd7d059226cc93677d721b_img.jpg)

The diagram illustrates the Di[M]O Pipeline. On the left, a legend indicates three token initialization strategies: 'All random' (marked with a red X), 'All masked' (marked with a red X), and 'Proposed' (marked with a green checkmark). The 'Proposed' strategy is shown as a 4x4 grid where some tokens are masked (grey) and others are random (blue), with a mask ratio  $0 < r_{\text{init}} < 1$ . This strategy is used to sample an initial state  $x_{\text{init}}$  from a Gaussian distribution  $\mathcal{N}(0, \sigma_{\text{init}} I)$ . The initial state is fed into the 'One-step Generator' (Student  $\theta$ ). The generator produces logits  $z_\theta$ , which are then sampled to form a sequence of tokens  $x_\theta$ . These tokens are processed by a 'Forward' mask diffusion process to produce 'Masked Tokens  $\tilde{x}_t$ '. The 'Masked Tokens  $\tilde{x}_t$ ' are used to update the generator and an 'Auxiliary Model  $\psi$ '. The generator is updated by minimizing the conditional divergence  $D(p_\phi || p_\psi)(\tilde{x}_t)$  at the token-level. The auxiliary model is trained using a cross-entropy loss to model the distribution of generated tokens  $x_\theta$ . The teacher model  $\phi$  is frozen during training.

Figure 3: Di[M]O Pipeline diagram. The diagram shows the flow from Token Initialization Strategy to the One-step Generator, then to Logits, Tokens, Masked Tokens, and finally to the Teacher and Auxiliary Model. It includes a legend for initialization strategies and a detailed view of the generator's internal process.

Figure 3. **Di[M]O Pipeline.** Our method distills a costly multi-step MDM teacher into a one-step generator. Given  $x_{\text{init}}$  sampled using our proposed token initialization strategy, the one-step generator (student  $\theta$ ) produces logits  $z_\theta$ , from which image token sequence  $x_\theta = \{x_\theta^i\}_{i=1}^L$  are sampled. These tokens are then processed to obtain an intermediate state  $\tilde{x}_t$  through a forward mask diffusion process. For each intermediate state  $\tilde{x}_t$ , we update the one-step generator  $\theta$  and auxiliary model  $\psi$  alternately: the one-step generator is optimized by minimizing the conditional divergence  $D(p_\phi || p_\psi)(\tilde{x}_t)$  at *token-level*, while the auxiliary model is trained using a cross-entropy loss to model the distribution of generated tokens  $x_\theta$  and to form the gradient to update  $\theta$ . The teacher  $\phi$  is frozen during training.

2. **Approximation of  $z_\theta(\tilde{x}_t)$ :** The approximation of the model’s Jacobian term must satisfy a critical requirement: it needs to provide gradient information to update the model parameters  $\theta$ . Given this constraint, we propose approximating the intractable implicit term  $z_\theta(\tilde{x}_t)$  by substituting it with the one-step output  $z_\theta(x_{\text{init}})$ , under the assumption of consistency property [102] (see Fig. 15 for illustration).

These approximations lead to the following gradient of loss:

$$\nabla_\theta \mathcal{L}_{\text{Di[M]O}} \approx \mathbb{E}_{x_{\text{init}}, t} \left[ w(t) \left( \mathbb{E}_{q_t | 0} \left[ \nabla_{z_\psi} D(p_\phi || p_\psi)(\tilde{x}_t) \frac{dz_\theta(x_{\text{init}})}{d\theta} \right] \right) \right]. \quad (7)$$

To effectively transfer prior knowledge from teacher model, we initialize both the generator  $\theta$  and auxiliary model  $\psi$  with the pre-trained weights of the teacher model  $\phi$ .

**Generalized Jeffrey Divergence.** Unlike similar works on continuous diffusion distillation [60, 66, 119] that rely on the Reverse KL (RKL) divergence, the proposed Di[M]O is not limited to this choice. Instead, it can leverage alternative token-level divergence measures, as we do not impose restrictions on the form of divergence in Eq. (7). This is an especially interesting feature as the RKL is known to exhibit unwanted mode-seeking behaviors [18].

In particular, we propose to explore the use of Generalized Jeffrey Divergence [44, 95], which is defined as a linear combination of the Forward KL (FKL) and RKL divergences and investigate its effect beyond mode-seeking (see Appendix F):

$$D_{\text{Jeffrey}}^\beta = (1 - \beta) D_{\text{FKL}} + \beta D_{\text{RKL}}, \quad (8)$$

with  $\beta$  the hyperparameter. The explicit gradient of these divergences is provided in Appendix A.2.

### 4.3. Token Initialization Strategy

Unlike continuous diffusion models, where the initial distribution is typically a standard Gaussian, the initial state  $x_1$  of MDM consists of a *sequence* of deterministic identical  $[M]$  tokens [3, 31]. This fixed initialization leads to mode collapse for one-step generation, as the generator can only out-

put fixed logits  $\hat{z}_\theta$  with limited sample diversity. We therefore explore three possible strategies for the token sequence initialization, including the fixed one, as follows (see left panel of Fig. 3):

1. **All masked tokens:** following the standard MDM initialization [86, 96], we initialize the sequence with only  $[M]$  tokens.
2. **All random tokens:** aiming for maximizing entropy, we choose image tokens with random values to fill the initial token sequence.
3. **Hybrid strategy:** combining the two previous strategies by fixing an initial mask ratio  $r_{\text{init}}$  and replacing the remaining  $1 - r_{\text{init}}$  of the  $[M]$  tokens with random image tokens.

In our work, we adopt the **hybrid strategy**, guided by two key hypotheses and empirically supported by our ablation analysis in Fig. 5: (1) To prevent mode collapse, the initialization of one-step generator should include a certain level of *randomness*; (2) as the one-step generator is inherited from the teacher, whose objective is to predict all tokens from *partially masked inputs*, the initialization must preserve a similar pattern, i.e., include *some mask tokens* to avoid input distribution mismatch.

Moreover, inspired by prior auto-regressive distillation work [53], which proposed to replace all initial token embeddings with random Gaussian noise, we therefore add Gaussian perturbation upon our current scheme to further increase the randomness from a discrete space to a real one. Specifically, after randomly selecting  $r_{\text{init}}$   $[M]$  tokens and  $1 - r_{\text{init}}$  random image tokens, we perturb all token embeddings  $\mathbf{e}$  with Gaussian noise  $\epsilon$  at a fixed noise level  $\sigma_{\text{init}}$  using a variance-preserving scheme [38, 101]  $\hat{\mathbf{e}} = \sqrt{1 - \sigma_{\text{init}}^2} \mathbf{e} + \sigma_{\text{init}} \epsilon$ , as illustrated in Figure 3.

**Overview of Algorithm.** We now summarize our method in Algorithm 1 where the generator and auxiliary model are updated iteratively. In each iteration, the generator produces tokens  $x_\theta$  from initial state  $x_{\text{init}}$  in one step and is optimized with the loss gradient in Eq. (7), while the auxiliary model is trained on these generated tokens as targets.

#### Algorithm 1 Di[M]O Distillation

**Require:** Pre-trained teacher model  $\phi$ , condition dataset  $\mathcal{D}$

- 1:  $\theta \leftarrow \text{copyWeights}(\phi)$ ,  $\psi \leftarrow \text{copyWeights}(\phi)$  // initialize
- 2: **repeat**
- 3:    *// Generate tokens  $x_0$*
- 4:    Sample  $x_{\text{init}} \sim p_{\text{init}}$ ,  $c \sim \mathcal{D}$  // with strategy in Sec. 4.3
- 5:    Get generator logits  $z_\theta(x_{\text{init}}, c) \in \mathbb{R}^{B \times h \times w \times V}$
- 6:     $x_\theta \in \mathbb{R}^{B \times h \times w} \xleftarrow{\text{sample}} p_\theta(x_0 | x_{\text{init}}) = \text{softmax}(z_\theta(x_{\text{init}}, c))$
- 7:    *// Update generator  $\theta$*
- 8:    Sample  $t \sim \mathcal{U}[0, 1]$ ,  $\tilde{x}_t \sim q_{t|0}(\tilde{x}_t | x_\theta(x_{\text{init}}, c))$  // Forward
- 9:    Calculate  $p_\phi(x_0 | \tilde{x}_t, c)$  and  $p_\psi(x_0 | \tilde{x}_t, c)$
- 10:    Update  $\theta$  with the loss gradient  $\nabla_\theta \mathcal{L}_{\text{Di[M]O}}$  (Eq. (7))
- 11:    *// Update auxiliary model  $\psi$*
- 12:    Sample  $t' \sim \mathcal{U}[0, 1]$ ,  $\tilde{x}_{t'} \sim q_{t'|0}(\tilde{x}_{t'} | x_\theta(x_{\text{init}}, c))$
- 13:    Update  $\psi$  with cross entropy loss (Eq. (3))
- 14: **until** convergence
- 15: **Return** one-step generator  $\theta$

## 5. Experiments

**Teacher Models.** We conduct extensive distillation experiments on both class-conditional image generation and text-to-image generation. For class-conditional generation, we adopt the best-performing and publicly available class-conditional MDM, MaskGit [7], as the teacher model. For text-to-image generation, we employ the recent Meissonic [4] as the teacher and use the LAION-Aesthetics-6+ prompt dataset [14] for prompting during the distillation.

**Evaluation Metrics.** The metrics we use for comparing ImageNet results are Fréchet Inception Distance (FID) [34] and Inception Score (IS) [88]. We also use precision (Prec.) and recall (Rec.) [48], density (Den.) and coverage (Cov.) [65] to further evaluate the fidelity and diversity of generated images. For text-to-image generation, we follow [4] and measure the HPSv2 [112] and Geneval [26] score. See Appendix D for more metrics and results of both tasks. In our experiments on ImageNet  $256 \times 256$ , we calculate the FID using 5k generated images for ablations and 50k generated for benchmarking, comparing them against 50k images from the ImageNet validation set with Clean-FID [73].

**Experiment Setup.** Both teacher models adopt Classifier-Free Guidance (CFG) [36]. For the MaskGit teacher, we maintain a CFG scale of 2 during distillation, while for the Meissonic teacher, we use a CFG scale of 4. Our generator is always trained with conditioning, eliminating the need for CFG during inference, further improving the sampling efficiency. All of the ablation experiments on MaskGit are trained with a batch size of 64 and 30000 iterations. We evaluate the final checkpoints with different temperature to get the best metric results. The learning rates for MaskGit experiments and Meissonic experiments are  $1 \times 10^{-5}$  and  $1 \times 10^{-6}$ , respectively. See Appendix C for more details.

Table 1. Quantitative results on class-conditional ImageNet-256. \* denotes numbers estimated from the original plot.

| ImageNet     | Method                      | Step (↓) | FID (↓) | IS (↑) | Prec. (↑) | Rec. (↑) | Den. (↑) | Cov. (↑) |
|--------------|-----------------------------|----------|---------|--------|-----------|----------|----------|----------|
| Teacher      | MaskGit [7]                 | 16       | 6.60    | 224.07 | 0.831     | 0.402    | 1.246    | 0.977    |
|              | MaskGit [7]                 | 8        | 6.66    | 221.57 | 0.827     | 0.397    | 1.233    | 0.974    |
|              | MaskGit [7]                 | 4        | 10.73   | 192.29 | 0.748     | 0.313    | 1.011    | 0.920    |
|              | MaskGit [7]                 | 2        | 91.35   | 13.37  | 0.178     | 0.164    | 0.091    | 0.122    |
| Sampler      | $\theta$ -trapezoidal [81]* | 64       | 6.7     | -      | -         | -        | -        | -        |
|              | $\theta$ -trapezoidal [81]* | 32       | 7.1     | -      | -         | -        | -        | -        |
| Distillation | di4c [31]                   | 4        | 6.79    | 209.2  | -         | -        | -        | -        |
|              | di4c-d [31]                 | 4        | 6.57    | 213.6  | -         | -        | -        | -        |
|              | Di[M]O                      | 1        | 6.91    | 214.0  | 0.828     | 0.377    | 1.255    | 0.967    |

### 5.1. Class-conditional Image Generation

In Tab. 1, we present a quantitative performance comparison of our Di[M]O with various acceleration methods, including the high-order sampler  $\theta$ -trapezoidal [81], other distillation techniques, and the teacher model. Compared to the high-order sampler, which requires 32 steps to achieve an FID of 7.1, our method significantly reduces the number of steps to just 1 while maintaining competitive performance, with an FID of 6.91 and an IS of 214.0. When comparing to distillation methods, our approach shows comparable results to di4c [31], which uses 4 steps and achieves an FID of 6.79 and an IS of 209.2. Our method yields an FID of 6.91 with a higher IS of 214.0 in one step. Additionally, when compared to the teacher model (MaskGit [7]) with varying steps, our method performs very closely to the teacher’s performance of 6.60 FID at 16 steps, while achieving this with just a single step. These comparisons highlights the effectiveness of our method in matching the performance of multi-step teacher models while dramatically reducing the computational cost. For further details, refer to Appendix C.

### 5.2. Ablations

In this section, we present ablation studies on key hyperparameters of our method to validate our design choices: (1) the initial mask ratio  $r_{\text{init}}$ ; (2) the Jeffrey Coefficient  $\beta$ ; and (3) the Gaussian perturbation Strength  $\sigma_{\text{init}}$ :

**Initial Mask Ratio  $r_{\text{init}}$ .** We begin our ablation study by analyzing the impact of the initial mask ratio  $r_{\text{init}}$ , which is a crucial factor in our algorithm. As shown in Fig. 5a, our results confirm the hypothesis from Sec. 4.3 that both extreme values,  $r_{\text{init}} = 0$  and  $r_{\text{init}} = 1$  fail to work. Specifically, setting  $r_{\text{init}} = 1$  results in mode collapse, while very low values ( $r_{\text{init}} \approx 0$ ) cause unstable training, as evidenced by the broken curves in the subfigures of Fig. 5a. The optimal choice appears to be  $r_{\text{init}} = 0.6$ , which achieves the lowest FID. See Fig. 4 for visual demonstrations.

**Jeffrey Coefficient  $\beta$ .** Building on this, we conduct an ablation on the generalized Jeffrey divergence coefficient  $\beta$  (Fig. 5b). Our experiments indicate that reducing  $\beta$  generally improves FID, aligning with observations from prior work [116]. Surprisingly, our method remains effective even for negative values of  $\beta$ , with the best performance

![Figure 4: Visual results of ImageNet. A 3x4 grid of images. The first column shows initial masks (r_init = 0). The second column shows one-step generation with r_init = 1. The third column shows one-step generation with r_init = 0.6. The fourth column shows teacher generation with 16 steps. Rows correspond to ImageNet classes: 388 (pandas), 979 (landscapes), and 207 (golden retrievers).](a68671ef36320b8515aa85d288d82cf1_img.jpg)

$r_{init} = 0$ 
Di[M]O ( $r_{init} = 1$ ) one step
Di[M]O ( $r_{init} = 0.6$ ) one step
Teacher 16 steps

Figure 4: Visual results of ImageNet. A 3x4 grid of images. The first column shows initial masks (r\_init = 0). The second column shows one-step generation with r\_init = 1. The third column shows one-step generation with r\_init = 0.6. The fourth column shows teacher generation with 16 steps. Rows correspond to ImageNet classes: 388 (pandas), 979 (landscapes), and 207 (golden retrievers).

Figure 4. **Visual results of ImageNet.** One-step generated images from the generator trained with different  $r_{init}$  in comparison with teacher generation with 16 sampling steps. The class labels of the samples from top to bottom are 388, 979 and 207 respectively.

![Figure 5: Ablation studies on ImageNet using FID as the evaluation metric. Three line plots (a, b, c) show FID (5k) vs Generator Temperature (2 to 10). (a) r_init: Initial Mask Ratio. (b) beta: Jeffrey Coefficient. (c) sigma_init: Perturbation Strength. Each plot contains multiple lines for different parameter values. Asterisks (*) indicate training that collapsed and is shown in inset plots at the top right of each graph.](c0843c6d138705289960d9f53a6e72a1_img.jpg)

(a)  $r_{init}$ : Initial Mask Ratio
(b)  $\beta$ : Jeffrey Coefficient
(c)  $\sigma_{init}$ : Perturbation Strength

Figure 5: Ablation studies on ImageNet using FID as the evaluation metric. Three line plots (a, b, c) show FID (5k) vs Generator Temperature (2 to 10). (a) r\_init: Initial Mask Ratio. (b) beta: Jeffrey Coefficient. (c) sigma\_init: Perturbation Strength. Each plot contains multiple lines for different parameter values. Asterisks (\*) indicate training that collapsed and is shown in inset plots at the top right of each graph.

Figure 5. Ablation studies on ImageNet using FID as the evaluation metric. \* means the training is collapsed and falls outside the comparable range with other results, we show these in the sub-figures at the right upper corner with the same x-axis range.

observed at  $\beta = -0.2$ .

**Gaussian Perturbation  $\sigma_{init}$ .** Finally, we examine the effect of Gaussian perturbation on token embeddings (Fig. 5c). Our results demonstrate that introducing perturbations provides additional improvements in FID, further refining the quality of the generated samples. For completeness, the corresponding results with the IS metric are provided in Fig. 7 in the Appendix.

### 5.3. Text-to-image Generation

We validate Di[M]O for text-to-image generation by evaluating our distilled one-step generator on the HPSv2 [112] and GenEval [26] benchmarks.

In Tab. 2, we show the HPSv2 [112] benchmark evaluating human preference for T2I models. Distilled from Meissonic [4], the only open-sourced text-to-image MDM, we compare our one-step generator with not only the teacher

model, but also recent diffusion models [19, 21, 76, 83] and their distilled one-step generators [18, 55, 90]. The results clearly show that our one-step generator achieves a competitive performance with teacher models generated using 16 to 32 steps, whereas teacher’s performance deteriorates rapidly when fewer generation steps: a 48-step generation yields 28.83, whereas the 4 steps’ drop to 24.66). Our one-step generator also outperforms other one-step continuous diffusion-based ones; however, direct comparisons are limited since they are distilled from different teachers.

To better evaluate the semantic expressiveness of our one-step generator, Tab. 3 presents a comparison with both diffusion models [21, 76, 79, 83] and our teacher, Meissonic [4]. The experimental results corroborate our previous observations: our one-step generator competes with teacher models’ performance using 16 to 32 steps, while the teacher’s performance degrades rapidly with reduced steps.

![Figure 6: A 3x6 grid of 18 diverse images generated by the distilled model. The images include a metallic robot, a classical portrait, a pink flower, a Pikachu-like creature, a woman with pink hair, a green frog, a snowy landscape, a man in a hat, a mountain landscape, a knight in armor, a cozy room, a woman with autumn leaves, a man with roses, a pink landscape, a monkey in a spacesuit, a sunset over water, a woman with a flower crown, and a vase of flowers.](c834b9abb4ddf70e5d10641f87d5ff5b_img.jpg)

Figure 6: A 3x6 grid of 18 diverse images generated by the distilled model. The images include a metallic robot, a classical portrait, a pink flower, a Pikachu-like creature, a woman with pink hair, a green frog, a snowy landscape, a man in a hat, a mountain landscape, a knight in armor, a cozy room, a woman with autumn leaves, a man with roses, a pink landscape, a monkey in a spacesuit, a sunset over water, a woman with a flower crown, and a vase of flowers.

Figure 6. Qualitative our method distilled from teacher Meissonic. Corresponding prompts can be found in Appendix H

Table 2. HPS v2.0 benchmark. Scores are collected from <https://github.com/tgxs002/HPSv2>. We highlight the **best**.

| HPS v2.0                   |      |              |              |              |              |              |
|----------------------------|------|--------------|--------------|--------------|--------------|--------------|
| Model                      | Step | Anim.        | Concept-art  | Painting     | Photo        | Averaged     |
| Latent Diffusion [83]      | 25   | 25.73        | 25.15        | 25.25        | 26.97        | 25.78        |
| DALL-E 2 [79]              | -    | 27.34        | 26.54        | 26.68        | 27.24        | 26.95        |
| Stable Diffusion v1.4 [83] | 50   | 27.26        | 26.61        | 26.66        | 27.27        | 26.95        |
| Stable Diffusion v2.0 [83] | 50   | 27.48        | 26.89        | 26.86        | 27.46        | 27.17        |
| DeepFloyd-XL [19]          | 25   | 27.64        | 26.83        | 26.86        | 27.75        | 27.27        |
| SDXL Base 1.0 [76]         | 50   | 28.88        | 27.88        | 27.92        | 28.31        | 28.25        |
| SDXL Refiner 1.0 [76]      | 50   | 28.93        | 27.89        | 27.90        | 28.38        | 28.27        |
| InstaFlow [55]             | 1    | 25.98        | 25.79        | 25.93        | 26.32        | 26.01        |
| SD Turbo [90]              | 1    | 27.98        | 27.59        | 27.16        | 27.19        | 27.48        |
| SwiftBrush v2 [18]         | 1    | 27.25        | 27.62        | 26.86        | 26.77        | 27.15        |
| Meissonic [4]              | 48   | <b>29.57</b> | <b>28.58</b> | <b>28.72</b> | <b>28.45</b> | <b>28.83</b> |
|                            | 32   | 29.18        | 28.32        | 28.28        | 27.96        | 28.44        |
|                            | 16   | 28.61        | 27.82        | 27.84        | 27.32        | 27.90        |
|                            | 8    | 25.62        | 26.49        | 26.67        | 27.07        | 26.46        |
|                            | 4    | 25.01        | 24.95        | 24.87        | 23.80        | 24.66        |
|                            | 2    | 23.06        | 23.28        | 23.22        | 22.38        | 22.98        |
| Di[M]O                     | 1    | 28.64        | 27.91        | 27.99        | 27.92        | 28.11        |

**Limitations and Future Works.** While our data-free distillation achieves teacher-level performances, its application has so far been limited to our current model scope. In the future, we aim to extend our approach to stronger MDM teachers, particularly for image and text generation. While our method is data-free, it would be beneficial to also introduce real data to boost the one-step student to outperform the teacher. Lastly, while we leverage generalized Jeffreys divergence to avoid mode seeking behaviors, we intend to explore more general  $f$ -divergences to increase our method’s flexibility and effectiveness.

Table 3. GenEval benchmark. We highlight the **best** result.

| GenEval       |      |             |             |             |             |             |             |                   |
|---------------|------|-------------|-------------|-------------|-------------|-------------|-------------|-------------------|
| Model         | Step | Overall     | Objects     |             | Counting    | Colors      | Position    | Color Attribution |
|               |      |             | Single      | Two         |             |             |             |                   |
| SD v1.5 [83]  | 50   | 0.43        | 0.97        | 0.38        | 0.35        | 0.76        | 0.04        | 0.06              |
| SD v2.1 [83]  | 50   | 0.50        | 0.98        | 0.51        | 0.44        | 0.85        | 0.07        | 0.17              |
| DALL-E2 [79]  | -    | 0.52        | 0.94        | 0.66        | <b>0.49</b> | 0.77        | 0.10        | 0.19              |
| SDXL [76]     | 50   | <b>0.55</b> | 0.98        | <b>0.74</b> | 0.39        | 0.85        | <b>0.15</b> | <b>0.23</b>       |
| Meissonic [4] | 48   | 0.54        | <b>0.99</b> | 0.66        | 0.42        | <b>0.86</b> | 0.10        | 0.22              |
|               | 32   | 0.46        | 0.92        | 0.53        | 0.33        | 0.80        | 0.08        | 0.13              |
|               | 16   | 0.37        | 0.82        | 0.39        | 0.20        | 0.70        | 0.05        | 0.08              |
|               | 8    | 0.20        | 0.58        | 0.12        | 0.05        | 0.40        | 0.02        | 0.04              |
|               | 4    | 0.09        | 0.31        | 0.02        | 0.01        | 0.18        | 0.01        | 0.01              |
|               | 2    | 0.03        | 0.14        | 0.01        | 0.00        | 0.05        | 0.00        | 0.00              |
| Di[M]O        | 1    | 0.43        | 0.91        | 0.53        | 0.22        | 0.75        | 0.07        | 0.11              |

## 6. Conclusion

In this work, we proposed Di[M]O, a novel method that leverages token-level distribution matching to distill the inference process of MDMs into one-step. Specifically, inspired by the concept of on-policy distillation, we match the token-level distribution conditioned on a pseudo intermediate state obtained from the student’s one-step generation. We also conducted extensive experiments on the choice of initialization strategy and distillation objective, to increase the robustness of Di[M]O. Our experimental results demonstrate that our distilled model generates images with comparable quality to the teacher models, while requiring only 1 sampling step during inference. This work demonstrates the power of distribution matching methods on distillation for MDM, contributes to the growing community of research exploring efficient generation of discrete data.

## Acknowledgements

This work was supported by ANR-22-CE23-0007, ANR-22-CE39-0016, Hi!Paris grant and fellowship, DATAIA Convergence Institute as part of the “Programme d’Investissement d’Avenir” (ANR-17-CONV-0003) operated by Ecole Polytechnique, IP Paris, and was granted access to the IDRIS High-Performance Computing (HPC) resources under the allocation 2024-AD011014300R1 and 2025-AD011015894 made by GENCI and mesoGIP of IP Paris. We also sincerely thank Nacereddine Laddaoui for the help with infrastructure, Haoge Deng and Yao Teng for their insightful discussions that contributed to this work. We are also grateful to Nicolas Dufour, Robin Courant, and Lucas Degeorge for their meticulous proofreading.

## Broader Impacts

Our work focuses on distilling the multi-step generation process of MDMs into one step, significantly reducing inference time and computational costs, therefore lowering the carbon footprint during the inference. This advancement has the potential to make high-quality generative models more accessible, facilitating applications in creative industries, content generation, and real-time systems. However, as with many generative modeling techniques, our method inherits biases from the teacher models. This could potentially lead to ethical concerns, including the generation of misleading or harmful content. Additionally, by enabling faster and more efficient content generation, our approach could lower the barrier to misuse, such as the creation of deepfakes or other deceptive media.

## References

- [1] Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In *The Twelfth International Conference on Learning Representations*, 2024.
- [2] Kushal Arora, Layla El Asri, Hareesh Bahuleyan, and Jackie Chi Kit Cheung. Why exposure bias matters: An imitation learning perspective of error accumulation in language generation. *arXiv preprint arXiv:2204.01171*, 2022.
- [3] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. *Advances in Neural Information Processing Systems*, 34:17981–17993, 2021.
- [4] Jinbin Bai, Tian Ye, Wei Chow, Enxin Song, Qing-Guo Chen, Xiangtai Li, Zhen Dong, Lei Zhu, and Shuicheng Yan. Meissonic: Revitalizing masked generative transformers for efficient high-resolution text-to-image synthesis. *arXiv preprint arXiv:2410.08261*, 2024.
- [5] Ashwin Balakrishna, Brijen Thananjeyan, Jonathan Lee, Felix Li, Arsh Zahed, Joseph E Gonzalez, and Ken Goldberg. On-policy robot imitation learning from a converging supervisor. In *Conference on Robot Learning*, pages 24–41. PMLR, 2020.
- [6] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. Beit: Bert pre-training of image transformers. *arXiv preprint arXiv:2106.08254*, 2021.
- [7] Victor Besnier and Mickael Chen. A pytorch reproduction of masked generative image transformer. *arXiv preprint arXiv:2310.14400*, 2023.
- [8] Victor Besnier, Mickael Chen, David Hurych, Eduardo Valle, and Matthieu Cord. Halton scheduler for masked generative image transformer. *iclr*, 2025.
- [9] Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. *Advances in Neural Information Processing Systems*, 35:28266–28279, 2022.
- [10] Andrew Campbell, Jason Yim, Regina Barzilay, Tom Rainforth, and Tommi Jaakkola. Generative flows on discrete state-spaces: Enabling multimodal flows with applications to protein co-design. *arXiv preprint arXiv:2402.04997*, 2024.
- [11] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 11315–11325, 2022.
- [12] Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. *arXiv preprint arXiv:2301.00704*, 2023.
- [13] Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. *ACM transactions on Graphics (TOG)*, 42(4):1–10, 2023.
- [14] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 2818–2829, 2023.
- [15] Seunggeun Chi, Hyung-gun Chi, Hengbo Ma, Nakul Agarwal, Faizan Siddiqui, Karthik Ramani, and Kwonjoon Lee. M2d2m: Multi-motion generation from text with discrete diffusion models. In *European Conference on Computer Vision*, pages 18–36. Springer, 2024.
- [16] Jacob K Christopher, Brian R Bartoldson, Bhavya Kaikhura, and Ferdinando Fioretto. Speculative diffusion decoding: Accelerating language generation through diffusion. *arXiv preprint arXiv:2408.05636*, 2024.
- [17] Imre Csiszár, Paul C Shields, et al. Information theory and statistics: A tutorial. *Foundations and Trends® in Communications and Information Theory*, 1(4):417–528, 2004.
- [18] Trung Dao, Thuan Hoang Nguyen, Thanh Le, Duc Vu, Khoi Nguyen, Cuong Pham, and Anh Tran. Swiftbrush v2: Make your one-step diffusion model better than its teacher.

- In *European Conference on Computer Vision*, pages 176–192. Springer, 2025.
- [19] IF DeepFloyd. Deepfloyd if, 2023. <https://huggingface.co/DeepFloyd>, 2023.
- [20] Justin Deschenaux and Caglar Gulcehre. Beyond autoregression: Fast llms via self-distillation through time. *arXiv preprint arXiv:2410.21035*, 2024.
- [21] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. *Advances in Neural Information Processing Systems*, 34:8780–8794, 2021.
- [22] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 12873–12883, 2021.
- [23] Guhao Feng, Yihan Geng, Jian Guan, Wei Wu, Liwei Wang, and Di He. Theoretical benefit and limitation of diffusion language model. *arXiv preprint arXiv:2502.09622*, 2025.
- [24] Michael Fuest, Vincent Tao Hu, and Björn Ommer. Mask-flow: Discrete flows for flexible and efficient long video generation. *arXiv preprint arXiv:2502.11234*, 2025.
- [25] Marjan Ghazvininejad, Omer Levy, Yinhan Liu, and Luke Zettlemoyer. Mask-predict: Parallel decoding of conditional masked language models. *arXiv preprint arXiv:1904.09324*, 2019.
- [26] Dhruva Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. *Advances in Neural Information Processing Systems*, 36:52132–52152, 2023.
- [27] Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao, Wei Bi, Jiawei Han, et al. Scaling diffusion language models via adaptation from autoregressive models. *arXiv preprint arXiv:2410.17891*, 2024.
- [28] Nate Gruver, Samuel Stanton, Nathan Frey, Tim GJ Rudner, Isidro Hotzel, Julien Lafrance-Vanasse, Arvind Rajpal, Kyunghyun Cho, and Andrew G Wilson. Protein design with guided discrete diffusion. *Advances in neural information processing systems*, 36:12489–12517, 2023.
- [29] Jiatao Gu, Shuangfei Zhai, Yizhe Zhang, Lingjie Liu, and Joshua M Susskind. Boot: Data-free distillation of denoising diffusion models with bootstrapping. In *ICML 2023 Workshop on Structured Probabilistic Inference {\&} Generative Modeling*, 2023.
- [30] Jiaqi Han, Mingjian Jiang, Yuxuan Song, Jure Leskovec, Stefano Ermon, and Minkai Xu.  $f$ -po: Generalizing preference optimization with  $f$ -divergence minimization. *arXiv preprint arXiv:2410.21662*, 2024.
- [31] Satoshi Hayakawa, Yuhta Takida, Masaaki Imaizumi, Hiromi Wakaki, and Yuki Mitsufuji. Distillation of discrete diffusion through dimensional correlations. *arXiv preprint arXiv:2410.08709*, 2024.
- [32] Jiajun He, Wenlin Chen, Mingtian Zhang, David Barber, and José Miguel Hernández-Lobato. Training neural samplers with reverse diffusive kl divergence. *arXiv preprint arXiv:2410.12456*, 2024.
- [33] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. *arXiv preprint arXiv:2104.08718*, 2021.
- [34] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. *Advances in neural information processing systems*, 30, 2017.
- [35] Geoffrey Hinton. Distilling the knowledge in a neural network. *arXiv preprint arXiv:1503.02531*, 2015.
- [36] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. *arXiv preprint arXiv:2207.12598*, 2022.
- [37] Jonathan Ho, Jayesh Gupta, and Stefano Ermon. Model-free imitation learning with policy optimization. In *International conference on machine learning*, pages 2760–2769. PMLR, 2016.
- [38] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems*, 33:6840–6851, 2020.
- [39] Minghui Hu, Chuanxia Zheng, Heliang Zheng, Tat-Jen Cham, Chaoyue Wang, Zuopeng Yang, Dacheng Tao, and Ponnuthurai N Suganthan. Unified discrete diffusion for simultaneous vision-language generation. *arXiv preprint arXiv:2211.14842*, 2022.
- [40] Vincent Tao Hu and Björn Ommer. [mask] is all you need. *arXiv preprint arXiv:2412.06787*, 2024.
- [41] Zemin Huang, Zhengyang Geng, Weijian Luo, and Guojun Qi. Flow generator matching. *arXiv preprint arXiv:2410.19310*, 2024.
- [42] Jiwan Hur, DongJae Lee, Gyojin Han, Jaehyun Choi, Yunho Jeon, and Junmo Kim. Unlocking the capabilities of masked generative models for image synthesis via self-guidance. *Advances in Neural Information Processing Systems*, 37:130977–130999, 2024.
- [43] Thibaut Issenhuth, Ugo Tanielian, Jérémie Mary, and David Picard. Edibert, a generative model for image editing. *arXiv preprint arXiv:2111.15264*, 2021.
- [44] Harold Jeffreys. An invariant form for the prior probability in estimation problems. *Proceedings of the Royal Society of London. Series A. Mathematical and Physical Sciences*, 186(1007):453–461, 1946.
- [45] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 10124–10134, 2023.
- [46] Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability flow ode trajectory of diffusion. *arXiv preprint arXiv:2310.02279*, 2023.
- [47] Jaeyeon Kim, Kulin Shah, Vasilis Kontonis, Sham Kakade, and Sitan Chen. Train for the worst, plan for the best: Understanding token ordering in masked diffusions. *arXiv preprint arXiv:2502.06768*, 2025.
- [48] Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. *Advances in neural information processing systems*, 32, 2019.

- [49] Xiner Li, Yulai Zhao, Chenyu Wang, Gabriele Scalia, Gokcen Eraslan, Surag Nair, Tommaso Biancalani, Shuiwang Ji, Aviv Regev, Sergey Levine, et al. Derivative-free guidance in continuous and discrete diffusion models with soft value-based decoding. *arXiv preprint arXiv:2408.08252*, 2024.
- [50] Zijie Li, Henry Li, Yichun Shi, Amir Barati Farimani, Yuval Kluger, Linjie Yang, and Peng Wang. Dual diffusion for unified image generation and understanding. *arXiv preprint arXiv:2501.00289*, 2024.
- [51] Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxl-lightning: Progressive adversarial diffusion distillation. *arXiv preprint arXiv:2402.13929*, 2024.
- [52] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In *Computer vision—ECCV 2014: 13th European conference, zurich, Switzerland, September 6–12, 2014, proceedings, part v 13*, pages 740–755. Springer, 2014.
- [53] Enshu Liu, Xuefei Ning, Yu Wang, and Zinan Lin. Distilled decoding 1: One-step sampling of image autoregressive models with flow matching. *arXiv preprint arXiv:2412.17153*, 2024.
- [54] Qiang Liu. Rectified flow: A marginal preserving approach to optimal transport. *arXiv preprint arXiv:2209.14577*, 2022.
- [55] Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, and Qiang Liu. InstafLOW: One step is enough for high-quality diffusion-based text-to-image generation. *arXiv preprint arXiv:2309.06380*, 2023.
- [56] Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. In *Forty-first International Conference on Machine Learning*, 2024.
- [57] Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. *arXiv preprint arXiv:2410.11081*, 2024.
- [58] Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 11461–11471, 2022.
- [59] Eric Luhman and Troy Luhman. Knowledge distillation in iterative generative models for improved sampling speed. *arXiv preprint arXiv:2101.02388*, 2021.
- [60] Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, and Zhihua Zhang. Diff-instruct: A universal approach for transferring knowledge from pre-trained diffusion models. *arXiv preprint arXiv:2305.18455*, 2023.
- [61] Weijian Luo, Zemin Huang, Zhengyang Geng, J Zico Kolter, and Guo-jun Qi. One-step diffusion distillation through score implicit matching. *Advances in Neural Information Processing Systems*, 37:115377–115408, 2025.
- [62] Nanye Ma, Shangyuan Tong, Haolin Jia, Hexiang Hu, Yu-Chuan Su, Mingda Zhang, Xuan Yang, Yandong Li, Tommi Jaakkola, Xuhui Jia, et al. Inference-time scaling for diffusion models beyond scaling denoising steps. *arXiv preprint arXiv:2501.09732*, 2025.
- [63] Chenlin Meng, Kristy Choi, Jiaming Song, and Stefano Ermon. Concrete score matching: Generalized score matching for discrete data. *Advances in Neural Information Processing Systems*, 35:34532–34545, 2022.
- [64] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 14297–14306, 2023.
- [65] Muhammad Ferjad Naeem, Seong Joon Oh, Youngjung Uh, Yunje Choi, and Jaesun Yoo. Reliable fidelity and diversity metrics for generative models. In *International conference on machine learning*, pages 7176–7185. PMLR, 2020.
- [66] Thuan Hoang Nguyen and Anh Tran. Swiftbrush: One-step text-to-image diffusion model with variational score distillation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 7807–7816, 2024.
- [67] Shen Nie, Fengqi Zhu, Chao Du, Tianyu Pang, Qian Liu, Guangtao Zeng, Min Lin, and Chongxuan Li. Scaling up masked diffusion models on text. *arXiv preprint arXiv:2410.18514*, 2024.
- [68] Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. *arXiv preprint arXiv:2502.09992*, 2025.
- [69] Hunter Nisonoff, Junhao Xiong, Stephan Allenspach, and Jennifer Listgarten. Unlocking guidance for discrete state-space diffusion and flow models. *arXiv preprint arXiv:2406.01572*, 2024.
- [70] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. *arXiv preprint arXiv:2304.07193*, 2023.
- [71] Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li. Your absorbing discrete diffusion secretly models the conditional distributions of clean data. *arXiv preprint arXiv:2406.03736*, 2024.
- [72] Yong-Hyun Park, Chieh-Hsin Lai, Satoshi Hayakawa, Yuhta Takida, and Yuki Mitsufuji. : Optimizing sampling schedule of discrete diffusion models. *CoRR*, 2024.
- [73] Gaurav Parmar, Richard Zhang, and Jun-Yan Zhu. On aliased resizing and surprising subtleties in gan evaluation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 11410–11420, 2022.
- [74] Suraj Patil, William Berman, Robin Rombach, and Patrick von Platen. amused: An open muse reproduction. *arXiv preprint arXiv:2401.01808*, 2024.
- [75] Ekkasit Pinyoanuntapong, Pu Wang, Minwoo Lee, and Chen Chen. Mmm: Generative masked motion model. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 1546–1555, 2024.
- [76] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and

- Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In *The Twelfth International Conference on Learning Representations (ICLR)*, 2024.
- [77] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. *arXiv preprint arXiv:2209.14988*, 2022.
- [78] Zipeng Qi, Lichen Bai, Haoyi Xiong, and Zeke Xie. Not all noises are created equally: Diffusion noise selection and optimization. *arXiv preprint arXiv:2407.14041*, 2024.
- [79] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. *arXiv preprint arXiv:2204.06125*, 2022.
- [80] Jarrod Rector-Brooks, Mohsin Hasan, Zhangzhi Peng, Zachary Quinn, Chenghao Liu, Sarthak Mittal, Nouha Dziri, Michael Bronstein, Yoshua Bengio, Pranam Chatterjee, et al. Steering masked discrete diffusion models via discrete denoising posterior prediction. *arXiv preprint arXiv:2410.08134*, 2024.
- [81] Yinuo Ren, Haoxuan Chen, Yuchen Zhu, Wei Guo, Yongxin Chen, Grant M Rotskoff, Molei Tao, and Lexing Ying. Fast solvers for discrete diffusion models: Theory and applications of high-order algorithms. *arXiv preprint arXiv:2502.00234*, 2025.
- [82] Alfréd Rényi. On measures of entropy and information. In *Proceedings of the fourth Berkeley symposium on mathematical statistics and probability, volume 1: contributions to the theory of statistics*, pages 547–562. University of California Press, 1961.
- [83] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 10684–10695, 2022.
- [84] Stéphane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In *Proceedings of the fourteenth international conference on artificial intelligence and statistics*, pages 627–635. JMLR Workshop and Conference Proceedings, 2011.
- [85] Samir Sadok, Simon Leglaive, Laurent Girin, Gaël Richard, and Xavier Alameda-Pineda. Ancogen: Analysis, control and generation of speech with a masked autoencoder. In *IEEE International Conference on Acoustics, Speech, and Signal Processing*, 2025.
- [86] Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. *arXiv preprint arXiv:2406.07524*, 2024.
- [87] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. *arXiv preprint arXiv:2202.00512*, 2022.
- [88] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. *Advances in neural information processing systems*, 29, 2016.
- [89] Tim Salimans, Thomas Mensink, Jonathan Heek, and Emiel Hooeboom. Multistep distillation of diffusion models via moment matching. *Advances in Neural Information Processing Systems*, 37:36046–36070, 2025.
- [90] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In *European Conference on Computer Vision*, pages 87–103. Springer, 2024.
- [91] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In *European Conference on Computer Vision*, pages 87–103. Springer, 2025.
- [92] Yair Schiff, Subham Sekhar Sahoo, Hao Phung, Guanghan Wang, Sam Boshar, Hugo Dalla-torre, Bernardo P de Almeida, Alexander Rush, Thomas Pierrot, and Volodymyr Kuleshov. Simple guidance mechanisms for discrete diffusion models. *arXiv preprint arXiv:2412.10193*, 2024.
- [93] Florian Schmidt. Generalization in generation: A closer look at exposure bias. *arXiv preprint arXiv:1910.00292*, 2019.
- [94] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*, 2017.
- [95] Pier Giuseppe Sessa, Robert Dadashi, Léonard Hussenot, Johan Ferret, Nino Vieillard, Alexandre Ramé, Bobak Shariari, Sarah Perrin, Abe Friesen, Geoffrey Cideron, et al. Bond: Aligning llms with best-of-n distillation. *arXiv preprint arXiv:2407.14622*, 2024.
- [96] Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis Titsias. Simplified and generalized masked diffusion for discrete data. *Advances in Neural Information Processing Systems*, 37:103131–103167, 2025.
- [97] Raghav Singhal, Zachary Horvitz, Ryan Teehan, Mengye Ren, Zhou Yu, Kathleen McKeown, and Rajesh Ranganath. A general framework for inference-time scaling and steering of diffusion models. *arXiv preprint arXiv:2501.06848*, 2025.
- [98] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In *International Conference on Machine Learning*, pages 2256–2265. PMLR, 2015.
- [99] Jiaming Song and Linqi Zhou. Ideas in inference-time scaling can benefit generative pre-training algorithms. *arXiv preprint arXiv:2503.07154*, 2025.
- [100] Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models. *arXiv preprint arXiv:2310.14189*, 2023.
- [101] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. *arXiv preprint arXiv:2011.13456*, 2020.
- [102] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. *arXiv preprint arXiv:2303.01469*, 2023.
- [103] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model

- beats diffusion: Llama for scalable image generation. *arXiv preprint arXiv:2406.06525*, 2024.
- [104] Onkar Susladkar, Jishu Sen Gupta, Chirag Sehgal, Sparsh Mittal, and Rekha Singhal. Motionaura: Generating high-quality and motion consistent videos using discrete diffusion. *arXiv preprint arXiv:2410.07659*, 2024.
- [105] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. *Advances in neural information processing systems*, 37:84839–84865, 2024.
- [106] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. *Advances in neural information processing systems*, 30, 2017.
- [107] Guanghan Wang, Yair Schiff, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Remasking discrete diffusion models with inference-time scaling. *arXiv preprint arXiv:2503.00307*, 2025.
- [108] Xinyou Wang, Zaixiang Zheng, Fei Ye, Dongyu Xue, Shujian Huang, and Quanquan Gu. Dplm-2: A multi-modal diffusion protein language model. *arXiv preprint arXiv:2410.13782*, 2024.
- [109] Zhendong Wang, Huangjie Zheng, Pengcheng He, Weizhu Chen, and Mingyuan Zhou. Diffusion-gan: Training gans with diffusion. *arXiv preprint arXiv:2206.02262*, 2022.
- [110] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. *Advances in Neural Information Processing Systems*, 36, 2024.
- [111] Taiqiang Wu, Chaofan Tao, Jiahao Wang, Runming Yang, Zhe Zhao, and Ngai Wong. Rethinking kullback-leibler divergence in knowledge distillation for large language models. *arXiv preprint arXiv:2404.02657*, 2024.
- [112] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. *arXiv preprint arXiv:2306.09341*, 2023.
- [113] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. *arXiv preprint arXiv:2408.12528*, 2024.
- [114] Sirui Xie, Zhisheng Xiao, Diederik P Kingma, Tingbo Hou, Ying Nian Wu, Kevin Patrick Murphy, Tim Salimans, Ben Poole, and Ruiqi Gao. Em distillation for one-step diffusion models. *arXiv preprint arXiv:2405.16852*, 2024.
- [115] Yanwu Xu, Yang Zhao, Zhisheng Xiao, and Tingbo Hou. Ufogen: You forward once large scale text-to-image generation via diffusion gans. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 8196–8206, 2024.
- [116] Yilun Xu, Weili Nie, and Arash Vahdat. One-step diffusion models with  $f$ -divergence distribution matching. *arXiv preprint arXiv:2502.15681*, 2025.
- [117] Hanshu Yan, Xingchao Liu, Jiachun Pan, Jun Hao Liew, Qiang Liu, and Jiashi Feng. Perflow: Piecewise rectified flow as universal plug-and-play accelerator. *arXiv preprint arXiv:2405.07510*, 2024.
- [118] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. *arXiv preprint arXiv:2405.14867*, 2024.
- [119] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 6613–6623, 2024.
- [120] Mingtian Zhang, Peter Hayes, Thomas Bird, Raza Habib, and David Barber. Spread divergence. In *International Conference on Machine Learning*, pages 11106–11116. PMLR, 2020.
- [121] Mingtian Zhang, Jiajun He, Wenlin Chen, Zijing Ou, José Miguel Hernández-Lobato, Bernhard Schölkopf, and David Barber. Towards training one-step diffusion models without distillation. *arXiv preprint arXiv:2502.08005*, 2025.
- [122] Yixiu Zhao, Jiaxin Shi, Lester Mackey, and Scott Linderman. Informed correctors for discrete diffusion models. *arXiv preprint arXiv:2407.21243*, 2024.
- [123] Mingyuan Zhou, Zhendong Wang, Huangjie Zheng, and Hai Huang. Long and short guidance in score identity distillation for one-step text-to-image generation. *arXiv preprint arXiv:2406.01561*, 2024.
- [124] Mingyuan Zhou, Huangjie Zheng, Yi Gu, Zhendong Wang, and Hai Huang. Adversarial score identity distillation: Rapidly surpassing the teacher in one step. *arXiv preprint arXiv:2410.14919*, 2024.
- [125] Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, and Hai Huang. Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation. In *Forty-first International Conference on Machine Learning*, 2024.
- [126] Zikai Zhou, Shitong Shao, Lichen Bai, Zhiqiang Xu, Bo Han, and Zeke Xie. Golden noise for diffusion models: A learning framework. *arXiv preprint arXiv:2411.09502*, 2024.
- [127] Yuanzhi Zhu, Hanshu Yan, Huan Yang, Kai Zhang, and Junnan Li. Accelerating video diffusion models via distribution matching. *arXiv preprint arXiv:2412.05899*, 2024.
- [128] Yuanzhi Zhu, Xingchao Liu, and Qiang Liu. Slimflow: Training smaller one-step diffusion models with rectified flow. In *European Conference on Computer Vision*, pages 342–359. Springer, 2025.

# Di[M]O: Distilling Masked Diffusion Models into One-step Generator

## Supplementary Material

The supplementary material is organized as follows:

- Appendix A: Relevant derivations of the approximated divergence gradient.
- Appendix B: Discussion of additional related works.
- Appendix C: Detailed experiment setup.
- Appendix D: Additional experiments and corresponding findings.
- Appendix E: Failure cases where the generation quality does not match that of the teacher model.
- Appendix F: Visualization and example of the mode-seeking/covering behaviors of the generalized Jeffrey divergence.
- Appendix G: Additional visual results of one-step generations from our distilled models.
- Appendix H: List of all prompts used in this paper for image generation.

### A. Relevant Derivations to the Token-level Distillation Loss

#### A.1. Loss Gradient

Given Eq. (2) and Eq. (4), by assuming  $D$  is differentiable with respect to  $p_\theta(x_0^i|\tilde{x}_t)$ , we can calculate Eq. (6) using chain rule as:

$$\begin{aligned}
 \nabla_\theta \mathcal{L}_{\text{Di[M]O}} &= \nabla_\theta \mathbb{E}_{x_{\text{init}}, t} [w(t) (\mathbb{E}_{q_{t|0}} [D((p_\phi||p_\theta)(\tilde{x}_t))])] \\
 &= \mathbb{E}_{x_{\text{init}}, t} [w(t) (\mathbb{E}_{q_{t|0}} [\nabla_\theta D((p_\phi||p_\theta)(\tilde{x}_t))])] \\
 &= \mathbb{E}_{x_{\text{init}}, t} \left[ w(t) \left( \mathbb{E}_{q_{t|0}} \left[ \frac{1}{L_M} \sum_{\substack{i=1 \\ \tilde{x}_t^i=[M]}}^L \nabla_\theta D(p_\phi(x_0^i|\tilde{x}_t)||p_\theta(x_0^i|\tilde{x}_t)) \right] \right) \right] \\
 &= \mathbb{E}_{x_{\text{init}}, t} \left[ w(t) \left( \mathbb{E}_{q_{t|0}} \left[ \frac{1}{L_M} \sum_{\substack{i=1 \\ \tilde{x}_t^i=[M]}}^L \nabla_{p_\theta(x_0^i|\tilde{x}_t)} D(p_\phi(x_0^i|\tilde{x}_t)||p_\theta(x_0^i|\tilde{x}_t)) \frac{dp_\theta(x_0^i|\tilde{x}_t)}{dz_\theta^i} \frac{dz_\theta^i}{d\theta} \right] \right) \right].
 \end{aligned} \tag{9}$$

By defining  $\nabla_{z_\theta} D((p_\phi||p_\theta)(\tilde{x}_t))$  as a vector with the  $i$ -th element  $[\nabla_{z_\theta} D((p_\phi||p_\theta)(\tilde{x}_t))]_i = \frac{1}{L_M} \nabla_{p_\theta(x_0^i|\tilde{x}_t)} D(p_\phi(x_0^i|\tilde{x}_t)||p_\theta(x_0^i|\tilde{x}_t)) \frac{dp_\theta(x_0^i|\tilde{x}_t)}{dz_\theta^i}$ , and  $\frac{dz_\theta}{d\theta}$  as a vector with the  $i$ -th element  $\frac{dz_\theta^i}{d\theta}$ , we have:

$$\nabla_\theta \mathcal{L}_{\text{Di[M]O}} = \mathbb{E}_{x_{\text{init}}, t} \left[ w(t) \left( \mathbb{E}_{q_{t|0}} \left[ \nabla_{z_\theta} D(p_\phi||p_\theta)(\tilde{x}_t) \frac{dz_\theta(\tilde{x}_t)}{d\theta} \right] \right) \right]. \tag{10}$$

By default, we apply stop-gradient to the  $\tilde{x}_t$ , since the sample operation from  $z_\theta$  to  $x_\theta(x_{\text{init}})$  is non-differentiable.

#### A.2. Explicit Form of Divergence

We start the derivation with the FKL and RKL in the generalized Jeffrey divergence. The forward and reverse KL between the teacher  $\phi$  and the one-step generator  $\theta$  at each output location  $i$  are:

$$\begin{aligned}
 D_{\text{FKL}_i}(\tilde{x}_t) &= \sum_{k=1}^V p_\phi(x_0^i = k|\tilde{x}_t) \log \left( \frac{p_\phi(x_0^i = k|\tilde{x}_t)}{p_\theta(x_0^i = k|\tilde{x}_t)} \right), \\
 D_{\text{RKL}_i}(\tilde{x}_t) &= \sum_{k=1}^V p_\theta(x_0^i = k|\tilde{x}_t) \log \left( \frac{p_\theta(x_0^i = k|\tilde{x}_t)}{p_\phi(x_0^i = k|\tilde{x}_t)} \right).
 \end{aligned} \tag{11}$$

The derivation of these KLs with respect to the student parameter  $\theta$  can be written as:

$$\begin{aligned}\nabla_{\theta} D_{FKL_i} &= \sum_{j=1}^V \frac{\partial D_{FKL_i}}{\partial z_{\theta_j^i}} \frac{\partial z_{\theta_j^i}}{\partial \theta}, \\ \nabla_{\theta} D_{RKL_i} &= \sum_{j=1}^V \frac{\partial D_{RKL_i}}{\partial z_{\theta_j^i}} \frac{\partial z_{\theta_j^i}}{\partial \theta}.\end{aligned}\tag{12}$$

Given that the probability corresponding to the logits  $z$  as  $p_{\theta}(x_0^i = k|\tilde{x}_t) = \frac{\exp(z_{\theta_k^i})}{\sum_{n=1}^V \exp(z_{\theta_n^i})}$ , we can precalculate the following quantity for  $j \neq k$ :

$$\begin{aligned}\frac{\partial}{\partial z_{\theta_j^i}} p_{\theta}(x_0^i = j|\tilde{x}_t) &= \frac{\exp(z_{\theta_j^i})}{\sum_{n=1}^V \exp(z_{\theta_n^i})} - p_{\theta}(x_0^i = j|\tilde{x}_t)^2 = p_{\theta}(x_0^i = j|\tilde{x}_t)(1 - p_{\theta}(x_0^i = j|\tilde{x}_t)), \\ \frac{\partial}{\partial z_{\theta_j^i}} p_{\theta}(x_0^i = k|\tilde{x}_t) &= -\frac{\exp(z_{\theta_k^i}) \exp(z_{\theta_j^i})}{(\sum_{n=1}^V \exp(z_{\theta_n^i}))^2} = -p_{\theta}(x_0^i = k|\tilde{x}_t)p_{\theta}(x_0^i = j|\tilde{x}_t).\end{aligned}\tag{13}$$

These are the gradients of the softmax function that we will use below to help the derivation.

##### A.2.1. Gradient of FKL

For each possible token  $i$ , we have:

$$\begin{aligned}\frac{\partial D_{FKL_i}}{\partial z_{\theta_j^i}} &= \frac{\partial}{\partial z_{\theta_j^i}} \sum_{k=1}^V p_{\phi}(x_0^i = k|\tilde{x}_t) \log \left( \frac{p_{\phi}(x_0^i = k|\tilde{x}_t)}{p_{\theta}(x_0^i = k|\tilde{x}_t)} \right) \\ &= \frac{\partial}{\partial z_{\theta_j^i}} \sum_{k=1}^V -p_{\phi}(x_0^i = k|\tilde{x}_t) \log p_{\theta}(x_0^i = k|\tilde{x}_t) \quad // \text{From KL to cross entropy} \\ &= \frac{\partial}{\partial z_{\theta_j^i}} \left( \sum_{k=1, k \neq j}^V -p_{\phi}(x_0^i = k|\tilde{x}_t) \log p_{\theta}(x_0^i = k|\tilde{x}_t) \right) - \frac{\partial}{\partial z_{\theta_j^i}} p_{\phi}(x_0^i = j|\tilde{x}_t) \log p_{\theta}(x_0^i = j|\tilde{x}_t) \\ &= \left( \sum_{k=1, k \neq j}^V -\frac{p_{\phi}(x_0^i = k|\tilde{x}_t)}{p_{\theta}(x_0^i = k|\tilde{x}_t)} \frac{\partial}{\partial z_{\theta_j^i}} p_{\theta}(x_0^i = k|\tilde{x}_t) \right) - \frac{p_{\phi}(x_0^i = j|\tilde{x}_t)}{p_{\theta}(x_0^i = j|\tilde{x}_t)} \frac{\partial}{\partial z_{\theta_j^i}} p_{\theta}(x_0^i = j|\tilde{x}_t) \quad // \text{derivative of } \log(\cdot) \\ &= \sum_{k=1, k \neq j}^V p_{\phi}(x_0^i = k|\tilde{x}_t)p_{\theta}(x_0^i = j|\tilde{x}_t) - p_{\phi}(x_0^i = j|\tilde{x}_t)(1 - p_{\theta}(x_0^i = j|\tilde{x}_t)) \quad // \text{substitute Eq. (13)} \\ &= \sum_{k=1}^V p_{\phi}(x_0^i = k|\tilde{x}_t)p_{\theta}(x_0^i = j|\tilde{x}_t) - p_{\phi}(x_0^i = j|\tilde{x}_t) \\ &= p_{\theta}(x_0^i = j|\tilde{x}_t) - p_{\phi}(x_0^i = j|\tilde{x}_t). \quad // \text{probability sums up to 1}\end{aligned}\tag{14}$$

##### A.2.2. Gradient of RKL

For each possible token  $i$ , we have:

$$\begin{aligned}
\frac{\partial D_{RKL_i}}{\partial z_{\theta_j^i}} &= \frac{\partial}{\partial z_{\theta_j^i}} \sum_{k=1}^V p_{\theta}(x_0^i = k|\tilde{x}_t) \log \left( \frac{p_{\theta}(x_0^i = k|\tilde{x}_t)}{p_{\phi}(x_0^i = k|\tilde{x}_t)} \right) \\
&= \frac{\partial}{\partial z_{\theta_j^i}} \left( \sum_{k=1, k \neq j}^V p_{\theta}(x_0^i = k|\tilde{x}_t) \log \left( \frac{p_{\theta}(x_0^i = k|\tilde{x}_t)}{p_{\phi}(x_0^i = k|\tilde{x}_t)} \right) \right) + \frac{\partial}{\partial z_{\theta_j^i}} \left( p_{\theta}(x_0^i = j|\tilde{x}_t) \log \left( \frac{p_{\theta}(x_0^i = j|\tilde{x}_t)}{p_{\phi}(x_0^i = j|\tilde{x}_t)} \right) \right) \\
&= \sum_{k=1, k \neq j}^V \left( -p_{\theta}(x_0^i = k|\tilde{x}_t) p_{\theta}(x_0^i = j|\tilde{x}_t) \log \left( \frac{p_{\theta}(x_0^i = k|\tilde{x}_t)}{p_{\phi}(x_0^i = k|\tilde{x}_t)} \right) - p_{\theta}(x_0^i = k|\tilde{x}_t) p_{\theta}(x_0^i = j|\tilde{x}_t) \right) \\
&\quad + p_{\theta}(x_0^i = j|\tilde{x}_t) (1 - p_{\theta}(x_0^i = j|\tilde{x}_t)) \log \left( \frac{p_{\theta}(x_0^i = j|\tilde{x}_t)}{p_{\phi}(x_0^i = j|\tilde{x}_t)} \right) + p_{\theta}(x_0^i = j|\tilde{x}_t) (1 - p_{\theta}(x_0^i = j|\tilde{x}_t)) \quad // \text{substitute Eq. (13)} \\
&= \sum_{k=1}^V \left( -p_{\theta}(x_0^i = k|\tilde{x}_t) p_{\theta}(x_0^i = j|\tilde{x}_t) \log \left( \frac{p_{\theta}(x_0^i = k|\tilde{x}_t)}{p_{\phi}(x_0^i = k|\tilde{x}_t)} \right) - p_{\theta}(x_0^i = k|\tilde{x}_t) p_{\theta}(x_0^i = j|\tilde{x}_t) \right) \\
&\quad + p_{\theta}(x_0^i = j|\tilde{x}_t) \log \left( \frac{p_{\theta}(x_0^i = j|\tilde{x}_t)}{p_{\phi}(x_0^i = j|\tilde{x}_t)} \right) + p_{\theta}(x_0^i = j|\tilde{x}_t) \\
&= \sum_{k=1}^V \left( -p_{\theta}(x_0^i = k|\tilde{x}_t) p_{\theta}(x_0^i = j|\tilde{x}_t) \log \left( \frac{p_{\theta}(x_0^i = k|\tilde{x}_t)}{p_{\phi}(x_0^i = k|\tilde{x}_t)} \right) \right) - \cancel{p_{\theta}(x_0^i = j|\tilde{x}_t)} \quad // \text{probability sums up to 1} \\
&\quad + p_{\theta}(x_0^i = j|\tilde{x}_t) \log \left( \frac{p_{\theta}(x_0^i = j|\tilde{x}_t)}{p_{\phi}(x_0^i = j|\tilde{x}_t)} \right) + \cancel{p_{\theta}(x_0^i = j|\tilde{x}_t)} \\
&= p_{\theta}(x_0^i = j|\tilde{x}_t) \left( \sum_{k=1}^V \left( -p_{\theta}(x_0^i = k|\tilde{x}_t) \log \left( \frac{p_{\theta}(x_0^i = k|\tilde{x}_t)}{p_{\phi}(x_0^i = k|\tilde{x}_t)} \right) \right) + \log \left( \frac{p_{\theta}(x_0^i = j|\tilde{x}_t)}{p_{\phi}(x_0^i = j|\tilde{x}_t)} \right) \right) \\
&= p_{\theta}(x_0^i = j|\tilde{x}_t) \left( \log \left( \frac{p_{\theta}(x_0^i = j|\tilde{x}_t)}{p_{\phi}(x_0^i = j|\tilde{x}_t)} \right) - D_{RKL_i} \right). \quad // \text{definition of RKL}
\end{aligned} \tag{15}$$

This is similar to the results derived for AR LLM in [111].

As a result, the approximated token-level gradients of FKL and RKL at each masked position  $i$  in Eq. (5) can be calculated as follows:

$$\begin{aligned}
\nabla_{z_{\psi}} D_{FKL_i}(p_{\phi}(x_0^i|\tilde{x}_t) \| p_{\psi}(x_0^i|\tilde{x}_t)) &:= \nabla_{z_{\psi}} D_{FKL_i}(\tilde{x}_t) = (p_{\psi}(x_0^i|\tilde{x}_t) - p_{\phi}(x_0^i|\tilde{x}_t)), \\
\nabla_{z_{\psi}} D_{RKL_i}(p_{\phi}(x_0^i|\tilde{x}_t) \| p_{\psi}(x_0^i|\tilde{x}_t)) &:= \nabla_{z_{\psi}} D_{RKL_i}(\tilde{x}_t) = p_{\psi}(x_0^i|\tilde{x}_t) \left( \log \left( \frac{p_{\psi}(x_0^i|\tilde{x}_t)}{p_{\phi}(x_0^i|\tilde{x}_t)} \right) - D_{RKL_i}(\tilde{x}_t) \right).
\end{aligned} \tag{16}$$

##### A.2.3. Gradient of $f$ -divergence

Our proposed token-level distillation can be seamlessly extended to general  $f$ -divergence [17] with the form [30, 82, 116]:

$$D_{f_i} = \sum_{k=1}^V p_{\theta}(x_0^i = k|\tilde{x}_t) f \left( \frac{p_{\phi}(x_0^i = k|\tilde{x}_t)}{p_{\theta}(x_0^i = k|\tilde{x}_t)} \right). \tag{17}$$

When the generator function  $f$  is differentiable, we can calculate its gradient as:

$$\begin{aligned}
\frac{\partial D_{f_i}}{\partial z_{\theta_j^i}} &= \frac{\partial}{\partial z_{\theta_j^i}} \sum_{k=1}^V p_{\phi}(x_0^i = k|\tilde{x}_t) f \left( \frac{p_{\phi}(x_0^i = k|\tilde{x}_t)}{p_{\theta}(x_0^i = k|\tilde{x}_t)} \right) \\
&= \sum_{k=1}^V \left( \frac{\partial p_{\phi}(x_0^i = k|\tilde{x}_t)}{\partial z_{\theta_j^i}} \left[ f \left( \frac{p_{\phi}(x_0^i = k|\tilde{x}_t)}{p_{\theta}(x_0^i = k|\tilde{x}_t)} \right) - \left( \frac{p_{\phi}(x_0^i = k|\tilde{x}_t)}{p_{\theta}(x_0^i = k|\tilde{x}_t)} \right) f' \left( \frac{p_{\phi}(x_0^i = k|\tilde{x}_t)}{p_{\theta}(x_0^i = k|\tilde{x}_t)} \right) \right] \right) \\
&= p_{\theta}(x_0^i = j|\tilde{x}_t) \sum_{k=1}^V \left( -p_{\theta}(x_0^i = k|\tilde{x}_t) \left[ f \left( \frac{p_{\phi}(x_0^i = k|\tilde{x}_t)}{p_{\theta}(x_0^i = k|\tilde{x}_t)} \right) - \left( \frac{p_{\phi}(x_0^i = k|\tilde{x}_t)}{p_{\theta}(x_0^i = k|\tilde{x}_t)} \right) f' \left( \frac{p_{\phi}(x_0^i = k|\tilde{x}_t)}{p_{\theta}(x_0^i = k|\tilde{x}_t)} \right) \right] \right) \\
&\quad + p_{\theta}(x_0^i = j|\tilde{x}_t) \left[ f \left( \frac{p_{\phi}(x_0^i = j|\tilde{x}_t)}{p_{\theta}(x_0^i = j|\tilde{x}_t)} \right) - \left( \frac{p_{\phi}(x_0^i = j|\tilde{x}_t)}{p_{\theta}(x_0^i = j|\tilde{x}_t)} \right) f' \left( \frac{p_{\phi}(x_0^i = j|\tilde{x}_t)}{p_{\theta}(x_0^i = j|\tilde{x}_t)} \right) \right].
\end{aligned} \tag{18}$$

As shown in Tab. 4, the generalized Jeffrey divergence belongs to  $f$ -divergence, with generator function  $f(u) = ((1 - \beta)u - \beta) \log u$ .

Table 4. Summary of some typical  $f$ -divergences  $D_f(p||q)$  together with generator functions  $f$ , where  $f : (0, \infty) \rightarrow \mathbb{R}$  is a convex function satisfying the condition  $f(1) = 0$ . This table is mainly adapted from [30].

| Name                                              | $D_f(p  q)$                                                                                                                                                                | Generator $f(u)$                                                   |
|---------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| Forward Kullback-Leibler                          | $\int p(x) \log \frac{p(x)}{q(x)} dx$                                                                                                                                      | $u \log u$                                                         |
| Reverse Kullback-Leibler                          | $\int q(x) \log \frac{q(x)}{p(x)} dx$                                                                                                                                      | $-\log u$                                                          |
| $\alpha$ -divergence ( $\alpha \notin \{0, 1\}$ ) | $\frac{1}{\alpha(\alpha-1)} \int \left( q(x) \left[ \left( \frac{p(x)}{q(x)} \right)^{1-\alpha} - (1-\alpha) \left( \frac{p(x)}{q(x)} \right) - \alpha \right] \right) dx$ | $\frac{1}{\alpha(\alpha-1)} (u^{1-\alpha} - (1-\alpha)u - \alpha)$ |
| Generalized Jeffrey                               | $\int [(1-\beta)p(x) - \beta q(x)] \log \left( \frac{p(x)}{q(x)} \right) dx$                                                                                               | $((1-\beta)u - \beta) \log u$                                      |
| Jensen-Shannon                                    | $\frac{1}{2} \int p(x) \log \frac{2p(x)}{p(x)+q(x)} + q(x) \log \frac{2q(x)}{p(x)+q(x)} dx$                                                                                | $-(u+1) \log \frac{1+u}{2} + u \log u$                             |
| Squared Hellinger                                 | $\int \left( \sqrt{p(x)} - \sqrt{q(x)} \right)^2 dx$                                                                                                                       | $(\sqrt{u} - 1)^2$                                                 |

Table 5. Generative performance on class-conditional ImageNet-256. Results of methods in type AR are taken from the DD [53]. Percentage drop values (relative to the teacher) are shown in parentheses.

| Type | Model              | FID ( $\downarrow$ ) | IS ( $\uparrow$ ) | Precision ( $\uparrow$ ) | Recall ( $\uparrow$ ) | #Para | Step ( $\downarrow$ ) |
|------|--------------------|----------------------|-------------------|--------------------------|-----------------------|-------|-----------------------|
| AR   | VAR-d16 [105]      | 4.19                 | 230.2             | 0.84                     | 0.48                  | 310M  | 10                    |
| AR   | VAR-d20 [105]      | 3.35                 | 301.4             | 0.84                     | 0.51                  | 600M  | 10                    |
| AR   | VAR-d24 [105]      | 2.51                 | 312.2             | 0.82                     | 0.53                  | 1.03B | 10                    |
| AR   | VAR-d16-DD [53]    | 9.94 (137%)          | 193.6 (16%)       | 0.80 (5%)                | 0.37 (23%)            | 327M  | <b>1</b>              |
| AR   | VAR-d16-DD [53]    | 7.82 (87%)           | 197.0 (14%)       | 0.80 (5%)                | 0.41 (15%)            | 327M  | 2                     |
| AR   | VAR-d20-DD [53]    | 9.55 (185%)          | 197.2 (35%)       | 0.78 (7%)                | 0.38 (26%)            | 635M  | <b>1</b>              |
| AR   | VAR-d20-DD [53]    | 7.33 (119%)          | 204.5 (32%)       | 0.82 (2%)                | 0.40 (22%)            | 635M  | 2                     |
| AR   | VAR-d24-DD [53]    | 8.92 (255%)          | 202.8 (35%)       | 0.78 (5%)                | 0.39 (26%)            | 1.09B | <b>1</b>              |
| AR   | VAR-d24-DD [53]    | 6.95 (177%)          | 222.5 (29%)       | 0.83 (-1%)               | 0.43 (19%)            | 1.09B | 2                     |
| AR   | LlamaGen-B [103]   | 5.42                 | 193.5             | 0.83                     | 0.44                  | 111M  | 256                   |
| AR   | LlamaGen-L [103]   | 4.11                 | 283.5             | 0.85                     | 0.48                  | 343M  | 256                   |
| AR   | LlamaGen-B-DD [53] | 15.50 (186%)         | 135.4 (30%)       | 0.76 (8%)                | 0.26 (41%)            | 98.3M | <b>1</b>              |
| AR   | LlamaGen-B-DD [53] | 11.17 (106%)         | 154.8 (20%)       | 0.80 (4%)                | 0.31 (30%)            | 98.3M | 2                     |
| AR   | LlamaGen-L-DD [53] | 11.35 (176%)         | 193.6 (32%)       | 0.81 (5%)                | 0.30 (38%)            | 326M  | <b>1</b>              |
| AR   | LlamaGen-L-DD [53] | 7.58 (84%)           | 237.5 (16%)       | 0.84 (1%)                | 0.37 (23%)            | 326M  | 2                     |
| MDM  | MaskGit [11]       | 6.60                 | 224.07            | 0.831                    | 0.402                 | 174M  | 16                    |
| MDM  | Di[M]O             | 6.91 (5%)            | 214.05 (4%)       | 0.828 (0.4%)             | 0.377 (6.2%)          | 174M  | <b>1</b>              |

### B. More Discussion on Related Works

Recent research has explored the possibility of converting AR models into discrete diffusion versions [27]. Given this ongoing work, we plan to investigate the applicability of our approach to AR models. Furthermore, while existing limitations of MDM are well known, an efficient MDM could still serve as a draft model for AR speculative decoding [16]. Diffusion distillation has seen widespread adoption of the GAN objective, either directly for distillation [91, 109, 115] or to enhance performance [51, 118, 124]. However, these methods are not directly applicable to MDM due to its discrete nature. In a similar vein, recent works based on SiD [60, 61, 123–125] aim to minimize the Fisher divergence or a generalized score-based divergence for distillation. However, these approaches require backpropagating the gradient through the teacher model—akin to adversarial training—which is infeasible for MDM due to the non-differentiable sampling operation. Xu *et al.* [116] recently introduced a general framework for distilling continuous diffusion models using  $f$ -divergence. At the cost of training another additional discriminator, this method successfully extends the DMD framework from RKL to general  $f$ -divergence by utilizing the output of the discriminator to weight the loss gradient in DMD. However, their method relies on the assumption that the teacher model and the real data used for training the discriminator obey the same underline distribution (otherwise need to simulate the teacher model to get more accurate synthetic data). The training of additional discriminator in general increases the computational overhead. Moreover, while effective for continuous models, their method relies on an additional GAN loss to achieve better performance, making it unsuitable for MDMs. To show the performance of our method, we compare it with DD [53], which distills AR models into one-step or few-step generators. As demonstrated in Tab. 5, our method yields significantly smaller performance drops compared to the teacher models than [53]. To summarize, our method successfully address the multi-token prediction challenge pointed out in [31, 99], by transfer the stochasticity into the model input and always predict tokens from the correct joint distribution of multiple tokens.

Table 6. Control experiments on the initialization choices to validate the hypothesis in Sec. 4.3.

| Initial Method                                                                                        | Step ( $\downarrow$ ) | FID-5k ( $\downarrow$ ) | IS-5k ( $\uparrow$ ) |
|-------------------------------------------------------------------------------------------------------|-----------------------|-------------------------|----------------------|
| use VAE encoded code of random noise                                                                  | 1                     | 189.39                  | 4.67                 |
| use $1 - r_{\text{init}}$ random image tokens, $r_{\text{init}}$ fixed class token (e.g. 1025)        | 1                     | 173.02                  | 5.00                 |
| use $1 - r_{\text{init}}$ random image tokens, $r_{\text{init}}$ fixed image token (e.g. 512)         | 1                     | 188.3                   | 4.65                 |
| use $1 - r_{\text{init}}$ random class tokens, $r_{\text{init}}$ fixed mask token [M]                 | 1                     | 174.47                  | 4.73                 |
| use $1 - r_{\text{init}}$ random image tokens, $r_{\text{init}}$ fixed mask token [M] (ours strategy) | 1                     | 12.01                   | 132.44               |

### C. More Experiment Setup

For the sampling of teacher model, we use the heuristic parallel sampler by default, with temperature=1.3, schedule mode arccos, and choose token to remask randomly during sampling (as the greedy approach by keeping the most confident tokens leads to degraded generation). In Fig. 14a, we show the teacher’s FID with 5k generated images under-difference inference steps with different CFG. This result suggests the limitation of the parallel sampler for test time scaling, as pointed out in [81]. Given these results, we use 16 steps and CFG=2.5 for the teacher model by default in the paper. In order to add random perturbation to the token embeddings, we fix the embedding layer of all the models during distillation. During the experiment, we use the same mask schedule for getting the intermediate state  $\tilde{x}_t$  from one-step model generation and training the auxiliary model. In addition, we choose the same schedule when training the teacher model. We use the arccos schedule for MaskGit distillation and the cosine schedule for Meisssonic distillation, respectively. We adopt the loss weight from [119] with  $w(t) = \frac{1}{p_{\theta}(x_0|\tilde{x}_t) - p_{\phi}(x_0|x_{\text{init}})}$ . Our experiment suggests that this weight can prevent the gradient of the generator from exploding (while the one with 1000 times the gradient value can still generate a regular image). We use by default mixed precision training with *bf16* and a gradient clipping gradient normalization 1. We use a constant learning rate scheduler with a linear warmup of 100 steps. We use the *adam* optimizer with beta1 = 0.9 and beta2 = 0.999 and no weight decay for all experiments. All temperatures for the three models are fixed to 1 during distillation. Exponential Moving Average (EMA) is applied with a rate of 0.9999 for all experiments. The codebook sizes for MaskGit and Meisssonic are 1024 and 8192, and the sequence length of the latent codes for MaskGit and Meisssonic are 1024 and 4096, respectively. For a sequence length of  $L$ , a codebook size of  $V$ , and the number of [M] tokens to be replaced  $N \approx (1 - r_{\text{init}})L$ , the possible initial token configure is  $\binom{L}{N} V^N$ . Given that  $L$  and  $V$  are usually large numbers, the possible initial token configure are sufficient to be the initial state. This explains why the noise perturbation yield marginal improvement, even though stabilize the training. The embedding layer in the one-step generator is fixed during distillation to improve the stability. By default, the FID, precision, recall, density and coverage are all calculated with features extracted from the InceptionV3 network.

### D. More Experimental Results

![Figure 7: IS results of the ablations corresponding to Fig. 5. The figure consists of three subplots (a, b, c) showing IS (5k) vs Generator Temperature (2 to 10). Subplot (a) shows the effect of Initial Mask Ratio (r_init) from 0.0 to 1.0. Subplot (b) shows the effect of Jeffrey Coefficient (beta) from -0.4 to 0.5. Subplot (c) shows the effect of Perturbation Strength (sigma_init) from 0.0 to 1.0. In all plots, IS generally decreases as temperature increases, with some values marked as collapsed (*) in the subplots.](df7cb4ea9bd6c3f445f3e264773b125f_img.jpg)

Figure 7 displays three line plots showing the Inception Score (IS) at 5k generated images versus Generator Temperature (ranging from 2 to 10). The plots illustrate the results of ablation studies for different initialization parameters.

- (a)  $r_{\text{init}}$ : Initial Mask Ratio. The plot shows IS (5k) on the y-axis (120 to 140) and Generator Temperature on the x-axis (2 to 10). Multiple lines represent different  $r_{\text{init}}$  values (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0). Some values are marked with (\*) indicating training collapse. An inset shows a zoomed-in view of the high-temperature region (8 to 10) where IS values are lower (0 to 50).
- (b)  $\beta$ : Jeffrey Coefficient. The plot shows IS (5k) on the y-axis (120 to 140) and Generator Temperature on the x-axis (2 to 10). Multiple lines represent different  $\beta$  values (-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5). An inset shows a zoomed-in view of the high-temperature region (8 to 10) where IS values are lower (2.5 to 5.0).
- (c)  $\sigma_{\text{init}}$ : Perturbation Strength. The plot shows IS (5k) on the y-axis (110 to 140) and Generator Temperature on the x-axis (2 to 10). Multiple lines represent different  $\sigma_{\text{init}}$  values (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0). Some values are marked with (\*) indicating training collapse. An inset shows a zoomed-in view of the high-temperature region (8 to 10) where IS values are lower (5 to 7).

Figure 7: IS results of the ablations corresponding to Fig. 5. The figure consists of three subplots (a, b, c) showing IS (5k) vs Generator Temperature (2 to 10). Subplot (a) shows the effect of Initial Mask Ratio (r\_init) from 0.0 to 1.0. Subplot (b) shows the effect of Jeffrey Coefficient (beta) from -0.4 to 0.5. Subplot (c) shows the effect of Perturbation Strength (sigma\_init) from 0.0 to 1.0. In all plots, IS generally decreases as temperature increases, with some values marked as collapsed (\*) in the subplots.

Figure 7. IS results of the ablations corresponding to Fig. 5. \* means the training is collapsed and falls outside the comparable range with other results, we therefore show them in the sub-figures at the right upper corner with the same range of the x-axis.

In this section we show more experiment results.

- IS results from ablation.** We present the IS results of the ablation in Fig. 7. Unlike FID, we found that the best IS is achieved at a much lower temperature. This suggest that the temperature can control the trade-off between FID and IS. We show visually how the final output image changes with the generator temperature in Fig. 8.
- One-step generation metric score with different temperature settings.** In Fig. 14b, we show the FID and IS metric with varying temperatures of the generator. We observe that, as the temperature increases, the IS score decreases, while the FID score initially declines, reaching its minimum at a temperature of 7, before starting to rise again.

![Figure 8: A 7x8 grid of images showing one-step generation at different temperature scales. The first four rows show clear, distinct images of a goldfish, a rooster, a cat, an hourglass, two ships, a dog, a race car, and an airplane. The fifth row shows slightly more distorted versions of these images. The sixth and seventh rows show highly abstract, noisy, and pixelated versions of the same subjects, indicating increasing generation temperature from top to bottom.](3267a096e9ca525744d8cd820f12eb59_img.jpg)

Figure 8: A 7x8 grid of images showing one-step generation at different temperature scales. The first four rows show clear, distinct images of a goldfish, a rooster, a cat, an hourglass, two ships, a dog, a race car, and an airplane. The fifth row shows slightly more distorted versions of these images. The sixth and seventh rows show highly abstract, noisy, and pixelated versions of the same subjects, indicating increasing generation temperature from top to bottom.

Figure 8. One-step generation at different temperature scale. From top to bottom the temperature is [1e-6, 1e-3, 1, 10, 20, 50, 100]

3. **Top-k visualization.** We apply the top-k trick when sampling each token from the predicted distributions  $p_\phi(x_0^i | \tilde{x}_t)$  and  $p_\theta(x_0^i | x_{\text{init}})$ . As shown in Fig. 12, after distillation, our one-step generation with top-1 sampling and without top-k are nearly identical, unlike in the teacher model. This suggests that each initial code almost deterministically maps to a fixed set of image tokens, justifying the use of Gaussian perturbation for token embeddings in the generator.
4. **Analysis of output distribution.** To further investigate this, we examine the number of potential output tokens with probabilities greater than 0.001, as shown in Fig. 9a. Specifically, we analyze 256 randomly generated images (total 65536 tokens), where each token in every image has a total of 1024 possible output tokens (vocabulary size). We find that for nearly half of the tokens, the output logits from our distilled model collapse into a delta distribution, confirming the trends observed in Fig. 12 and Fig. 8. For comparison, Fig. 9b presents the corresponding distribution for the teacher model, where the potential output token probabilities are more evenly spread across different possible token values.
5. **Information in the initial sequence.** Similar to several recent works on the influence of initial noise on the final generated images

![Figure 9: Two histograms showing the distribution of potential tokens. (a) Histogram of student probabilities: The x-axis is 'Number of potential tokens' (0 to 140) and the y-axis is 'Number of tokens' (0 to 25000). The distribution is highly skewed towards zero, with a peak at 0. (b) Histogram of teacher probabilities: The x-axis is 'Number of potential tokens' (0 to 400) and the y-axis is 'Number of tokens' (0 to 700). The distribution is more spread out, peaking around 50-100 tokens.](3102c32204f998dba666e1e915d5babf_img.jpg)

Figure 9: Two histograms showing the distribution of potential tokens. (a) Histogram of student probabilities: The x-axis is 'Number of potential tokens' (0 to 140) and the y-axis is 'Number of tokens' (0 to 25000). The distribution is highly skewed towards zero, with a peak at 0. (b) Histogram of teacher probabilities: The x-axis is 'Number of potential tokens' (0 to 400) and the y-axis is 'Number of tokens' (0 to 700). The distribution is more spread out, peaking around 50-100 tokens.

Figure 9. Histogram of number of potential output with probability greater than 0.001

![Figure 10: A 3x3 grid of images comparing 'Real image', 'Reconstruction', and 'Random generation' for three different classes. The first row shows hot air balloons, the second row shows white fluffy dogs (Samoyeds), and the third row shows churches. The 'Reconstruction' column shows images that closely resemble the 'Real image' column, while the 'Random generation' column shows more diverse and less similar results.](eea8b24476b46def99046ef43c716b10_img.jpg)

Figure 10: A 3x3 grid of images comparing 'Real image', 'Reconstruction', and 'Random generation' for three different classes. The first row shows hot air balloons, the second row shows white fluffy dogs (Samoyeds), and the third row shows churches. The 'Reconstruction' column shows images that closely resemble the 'Real image' column, while the 'Random generation' column shows more diverse and less similar results.

Figure 10. One-step image generation from encoded image tokens and random image tokens. The class labels are 417, 279 and 497, respectively.

[13, 58, 78, 126]. In this work, we designed a token initialization strategy which injects randomness in the initial sequence  $x_0$ . We here investigate the influence in the initial code. For a distilled model trained with  $r_{\text{init}} = 0.6$ , we tested its performance using initial sequences composed of either 40% random image tokens or 40% image tokens derived from encoding real images with VQ-VAE. The results, shown in Fig. 10, present three image examples with three different random seeds applied to each for both the random image tokens case and the encoded image token case. We find that, unlike using random image tokens when the one-step generator can produce diverse images, using the real image tokens instead results in generations that closely resemble the original images. This suggests that, similar to continuous diffusion processes, the information contained in randomly initialized codes significantly influences the final generation. This property enables test-time scaling techniques for our model to improve performance [62, 107].

6. **Interpolation between initial token sequences.** In addition, we show the interpolation results between random initial sequence and encoded sequence in Fig. 11.
7. **Control experiments on token initialization strategy.** In our experiments, we conducted a controlled study on the token initialization strategy to verify our hypothesis in Sec. 4.3 that the initial sequence of the student should be similar to those used to train the teacher model (see Tab. 6). Similar to our main strategy, we replaced a fraction  $1 - r_{\text{init}}$  of the  $[M]$  tokens with random *visual* tokens. However, in this case, we replaced the remaining  $[M]$  tokens with a fixed token, such as image token 512 or class token 1025. Additionally, we tested another variation where we replaced  $1 - r_{\text{init}}$  of the  $[M]$  tokens with random *class* tokens while keeping the rest as  $[M]$  tokens. In both settings, the models either failed to generate meaningful images. This outcome reinforces the student model’s preference for a

![Figure 11: A 3x12 grid of images showing the interpolation between encoded image tokens and random image tokens. The first column shows the original images (a white dog, a golden retriever, and a Siamese cat). The next 10 columns show the interpolated images, which become increasingly noisy and abstract as they move towards the random image tokens in the final column. The rows correspond to class labels 279, 284, and 207, respectively.](793eb94053441c45bcf1e1fad773a7eb_img.jpg)

Figure 11: A 3x12 grid of images showing the interpolation between encoded image tokens and random image tokens. The first column shows the original images (a white dog, a golden retriever, and a Siamese cat). The next 10 columns show the interpolated images, which become increasingly noisy and abstract as they move towards the random image tokens in the final column. The rows correspond to class labels 279, 284, and 207, respectively.

Figure 11. Interpolation between the encoded image token (left most column) and random image token (right most column) in the initial sequence. The class labels are 279, 284 and 207, respectively.

‘familiar’ input, a hybrid of visual and [M] tokens, during distillation. Furthermore, we experimented with sampling random sequences encoded from Gaussian noise of the same size as an image. As shown in Tab. 6, the training of these control experiments leads to divergence metric values.

8. **Additional metrics (FID and CLIP score) on Meissonic.** We evaluated the FID, Fréchet Dino-v2 [70] Distance (FDD) and CLIP Score [33, 45] on the MsCoCo 30k validation dataset [52], as shown in Table 8. Two key observations emerge from the results: (1) as the number of generation steps decreases, the teacher model’s FID deteriorates rapidly (e.g., 8-step FID is 96.75 compared to 64-step FID of 48.27); and (2) our one-step student achieves superior FID/FDD while maintaining a CLIP Score comparable to the teacher. This suggests that (i) our generator performs competitively with the teacher and (ii) FID/FDD may not always be a reliable fidelity metric, particularly for the Meissonic teacher model. We hypothesize that the biased style of the teacher model causes discrepancies compared to realistic images from the MS COCO dataset. Consequently, the distilled student model records better performance relative to the teacher.
9. **Additional HPSV2 results on Meissonic.** In Tab. 7, we present a comparison of our teacher model, Meissonic [4], across different classifier-free guidance settings. Our results indicate that Meissonic operates optimally at CFG= 9 for the HPSV2 metric.
10. We experimented with the Two Time-scale Update Rule [118] to update the fake score multiple times per iteration. However, this did not improve model performance or further stabilize the training loss.
11. Inspired by [35], we added soft targets for training the auxiliary model using the following loss function:

$$\mathcal{L}_{\text{MDM}} = \mathbb{E}_{x_{\text{init}}, t} \left[ \gamma(t) \left( \mathbb{E}_{q_t|0} \left[ (1 - \alpha) (-\log p_{0|t}(x_\theta(x_{\text{init}}) | \tilde{x}_t, \phi)) + \alpha D_{\text{KL}}(p_\psi(x_0 | \tilde{x}_t) || p_\theta(x_0 | x_{\text{init}})) \right] \right) \right] \quad (19)$$

where  $\alpha$  is a hyperparameter controlling the interpolation between the hard target cross-entropy loss and the soft target KL loss.

12. The default CFG for MaskGit during distillation is 2. We also explored adaptive CFG, where the guidance scale varies with  $r_t$ , similar to the linear CFG used in MDM parallel sampling. However, this approach did not yield improvements.
13. Inspired by Proximal Policy Optimization (PPO) [94], we introduced an entropy bonus term  $-p_\theta(x_0 | x_{\text{init}}) \log p_\theta(x_0 | x_{\text{init}})$  to encourage generation diversity. However, this did not show practical benefits.
14. While  $\beta \in [-0.3, 1]$  works well for the ImageNet teacher, we found that in the Meissonic experiment, the distillation process diverges when  $\beta < -0.1$  or  $\beta > 0.2$ .

### E. Failure Cases

While the distilled one-step generator performs comparably to the teacher model in most cases, certain classes exhibit slight color shifts and mode collapse, as shown in Fig. 13. This discrepancy may explain the observed gap in evaluation metrics between the teacher and the one-step generator.

### F. Mode-seeking vs. Mode-covering

In Fig. 16, we show with a Gaussian example the visualization of the mode-seeking vs. mode-covering behaviors of the generalized Jeffrey divergence with different  $\beta$ . When  $\beta$  is small, the divergence approaches the FKL, which is known for its mode-covering tendency, assigning high importance to matching all modes of the target distribution. As  $\beta$  increases, the behavior transitions toward a balanced form, and for large  $\beta$ , the divergence exhibits a mode-seeking tendency, akin to the RKL, which focuses on fitting high-density regions while ignoring low-probability modes.

Table 7. Complete HPS v2.0 benchmark. Scores are collected from <https://github.com/tgxs002/HPSv2>. We highlight the best.

| HPS v2.0                   |     |              |              |              |              |              |
|----------------------------|-----|--------------|--------------|--------------|--------------|--------------|
| Model                      | NFE | Animation    | Concept-art  | Painting     | Photo        | Averaged     |
| Latent Diffusion [83]      | 25  | 25.73        | 25.15        | 25.25        | 26.97        | 25.78        |
| DALL-E 2 [79]              | -   | 27.34        | 26.54        | 26.68        | 27.24        | 26.95        |
| Stable Diffusion v1.4 [83] | 50  | 27.26        | 26.61        | 26.66        | 27.27        | 26.95        |
| Stable Diffusion v2.0 [83] | 50  | 27.48        | 26.89        | 26.86        | 27.46        | 27.17        |
| DeepFloyd-XL [19]          | 25  | 27.64        | 26.83        | 26.86        | 27.75        | 27.27        |
| SDXL Base 1.0 [76]         | 50  | 28.88        | 27.88        | 27.92        | 28.31        | 28.25        |
| SDXL Refiner 1.0 [76]      | 50  | 28.93        | 27.89        | 27.90        | 28.38        | 28.27        |
| InstaFlow [55]             | 1   | 25.98        | 25.79        | 25.93        | 26.32        | 26.01        |
| SD Turbo [90]              | 1   | 27.98        | 27.59        | 27.16        | 27.19        | 27.48        |
| SwiftBrush v2 [18]         | 1   | 27.25        | 27.62        | 26.86        | 26.77        | 27.15        |
| Meissonic (cfg=9) [4]      | 48  | <b>29.57</b> | <b>28.58</b> | <b>28.72</b> | <b>28.45</b> | <b>28.83</b> |
|                            | 32  | 29.18        | 28.32        | 28.28        | 27.96        | 28.44        |
|                            | 16  | 28.61        | 27.82        | 27.84        | 27.32        | 27.90        |
|                            | 8   | 25.62        | 26.49        | 26.67        | 27.07        | 26.46        |
|                            | 4   | 25.01        | 24.95        | 24.87        | 23.80        | 24.66        |
|                            | 2   | 23.06        | 23.28        | 23.22        | 22.38        | 22.98        |
| Meissonic (cfg=4) [4]      | 48  | 28.52        | 27.44        | 27.54        | 27.17        | 27.67        |
|                            | 32  | 28.59        | 27.54        | 27.60        | 27.22        | 27.74        |
|                            | 16  | 28.49        | 27.52        | 27.65        | 27.20        | 27.71        |
|                            | 8   | 27.99        | 27.24        | 27.31        | 26.54        | 27.27        |
|                            | 4   | 26.33        | 26.03        | 26.01        | 24.79        | 25.79        |
|                            | 2   | 23.61        | 23.87        | 23.72        | 22.50        | 23.43        |
| Di[M]O                     | 1   | 28.64        | 27.91        | 27.99        | 27.92        | 28.11        |

Table 8. Comparison of FID, FDD and CLIP-Score for Meissonic [4] across varying generation steps and our one-step generator. The results are evaluated on MSCOCO-val 30k dataset.

| Steps      | 64    | 32    | 16    | 8     | 1 ( <i>ours</i> ) |
|------------|-------|-------|-------|-------|-------------------|
| FID        | 48.27 | 50.13 | 63.29 | 96.75 | 38.45             |
| FDD        | 620.9 | 625.6 | 709.6 | 980.8 | 548.6             |
| CLIP-Score | 0.321 | 0.318 | 0.307 | 0.280 | 0.322             |

![Figure 12: A 4x8 grid of images showing burger generation results. The rows are labeled 'Teacher default', 'Teacher top-1', 'Ours default', and 'Ours top-1'. The columns show different generated burgers. The 'Teacher' rows show more varied and detailed results, while the 'Ours' rows show more consistent and detailed results, particularly in the 'top-1' sampling mode.](5f748e9d8e77de7b6a98e6039de7d0fd_img.jpg)

Figure 12: A 4x8 grid of images showing burger generation results. The rows are labeled 'Teacher default', 'Teacher top-1', 'Ours default', and 'Ours top-1'. The columns show different generated burgers. The 'Teacher' rows show more varied and detailed results, while the 'Ours' rows show more consistent and detailed results, particularly in the 'top-1' sampling mode.

Figure 12. The top two rows display results from the teacher model using 16-step generation with default setup, with the first row generated without top-k filtering and the other with top-1 sampling. The bottom two rows showcase results from our distilled one-step generator. The class label is 933.

![Figure 13: A 4x8 grid of images showing distribution shift in the student model. The top two rows (rows 1 and 2) show results from the teacher model with 16 steps, featuring various oranges and orange slices. The bottom two rows (rows 3 and 4) show results from the one-step generator, featuring various daisies and other flowers. The images illustrate the model's ability to generate diverse and high-quality samples for different classes.](b91e305493a4852dca57a801f54085c8_img.jpg)

Figure 13: A 4x8 grid of images showing distribution shift in the student model. The top two rows (rows 1 and 2) show results from the teacher model with 16 steps, featuring various oranges and orange slices. The bottom two rows (rows 3 and 4) show results from the one-step generator, featuring various daisies and other flowers. The images illustrate the model's ability to generate diverse and high-quality samples for different classes.

Figure 13. Distribution shift in the student. For both classes, the top two rows are results from the teacher with 16 steps and the lower two rows are the results from our one-step generator. The class labels are 950 and 985, respectively.

### G. More Qualitative Results

In Fig. 17, we present randomly sampled ImageNet images generated in a single step by our distilled models with MaskGit teacher. Figure 18 compares our one-step generator with the Meisssonic teacher model using different sampling steps. Notably, our one-step generation achieves superior visual quality compared to the teacher model’s 16-step generation. Finally, in Fig. 19, we provide additional text-to-image one-step generation results from our distilled model with the Meisssonic teacher.

![Figure 15(a): Line graph showing FID vs CFG for MaskGit teacher performance. The x-axis is CFG (2 to 10) and the y-axis is FID (12 to 17). Four series are shown: MaskGit 4 steps (blue), 8 steps (orange), 16 steps (green), and 32 steps (red). FID generally decreases as CFG increases, with more steps showing lower FID at lower CFG values.](9ce50bc10864dc86e1cdee4be08f1897_img.jpg)

| CFG | MaskGit 4 steps | MaskGit 8 steps | MaskGit 16 steps | MaskGit 32 steps |
|-----|-----------------|-----------------|------------------|------------------|
| 2   | 17.0            | 12.8            | 12.8             | 13.0             |
| 3   | 16.2            | 12.2            | 11.8             | 12.5             |
| 4   | 16.0            | 12.0            | 12.0             | 12.8             |
| 5   | 15.8            | -               | -                | -                |
| 6   | 15.8            | -               | -                | -                |
| 7   | 16.0            | -               | -                | -                |
| 8   | 16.0            | -               | -                | -                |
| 9   | 16.2            | -               | -                | -                |
| 10  | 16.2            | -               | -                | -                |

Figure 15(a): Line graph showing FID vs CFG for MaskGit teacher performance. The x-axis is CFG (2 to 10) and the y-axis is FID (12 to 17). Four series are shown: MaskGit 4 steps (blue), 8 steps (orange), 16 steps (green), and 32 steps (red). FID generally decreases as CFG increases, with more steps showing lower FID at lower CFG values.

(a) MaskGit teacher performance under difference inference steps with different CFG. Metric calculated with 5k generated samples.

![Figure 15(b): Line graph showing FID vs Inception Score for a one-step generator distilled from MaskGit teacher. The x-axis is Inception Score (190 to 225) and the y-axis is FID (7.0 to 8.0). A single blue series shows FID decreasing as Inception Score increases, reaching a minimum around Inception Score 215.](485c57a6add7e0bd7898009db1179ee6_img.jpg)

| Inception Score | FID  |
|-----------------|------|
| 190             | 7.42 |
| 195             | 7.25 |
| 200             | 7.10 |
| 205             | 7.02 |
| 210             | 6.98 |
| 215             | 7.02 |
| 220             | 7.15 |
| 225             | 7.35 |
| 230             | 7.58 |
| 235             | 7.78 |

Figure 15(b): Line graph showing FID vs Inception Score for a one-step generator distilled from MaskGit teacher. The x-axis is Inception Score (190 to 225) and the y-axis is FID (7.0 to 8.0). A single blue series shows FID decreasing as Inception Score increases, reaching a minimum around Inception Score 215.

(b) Performance of one-step generator distilled from MaskGit teacher under different temperatures. From left to right the temperatures are [11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.5, 4.0, 3.0, 2.0]. Metric calculated with 50k generated samples.

![Figure 15: A visual representation of the consistency assumption. It shows an initial state x_init entering a 'One-step Generator theta'. The generator produces two paths: one leading to an intermediate state x_tilde_t and another leading to a final state z_theta(x_init). The intermediate state x_tilde_t is then processed by the generator to produce a final state z_theta(x_tilde_t). The diagram illustrates that ideally, z_theta(x_tilde_t) should be identical to z_theta(x_init).](4b87467ad9642943235f48f7d4b59449_img.jpg)

Figure 15: A visual representation of the consistency assumption. It shows an initial state x\_init entering a 'One-step Generator theta'. The generator produces two paths: one leading to an intermediate state x\_tilde\_t and another leading to a final state z\_theta(x\_init). The intermediate state x\_tilde\_t is then processed by the generator to produce a final state z\_theta(x\_tilde\_t). The diagram illustrates that ideally, z\_theta(x\_tilde\_t) should be identical to z\_theta(x\_init).

Figure 15. A visual representation of the consistency assumption. Ideally, the model’s prediction based on the correct intermediate state,  $\tilde{x}_t$ , should be identical to its prediction derived from the initial sequence,  $x_{init}$ .

![Figure 16: A line graph titled 'General Jefferey Divergence Comparison'. The x-axis is 'x' (-10 to 20) and the y-axis is 'Density' (0.0 to 0.4). It shows a target distribution p (solid black line) and several generalized Jefferey divergence distributions for different beta values (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0). As beta increases, the distribution becomes more peaked and shifts towards higher x values.](a1a474be12b8992842992294b1d18592_img.jpg)

| $\beta$      | Peak Density (approx.) | Peak Position (approx.) |
|--------------|------------------------|-------------------------|
| target $p$   | 0.28                   | 10                      |
| $\beta: 0$   | 0.05                   | 10                      |
| $\beta: 0.1$ | 0.08                   | 10                      |
| $\beta: 0.3$ | 0.12                   | 10                      |
| $\beta: 0.5$ | 0.18                   | 10                      |
| $\beta: 0.7$ | 0.25                   | 10                      |
| $\beta: 0.9$ | 0.35                   | 10                      |
| $\beta: 1.0$ | 0.40                   | 10                      |

Figure 16: A line graph titled 'General Jefferey Divergence Comparison'. The x-axis is 'x' (-10 to 20) and the y-axis is 'Density' (0.0 to 0.4). It shows a target distribution p (solid black line) and several generalized Jefferey divergence distributions for different beta values (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0). As beta increases, the distribution becomes more peaked and shifts towards higher x values.

Figure 16. Toy example to visualize the mode-seeking VS mode-covering behavior of different  $\beta$  values in generalized Jefferey divergence. We grid search the mean and std to minimize the generalized Jefferey divergence.

![A large grid of 200 small images arranged in 20 rows and 10 columns, showing a wide variety of objects and animals generated by a class-conditional model on ImageNet.](ae02603e9e4b46477222bf72c1c7c7f6_img.jpg)

A large grid of 200 small images arranged in 20 rows and 10 columns. Each small image represents a sample generated by a class-conditional model on the ImageNet dataset. The samples are highly diverse, covering a wide range of categories including animals (e.g., dogs, cats, birds, insects, mammals), objects (e.g., tools, vehicles, food, household items), and scenes (e.g., landscapes, buildings, people). The images are of varying quality and composition, demonstrating the model's ability to generate a broad spectrum of visual concepts.

A large grid of 200 small images arranged in 20 rows and 10 columns, showing a wide variety of objects and animals generated by a class-conditional model on ImageNet.

Figure 17. One-step samples from our class-conditional model on ImageNet.

![Figure 18: A 4x8 grid of images comparing the results of the Meisssonc model at different steps (64, 16, 4, and 1 step) for eight different prompts. The prompts are: a raccoon in a spacesuit, a woman with a flower crown, a bear in a suit, a monkey in a spacesuit, a person in a car, a lake at night, a man in a suit, and a raccoon in a top hat. The images show that the quality of the generated images drops significantly as the number of steps decreases from 64 to 1.](7b517f15a2d307945bd78e30c13a75bd_img.jpg)

Figure 18 shows a comparison of the Meisssonc model's results at different steps (64, 16, 4, and 1 step) for eight different prompts. The prompts are: a raccoon in a spacesuit, a woman with a flower crown, a bear in a suit, a monkey in a spacesuit, a person in a car, a lake at night, a man in a suit, and a raccoon in a top hat. The images show that the quality of the generated images drops significantly as the number of steps decreases from 64 to 1.

Figure 18: A 4x8 grid of images comparing the results of the Meisssonc model at different steps (64, 16, 4, and 1 step) for eight different prompts. The prompts are: a raccoon in a spacesuit, a woman with a flower crown, a bear in a suit, a monkey in a spacesuit, a person in a car, a lake at night, a man in a suit, and a raccoon in a top hat. The images show that the quality of the generated images drops significantly as the number of steps decreases from 64 to 1.

Figure 18. Comparison with the teacher: Meisssonc [4] on different steps, we see clearly that the teacher model’s results drop very quickly (e.g., around 4 steps).

![Figure 19: A 4x8 grid of 32 diverse images generated by a one-step generator distilled from Meisssonc. The images include a variety of subjects such as a cat in a top hat, a woman with a flower crown, a bear in a suit, a monkey in a spacesuit, a person in a car, a lake at night, a man in a suit, and a raccoon in a top hat. The images show a high level of detail and creativity, demonstrating the quality of the one-step generator.](0ccbf2e1f1d9d0aae8865d824a1fc322_img.jpg)

Figure 19 shows a grid of 32 diverse images generated by a one-step generator distilled from Meisssonc. The images include a variety of subjects such as a cat in a top hat, a woman with a flower crown, a bear in a suit, a monkey in a spacesuit, a person in a car, a lake at night, a man in a suit, and a raccoon in a top hat. The images show a high level of detail and creativity, demonstrating the quality of the one-step generator.

Figure 19: A 4x8 grid of 32 diverse images generated by a one-step generator distilled from Meisssonc. The images include a variety of subjects such as a cat in a top hat, a woman with a flower crown, a bear in a suit, a monkey in a spacesuit, a person in a car, a lake at night, a man in a suit, and a raccoon in a top hat. The images show a high level of detail and creativity, demonstrating the quality of the one-step generator.

Figure 19. Qualitative results of our one-step generator distilled from Meisssonc [4].

### H. Misc.

**Prompts** Below is a collection of creative prompts we used to generate images in Figs. 1, 6, 18 and 19:

- A plushy tired owl sits on a pile of antique books in a humorous illustration.
- A photograph of a woman from Steven Universe with gigantic pink ringlets and a white dress.
- A white bichon frise puppy dog riding a black motorcycle in Hollywood at sundown with palm trees in the background.
- A photorealistic image of a giant floating glass sphere in a rocky landscape surrounded by a gentle mist.
- A cosmonaut otter poses for a portrait painted in intricate detail by Rembrandt.
- A beaver in formal attire stands next to a stack of books in a library.
- An egirl with pink hair and extensive makeup.
- Portrait of a monkey wearing a spacesuit and an astronaut helmet.
- Closeup of a seinen manga film still showing the interior of a shinkansen train with a leather seat and a window view, with a hyperrealistic film still from a Nepali movie projecting in the background.
- Swedish lake at night with heavy snowfall depicted in hyper-realistic and detailed art.
- Pencil sketch of Danny DeVito by Milt Kahl.
- A realistic anime painting of a cosmic woman wearing clothes made of universes with glowing red eyes.
- Photo of Ty Lee from Avatar.
- A green field with flowers and pink and yellow clouds under a bright sun at sunset, illustrated by Peter Chan in a colorful Day of the Tentacle style on Artstation.
- A raccoon in formal attire, carrying a bag and cane, depicted in a Rembrandt-style oil painting.
- A girl in school uniform standing in the city.
- Serene, anime-style landscape with vibrant flowers and trees, picturesque clouds, and no signs of human activity.
- A kangaroo wearing an orange hoodie and blue sunglasses holding a sign in front of the Sydney Opera House.
- A pikachu in a forest illustration.
- An oil painting close-up portrait of a young black woman wearing a crown of wildflowers, surrounded by hazy golden light.
- A capybara wearing sunglasses.
- A frog wearing an anime-inspired onesie.
- A blue-haired girl with soft features stares directly at the camera in an extreme close-up Instagram picture.
- Digital art of Prince of Roses.
- A landscape featuring a Kyoto Animation-style building.
- A path winding through a forest depicted in digital art.
- A close-up portrait of a beautiful girl with an autumn leaves headdress and melting wax.
- A neon-soaked cyberpunk alleyway with rain-drenched streets and futuristic holograms, gritty yet vibrant, hyper-realistic, ultra-detailed, cinematic scene.
- A serene mountain landscape at sunrise, mist rolling over rugged peaks, ultra-detailed, photorealistic, soft lighting, high-resolution, digital art.
- A hyper-detailed closeup of a dew-covered insect on a vibrant leaf, extreme macro photography style, ultra-realistic, high-resolution, intricate textures.
- An anime-style magical girl in a dynamic pose, vibrant colors, ultra-detailed costume and background, energetic, high-resolution, cinematic lighting.
- An enchanted autumn forest with falling leaves and warm, glowing light, ultra-detailed, photorealistic, rich textures, digital art, serene mood.
- An elegant Renaissance portrait of a noble figure, detailed textures, soft natural lighting, ultra-detailed, classical, high-resolution, oil painting style.
- A cybernetic humanoid robot portrait with metallic textures and neon accents, ultra-detailed, photorealistic, cinematic, futuristic digital art.
- A vibrant street art mural on an urban wall, ultra-detailed, energetic, bold colors, high-resolution, digital painting, modern art style.
- A dark fantasy warrior in intricately detailed armor standing in a stormy battlefield, ultra-detailed, hyper-realistic, cinematic, dynamic action scene.
- An intense lightning storm over a vast desert landscape, ultra-detailed, dramatic, high-resolution, cinematic, digital art, atmospheric.
- A detailed nature macro shot of a vibrant flower with dewdrops, ultra-detailed, photorealistic, high-resolution, digital painting, delicate textures.
- An ultra-realistic snowy mountain village under a starry sky, ultra-detailed, atmospheric, cinematic, high-resolution, digital winter wonderland.
- A humorous portrait of a cat dressed as a Victorian aristocrat, in vintage photorealism.
- A photorealistic shot of a bouquet of wildflowers in a clear glass vase on a sunlit windowsill.
- A mecha jet fighter engages in an air battle with an explosion as a backdrop, set against a dark, starry sky in a highly-detailed art piece by Stephan Martinieri.

- A young black woman stands in front of a ringed planet in space.
- Digital art of a cherry tree overlooking a valley with a waterfall at sunset.
- An astronaut in white futuristic cybernetic armor running on the surface of the moon, featured in an artwork illustration on Artstation.
- The image is a headshot of a happy girl with white hair in a school uniform, illustrated by Ilya Kuvshinov.
- A minimalistic heart drawing created using Adobe Illustrator.
- The image is a digital art headshot of an owlfolk character with high detail and dramatic lighting.
- Close up of an eye with the Earth inside the pupil, inspired by Wes Anderson's art.
- Landrover drives through a rain-soaked forest in a highly-detailed digital artwork by Greg Rutkowski and Artgerm.
- A snowy lake in Sweden captured in a vibrant, cinematic style with intense detail and raytracing technology showcased on Artstation.
- A heron silhouetted against a beautiful sunrise, created by Greg Rutkowski.
- A surreal portrait of a woman with a giant carnation face in a flower field at sunset with colorful clouds and a large sky, created by artist Simon Stålenhag.