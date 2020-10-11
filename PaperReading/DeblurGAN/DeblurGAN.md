

# DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks

*Orest Kupyn1,3, Volodymyr Budzan1,3, Mykola Mykhailych1 , Dmytro Mishkin2 , Jiˇri Matas2 1 Ukrainian Catholic University, Lviv, Ukraine {kupyn, budzan, mykhailych}@ucu.edu.ua 2 Visual Recognition Group, Center for Machine Perception, FEE, CTU in Prague {mishkdmy, matas}@cmp.felk.cvut.cz 3 ELEKS Ltd.*

## Abstract

We present DeblurGAN, an end-to-end learned method for motion deblurring. The learning is based on a conditional GAN and the content loss . DeblurGAN achieves state-of-the art performance both in the structural similarity measure and visual appearance. The quality of the deblurring model is also evaluated in a novel way on a real-world problem – object detection on (de-)blurred images. The method is 5 times faster than the closest competitor – DeepDeblur [25]. We also introduce a novel method for generating synthetic motion blurred images from sharp ones, allowing realistic dataset augmentation. The model, code and the dataset are available at https://github.com/KupynOrest/DeblurGAN

我们提出了一种端到端学习的运动去模糊方法DeblurGAN。学习是基于条件GAN和内容丢失。DeblurGAN在结构相似性度量和视觉外观方面都达到了最先进的性能。该模型的质量也通过一种新的方法来评估，即在（去）模糊图像上的目标检测。这种方法比最接近的竞争对手DeepDeblur[25]快5倍。我们还介绍了一种新的方法来产生合成的运动模糊图像从尖锐的，允许现实的数据集扩充。模型、代码和数据集位于https://github.com/KupynOrest/DeblurGAN

## 1. Introduction

This work is on blind motion deblurring of a single photograph. Significant progress has been recently achieved in related areas of image super-resolution [20] and inpainting [45] by applying generative adversarial networks (GANs) [10]. GANs are known for the ability to preserve texture details in images, create solutions that are close to the real image manifold and look perceptually convincing. Inspired by recent work on image super-resolution [20] and image-to-image translation by generative adversarial networks [16], we treat deblurring as a special case of such image-to-image translation. We present DeblurGAN – an approach based on conditional generative adversarial networks [24] and a multi-component loss function. Unlike previous work we use Wasserstein GAN [2] with the gradient penalty [11] and perceptual loss [17]. This encourages solutions which are perceptually hard to distinguish from real sharp images and allows to restore finer texture details than if using traditional MSE or MAE as an optimization target.

这项工作是关于一张照片的盲动去模糊。最近，在图像超分辨率[20]和修复[45]的相关领域，通过应用生成性对抗网络（GANs）[10]，取得了重大进展。GANs以能够在图像中保留纹理细节、创建接近真实图像流形的解决方案以及看起来令人信服而著称。受最近关于图像超分辨率[20]和生成对抗性网络的图像到图像转换的研究[16]的启发，我们将去模糊作为这种图像到图像转换的一个特例。我们提出了一种基于条件生成对抗网络[24]和多分量损失函数的DeblurGAN方法。与之前的研究不同，我们使用Wasserstein GAN[2]和梯度惩罚[11]和知觉损失[17]。这就鼓励了那些在视觉上很难与真实的清晰图像区分的解决方案，并允许恢复比使用传统的MSE或MAE作为优化目标的更精细的纹理细节。

![1](pic\1.JPG)

![1](pic\2.JPG)

We make three contributions. First, we propose a loss and architecture which obtain state-of-the art results in motion deblurring, while being 5x faster than the fastest competitor. Second, we present a method based on random trajectories for generating a dataset for motion deblurring training in an automated fashion from the set of sharp image. We show that combining it with an existing dataset for motion deblurring learning improves results compared to training on real-world images only. Finally, we present a novel dataset and method for evaluation of deblurring algorithms based on how they improve object detection results.

我们有三个贡献。首先，我们提出了一种损失和架构，它在运动去模糊方面获得了最先进的结果，同时比最快的竞争对手快5倍。其次，我们提出了一种基于随机轨迹的方法来自动生成一个用于运动去模糊训练的数据集。我们表明，与仅在真实图像上训练相比，将其与现有的数据集相结合进行运动去模糊学习可以提高结果。最后，我们提出了一个新的数据集和方法来评估去模糊算法如何改善目标检测结果。

## 2. Related work

### 2.1. Image Deblurring

The common formulation of non-uniform blur model is the following:非均匀模糊模型的常用公式如下： 

​															IB = k(M) ∗ IS + N, (1) 

where IB is a blurred image, k(M) are unknown blur kernels determined by motion field M. IS is the sharp latent image, ∗ denotes the convolution, N is an additive noise. The family of deblurring problems is divided into two types: blind and non-blind deblurring. Early work [37] mostly focused on non-blind deblurring, making an assumption that the blur kernels k(M) are known. Most rely on the classical Lucy-Richardson algorithm, Wiener or Tikhonov filter to perform the deconvolution operation and obtain IS estimate. Commonly the blur function is unknown, and blind deblurring algorithms estimate both latent sharp image IS and blur kernels k(M). Finding a blur function for each pixel is an ill-posed problem, and most of the existing algorithms rely on heuristics, image statistics and assumptions on the sources of the blur. Those family of methods addresses the blur caused by camera shake by considering blur to be uniform across the image. Firstly, the camera motion is estimated in terms of the induced blur kernel, and then the effect is reversed by performing a deconvolution operation. Starting with the success of Fergus et al. [8], many methods [44][42][28][3] has been developed over the last ten years. Some of the methods are based on an iterative approach [8] [44], which improve the estimate of the motion kernel and sharp image on each iteration by using parametric prior models. However, the running time, as well as the stopping criterion, is a significant problem for those kinds of algorithms. Others use assumptions of a local linearity of a blur function and simple heuristics to quickly estimate the unknown kernel. These methods are fast but work well on a small subset of images.

其中IB是模糊图像，k（M）是由运动场M确定的未知模糊核。is是锐利的潜像，*表示卷积，N是加性噪声。去模糊问题族分为两类：盲解模糊和非盲解模糊。早期的工作主要集中在非盲去模糊，假设模糊核k（M）已知。大多依靠经典的Lucy-Richardson算法、Wiener或Tikhonov滤波器进行反褶积运算，得到IS估计。模糊函数通常是未知的，盲去模糊算法同时估计潜在的锐化图像is和模糊核k（M）。寻找每个像素的模糊函数是一个不适定的问题，现有的算法大多依赖于启发式、图像统计和对模糊源的假设。这些方法通过考虑模糊在图像中是均匀的来解决由相机抖动引起的模糊。首先，根据诱导模糊核估计摄像机运动，然后通过反褶积来逆转效果。从费格斯等人的成功开始。[8] 在过去的十年里，人们发展了许多方法[44][42][28][3]。其中一些方法是基于迭代方法[8][44]，通过使用参数先验模型来改进每次迭代的运动核和锐化图像的估计。然而，这些算法的运行时间和停止准则是一个重要的问题。另一些则使用模糊函数的局部线性假设和简单的启发式来快速估计未知核。这些方法速度快，但对一小部分图像效果良好。

Recently, Whyte et al. [40] developed a novel algorithm for non-uniform blind deblurring based on a parametrized geometric model of the blurring process in terms of the rotational velocity of the camera during exposure. Similarly Gupta et al. [12] made an assumption that the blur is caused only by 3D camera movement. With the success of deep learning, over the last few years, there appeared some approaches based on convolutional neural networks (CNNs). Sun et al. [36] use CNN to estimate blur kernel, Chakrabarti [6] predicts complex Fourier coefficients of motion kernel to perform non-blind deblurring in Fourier space whereas Gong [9] use fully convolutional network to move for motion flow estimation. All of these approaches use CNN to estimate the unknown blur function. Recently, a kernel-free end-to-end approaches by Noorozi [27] and Nah [25] that uses multi-scale CNN to directly deblur the image. Ramakrishnan et al. [29] use the combination of pix2pix framework [16] and densely connected convolutional networks [15] to perform blind kernel-free image deblurring. Such methods are able to deal with different sources of the blur

最近，Whyte等人。[40]根据曝光过程中相机的旋转速度，基于模糊过程的参数化几何模型，开发了一种新的非均匀盲去模糊算法。同样，Gupta等人。[12] 假设模糊是由3D摄像机移动引起的。随着深度学习的成功，近年来出现了一些基于卷积神经网络的方法。Sun等人。[36]使用CNN估计模糊核，Chakrabarti[6]预测运动核的复Fourier系数以在Fourier空间中进行非盲去模糊，而Gong[9]则使用完全卷积网络进行运动流估计。所有这些方法都使用CNN来估计未知的模糊函数。最近，noorzi[27]和Nah[25]提出了一种无核端到端方法，它使用多尺度CNN直接去模糊图像。Ramakrishnan等人。[29]利用pix2pix框架[16]和密集连接卷积网络[15]相结合的方法进行盲无核图像去模糊。这种方法能够处理不同的模糊源

![1](pic\3.JPG)

### 2.2. Generative adversarial networks

The idea of generative adversarial networks, introduced by Goodfellow et al. [10], is to define a game between two competing networks: the discriminator and the generator. The generator receives noise as an input and generates a sample. A discriminator receives a real and generated sample and is trying to distinguish between them. The goal of the generator is to fool the discriminator by generating perceptually convincing samples that can not be distinguished from the real one. The game between the generator G and discriminator D is the minimax objective:

生成性对抗网络的概念，由Goodfellow等人提出。[10] ，定义了两个竞争网络之间的博弈：鉴别器和生成器。发生器接收噪声作为输入并生成样本。鉴别器接收真实的和生成的样本，并试图区分它们。生成器的目标是通过生成具有感知说服力的样本来愚弄鉴别器，这些样本无法与真实样本区分开。发生器G和鉴别器D之间的博弈是极小极大目标：

​						min G max D E xvPr [log(D(x))] + E x˜vPg [log(1 − D(˜x))]

where Pr is the data distribution and Pg is the model distribution, defined by x˜ = G(z), z v P(z), the input z is a sample from a simple noise distribution. GANs are known for its ability to generate samples of good perceptual quality, however, training of vanilla version suffer from many problems such as mode collapse, vanishing gradients etc, as described in [33]. Minimizing the value function in GAN is equal to minimizing the Jensen-Shannon divergence between the data and model distributions on x. Arjovsky et al. [2] discuss the difficulties in GAN training caused by JS divergence approximation and propose to use the Earth-Mover (also called Wasserstein-1) distance W(q, p). The value function for WGAN is constructed using Kantorovich-Rubinstein duality [39]:

其中，Pr是数据分布，Pg是模型分布，由x∮=G（z），z v P（z）定义，输入z是简单噪声分布的样本。GANs以其产生高感知质量样本的能力而闻名，然而香草版的训练却存在模式崩溃、梯度消失等问题，如[33]所述。最小化GAN中的值函数等于最小化x.Arjovsky等人的数据和模型分布之间的Jensen-Shannon散度。[2] 讨论了JS散度近似在GAN训练中的困难，并提出了使用地球移动器（也称Wasserstein-1）距离W（q，p）的方法。WGAN的值函数是使用Kantorovich-Rubinstein对偶[39]构造的：

​						min G max D∈D E xvPr [D(x)] − E x˜vPg [D(˜x)] (3)

where D is the set of 1−Lipschitz functions and Pg is once again the model distribution The idea here is that critic value approximates K · W(Pr, Pθ), where K is a Lipschitz constant and W(Pr, Pθ) is a Wasserstein distance. In this setting, a discriminator network is called critic and it approximates the distance between the samples. To enforce Lipschitz constraint in WGAN Arjovsky et al. add weight clipping to [−c, c]. Gulrajani et al. [11] propose to add a gradient penalty term instead:

其中D是1−Lipschitz函数的集合，Pg再次是模型分布，这里的思想是临界值近似K·W（Pr，Pθ），其中K是Lipschitz常数，W（Pr，Pθ）是Wasserstein距离。在这种情况下，鉴别器网络被称为critic，它近似于样本之间的距离。在希约夫斯基等人的约束下。将权重剪裁添加到[-c，c]。Gulrajani等人。[11] 建议增加一个梯度惩罚项：

​						λ E x˜vPx˜ [(k∇x˜D(˜x)k2 − 1)2 ] (4)

to the value function as an alternative way to enforce the Lipschitz constraint. This approach is robust to the choice of generator architecture and requires almost no hyperparameter tuning. This is crucial for image deblurring as it allows to use novel lightweight neural network architectures in contrast to standard Deep ResNet architectures, previously used for image deblurring [25].

作为强制Lipschitz约束的另一种方法。这种方法对生成器体系结构的选择是健壮的，并且几乎不需要超参数调整。这对于图像去模糊非常关键，因为它允许使用新的轻量级神经网络结构，与之前用于图像去模糊的标准深度ResNet架构相比[25]。

### 2.3. Conditional adversarial networks

Generative Adversarial Networks have been applied to different image-to-image translation problems, such as super resolution [20], style transfer [22], product photo generation [5] and others. Isola et al. [16] provides a detailed overview of those approaches and present conditional GAN architecture also known as pix2pix. Unlike vanilla GAN,cGAN learns a mapping from observed image x and random noise vector z, to y : G : x, z → y. Isola et al. also put a condition on the discriminator and use U-net architecture [31] for generator and Markovian discriminator which allows achieving perceptually superior results on many tasks, including synthesizing photos from label maps, reconstructing objects from edge maps, and colorizing images.

生成对抗网络已经应用于不同的图像到图像的翻译问题，例如超分辨率[20]、风格转换[22]、产品照片生成[5]等。Isola等人。[16] 提供了这些方法的详细概述和目前的条件GAN架构也称为pix2pix。与香草型GAN不同，cGAN从观察到的图像x和随机噪声向量z学习映射到y:G:x，z→y.Isola等。另外，在鉴别器上设置了一个条件，并将U-net体系结构[31]用于generator和Markovian鉴别器，这允许在许多任务上获得感知上优越的结果，包括从标签地图合成照片、从边缘地图重建对象和给图像着色。

## 3. The proposed method

The goal is to recover sharp image IS given only a blurred image IB as an input, so no information about the blur kernel is provided. Debluring is done by the trained CNN G<sub>θG</sub> , to which we refer as the Generator. For each IB it estimates corresponding IS image. In addition, during the training phase, we introduce critic the network D<sub>θD</sub> and train both networks in an adversarial manner.

该算法的目标是只给模糊图像IB作为输入，不提供模糊核的信息，以此恢复锐利图像。去模糊是由经过训练的CNN G<sub>θG</sub>来完成的，我们称之为生成器。对于每个IB，它估计出相应的IS图像。另外，在训练阶段，我们引入批评网络D<sub>θD</sub>，并以对抗的方式训练两个网络。

### 3.1. Loss function

We formulate the loss function as a combination of content and adversarial loss:

我们将损失函数表示为内容损失和对抗损失的组合：

![1](pic\4.JPG)

where the λ equals to 100 in all experiments. Unlike Isola et al. [16] we do not condition the discriminator as we do not need to penalize mismatch between the input and output. Adversarial loss Most of the papers related to conditional GANs, use vanilla GAN objective as the loss [20][25] function. Recently [47] provides an alternative way of using least aquare GAN [23] which is more stable and generates higher quality results. We use WGAN-GP [11] as the critic function, which is shown to be robust to the choice of generator architecture [2]. Our premilinary experiments with different architectures confirmed that findings and we are able to use architecture much lighter than ResNet152 [25], see next subsection. The loss is calculated as the following:

式中，在所有实验中λ等于100。不像Isola等人。[16] 我们没有条件的鉴别器，因为我们不需要惩罚不匹配的输入和输出。对抗性失利的论文大多与条件句有关，采用香草根客观作为损失函数[20][25]。最近[47]提供了一种使用最少水性GAN[23]的替代方法，这种方法更稳定，产生更高质量的结果。我们使用WGAN-GP[11]作为临界函数，它对生成器架构的选择是鲁棒的[2]。我们对不同体系结构的初步实验证实了这一发现，并且我们能够使用比ResNet152[25]轻得多的体系结构，见下一小节。损失计算如下：

![1](pic\5.JPG)

DeblurGAN trained without GAN component converges, but produces smooth and blurry images.

无GAN分量训练的DeblurGAN收敛，但产生平滑模糊的图像。

**Content loss**. Two classical choices for ”content” loss function are L1 or MAE loss, L2 or MSE loss on raw pixels. Using those functions as sole optimization target leads to the blurry artifacts on generated images due to the pixelwise average of possible solutions in the pixel space [20]. Instead, we adopted recently proposed Perceptual loss [17]. Perceptual loss is a simple L2-loss, but based on the difference of the generated and target image CNN feature maps. It is defined as following:

**内容丢失**。“内容”损失函数的两个经典选择是L1或MAE损失，L2或MSE损失原始像素。使用这些函数作为唯一的优化目标会导致生成图像上的模糊伪影，这是由于像素空间中可能解的像素平均值[20]。相反，我们采用了最近提出的知觉丧失[17]。感知损失是一种简单的L2损失，但基于生成图像和目标图像的CNN特征映射的差异。定义如下：

![1](pic\6.JPG)

where φi,j is the feature map obtained by the j-th convolution (after activation) before the i-th maxpooling layer within the VGG19 network, pretrained on ImageNet [7], Wi,j and Hi,j are the dimensions of the feature maps. In our work we use activations from V GG3,3 convolutional layer. The activations of the deeper layers represents the features of a higher abstraction [46][20]. The perceptual loss focuses on restoring general content [16] [20] while ad-versarial loss focuses on restoring texture details. DeblurGAN trained without Perceptual loss or with simple MSE on pixels instead doesn’t converge to meaningful state.

其中φi，j是VGG19网络内第i个最大池层之前通过第j次卷积（激活后）获得的特征图，在ImageNet上进行了预训练[7]，Wi，j和Hi，j是特征图的维数。在我们的工作中，我们使用vgg3，3卷积层的激活。更深层次的激活代表了更高抽象的特征[46][20]。感性损失侧重于恢复一般内容[16][20]，而广告损失侧重于恢复纹理细节。DeblurGAN训练时没有知觉损失，或者在像素上有简单的均方误差，而不是收敛到有意义的状态。

**Additional regularization**. We have also tried to add TV regularization and model trained with it yields worse performance – 27.9 vs. 28.7 w/o PSNR on GoPro dataset.

**额外的正规化**。我们还尝试添加TV正则化，使用它训练的模型会产生更差的性能-在GoPro数据集上，不带PSNR的27.9 vs.28.7。

![1](pic\7.JPG)

![1](pic\8.JPG)

### 3.2. Network architecture

Generator CNN architecture is shown in Figure 3. It is similar to one proposed by Johnson et al. [17] for style transfer task. It contains two strided convolution blocks with stride 1 2 , nine residual blocks [13] (ResBlocks) and two transposed convolution blocks. Each ResBlock consists of a convolution layer, instance normalization layer [38], and ReLU [26] activation. Dropout [35] regularization with a probability of 0.5 is added after the first convolution layer in each ResBlock. In addition, we introduce the global skip connection which we refer to as ResOut. CNN learns a residual correction IR to the blurred image IB, so IS = IB + IR. We find that such formulation makes training faster and resulting model generalizes better. During the training phase, we define a critic network DθD , which is Wasserstein GAN [2] with gradient penalty [11], to which we refer as WGAN-GP. The architecture of critic network is identical to PatchGAN [16, 22]. All the convolutional layers except the last are followed by InstanceNorm layer and LeakyReLU [41] with α = 0.2.

生成器CNN架构如图3所示。这与Johnson等人提出的类似。[17] 用于样式转换任务。它包含两个步长为12的跨步卷积块、9个剩余块[13]（ResBlocks）和两个转置卷积块。每个ResBlock由卷积层、实例规范化层[38]和ReLU[26]激活组成。在每个ResBlock中的第一个卷积层之后添加概率为0.5的Dropout[35]正则化。此外，我们还介绍了我们称之为ResOut的全局跳过连接。CNN学习模糊图像IB的残差校正IR，因此IS=IB+IR。我们发现，这样的公式化使得训练速度更快，得到的模型更易于推广。在训练阶段，我们定义了一个临界网络DθD，它是具有梯度惩罚的Wasserstein GAN[2]，我们称之为WGAN-GP。批评家网络的架构与PatchGAN相同[16,22]。除最后一层外，所有卷积层之后都是瞬变层和漏雨层[41]，α=0.2。

## 4. Motion blur generation

There is no easy method to obtain image pairs of corresponding sharp and blurred images for training.A typical approach to obtain image pairs for training is to use a high frame-rate camera to simulate blur using average of sharp frames from video [27, 25]. It allows to create realistic blurred images but limits the image space only to scenes present in taken videos and makes it complicated to scale the dataset. Sun et al. [36] creates synthetically blurred images by convolving clean natural images with one out of 73 possible linear motion kernels, Xu et al. [43] also use linear motion kernels to create synthetically blurred images.Chakrabarti [6] creates blur kernel by sampling 6 random points and fitting a spline to them. We take a step further and propose a method, which simulates more realistic and complex blur kernels. We follow the idea described by Boracchi and Foi [4] of random trajectories generation. Then the kernels are generated by applying sub-pixel interpolation to the trajectory vector. Each trajectory vector is a complex valued vector, which corresponds to the discrete positions of an object following 2D random motion in a continuous domain. Trajectory generation is done by Markov process, summarized in Algorithm 1. Position of the next point of the trajectory is randomly generated based on the previous point velocity and position, gaussian perturbation, impulse perturbation and deterministic inertial component.

获取相应的锐利和模糊图像的图像对进行训练并不是一种简单的方法，获得用于训练的图像对的一种典型方法是使用高帧速率相机，利用视频中的锐利帧的平均值来模拟模糊[27,25]。它允许创建逼真的模糊图像，但将图像空间仅限于拍摄视频中的场景，并使数据集的缩放变得复杂。Sun等人。[36]通过将干净的自然图像与73个可能的线性运动核中的一个进行卷积，创建合成模糊图像，Xu等人。[43]还可以使用线性运动核来创建合成模糊图像。查克拉巴蒂[6] 通过采样6个随机点并拟合样条线来创建模糊核。我们提出了一种更为复杂和逼真的核方法。我们遵循Boracchi和Foi[4]描述的随机轨迹生成的思想。然后对轨迹向量进行亚像素插值，生成核函数。每一个轨迹向量都是一个复值向量，它对应于物体在连续域中二维随机运动后的离散位置。轨迹生成采用马尔可夫过程，算法1总结。根据前一点的速度和位置、高斯摄动、脉冲摄动和确定性惯性分量，随机生成下一个轨迹点的位置。

![1](pic\9.JPG)

## 5. Training Details

We implemented all of our models using PyTorch[1] deep learning framework. The training was performed on a single Maxwell GTX Titan-X GPU using three different datasets. The first model to which we refer as DeblurGANWILD was trained on a random crops of size 256x256 from 1000 GoPro training dataset images [25] downscaled by a factor of two. The second one DeblurGANSynth was trained on 256x256 patches from MS COCO dataset blurred by method, presented in previous Section. We also trained DeblurGANComb on a combination of synthetically blurred images and images taken in the wild, where the ratio of synthetically generated images to the images taken by a high frame-rate camera is 2:1. As the models are fully convolutional and are trained on image patches they can be applied to images of arbitrary size. For optimization we follow the approach of [2] and perform 5 gradient descent steps on DθD , then one step on GθG , using Adam [18] as a solver. The learning rate is set initially to 10−4 for both generator and critic. After the first 150 epochs we linearly decay the rate to zero over the next 150 epochs. At inference time we follow the idea of [16] and apply both dropout and instance normalization. All the models were trained with a batch size = 1, which showed empirically better results on validation. The training phase took 6 days for training one DeblurGAN network.

我们使用PyTorch[1]深度学习框架实现了所有模型。训练是在一个Maxwell GTX Titan-xgpu上使用三个不同的数据集进行的。我们称之为DeblurGANWILD的第一个模型是在1000个GoPro训练数据集图像[25]中随机选取大小为256x256的作物进行训练的。第二个是DeblurGANSynth训练的256x256个斑块从MS-COCO数据集模糊的方法，在前面一节介绍。我们还对合成模糊图像和野外拍摄的图像进行了DeblurGANComb训练，其中合成图像与高帧频相机拍摄的图像的比率为2:1。由于模型是完全卷积的，并且是在图像块上训练的，所以它们可以应用于任意大小的图像。对于优化，我们遵循[2]的方法，在DθD上执行5个梯度下降步骤，然后在GθG上执行一个步骤，使用Adam[18]作为解算器。对于generator和critic，学习率最初设置为10-4。在第一个150个周期之后，在接下来的150个周期中，我们线性地将速率衰减到零。在推理时，我们遵循[16]的思想，同时应用了dropout和实例规范化。所有的模型都被训练成一个批大小=1，这表明在实验上验证结果更好。训练阶段用了6天的时间训练一个DeblurGAN网络。

![1](pic\10.JPG)

![1](pic\11.JPG)

![1](pic\12.JPG)![1](pic\13.JPG)

## 6. Experimental evaluation

### 6.1. GoPro Dataset

GoPro dataset[25] consists of 2103 pairs of blurred and sharp images in 720p quality, taken from various scenes. We compare the results of our models with state of the art models [36], [25] on standard metrics and also show the running time of each algorithm on a single GPU. Results are in Table1. DeblurGAN shows superior results in terms of structured self-similarity, is close to state-of-the-art in peak signal-to-noise-ratio and provides better looking results by visual inspection. In contrast to other neural models, our network does not use L2 distance in pixel space so it is not directly optimized for PSNR metric. It can handle blur caused by camera shake and object movement, does not suffer from usual artifacts in kernel estimation methods and at the same time has more than 6x fewer parameters comparing to Multi-scale CNN , which heavily speeds up the inference. Deblured images from test on GoPro dataset are shown in Figure 7.

GoPro数据集[25]由2103对720p质量的模糊和清晰图像组成，这些图像来自不同的场景。我们将我们的模型的结果与最先进的模型[36]、[25]进行了比较，并给出了每个算法在单个GPU上的运行时间。结果见表1。DeblurGAN在结构自相似性方面显示出优越的结果，在峰值信噪比方面接近最新水平，并且通过视觉检查提供了更好的效果。与其他神经网络模型相比，我们的网络不使用像素空间中的L2距离，因此没有直接针对PSNR指标进行优化。它能处理摄像机抖动和目标运动引起的模糊，不受核估计方法中常见的伪影影响，同时参数比多尺度CNN少6倍以上，大大加快了推理速度。图7显示了在GoPro数据集上测试的去模糊图像。

### 6.2. Kohler dataset

Kohler dataset [19] consists of 4 images blurred with 12 different kernels for each of them. This is a standard benchmark dataset for evaluation of blind deblurring algorithms. The dataset is generated by recording and analyzing real camera motion, which is played back on a robot platform such that a sequence of sharp images is recorded sampling the 6D camera motion trajectory. Results are in Table 2, similar to GoPro evaluation.

科勒数据集[19]由4幅模糊图像组成，每个图像都有12个不同的核。这是一个用于评估盲去模糊算法的标准基准数据集。数据集是通过记录和分析真实的摄像机运动来生成的，这些运动在机器人平台上回放，这样就可以记录一系列清晰的图像，并对6D摄像机的运动轨迹进行采样。结果如表2所示，与GoPro评估结果类似。

### 6.3. Object Detection benchmark on YOLO

Object Detection is one of the most well-studied problems in computer vision with applications in different domains from autonomous driving to security. During the last few years approaches based on Deep Convolutional Neural Networks showed state of the art performance comparing to traditional methods. However, those networks are trained on limited datasets and in real-world settings images are often degraded by different artifacts, including motion blur, Similar to [21] and [32] we studied the influence of motion blur on object detection and propose a new way to evaluate the quality of deblurring algorithm based on results of object detection on a pretrained YOLO [30] network.

目标检测是计算机视觉研究的热点问题之一，其应用领域从自主驾驶到安全等各个领域。在过去的几年中，与传统方法相比，基于深卷积神经网络的方法显示出了最先进的性能。然而，这些网络是在有限的数据集上训练的，在现实世界中，图像常常会因不同的伪影而退化，包括运动模糊，与文献[21]和[32]相似，我们研究了运动模糊对目标检测的影响，提出了一种基于预训练YOLO[30]网络的目标检测结果来评价去模糊算法质量的新方法。

For this, we constructed a dataset of sharp and blurred street views by simulating camera shake using a high frame-rate video camera. Following [14][25][27] we take a random between 5 and 25 frames taken by 240fps camera and compute the blurred version of a middle frame as an average of those frames. All the frames are gamma-corrected with γ = 2.2 and then the inverse function is taken to obtain the final blurred frame. Overall, the dataset consists of 410 pairs of blurred and sharp images, taken from the streets and parking places with different number and types of cars.

为此，我们使用高帧频摄像机模拟摄像机抖动，构建了一个清晰和模糊的街景数据集。在[14][25][27]之后，我们随机选取了240fps相机拍摄的5到25帧，然后计算中间帧的模糊版本作为这些帧的平均值。用γ=2.2对所有帧进行gamma校正，然后取逆函数得到最终的模糊帧。总体而言，该数据集由410对模糊和清晰的图像组成，这些图像分别来自街道和停车场，这些图像都有不同数量和类型的汽车。

Blur source includes both camera shake and blur caused by car movement. The dataset and supplementary code are available online. Then sharp images are feed into the YOLO network and the result after visual verification is assigned as ground truth. Then YOLO is run on blurred and recovered versions of images and average recall and precision between obtained results and ground truth are calculated. This approach corresponds to the quality of deblurring models on real-life problems and correlates with the visual quality and sharpness of the generated images, in contrast to standard PSNR metric. The precision, in general, is higher on blurry images as there are no sharp object boundaries and smaller object are not detected as it shown in Figure 9.

模糊源包括摄像机抖动和汽车运动引起的模糊。数据集和补充代码可在线获取。然后将锐利的图像输入到YOLO网络中，视觉验证后的结果被指定为地面真实。然后对模糊和恢复后的图像运行YOLO，计算得到的结果与实际情况之间的平均召回率和精确度。与标准的PSNR度量相比，这种方法对应于实际问题上的去模糊模型的质量，并与生成图像的视觉质量和清晰度相关。一般来说，在模糊图像上精度更高，因为没有清晰的物体边界，小的物体没有被检测出来，如图9所示。

Results are shown in Table 3. DeblurGAN significantly outperforms competitors in terms of recall and F1 score.

结果如表3所示。DeblurGAN 在召回率和F1得分方面明显优于竞争对手。

![1](pic\14.JPG)

![1](pic\15.JPG)

## 7. Conclusion

We described a kernel-free blind motion deblurring learning approach and introduced DeblurGAN which is a Conditional Adversarial Network that is optimized using a multi-component loss function. In addition to this, we implemented a new method for creating a realistic synthetic motion blur able to model different blur sources. We introduce a new benchmark and evaluation protocol based on results of object detection and show that DeblurGAN significantly helps detection on blurred images.

我们描述了一种无核盲运动去模糊学习方法，并介绍了一种利用多分量损失函数优化的条件对抗网络DeblurGAN。除此之外，我们实现了一种新的方法来创建一个真实的合成运动模糊，能够建模不同的模糊源。提出了一种新的基于基准的模糊目标检测方法，并给出了一种新的模糊目标检测方法。

## 8. Acknowledgements

The authors were supported by the ELEKS Ltd., ARVI Lab, Czech Science Foundation Project GACR P103/12/G084, the Austrian Ministry for Transport, Innovation and Technology, the Federal Ministry of Science, Research and Economy, and the Province of Upper Austria in the frame of the COMET center, the CTU student grant SGS17/185/OHK3/3T/13. We thank Huaijin Chen for finding the bug in peak-signal-to-noise ratio evaluation.

作者得到了ELEKS有限公司、ARVI实验室、捷克科学基金项目GACR P103/12/G084、奥地利交通、创新和技术部、联邦科学、研究和经济部以及上奥地利省在彗星中心框架下的支持，CTU学生资助SGS17/185/OHK3/3T/13。感谢陈怀进在峰值信噪比评估中发现了缺陷。

## References

[1] PyTorch. http://pytorch.org. 5 

[2] M. Arjovsky, S. Chintala, and L. Bottou. Wasserstein GAN. ArXiv e-prints, Jan. 2017. 1, 3, 4, 5, 7 

[3] S. D. Babacan, R. Molina, M. N. Do, and A. K. Katsaggelos. Bayesian blind deconvolution with general sparse image priors. In European Conference on Computer Vision (ECCV), Firenze, Italy, October 2012. Springer. 2 

[4] G. Boracchi and A. Foi. Modeling the performance of image restoration from motion blur. Image Processing, IEEE Transactions on, 21(8):3502 –3517, aug. 2012. 5 

[5] K. Bousmalis, N. Silberman, D. Dohan, D. Erhan, and D. Krishnan. Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Networks. ArXiv e-prints, Dec. 2016. 3 

[6] A. Chakrabarti. A neural approach to blind motion deblurring. In Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 2016. 3, 5 [7] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. ImageNet: A Large-Scale Hierarchical Image Database. In CVPR09, 2009. 4 

[8] R. Fergus, B. Singh, A. Hertzmann, S. T. Roweis, and W. T. Freeman. Removing camera shake from a single photograph. ACM Trans. Graph., 25(3):787–794, July 2006. 2, 5

 [9] D. Gong, J. Yang, L. Liu, Y. Zhang, I. Reid, C. Shen, A. Van Den Hengel, and Q. Shi. From Motion Blur to Motion Flow: a Deep Learning Solution for Removing Heterogeneous Motion Blur. 2016. 3 

[10] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative Adversarial Networks. June 2014. 1, 3 

[11] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville. Improved Training of Wasserstein GANs. ArXiv e-prints, Mar. 2017. 1, 3, 4, 5 

[12] A. Gupta, N. Joshi, C. L. Zitnick, M. Cohen, and B. Curless. Single image deblurring using motion density functions. In Proceedings of the 11th European Conference on Computer Vision: Part I, ECCV’10, pages 171–184, Berlin, Heidelberg, 2010. Springer-Verlag. 3 

[13] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. arXiv preprint arXiv:1512.03385, 2015. 3, 5 

[14] M. Hirsch, C. J. Schuler, S. Harmeling, and B. Scholkopf. Fast removal of non-uniform camera shake. In Proceedings of the 2011 International Conference on Computer Vision, ICCV ’11, pages 463–470, Washington, DC, USA, 2011. IEEE Computer Society. 8 

[15] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger. Densely connected convolutional networks. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2261–2269, 2017. 3 

[16] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros. Image-to-image translation with conditional adversarial networks. arxiv, 2016. 1, 3, 4, 5, 7 

[17] J. Johnson, A. Alahi, and L. Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In European Conference on Computer Vision, 2016. 1, 4, 5 

[18] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. CoRR, abs/1412.6980, 2014. 7 

[19] R. Kohler, M. Hirsch, B. Mohler, B. Sch ¨ olkopf, and ¨ S. Harmeling. Recording and playback of camera shake: Benchmarking blind deconvolution with a real-world database. In Proceedings of the 12th European Conference on Computer Vision - Volume Part VII, ECCV’12, pages 27– 40, Berlin, Heidelberg, 2012. Springer-Verlag. 7 

[20] C. Ledig, L. Theis, F. Huszar, J. Caballero, A. Cunningham, A. Acosta, A. Aitken, A. Tejani, J. Totz, Z. Wang, and W. Shi. Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network. ArXiv e-prints, Sept. 2016. 1, 3, 4 

[21] B. Li, X. Peng, Z. Wang, J. Xu, and D. Feng. An All-inOne Network for Dehazing and Beyond. ArXiv e-prints, July 2017. 7 

[22] C. Li and M. Wand. Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks. ArXiv e-prints, Apr. 2016. 3, 5 

[23] X. Mao, Q. Li, H. Xie, R. Y. K. Lau, and Z. Wang. Least squares generative adversarial networks, 2016. cite arxiv:1611.04076. 4 

[24] M. Mirza and S. Osindero. Conditional generative adversarial nets. CoRR, abs/1411.1784, 2014. 1 

[25] S. Nah, T. Hyun, K. Kyoung, and M. Lee. Deep Multi-scale Convolutional Neural Network for Dynamic Scene Deblurring. 2016. 1, 2, 3, 4, 5, 6, 7, 8 

[26] V. Nair and G. E. Hinton. Rectified linear units improve restricted boltzmann machines. In International Conference on Machine Learning (ICML), pages 807–814, 2010. 5 

[27] M. Noroozi, P. Chandramouli, and P. Favaro. Motion Deblurring in the Wild. 2017. 3, 5, 8 

[28] D. Perrone and P. Favaro. Total variation blind deconvolution: The devil is in the details. In EEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014. 2 

[29] S. Ramakrishnan, S. Pachori, A. Gangopadhyay, and S. Raman. Deep generative filter for motion deblurring. 2017 IEEE International Conference on Computer Vision Workshops (ICCVW), pages 2993–3000, 2017. 3 

[30] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi. You Only Look Once: Unified, Real-Time Object Detection. ArXiv e-prints, June 2015. 1, 7, 8 

[31] O. Ronneberger, P. Fischer, and T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. ArXiv e-prints, May 2015. 4 

[32] M. S. M. Sajjadi, B. Scholkopf, and M. Hirsch. Enhancenet: ¨ Single image super-resolution through automated texture synthesis. 2017 IEEE International Conference on Computer Vision (ICCV), pages 4501–4510, 2017. 7 

[33] T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen. Improved Techniques for Training GANs. ArXiv e-prints, June 2016. 3 

[34] K. Simonyan and A. Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition. ArXiv eprints, Sept. 2014. 4 

[35] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. Dropout: A simple way to prevent neural networks from overfitting. J. Mach. Learn. Res., 15(1):1929– 1958, Jan. 2014. 5 

[36] J. Sun, W. Cao, Z. Xu, and J. Ponce. Learning a Convolutional Neural Network for Non-uniform Motion Blur Removal. 2015. 3, 5, 7, 8 

[37] R. Szeliski. Computer Vision: Algorithms and Applications. Springer-Verlag New York, Inc., New York, NY, USA, 1st edition, 2010. 2 

[38] D. Ulyanov, A. Vedaldi, and V. S. Lempitsky. Instance normalization: The missing ingredient for fast stylization. CoRR, abs/1607.08022, 2016. 5 

[39] C. Villani. Optimal Transport: Old and New. Grundlehren der mathematischen Wissenschaften. Springer Berlin Heidelberg, 2008. 3 

[40] O. Whyte, J. Sivic, A. Zisserman, and J. Ponce. Non-uniform deblurring for shaken images. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2010. 2, 8 

[41] B. Xu, N. Wang, T. Chen, and M. Li. Empirical evaluation of rectified activations in convolutional network. arXiv preprint arXiv:1505.00853, 2015. 5 

[42] L. Xu and J. Jia. Two-phase kernel estimation for robust motion deblurring. In Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 2010. 2

[43] L. Xu, J. S. J. Ren, C. Liu, and J. Jia. Deep convolutional neural network for image deconvolution. In Proceedings of the 27th International Conference on Neural Information Processing Systems - Volume 1, NIPS’14, pages 1790–1798, Cambridge, MA, USA, 2014. MIT Press. 5 

[44] L. Xu, S. Zheng, and J. Jia. Unnatural L0 Sparse Representation for Natural Image Deblurring. 2013. 2, 7, 8 

[45] R. A. Yeh, C. Chen, T. Lim, M. Hasegawa-Johnson, and M. N. Do. Semantic image inpainting with perceptual and contextual losses. CoRR, abs/1607.07539, 2016. 1 

[46] M. D. Zeiler and R. Fergus. Visualizing and understanding convolutional networks. CoRR, abs/1311.2901, 2013. 4 

[47] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros. Unpaired imageto-image translation using cycle-consistent adversarial networks. arXiv preprint arXiv:1703.10593, 2017. 4