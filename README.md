# V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration

<p align="center">
  📄 <a href="https://arxiv.org" target="_blank">Paper</a> &nbsp; | &nbsp;
  🤗 <a href="https://huggingface.co/desimfj/V-Bridge" target="_blank">Model</a> &nbsp; | &nbsp;
</p>

This repo contains the code for the paper V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration.

## Overview

Large-scale video generative models are trained on vast and diverse visual data, enabling them to internalize rich structural, semantic, and dynamic priors of the visual world. While these models have demonstrated impressive generative capability, their potential as general-purpose visual learners remains largely untapped. In this work, we introduce V-Bridge, a framework that bridges this latent capacity to versatile few-shot image restoration tasks. We reinterpret image restoration not as a static regression problem, but as a progressive generative process, and leverage video models to simulate the gradual refinement from degraded inputs to high-fidelity outputs. Surprisingly, with only 1,000 multi-task training samples (less than 2\% of existing restoration methods), pretrained video models can be induced to perform competitive image restoration, achieving multiple tasks with a single model, rivaling specialized architectures designed explicitly for this purpose. Our findings reveal that video generative models implicitly learn powerful and transferable restoration priors that can be activated with only extremely limited data, challenging the traditional boundary between generative modeling and low-level vision, and opening a new design paradigm for foundation models in visual tasks.


<p align="center"><img src="img/teaser.png" style="width: 100%;"></p>
<p align="center">
  <em>Figure 1: Left: Image restoration is formulated as progressive video generation with frame drift correction. Right: Leveraging video generative priors leads to stronger generalization under limited data compared to current image restoration method.</em>
</p>

## ToDO
- [x] Release test code.
- [ ] Release training code.
- [ ] Release training dataset.
- [ ] Optimize inference speed.

## Experiments

### Eval
Please refer to V-Bridge/README.md:

The key visual results of V-Bridge are as follows:

<p align="center"><img src="img/results.png" style="width: 100%;"></p>

## Contact

If interested in our work, please contact us at:

\- Shenghe Zheng: shenghez.zheng@gmail.com

## Citation
```

```

## Reference

Thanks to the open-source project [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun).
