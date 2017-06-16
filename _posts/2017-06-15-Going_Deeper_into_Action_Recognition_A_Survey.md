---
layout: post
title: "Going Deeper into Action Recognition: A Survey"
categories:
- Paper Summary
tags:
- Action Recognition
- Deep Learning
- Survey paper
---

# Going Deeper into Action Recognition: A Survey

## Abstract
* The broad range of applications
  * video surveillance
  * human-computer interaction
  * retail analytics
  * user interface design
  * learning for robotics
  * web-video search and retrieval
  * medical diagnosis
  * quality-of-life improvement for elderly care
  * sports analytics
* Comprehensive review
  * handcrafted representations
  * deep learning based approaches



## Introduction

### What is an action?
* Human motions
  * from the simplest movement of a limb
  * to complex joint movement of a group of limbs and body
  * *action* seems to be hard to define

* Moeslund and Granum (2006); Poppe (2010)
  * *action primitives* as "an atomic movement that can be described at the limb level"
  * *action*: a diverse range of movements, from "simple and primitive ones" to "cyclic body movements"
  * *activity*: "a number of subsequent actions, representing a complex movement
    * Ex) left leg forward: action primitive of running
    * Ex) jumping hurdles: activity performed with the actions starting, running and jumping
* Turaga et al. (2008)
  * *action*: "simple motion patterns usually executed by a single person and typically lasting for a very short duration (order of tens of seconds)"
  * *activity*: "a complex sequence of actions performed by several humans who could be interacting with each other in a constrained manner.
    * Ex) actions: walking, running, or swimming
    * Ex) activity: tow persons shaking hands or a football team scoring a goal
* Wang et al. (2016)
  * *action*: "the change of transformation an action brings to the environment"
    * Ex) kicking a ball

* Authors
  * **Action**: "the most elementary and meaningful interactions" between humans and the environment
  * the category of the action: the meaning associated with this interaction


### Taxonomy
![taxonomy]({{ url }}/assets/2017-06-15/taxonomy.png){:width="80%"}



## 1. Where to start from?
* A good representation
  * be easy to compute
  * provide description for a sufficiently large class of actions
  * reflect the similarity between two like actions
  * be robust to various variations (e.g., view-point, illumination)
* Earliest works
  * make use of 3D models
  * but 3D models is difficult and expensive
* Another solutions without 3D
  * **Holistic representations**
  * **Local representations**


### 1.1 Holistic representations
* Motion Energy Image (MEI) and Motion History Image (MHI)
* MEI equation

$$ E_{\tau}(x, y, t) = \bigcup_{i=0}^{\tau - 1} D(x, y, t-i) $$

* $ D(x, y, t) $: a binary image sequence representing the detected object pixels

![MEI&MHI]({{ url }}/assets/2017-06-15/MEI_MHI.png){:width="80%"}

* Spatiotemporal volumes & spatiotemporal surfaces

![spatiotemporal]({{ url }}/assets/2017-06-15/spatiotemporal.png){:width="80%"}

* The holistic approaches are too rigid to capture possible variations of actions (e.g., view point, appearance, occlusions)
* Silhouette based representations are not capable of capturing fine details within the silhouette



## 2. Local Representation based Approaches
* interest point detection → local descriptor extraction → aggregation of local descriptors.


### 2.1 Interest Point Detection
* Space-Time Interest Points (STIPs)
  * 3D-Harris detector: extend version of Harris corner detector

![STIPs]({{ url }}/assets/2017-06-15/STIPs.png){:width="70%"}


### 2.2 Local Descriptors

#### Edge and Motion Descriptors
* HoG3D: extended version of Histogram of Gradient Orientations
* HoF: Histogram of Optical Flow
* Motion Boundary Histogram (HBM)

![MBH]({{ url }}/assets/2017-06-15/MBH.png){:width="80%"}

#### Pixel Pattern Descriptors
* Volume local binary patterns (VLBP)
* Region Covariance Descriptor (RCD)

![PPD]({{ url }}/assets/2017-06-15/PPD.png){:width="80%"}



## 3. Deep Architectures for Action Recognition
* Four categories
  * Spatiotemporal networks
  * Multiple stream networks
  * Deep generative networks
  * Temporal coherency networks


### 3.1 Spatiotemporal networks
* Filters learned by CNN
  * In very first layers: *low* level features (ex. Gabor-like filters)
  * In top layers: *high* level semantics
* Direct approach
  * apply convolution operation with temporal information
  * *3D convolution* [Ji et al. (2013)](http://ieeexplore.ieee.org/document/6165309/) [^fn1]
    * 3D kernels: extract features from both spatial and temporal dimensions
    * *conv3d* has fixed temporal domain (ex. fixed 10 frame input)
    * it is unclear why a similar assumption should be made across the temporal domain
* Various Fusion Schemes
  * Temporal pooling [Ng et al. (2015)](https://arxiv.org/pdf/1503.08909.pdf) [^fn2]
  * Slow fusion [Karpathy et al. (2014)](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42455.pdf) [^fn3]

![SlowFusion]({{ url }}/assets/2017-06-15/slowFusion.png){:width="90%"}

* C3D [Tran et al. (2015)](https://arxiv.org/pdf/1412.0767.pdf) [^fn4]
* Factorizing conv3d into conv2d and conv1d [Sun et al. (2015)](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Sun_Human_Action_Recognition_ICCV_2015_paper.pdf) [^fn5]
* Recurrent structure 
  * Long-term Recurrent Convolutional Network [Donahue et al. (2015)](https://arxiv.org/pdf/1411.4389.pdf) [^fn6]
  * Delving (GRU-RCN) [Ballas et al. (2016)](https://arxiv.org/pdf/1511.06432.pdf) [^fn7]

![Long-temRCN]({{ url }}/assets/2017-06-15/Long-termRCN.png){:width="70%"}
![GRU-RCN]({{ url }}/assets/2017-06-15/GRU-RCN.png){:width="90%"}


### 3.2 Multiple Stream Networks
* Fig(b) Two-stream network [Simonyan and Zisserman (2014)](https://arxiv.org/pdf/1406.2199.pdf) [^fn8]
* Fig(c) Two-stream fusion network [Feichtenhofer et al. (2016)](https://arxiv.org/pdf/1604.06573.pdf) [^fn9]

![Two-stream]({{ url }}/assets/2017-06-15/Two-stream.png)


### 3.3 Deep Generative Models
less relevant to my works


### 3.4 Temporal Coherency Networks
less relevant to my works



## 4. A Quantitative Analysis

### 4.1 What is measured by action dataset?

* Comprehensive list of available datasets

| Dataset | Source | No. of Videos | No. of Classes |
| ------- | ------ | ------------- | -------------- |
| KTH | Both outdoors and indoors | 600 | 6 |
| Weizmann | Outdoor vid. on still backgrounds | 90 | 10 |
| UCF-Sports | TV sports broadcasts (780x480) | 150 | 10 |
| Hollywood2 | 69 movie clips | 1707 | 12 |
| Olympic Sports | YouTube | | 16 |
| HMDB-51 | YouTube, Movies | 7000 | 51 |
| UCF-50 | YouTube | - | 50 |
| UCF-101 | YouTube | 13320 | 101 |
| Sports-1M | YouTube | 1133158 | 487 |

* The complexity of a datasets
  * KTH and Weizmann (low complexity)
    * limited camera motion, almost zero background clutter
    * scope is limited
    * basic actions: waliking, running and jumping
  * YouTube, movies and TV (ex. HMDB-51, UCF-101)
    * camera motion (and shake)
    * view-point variations
    * resolution inconsistencies
  * HMDB-51 and UCF-101 (medium complexity)
    * the actions are well cropped in the temporal domain
    * NOT well-suited: measuring the performance of action localization
    * exist *subtle class*
      * chewing and talking
      * playing violin and playing cello
  * Hollywood2 and Sports-1M (high complexity)
    * view-point/editing complexities
    * the action usually occur in a **small portion** of the clip
    * Sports-1M has scenes of spectators and banner adverts
  * HMDB-51, UCF-101, Hollywood2 and Sports-1M
    * *cannot* be distinguished by motion clues
    * the objects contributed to the actions become important
  * Deep learning need to very very much dataset
    * training on small and medium size datasets (KTH and Wiezmann) is difficult
    * Many researches exploit the pretrained model using Sports-1M dataset


### 4.2 Recognition Results
* column *Type*: Deep-net based(D), Representation based(R), Fused solution(F)

![Comprehensive-Results]({{ url }}/assets/2017-06-15/comprehensiveResults.png)

#### State-of-the-art solutions
* Deep-net solutions
  * The spatiotemporal networks
    * [Karpathy et al. (2014)](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42455.pdf) [^fn3]
    * [Tran et al. (2015)](https://arxiv.org/pdf/1412.0767.pdf) [^fn4]
    * [Varol et al. (2016)](https://arxiv.org/pdf/1604.04494.pdf) [^fn10]
  * Two-stream networks 
    * [Simonyan and Zisserman (2014)](https://arxiv.org/pdf/1406.2199.pdf) [^fn8]
    * [Feichtenhofer et al. (2016)](https://arxiv.org/pdf/1604.06573.pdf) [^fn9]
  * More rigorous data augmentation
    * temporal crops by random clip sampling
    * frame skipping


### 4.3 What algorithmic changes to expect in future?


### 4.4 Bringing action recognition into life
We must fully understand the following areas in order to apply action recognition in real-life scenarios

* joint detection and recognition from a sequence
* constraining into a refined set of actions instead of big pool of classes






## References
[^fn1]: S. Ji, W. Xu, M. Yang, and K. Yu. 3d convolutional neural networks for human action recognition. IEEE Trans. Pattern Analysis and Machine Intelligence, 35(1):221–231, Jan 2013. ISSN 0162-8828.
[^fn2]: Joe Yue-Hei Ng, M. Hausknecht, S. Vijayanarasimhan, O. Vinyals, R. Monga, and G. Toderici. Beyond short snippets: Deep networks for video classification. In Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 4694–4702, 2015.
[^fn3]: Andrej Karpathy, George Toderici, Sanketh Shetty, Thomas Leung, Rahul Sukthankar, and Li Fei-Fei. Large-scale video classification with convolutional neural networks. In Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1725–1732, 2014.
[^fn4]: D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri. Learning spatiotemporal features with 3d convolutional networks. In Proc. Int. Conference on Computer Vision (ICCV), pages 4489–4497, Dec 2015.
[^fn5]: L. Sun, K. Jia, D. Y. Yeung, and B. E. Shi. Human action recognition using factorized spatiotemporal convolutional networks. In Proc. Int. Conference on Computer Vision (ICCV), pages 4597–4605, Dec 2015.
[^fn6]: Jeff Donahue, Lisa Anne Hendricks, Sergio Guadarrama, Marcus Rohrbach, Subhashini Venugopalan, Kate Saenko, and Trevor Darrell. Long-term recurrent convolutional networks for visual recognition and description. In Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2625–2634, 2015.
[^fn7]: Nicolas Ballas, Li Yao, Chris Pal, Aaron Courville. Delving deeper into convolutional networks for learning video representations. arXiv:1511.06432, 2016.
[^fn8]: Karen Simonyan and Andrew Zisserman. Two-stream convolutional networks for action recognition in videos. In Proc. Advances in Neural Information Processing Systems (NIPS), pages 568–576, 2014.
[^fn9]: Christoph Feichtenhofer, Axel Pinz, and Andrew Zisserman. Convolutional two-stream network fusion for video action recognition. In Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1933–1941, 6 2016.
[^fn10]: Gül Varol, Ivan Laptev, and Cordelia Schmid. Long-term Temporal Convolutions for Action Recognition. arXiv:1604.04494, 2016.
