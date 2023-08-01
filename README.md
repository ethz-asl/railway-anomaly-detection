# Railway Anomaly Detection
Local and Global Information in Obstacle Detection on Railway Tracks


## Abstract

Reliable obstacle detection on railways could help prevent collisions that result in injuries and potentially damage or derail the train. Unfortunately, generic object detectors do not have enough classes to account for all possible scenarios, and datasets featuring objects on railways are challenging to obtain. We propose utilizing a shallow network to learn railway segmentation from normal railway images. The limited receptive field of the network prevents overconfident predictions and allows the network to focus on the locally very distinct and repetitive patterns of the railway environment. Additionally, we explore the controlled inclusion of global information by learning to hallucinate obstacle-free images. We evaluate our method on a custom dataset featuring railway images with artificially augmented obstacles. Our proposed method outperforms other learning-based baseline methods.


![Anomaly detection pipeline overview.](/images/anomaly-direct.png)

## Paper

The anomaly detection pipeline is described in the following publication:

- Matthias Brucker, Andrei Cramariuc, Cornelius von Einem, Roland Siegwart, Cesar Cadena, **Local and Global Information in Obstacle Detection on Railway Tracks**, in _2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)_, October 2023. [[PDF]([https://arxiv.org/pdf/2102.08145.pdf](https://arxiv.org/abs/2307.15478))]


```bibtex
@inproceedings{brucker2023local,
  title={Local and Global Information in Obstacle Detection on Railway Tracks},
  author={Brucker, Matthias and Cramariuc, Andrei and von Einem, Cornelius and Siegwart, Roland and Cadena, Cesar},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2023},
  organization={IEEE}
}
```

## Use

In order to **re-generate the datasets**, have a look at the dataset_generation directory.

In order to **train or evaluate** any of our patch-wise classification / semantic difference models, have a look at the patch_classification directory.
