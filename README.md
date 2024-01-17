# SA-DGNet

SA-DGNet implements a deep gated neural network with self-attention mechanism for survival analysis with single risk and competing risks.

# Usage

## Requirement

* Python 3.8.16
* Pytorch 1.12.1
* Pycox 0.2.3
* scikit-learn 1.2.2

## Datasets

The METARIC and SUPPORT datasets can be downloaded from [Pycox](https://github.com/havakv/pycox), while the SEER dataset needs to be downloaded from the [official SEER website](https://seer.cancer.gov/data/access.html).

## Training and evaluation
For single risk scenarios, please refer to [SingleRiskDemo.ipynb](https://github.com/yangxulin/SA-DGNet/blob/main/examples/SingleRiskDemo.ipynb) for training and evaluation steps; for competing risk scenarios, please refer to [CompetingRisksDemo.ipynb](https://github.com/yangxulin/SA-DGNet/blob/main/examples/CompetingRisksDemo.ipynb).
