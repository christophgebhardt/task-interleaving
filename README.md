# HRL Explains Task Interleaving Behavior
The code accompanying the paper "[Hierarchical Reinforcement Learning Explains Task Interleaving Behavior](https://link.springer.com/article/10.1007/s42113-020-00093-9)" which appeared in Computational Brain & Behavior (2021) 4:284–304. 

- Authors: [Christoph Gebhardt](https://ait.ethz.ch/people/gebhardt/), [Antti Oulasvirta](http://users.comnet.aalto.fi/oulasvir/), [Otmar Hilliges](https://ait.ethz.ch/people/hilliges/)
- Project page: https://ait.ethz.ch/projects/2020/hrl-task-interleaving/

## Hierarchical Reinforcement Learning Model
### Setup
Please note that we have tested this code-base under Python 3.9.

Clone this repository somewhere with:
```
git clone https://github.com/christophgebhardt/task-interleaving.git
cd task-interleaving
```

Then from the base directory of this repository, install all dependencies with:
```
pip install -r requirements.txt
```

### Usage
Please see the Jupyter notebook [task-interleaving.ipynb](task-interleaving.ipynb) for an instructions on how to use the model.
The code of this notebook (without instructions) is also available in [run.py](run.py).

## Data
The dataset of this work is available on request. Please fill in this [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSeyZ8A4Uw1L7a9ssmW-g6Yzdhi68mteQuYBmnpu8GaBrUy-cA/viewform?usp=sf_link) to gain access to the dataset.

## Model Fitting with Approximate Bayesian Computation
To fit the HRL model to a specific user in our dataset, you can use [ELFI](https://elfi.readthedocs.io/en/latest/index.html) and follow the approach descriped in [1]. The code of our fitting procedure is still to be published.

[1] Kangasrääsiö, A., Athukorala, K., Howes, A., Corander, J., Kaski, S., & Oulasvirta, A. (2017, May). Inferring cognitive models from data using approximate Bayesian computation. In Proceedings of the 2017 CHI conference on human factors in computing systems (pp. 1295-1306).

## Citation
If you are using the code-base and/or the dataset in research, please use the following citation:
```
@article{gebhardt2021hierarchical,
  title={Hierarchical reinforcement learning explains task interleaving behavior},
  author={Gebhardt, Christoph and Oulasvirta, Antti and Hilliges, Otmar},
  journal={Computational Brain \& Behavior},
  volume={4},
  number={3},
  pages={284--304},
  year={2021},
  publisher={Springer}
}
```
