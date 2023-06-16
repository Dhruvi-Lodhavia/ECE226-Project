# NAS for transformers - Adding backbone architectures to LiteTransformerSearch


## Package Guidelines

### Installation

Install all the requirements using:

```Python
pip install -e .
```

*If you encounter any problems with the automatic installation of the [archai](https://github.com/microsoft/archai) package, contact us.*



### Corpus

Corpus stands for a collection of pre-processed texts, which will be converted/trained into vocabularies and tokenizers. Although it is straightforward to add a new corpus, LTS uses the following ones provided by the `datasets/corpus` module:

* [WikiText-2](https://drive.google.com/drive/folders/1BuQmR5ASwDSK6-j2VMouM6VvQFFacJgp?usp=sharing)
(change path appropriately in notebooks)
Download this in the archai\nlp\experiments folder

### Baseline logs


We trained the baseline models and they can be found [here](https://drive.google.com/drive/folders/11MQtPrkGaz4idzEIrZYbdpWZ1podZVpO?usp=sharing).


### Logdir


We ran the search space for gpt2, ctrl and opt. You can find them [here](https://drive.google.com/drive/folders/11MQtPrkGaz4idzEIrZYbdpWZ1podZVpO?usp=sharing).
It also contains the training files after training the pareto architectures and final plots

### Changes

in search.py Plot pareto baseline - change the search experiment to path you want to store your results after plotting

in train.py - change paths in line 68 and 75

We have provided jupyter notebooks for each model - gpt2, opt and ctrl. You can use them as a starting point.


Codes belong to LiteTransformerSearch.
Javaheripi, Mojan, et al. ”LiteTransformerSearch: Training-free Neural Architecture Search for Efficient Language Models.” Advances in Neural Information Processing Systems 35 (2022): 24254-24267.