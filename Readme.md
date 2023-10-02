# Introduction

This is the repository for [Human-like Few-Shot Learning via
Bayesian Reasoning over Natural Language](https://arxiv.org/abs/2306.02797)

It contains source code and data, critically including precomputed samples from expensive LLMs in `completions.db`.

Released under GPLv3 (see `LICENSE.txt`)

# OpenAI Setup

OpenAI keys should be placed in a file called `secret_keys.py`. *You need to make this file.* Structure it like this:

```
def retrieve_keys():
   return ["<my-first-key>", "<my-second-key-as-backup>", ...]
```

# Number Game

## Training

Train the model using:
```
python number.py --n_proposals <number_of_samples>  --methods <a_method> --export <export_path_to_logs>  --iterations <gradient_descent_steps>
```


Further options:
* `--deduplication` preforms deduplication of samples from $q$ instead of importance sampling (model in paper used this option)
* `--methods "latent lang2code"` uses both a latent language representation of the concept and a Python likelihood (model in paper used this option)
* `--prior fixed` forces it to use the pretrained prior
* `--methods "latent code"` uses only a latent Python representation of the concept

## Visualization

To plot model-human correlations, do:
```
python plotting.py --correlation <path(s)_to_csv_file_created_by_training> --export <filename.pdf>
```

To visualize the model predictions (like Figure 2), do:
```
python plotting.py --predictions <path_to_a_csv_file_created_by_training> --export <filename.pdf> --examples 16 16_8_2_64 16_23_19_20 60 60_80_10_30 60_52_57_55 98_81_86_93 25_4_36_81 
```

# Logical Concepts

## Training

Train the model using:
```
python shape.py --examples 15 --set 2 --methods "latent lang2code" \ # always use these options
                --n_proposals <number_of_samples> --prior <learned_or_fixed> \
                --iterations 100
```

Further options:
* `--performance` optimizes for task performance instead of optimizing for fit to human data
* `--force_higher_order` samples using a single prompt for every task that only shows example higher order rules. Otherwise, a different prompt is used for propositional and higher order tasks

## Visualization

To plot model-human correlations, do:
```
python visualize_shape.py --export <filename.pdf> --compare <csv_files_produced_from_training>
```

To visualize individual learning curves, do:
```
python visualize_shape.py <a_single_csv_file_produced_from_training> --curve <concept_number>_2 --export <filename.pdf>
```
Concept number ranges from 1-112. The suffix `_2` tells it to use the holdout testing learning curve (split 2, which was designated as test data by `--set 2` when invoking `python shape.py`). 

## Human Study

Data from the human study can be found in `special_human_data.tsv`. It is processed by `human.py` in the function `special_concept`.

The web app for the human study can be found under `human_experiment_webpage/`

To run the model on this data, execute:
```
python shape.py --examples 15 --set 2 --methods "latent lang2code" \ # always use these options
                --n_proposals 100 --prior learned \
                --iterations 100 \
                --concept 200 201 --transfer_prior # this tells it to run on the special human data
```
The option `--transfer_prior` tells it to load the learned prior from the most recent `shape.py` training run, as done in the paper.
