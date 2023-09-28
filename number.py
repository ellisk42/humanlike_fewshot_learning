"""https://dspace.mit.edu/bitstream/handle/1721.1/16714/42471842-MIT.pdf?sequence=2&isAllowed=y"""
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import scipy.special
import random
import scipy.special

from scipy.special import log_softmax

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


# import matplotlib
# matplotlib.use('QtAgg')

from proposal import set_sampling_seed
from human import get_human_number_data
from prior import code_prior, nl_prior
from proposal import propose_code_hypotheses, propose_nl_hypotheses, sample_completions
from likelihood import execution_likelihood, save_likelihood_cache, compute_support, nl_likelihood, marginal_lm_likelihood, transpiler_likelihood, propose_next_number
from utilities import logsumexp
from parameter_fitting import NumberModel, number_data_loaders

from collections import namedtuple

Config = namedtuple("Config", "n_proposals epsilon nl_hypotheses code_execution deduplicate reweigh use_prior per_token use_likelihood no_latent propose_from_prior MAP")

def nl2py(nl):
    # compute the engine, defaulting to "code-davinci-002" unless the command line options say "llama"
    engine = "code-davinci-002" if not arguments.llama else "llama"
    return sample_completions(
        engine=engine,
        prompt=f"""# Write a python function to check if a number is {nl}.
def check_number(num):
    return""",
        temperature=0,
        n=1,
        stop="\n",
        max_tokens=128
    )[0][0]

def likelihood(h, E, config):
    if not config.code_execution:
        return nl_likelihood(h, E)

    if config.nl_hypotheses:
        return transpiler_likelihood(h, E, epsilon=config.epsilon)
    else:
        return execution_likelihood(h, E, epsilon=config.epsilon)

def likelihood_in_concept(h, E, config):
    if not config.code_execution:
        return nl_likelihood(h, E)

    if config.nl_hypotheses:
        log_probability = transpiler_likelihood(h, E)
    else:
        log_probability = execution_likelihood(h, E)

    if math.isinf(log_probability):
        return log_probability
    else:
        return 0.

def prior(h, config):
    if config.nl_hypotheses:
        return nl_prior(h)
    else:
        return code_prior(h)

Result = namedtuple("Result", "hypotheses log_prior log_weight support log_posterior predictions")
def important_bayesian_stuff(config, E):

    # in the event that we are doing a non-Bayes model, 
    # we can still pretend it is by saying there has a single latent hypothesis that directly predicts the data
    if config.no_latent:
        probability_in_concept = [0 for _ in range(101) ]
        if config.use_likelihood:
            samples = propose_next_number(E, config.n_proposals)
            for i in range(len(probability_in_concept)):
                probability_in_concept[i] = samples[i]
        else:
            for i in range(len(probability_in_concept)):
                probability_in_concept[i] = marginal_lm_likelihood(E, i)
        probability_in_concept = np.exp(np.array([probability_in_concept]))
        result = Result([("no latent", 0, 0)], np.array([0]), np.array([0]),
                        probability_in_concept, np.array([0]), 
                        probability_in_concept)
        return result

    if config.nl_hypotheses:
        hs = propose_nl_hypotheses(E if not config.propose_from_prior else [], config.n_proposals, 
                                   engine="code-davinci-002")# if not arguments.llama else "llama")
    else:
        assert not config.propose_from_prior
        hs = propose_code_hypotheses(E, config.n_proposals)

    if config.deduplicate:
        hs = {hypothesis: stats for hypothesis, *stats in hs }
        hs = [(hypothesis, *stats) for hypothesis, stats in hs.items()]
    
    print("about to compute priors")
    priors = prior([h for h, q_h, l_h in hs ], config)
    print("done computing priors")

    if config.per_token:
        priors = [ pr/ln for pr, ln in priors]
    else:
        priors = [ pr for pr, ln in priors]

    if config.reweigh:
        log_weight = np.array([ -q_h for h, q_h, l_h in hs ])
    else:
        log_weight = np.array([ 0. for h, q_h, l_h in hs ])

    if config.nl_hypotheses and config.code_execution:
        print("computed priors, about to compute program translations")
        code_hypotheses = {nl: nl2py(nl) for nl in set(nl_ for nl_, _, _ in hs)}
        # convert to list
        code_hypotheses = [code_hypotheses[nl] for nl, _, _ in hs]
    if not config.nl_hypotheses and config.code_execution:
        code_hypotheses = [h for h, q_h, l_h in hs]

    # now we compute support
    print("about to compute support")
    if config.code_execution:
        support = [ [ (n in s)*1 for n in range(101) ] for h in code_hypotheses for s in [compute_support(h)] ]
    else:
        support = [ [ math.exp(nl_likelihood(h, n)) for n in range(101) ]
                     for h, q_h, l_h in hs ]

    save_likelihood_cache()

    print("done computing support")

    support = np.array(support)
    pN_H = support/np.maximum(np.sum(support, -1)[:,None], 1e-10)

    priors = np.array(priors)
    evidence = np.array([n in E for n in range(101) ])

    # likelihood over data includes noise model
    like = np.log(pN_H * (1-config.epsilon) + 0.01 * config.epsilon)
    like = np.sum(like*evidence, -1)
    
    # posterior over hypotheses
    posterior = like + priors + log_weight
    # normalize posterior
    posterior = log_softmax(posterior)

    # construct posterior predictive
    # this is a number between 0 and 1, for each index 0--100
    posterior_predictive = np.sum(support * np.exp(posterior)[:,None], 0)

    if config.nl_hypotheses and config.code_execution:
        hypotheses = list(zip([h for h, q_h, l_h in hs], code_hypotheses))
    else:
        hypotheses = [h for h, q_h, l_h in hs]

    best = np.max(posterior)
    # find top k=3 and show them
    top_indices = np.argsort(posterior)[::-1][:3]
    for i in top_indices[::-1]:
        h = hypotheses[i]
        print("BEST", h)
        print(posterior[i], f"p({E}|h)={like[i]}",
        priors[i], log_weight[i])
        print()
    print()
    #assert False



    return Result(hypotheses, priors, log_weight, support, posterior, posterior_predictive)


def optimized_posterior_predictive(Es, config, prior, neural_network_options, experiment_name):

    results = [important_bayesian_stuff(config, E) for E in Es]

    human_predictions = [get_human_number_data(E) for E in Es]
    

    # we are going to return the average predictions over every fold
    # but never average in the prediction for a human judgment when we trained on that judgment
    # to do that, we are going to collect the predictions for each fold, for each dataset, for each number
    # finally we return the average
    all_predictions = [ [ list() for _ in range(101) ] for _ in Es]

    folds = 10
    for fold, (train, validation) in enumerate(number_data_loaders(results, human_predictions, batch_size=1, folds=folds)):
        model = NumberModel(results, Es, prior=prior, MAP=config.MAP, **neural_network_options)
        logger = TensorBoardLogger("lightning_logs", name=experiment_name+f"_fold_{fold}of{folds}")
        trainer = pl.Trainer(max_epochs=model.iterations, val_check_interval=len(train)*20, check_val_every_n_epoch=None, logger=logger, accelerator="cpu")
        trainer.fit(model, train, validation, ckpt_path=None)

        predictions = [model.posterior_predictive(index).detach().cpu().exp().numpy() for index in range(len(Es))]

        for d in range(len(Es)):
            for n in range(101):
                assert validation.dataset.pairs[d][0] == d

                # we don't have human data for this number, in which case it was never trained on
                # or it was part of validation for this fold
                no_human_data = human_predictions[d][n]<0
                # in theory should be able to get rid of this logic but I don't want to break everything in case it is buggy
                in_validation = validation.dataset.pairs[d][1][n] >= 0
                if no_human_data or in_validation:
                    all_predictions[d][n].append(predictions[d][n])

    # last we train a single model on all the data
    # we do this so that we can get a single posterior over all hypotheses
    # all quantitative metrics are always reported on test data and not on this special model
    model = NumberModel(results, Es, prior=prior, MAP=config.MAP,  **neural_network_options)
    logger = TensorBoardLogger("lightning_logs", name=experiment_name+f"_all_data")
    trainer = pl.Trainer(max_epochs=model.iterations, val_check_interval=None, check_val_every_n_epoch=None, logger=logger, accelerator="cpu")
    trainer.fit(model, number_data_loaders(results, human_predictions, batch_size=1, folds=0), None, ckpt_path=None)

    posteriors = [model.posterior(index).detach().cpu().exp().numpy() for index in range(len(Es))]
    priors = [model.prior(index).detach().cpu().numpy() for index in range(len(Es))]
    likelihoods = [model.likelihood(index).detach().cpu().numpy() for index in range(len(Es))]

    all_posterior_samples = [ list() for _ in range(len(Es)) ]
    for d in range(len(Es)):
        def detuple(hh): return hh[0]  if isinstance(hh, tuple) else hh
        all_posterior_samples[d].extend(list(np.random.choice(list(map(detuple, results[d].hypotheses)), 
                                                                  1000, p=posteriors[d])))
    
    # convert all_posterior_samples to a dictionary mapping a sample to its frequency
    all_posterior_samples = [ {h: all_posterior_samples[d].count(h) for h in set(all_posterior_samples[d])} for d in range(len(Es)) ]

    return [ [ np.mean(np.array(prediction_list)) for prediction_list in predictions ] for predictions in all_predictions ], all_posterior_samples
                    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument('--n_proposals', type=int, default=100)
    parser.add_argument('--epsilon', "-e", type=float, default=0.1)
    parser.add_argument('--seed', "-s", type=int, default=None)
    parser.add_argument('--export', default=None)
    parser.add_argument('--data', "-d", default="Tenenbaum", choices=["Tenenbaum"])
    parser.add_argument('--per_token', default=False, action="store_true")
    parser.add_argument('--log', default=False, action="store_true")
    parser.add_argument("--prior", default="learned", choices=["learned", "uniform", "fixed"])
    
    parser.add_argument('--hidden', default=0, type=int)
    parser.add_argument('--iterations', default=1000, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)

    parser.add_argument('--propose_from_prior', default=False, action="store_true")
    parser.add_argument('--deduplicate', default=False, action="store_true")
    parser.add_argument('--methods', nargs='+', 
    choices=["latent code", "latent lang", "latent lang2code", 
             "sample+filter", "raw LLM/classify", "raw LLM/generate", "L3", "human"],
    default=["latent code", "latent lang", "latent lang2code", "raw LLM/generate", "raw LLM/classify"])

    parser.add_argument('--llama', default=False, action="store_true")

    arguments = parser.parse_args()

    if arguments.seed is not None:
        set_sampling_seed(arguments.seed)

    # store the neural network options into a dictionary
    neural_network_options = {k:v for k,v in vars(arguments).items() if k in ["hidden", "dropout", "iterations"]}

    nn_options_string ="_"+ "_".join([f"{k}={v}" for k,v in neural_network_options.items()])
    experiment_name = f"numbergame_seed={arguments.seed}_{arguments.data}_n={arguments.n_proposals}{nn_options_string}"
    if arguments.propose_from_prior:
        experiment_name += "_PfP"

    if arguments.llama:
        experiment_name += "_llama"

    configs = {"latent code": Config(n_proposals=arguments.n_proposals,
                                    epsilon=arguments.epsilon,
                                    nl_hypotheses=False,
                                    code_execution=True,
                                    deduplicate=arguments.deduplicate,
                                    reweigh=not arguments.deduplicate,
                                    use_prior=True,
                                    per_token=arguments.per_token,
                                    use_likelihood=True,
                                    no_latent=False,
                                    propose_from_prior=arguments.propose_from_prior,
                                    MAP=False),
               "L3": Config(n_proposals=arguments.n_proposals,
                                    epsilon=arguments.epsilon,
                                    nl_hypotheses=True,
                                    code_execution=True,
                                    deduplicate=arguments.deduplicate,
                                    reweigh=not arguments.deduplicate,
                                    use_prior=False,
                                    per_token=arguments.per_token,
                                    use_likelihood=True,
                                    no_latent=False,
                                    propose_from_prior=arguments.propose_from_prior, 
                                    MAP=True),
              "latent lang": Config(n_proposals=arguments.n_proposals,
                                    epsilon=arguments.epsilon,
                                    nl_hypotheses=True,
                                    code_execution=False,
                                    deduplicate=arguments.deduplicate,
                                    reweigh=not arguments.deduplicate,
                                    use_prior=True,
                                    per_token=arguments.per_token,
                                    use_likelihood=True,
                                    no_latent=False,
                                    propose_from_prior=arguments.propose_from_prior, 
                                    MAP=False),
            "latent lang2code": Config(n_proposals=arguments.n_proposals,
                                    epsilon=arguments.epsilon,
                                    nl_hypotheses=True,
                                    code_execution=True,
                                    deduplicate=arguments.deduplicate,
                                    reweigh=not arguments.deduplicate,
                                    use_prior=True,
                                    per_token=arguments.per_token,
                                    use_likelihood=True,
                                    no_latent=False,
                                    propose_from_prior=arguments.propose_from_prior, 
                                    MAP=False),
            "sample+filter": Config(n_proposals=arguments.n_proposals,
                                    epsilon=arguments.epsilon,
                                    nl_hypotheses=False,
                                    code_execution=True,
                                    deduplicate=False,
                                    reweigh=False,
                                    use_prior=False,
                                    per_token=arguments.per_token,
                                    use_likelihood=False,
                                    no_latent=False,
                                    propose_from_prior=False ,
                                    MAP=False),
            "raw LLM/classify": Config(n_proposals=arguments.n_proposals,
                                    epsilon=arguments.epsilon,
                                    nl_hypotheses=False,
                                    code_execution=False,
                                    deduplicate=False,
                                    reweigh=False,
                                    use_prior=False,
                                    per_token=arguments.per_token,
                                    use_likelihood=False,
                                       no_latent=True,
                                    propose_from_prior=False, 
                                    MAP=False),
               "raw LLM/generate": Config(n_proposals=arguments.n_proposals,
                                    epsilon=arguments.epsilon,
                                    nl_hypotheses=False,
                                    code_execution=False,
                                    deduplicate=False,
                                    reweigh=False,
                                    use_prior=False,
                                    per_token=arguments.per_token,
                                    use_likelihood=True,
                                    no_latent=True,
                                    propose_from_prior=False, 
                                    MAP=False)}
    datasets = [[16, 8, 2, 64],
                [60],
                [16, 23, 19, 20],
                [16],
                [60, 80, 10, 30],
                [60, 52, 57, 55],
                [98,81,86,93],
                [25,4,36,81],
            ]

    # compute the posterior predictive curves for each method on each dataset
    posterior_predictives = {}
    posterior_samples = {}
    for method in arguments.methods:
        if method == "human":
            posterior_predictives.update({(method, tuple(E)): get_human_number_data(E)
                                            for E in datasets})
            continue
        config = configs[method]
        pps, these_posterior_samples = optimized_posterior_predictive(datasets, config, arguments.prior, neural_network_options, experiment_name)
        posterior_predictives.update({(method, tuple(E)): pp
                                       for E, pp in zip(datasets, pps)})
        posterior_samples.update({(method, tuple(E)): this_posterior_samples
                                        for E, this_posterior_samples in zip(datasets, these_posterior_samples)})

    # now we are going to plot everything

    # make the horizontal size of the figure proportional to the number of methods, 
    # and the vertical size proportional to the number of datasets
    plt.figure(figsize=(len(arguments.methods)*8, len(datasets)*2))
    index=1
    for E in datasets:
        for method in arguments.methods:
            pp = posterior_predictives[(method, tuple(E))]
            
            plt.subplot(len(datasets), len(arguments.methods), index)
            index+=1
            for e in E: pp[e] = 1
            plt.bar(list(range(len(pp)))[1:], pp[1:], color ='blue',
                    width = 0.4)
            if 'human' in arguments.methods:
                pp = np.array(posterior_predictives[('human', tuple(E))])
                pp[pp < 0] = 0
                plt.bar(np.array(list(range(len(pp)))[1:])+0.4, pp[1:], color ='green',
                    width = 0.4)
            plt.title(f"{E}\n{method}")
    if arguments.export:
        plt.tight_layout()
        plt.savefig(arguments.export+".png")

        # also save the posterior predictive probabilities in a csv
        with open(arguments.export+".csv", "w") as f:
            f.write("method,dataset,example,pp\n")
            for (method, E), pp in posterior_predictives.items():
                for i, p in enumerate(pp):
                    f.write(f"{method},{'_'.join(map(str, E))},{i},{p}\n")

        # also save the posterior samples in a tsv
        with open(arguments.export+"_samples.tsv", "w") as f:
            f.write("method\tdataset\tsample\tfrequency\n")
            for (method, E), samples in posterior_samples.items():
                for frequency, sample in samples.items():
                    f.write(f"{method}\t{'_'.join(map(str, E))}\t{sample}\t{frequency}\n")
    else:
        plt.show()

    # show a figure with model predictions on the horizontal axis and human predictions on the vertical axis
    
    correlations = {} # one for each method
    for method in arguments.methods:
        if method == "human":
            continue

        relevant_posterior_predictive = {k: v for k, v in posterior_predictives.items() if k[0] == method}
        for (method, E), pp in relevant_posterior_predictive.items():
            human = get_human_number_data(E)
            for n in range(1, 101):
                if n in E: continue
                if human[n] < 0: continue
                correlations[method] = correlations.get(method, list())
                correlations[method].append([pp[n], human[n]])
    plt.figure()
    for method, data in correlations.items():
        data = np.array(data)
        r = np.corrcoef(data[:,0], data[:,1])[0,1]
        plt.scatter(data[:,0], data[:,1], label=method+f"(r={r:.2f})")
        
    plt.plot([0,1], [0,1], color="black")
    plt.xlabel("model prediction")
    plt.ylabel("human prediction")
    plt.legend()
    if arguments.export:
        plt.tight_layout()
        plt.savefig(arguments.export+"_correlations.png")
    else:
        plt.show()
