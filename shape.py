import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import random

from scipy.special import log_softmax

# we are going to use pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


from human import *
from proposal import propose_nl_set_hypothesis, sample_completions, propose_nl_higher_order_set_hypothesis, batch_program_translations
from cached_eval import cached_eval
from prior import nl_prior
from utilities import logsumexp, timing
from likelihood import marginal_lm_likelihood
from parameter_fitting import ShapeModel, shape_data_loader

from collections import namedtuple

Config = namedtuple("Config", "n_proposals proposal_distribution epsilon nl_hypotheses code_execution deduplicate reweigh use_prior per_token use_likelihood no_latent ML")

def binary_prediction(code_hypothesis, shape, color, size):
    expression = f"shape='{shape}'; color='{color}'; size={size}; print({code_hypothesis})"
    return cached_eval(expression)

def set_predictions(code_hypothesis, objects):
    all_examples = [f"('{shape}', '{color}', {size})" for shape, color, size in objects]
    all_examples = "[" + ", ".join(all_examples) + "]"
    code = f"""{code_hypothesis}
    
all_examples = {all_examples}
results = []
for n in range(len(all_examples)):
    probe = all_examples[n]
    remainder = all_examples[:n] + all_examples[n+1:]
    results.append(check_object(probe, remainder))
print(results)
"""
    value = cached_eval(code, statement=True)
    if value is None:
        # print("invalid code hypothesis")
        # print(code)    
        return [False]*len(objects)
    # another alternative would be to discard these programs
    return [bool(v) for v in value]

_nl2py_cache = {}
_set_synthesis_prompt = None
_replication_synthesis_prompt = None
def nl2py(nl, higher_order=False, replication=False):
    global _nl2py_cache, _set_synthesis_prompt, _replication_synthesis_prompt

    if _set_synthesis_prompt is None:
        with open("set_synthesis_prompt.py", "r") as handle:
            _set_synthesis_prompt = handle.read()

    if _replication_synthesis_prompt is None:
        with open("replication_synthesis_prompt.py", "r") as handle:
            _replication_synthesis_prompt = handle.read()

    assert isinstance(nl, list)

    # figure out which members of nl are not in the cache
    missing_indices = [i for i, nl_ in enumerate(nl) if (nl_, higher_order, replication) not in _nl2py_cache]
    # we are going to translate these into python
    to_translate = [nl[i] for i in missing_indices]

    # first: check that we actually have to do translation
    if len(to_translate)>0 and higher_order:
        if replication:
            prompts = [_replication_synthesis_prompt%(_nl,_nl,_nl) for _nl in to_translate ]
            completions = [sample_completions(pr, 1, temperature=0, stop="", max_tokens=512, engine="gpt-4")[0] for pr in prompts ]
            # now we have to clean up the completions
            cleaned_completions = []
            for k in completions:
                k = k[0]
                lines = k.split("\n")
                
                if "```" in k:
                    # just extract within the code blocks
                    delimiting_lines = [i for i, l in enumerate(lines) if "```" in l]
                    assert len(delimiting_lines)==2
                    lines = lines[delimiting_lines[0]+1:delimiting_lines[1]]                
                
                #sometimes it does do stupid things like add test cases at the end, which we can detect because they don't start with new whitespace
                lines = [ln for index, ln in enumerate(lines)
                         if index==0 or len(ln)==0 or ln[0].isspace() ]
                cleaned_completions.append("\n"+"\n".join(lines)+"\n")
            completions = cleaned_completions
        else:
            prompts = [_set_synthesis_prompt%(_nl,_nl) for _nl in to_translate ]
            print(f"calling batched translations, number of prompts={len(prompts)}")
            completions = batch_program_translations(prompts, 
                                                max_tokens=256 ,
                                                stop=["#DONE"]
                    )
        
            completion_pattern = f"""def check_object(this_object, other_objects):
    this_shape, this_color, this_size = this_object
    all_example_objects = other_objects + [this_object]
    %s"""
            completions = [completion_pattern%completion for completion, _, _ in completions]

        for _nl, k in zip(to_translate, completions):
            # print(_nl)
            # print(k)
            # print()
            _nl2py_cache[(_nl, higher_order, replication)] = k

    if len(to_translate)>0 and not higher_order:
        propositional_synthesis_prompt = '''# Python 3
def check_shape(shape, color, size):
    """
    shape: a string, either "circle", "rectangle", or "triangle"
    color: a string, either "yellow", "green", or "blue"
    size: an int, either 1 (small), 2 (medium), or 3 (large)
    returns: True if the the input obeys the following rule:
        %s
    """
    return'''
        prompts = [propositional_synthesis_prompt%_nl for _nl in to_translate]
        completions = batch_program_translations(prompts, max_tokens=128, stop="\n")
        for _nl, k in zip(to_translate, completions):
            _nl2py_cache[(_nl, higher_order, replication)] = k[0]

    return_value = [_nl2py_cache[(nl_, higher_order, replication)] for nl_ in nl]
    return return_value

Result = namedtuple("Result", "hypotheses log_prior num_correct num_incorrect log_likelihood log_posterior predictions hypothesis_gives_correct")
def important_bayesian_stuff(config, prompt, examples, test, bonus_examples=[], higher_order=False, only_propose_hypotheses=False, is_replication=False):
    """
    returns `Result` object with the following fields:
        - hypotheses: a list of hypotheses
        - log_prior: a list of priors
        - num_correct: for each hypothesis, the number of examples correctly predicted
        - num_incorrect: for each hypothesis, the number of examples incorrectly predicted
        - log_likelihood: for each hypothesis, the likelihood of the examples
        - log_posterior: for each hypothesis, the posterior given the examples
        - predictions: for each hypothesis, and for each test example, the prediction (0/1)
        - hypothesis_gives_correct: for each hypothesis, and for each test example, whether the hypothesis gives the correct prediction (0/1)
    """
    
    # in the event that we are doing a non-Bayes model, 
    # we can still pretend it is by saying there has a single latent hypothesis that directly predicts the data
    if config.no_latent:
        test_outputs = [ judgment for judgment, _ in test ]
        test_inputs = [ (shape, color, size) for _, (shape, color, size) in test ]

        if higher_order:
            log_probability_in_concept = marginal_lm_likelihood(examples, test_inputs, domain="higher_order_shapes" if higher_order else "shapes")
        else:
            log_probability_in_concept = [ marginal_lm_likelihood(examples, test_input, domain="shapes") for test_input in test_inputs ]
        
        predictions = np.exp(np.array(log_probability_in_concept)) # for each test example, this is supposed to be the probability of predicting true
        gives_correct = np.array(test_outputs)*predictions + (1-np.array(test_outputs))*(1-predictions)

        number_correct_or_incorrect = np.ones((1, len(examples)))
        log_hypothesis_probability = np.zeros((1,))
        result = Result([("no latent","no latent")], log_hypothesis_probability, 
                        number_correct_or_incorrect, number_correct_or_incorrect,
                        log_hypothesis_probability, log_hypothesis_probability, 
                        predictions[None, :], gives_correct[None, :]
        )
        return result


    print("proposing hypotheses")

    if higher_order: q = propose_nl_higher_order_set_hypothesis
    else: q = propose_nl_set_hypothesis

    with timing("proposing hypotheses"):
        if config.nl_hypotheses:
            # Models an online hypothesis generation strategy where, at each step, we propose a new set of hypothesis
            # therefore, for each prefix of the examples, we propose hypotheses and union them together

            # always start with these
            hs = propose_nl_set_hypothesis([], 100, temperature=1)[:config.n_proposals]
            hs = [("Rule: every shape (always return true).", 0, 0), ("Rule: no shape (always return false).", 0, 0)] + hs
            
            if config.proposal_distribution:
                for i in range(len(examples)):
                    hs += q(examples[:i+1], config.n_proposals, prompt_index=prompt)
                    # if i==len(examples)-1:
                    #     print("after", i, "examples we proposed these hypotheses")
                    #     for h in q(examples[:i+1], config.n_proposals, prompt_index=prompt):
                    #         print(h[0].replace("Something is positive if it", ""))
                    #     print()
                for n_bonus in range(1, len(bonus_examples)+1):
                    assert False, "bonus bad, cheating"
                    hs += q(examples+bonus_examples[:n_bonus], config.n_proposals, prompt=prompt)
            elif len(examples) > 0:
                hs += q([], config.n_proposals*len(examples))
        else:
            assert False, "not implemented"
            hs = propose_py_set_hypotheses(examples, config.n_proposals)

    if config.deduplicate:
        hs = list(set(hs))

    print("proposed hypotheses, about to compute priors")

    with timing("computing priors"):
        if config.ML or arguments.prior != "fixed":
            print("WARNING: skipping priors")
            priors = [(0.,1)]*len(hs)
        else:
            priors = nl_prior([nl for nl, _, _ in hs], domain="shape")
        if config.per_token:
            priors = [ pr/ln for pr, ln in priors]
        else:
            priors = [ pr for pr, ln in priors]

    print("computed priors, about to compute program translations")

    with timing("code generation"):
        nls = list(set(nl_ for nl_, _, _ in hs))
        code_hypothesis = nl2py(nls, higher_order=higher_order, replication=is_replication)
        code_hypotheses = dict(zip(nls, code_hypothesis))
                
    
    if False and config.deduplicate:
        # also remove duplicate hypotheses which have the same code
        mapping = {} # maps code to (nl, log_prior)
        for nl, log_prior in zip(hs, priors):
            code = code_hypotheses[nl[0]]
            if code not in mapping or mapping[code][1] < log_prior:
                mapping[code] = (nl, log_prior)

        hs = list(mapping.values())
        priors = [log_prior for _, log_prior in hs]
        hs = [nl for nl, _ in hs]
        
    # convert to list
    code_hypotheses = [code_hypotheses[nl] for nl, _, _ in hs]

    num_correct, num_incorrect, log_likelihood = [[0 for _ in examples] for _ in hs], [[0 for _ in examples] for _ in hs], []
    print("computed program translations, about to compute correct/incorrect")

    with timing("program execution"):
        n_executions = 0
        
        for hypothesis_index, code_hypothesis in enumerate(code_hypotheses):
            for batch_index, batch in enumerate(examples):
                if higher_order: predictions = set_predictions(code_hypothesis, [(shape, color, size) for _, (shape, color, size) in batch])
                else: predictions = [ binary_prediction(code_hypothesis, shape, color, size) for _, (shape, color, size) in batch ]

                n_executions += 1

                for predicted_value, (gt_judgment, (shape, color, size)) in zip(predictions, batch):
                    if predicted_value == gt_judgment: num_correct[hypothesis_index][batch_index] += 1
                    else: num_incorrect[hypothesis_index][batch_index] += 1
            log_likelihood.append(math.log(config.epsilon)*sum(num_incorrect[hypothesis_index]) + \
                                math.log(1-config.epsilon)*sum(num_correct[hypothesis_index]))

        if test is not None:
            print("evaluating on test set")

            test_outputs = [ judgment for judgment, _ in test ]
            test_inputs = [ (shape, color, size) for _, (shape, color, size) in test ]

            predictions = np.zeros((len(hs), len(test)), dtype=np.int32)
            for hypothesis_index, code_hypothesis in enumerate(code_hypotheses):
                if higher_order:
                    predictions[hypothesis_index] = np.array(set_predictions(code_hypothesis, test_inputs))*1
                else:
                    predictions[hypothesis_index] = np.array([ binary_prediction(code_hypothesis, shape, color, size) == True for shape, color, size in test_inputs ])*1

                n_executions += 1
            hypothesis_gives_correct = predictions == np.array(test_outputs)[None,:]
        else:
            predictions = None
            hypothesis_gives_correct = None
    print(f"executed {n_executions} programs-example pairs")
    if config.ML:
        # find the maximum likelihood hypothesis
        max_likelihood_index = np.argmax(log_likelihood)
        # now we are going to remove all other hypotheses
        hs = [hs[max_likelihood_index]]
        code_hypotheses = [code_hypotheses[max_likelihood_index]]
        priors = [priors[max_likelihood_index]]
        num_correct = [num_correct[max_likelihood_index]]
        num_incorrect = [num_incorrect[max_likelihood_index]]
        log_likelihood = [log_likelihood[max_likelihood_index]]
        predictions = predictions[max_likelihood_index][None,:]
        hypothesis_gives_correct = hypothesis_gives_correct[max_likelihood_index][None,:]

    return Result(
           hypotheses=[(nl, code) for (nl, _, _), code in zip(hs, code_hypotheses) ], 
           log_prior=np.array(priors), 
           num_correct=np.array(num_correct), 
           num_incorrect=np.array(num_incorrect),
           log_likelihood=np.array(log_likelihood),
           log_posterior=log_softmax(np.array(log_likelihood) + np.array(priors)),
           predictions=predictions, 
           hypothesis_gives_correct=hypothesis_gives_correct)
    

def optimized_model_curves(examples_train, human_train, 
                           examples_test, human_test,
                           config, prior, prompt, cheat=False, checkpoint=None, 
                           neural_network_options=None, experiment_name=None, higher_order=False,
                           only_propose_hypotheses=False):

    if neural_network_options is None:
        neural_network_options = dict()

    # number of datasets
    D_train = len(human_train)
    D_test = len(human_test)
    D = max(D_train, D_test)
    assert D_test >= D_train # there might be heldout concepts
    assert len(human_train) == len(examples_train)
    assert len(human_test) == len(examples_test)

    precomputed_results_train = [list() for _ in range(D_train)]
    precomputed_results_test = [list() for _ in range(D_test)]

    for d in range(D):
        print("computing dataset %d/%d" % (d+1, D))

        if d < D_train:
            for e in range(len(examples_train[d])):
                train, tests = examples_train[d][:e], examples_train[d][e]
                if cheat: bonus = examples_train[d][e:]
                else: bonus = []

                result = important_bayesian_stuff(config, prompt, train, tests, bonus, higher_order=higher_order[d], only_propose_hypotheses=only_propose_hypotheses,
                                                  is_replication=any( len(ex) > 5 for ex in examples_train[d]))
                precomputed_results_train[d].append(result)

        assert d < D_test
        
        for e in range(len(examples_test[d])):
            train, tests = examples_test[d][:e], examples_test[d][e]

            if cheat: bonus = examples_test[d][e:]
            else: bonus = []

            result = important_bayesian_stuff(config, prompt, train, tests, bonus, higher_order=higher_order[d], only_propose_hypotheses=only_propose_hypotheses,
                                              is_replication=any( len(ex) > 5 for ex in examples_test[d]))
            precomputed_results_test[d].append(result)

    if only_propose_hypotheses:
        # terminate the process
        # we got what we came for
        os._exit(0)

    #precomputed_results = [pr[1:] for pr in precomputed_results]
    model = ShapeModel(precomputed_results_train+precomputed_results_test, prior, load_fixed_prior=arguments.transfer_prior, **neural_network_options)
    train, validation = shape_data_loader(precomputed_results_train, precomputed_results_test, 
                                          human_train, human_test, batch_size=32)

    logger = TensorBoardLogger("lightning_logs", name=experiment_name)
    trainer = pl.Trainer(max_epochs=model.iterations, check_val_every_n_epoch=10, logger=logger, accelerator="cpu")
    trainer.fit(model, train, validation, ckpt_path=checkpoint)

    best_nls, best_codes = model.map_expressions()
    final_curves = model.predict_accuracy_curves()
    
    # the first half of the curves are the training curves, the second half are the test curves
    # same for the nls and codes
    # split them into train and test
    final_curves = [final_curves[:D_train], final_curves[D_train:]]
    best_nls = [best_nls[:D_train], best_nls[D_train:]]
    best_codes = [best_codes[:D_train], best_codes[D_train:]]

    return final_curves, (best_nls, best_codes)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument('--n_proposals', "-n", type=int, default=10)
    parser.add_argument('--set', choices=[1, 2], default=2, type=int, help="which dataset to use as holdout data.")
    parser.add_argument('--concept', type=int, nargs='+', default=list(range(1, 112+1))) #121
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--cheat', default=False, action="store_true")
    parser.add_argument('--per_token', default=False, action="store_true")
    parser.add_argument("--transfer_prior", default=False, action="store_true")
    parser.add_argument('--examples', "-e", type=int, default=15)
    parser.add_argument('--prompt', default=0, type=int)
    parser.add_argument('--force_higher_order', default=False, action="store_true")
    parser.add_argument('--checkpoint')
    parser.add_argument("--test_on_bonus", default=False, action="store_true", help="run testing on bonus concepts (200+201)")
    parser.add_argument('--prior', default="fixed", choices=["fixed", "uniform", "learned"])
    parser.add_argument('--methods', nargs='+', 
                        choices=["latent code", "latent lang", "latent lang2code", 
                                 "sample+filter", "raw LLM",  "prior sampling", "L3"],
    default=["latent lang2code"]) #"latent lang2code", "prior sampling, lang2code"

    # neural network options
    parser.add_argument('--hidden', type=int, default=0, help="number of hidden units for the learned prior")
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--performance', default=False, action="store_true", help="whether to use performance as a loss instead of fit to human data")

    parser.add_argument('--only_propose_hypotheses', default=False, action="store_true", help="whether to only propose hypotheses and then terminate")
    
    arguments = parser.parse_args()

    from proposal import set_sampling_seed
    set_sampling_seed(arguments.seed)

    # store the neural network options into a dictionary
    neural_network_options = {k:v for k,v in vars(arguments).items() if k in ["hidden", "dropout", "iterations", "performance"]}

    if arguments.prior == "learned": nn_options_string ="_"+ "_".join([f"{k}={v}" for k,v in neural_network_options.items()])
    else: nn_options_string = ""
    experiment_name = f"{arguments.prior}_prompt={arguments.prompt}_n={arguments.n_proposals}{nn_options_string}"
    if len(arguments.methods)==1:
        experiment_name = arguments.methods[0].replace(" ", "_") + "_" + experiment_name
    if arguments.cheat: experiment_name += "_cheat"
    if arguments.force_higher_order: experiment_name += "_all_higher_order"
    if arguments.seed is not None: experiment_name += f"_seed={arguments.seed}"

    export = f"figures/shape/{experiment_name}"
    os.system(f"mkdir -p {export}")

    if arguments.set == 1:
        train_set, test_set = 2, 1
    else:
        train_set, test_set = 1, 2

    configs = {"latent lang2code":
                    Config(n_proposals=arguments.n_proposals,
                    proposal_distribution=True,
                    nl_hypotheses=True,
                    code_execution=True,
                    deduplicate=True,
                    reweigh=True,
                    ML=False, 
                    use_prior=True,
                    per_token=arguments.per_token,
                    use_likelihood=True,
                    no_latent=False,
                    epsilon=arguments.epsilon), 
                "prior sampling":
                    Config(n_proposals=arguments.n_proposals,
                    proposal_distribution=False,
                    nl_hypotheses=True,
                    code_execution=True,
                    deduplicate=True,
                    reweigh=True,
                    use_prior=True,
                    ML=False, 
                    per_token=arguments.per_token,
                    use_likelihood=True,
                    no_latent=False,
                    epsilon=arguments.epsilon), 
                "raw LLM":
                    Config(n_proposals=10, # we are only interested in what fraction of samples are yes/no
                    proposal_distribution=False,
                    nl_hypotheses=True,
                    code_execution=True,
                    deduplicate=True,
                    reweigh=True,
                    use_prior=True,
                    per_token=arguments.per_token,
                    use_likelihood=True,
                    ML=False, 
                    no_latent=True, # IMPORTANT
                    epsilon=arguments.epsilon), 
                    "L3":
                    Config(n_proposals=arguments.n_proposals,
                    proposal_distribution=True,
                    nl_hypotheses=True,
                    code_execution=True,
                    deduplicate=True,
                    reweigh=True,
                    use_prior=True,
                    per_token=arguments.per_token,
                    use_likelihood=True,
                    no_latent=False,
                    ML=True, # IMPORTANT
                    epsilon=arguments.epsilon), 
                    }


    # get the human accuracy curves because we fit model parameters to the human data
    human_accuracies_train, human_accuracies_test = [], []
    example_list_train, example_list_test = [], []
    expressions_train, expressions_test = [], []

    higher_order = []

    for concept in arguments.concept:
        human_accuracy, expression, examples = get_learning_curve(concept,train_set)
        human_accuracies_train.append(human_accuracy[:arguments.examples])
        example_list_train.append(examples[:arguments.examples])
        expressions_train.append(expression)

        human_accuracy, expression, examples = get_learning_curve(concept,test_set)
        human_accuracies_test.append(human_accuracy[:arguments.examples])
        example_list_test.append(examples[:arguments.examples])
        expressions_test.append(expression)

        is_higher_order = "S" in expression
        if arguments.force_higher_order: is_higher_order = True
        higher_order.append(is_higher_order)

    if arguments.test_on_bonus:
        for concept in [200, 201]:
            human_accuracy, expression, examples = get_learning_curve(concept,test_set)
            human_accuracies_test.append(human_accuracy[:arguments.examples])
            example_list_test.append(examples[:arguments.examples])
            expressions_test.append(expression)
            higher_order.append(True)

    model_accuracy = {} # map from (concept, method) to list of accuracies
    best_nl, best_code = {}, {} # map from method to best nl and code dictionaries
    for method in arguments.methods:
        config = configs[method]
        print(f"method: {method}")
        
        (train_curves, test_curves), (_best_nl, _best_code) = optimized_model_curves(example_list_train, human_accuracies_train, 
                                                                                    example_list_test, human_accuracies_test, 
                                                                                    config, arguments.prior, arguments.prompt, 
                                                                                    checkpoint=arguments.checkpoint, 
                                                                                    cheat=arguments.cheat, 
                                                                                    neural_network_options=neural_network_options, 
                                                                                    experiment_name=experiment_name, 
                                                                                    higher_order=higher_order,
                                                                                    only_propose_hypotheses=arguments.only_propose_hypotheses)
        best_nl[method], best_code[method] = _best_nl, _best_code
        for concept_index, concept in enumerate(arguments.concept):
            model_accuracy[(concept, method)] = ([item for sublist in train_curves[concept_index] for item in sublist],
                                                 [item for sublist in test_curves[concept_index] for item in sublist])

        # might also have those bonus concepts
        if arguments.test_on_bonus:
            for concept_index, concept in enumerate([200, 201]):
                model_accuracy[(concept, method)] = (None,
                                                     [item for sublist in test_curves[concept_index+len(arguments.concept)] for item in sublist])
    
    for plotting_train in [True, False]:
        if plotting_train:
            human_accuracies = human_accuracies_train
            example_list = example_list_train
            expressions = expressions_train
            this_set = train_set
            traintest_string = "train"
        else:
            human_accuracies = human_accuracies_test
            example_list = example_list_test
            expressions = expressions_test
            this_set = test_set
            traintest_string = "test"
        
        for concept_index, concept in enumerate(arguments.concept):
            plt.figure()
            if _best_nl is None:
                plt.title(expressions[concept_index])
            else:
                plt.title(f"{expressions[concept_index]}\n{_best_nl[int(not plotting_train)][concept_index]}")
                print(f"natural language {_best_nl[int(not plotting_train)][concept_index]} translates to:\n{_best_code[int(not plotting_train)][concept_index]}\n")

            example_sizes = [ len(example_list[concept_index][i]) for i in range(arguments.examples) ]
            # compute running sum
            example_sizes = [sum(example_sizes[:i]) for i in range(1, len(example_sizes))]
            # put a vertical dashed line at each example size
            for x in example_sizes:
                plt.axvline(x-0.5, color="black", linestyle=":")

            # flatten human_accuracy, which is a list of lists
            human_accuracy = [item for sublist in human_accuracies[concept_index] for item in sublist]
            
            max_x = max([len(model_accuracy[(concept, method)][int(not plotting_train)]) for method in arguments.methods]) if arguments.methods else len(human_accuracy)
    
            # plot human accuracy
            plt.plot(list(range(0,max_x)), 
                    human_accuracy[:max_x], 
                    label="human")

            for method in arguments.methods:
                # compute correlation between human and model accuracy
                correlation = np.corrcoef(human_accuracy[:len(model_accuracy[(concept, method)][int(not plotting_train)])], 
                                        model_accuracy[(concept, method)][int(not plotting_train)])[0,1]
                plt.plot(range(len(model_accuracy[(concept, method)][int(not plotting_train)])), 
                         model_accuracy[(concept, method)][int(not plotting_train)],
                         label=f"{method} (r={correlation:.2f})")
            
            plt.legend()
            plt.xlabel("Number of examples")
            plt.ylabel("Accuracy")
            plt.ylim(0,1.1)

            if export is not None:
                plt.tight_layout()
                fn = Path(export) / f"{concept}_{this_set}_{traintest_string}.png"
                plt.savefig(fn)
            else:
                plt.show()

        # make a figure with one panel per method, showing the correlation between human and model response
        plt.figure()
        for method in arguments.methods:
            axis = plt.subplot(1, len(arguments.methods), arguments.methods.index(method)+1)
            xs = [ x for concept_index, concept in enumerate(arguments.concept) for x in model_accuracy[(concept, method)][int(not plotting_train)] ]
            ys = [ y for concept_index, concept in enumerate(arguments.concept) for y_ in human_accuracies[concept_index] for y in y_ ][:len(xs)]
            xs = xs[:len(ys)]

            # xs,ys tell us the probability of getting it correct
            # we want to know the probability of predicting true
            # so we need to collect the ground truth responses, in order to sort out whether true is correct or not
            ground_truth = [gt for concept_index in range(len(arguments.concept)) for ex in example_list[concept_index] for gt, _ in ex]
            xs = [x if gt else 1-x for x, gt in zip(xs, ground_truth)]
            ys = [y if gt else 1-y for y, gt in zip(ys, ground_truth)]

            correlation = np.corrcoef(xs, ys)[0,1]**2
            # make a heatmap of xs,ys
            heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=10)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            plt.imshow(heatmap, extent=extent, origin='lower', cmap="Blues")
            plt.xlabel("Model response")
            plt.ylabel("Human response")
            axis.set_title(f"{method} (r2={correlation:.3f})")
        if export is not None:
            plt.tight_layout()
            fn = Path(export) / f"shape_response_correlation_{this_set}_{traintest_string}.png"
            plt.savefig(fn)
        else:
            plt.show()

        # make a figure with one panel per method, showing the correlation between human and model accuracy
        plt.figure()
        for method in arguments.methods:
            axis = plt.subplot(1, len(arguments.methods), arguments.methods.index(method)+1)
            xs = [ x for concept_index, concept in enumerate(arguments.concept) for x in model_accuracy[(concept, method)][int(not plotting_train)] ]
            ys = [ y for concept_index, concept in enumerate(arguments.concept) for y_ in human_accuracies[concept_index] for y in y_ ][:len(xs)]
            xs = xs[:len(ys)]

            correlation = np.corrcoef(xs, ys)[0,1]**2
            # make a heatmap of xs,ys, but on a log scale            

            heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=10)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            plt.imshow(heatmap, extent=extent, origin='lower', cmap="Blues")
            plt.xlabel("Model accuracy")
            plt.ylabel("Human accuracy")
            axis.set_title(f"{method} (r2={correlation:.3f})")
        if export is not None:
            plt.tight_layout()
            fn = Path(export) / f"shape_accuracy_correlation_{this_set}_{traintest_string}.png"
            plt.savefig(fn)
        else:
            plt.show()

        # also dump out to a CSV file
        if export is not None:
            if arguments.seed is not None:
                fn = Path(export) / f"{this_set}_{traintest_string}_seed{arguments.seed}_{arguments.n_proposals}.csv"
            else:
                fn = Path(export) / f"{this_set}_{traintest_string}_{arguments.n_proposals}.csv"
            
            with open(fn, "w") as f:
                f.write("method,concept,split,trainortest,index,groundtruth,modelaccuracy,humanaccuracy,maplanguage,mapcode\n")
                for method in arguments.methods:
                    if method == "latent lang2code" and arguments.prior == "fixed":
                        method_name = method + " (fixed prior)"
                    else:
                        method_name = method

                    list_of_concepts = list(arguments.concept)
                    # take into account the bonus but only if we are doing test data right now, because we never train on the bonus
                    if not plotting_train and arguments.test_on_bonus:
                        list_of_concepts += [200, 201]
                    for concept_index, concept in enumerate(list_of_concepts):
                        ma = model_accuracy[(concept, method)][int(not plotting_train)]
                        ha = [ y for y_ in human_accuracies[concept_index] for y in y_ ]
                        ground_truth = [gt*1 for ex in example_list[concept_index] for gt, _ in ex]
                        for i, (gt, modelaccuracy, humanaccuracy) in enumerate(zip(ground_truth, ma, ha)):
                            if i == 0:
                                bestnl = best_nl[method][int(not plotting_train)][concept_index].replace(",", " ").replace("\n", "\t")
                                bestcode = best_code[method][int(not plotting_train)][concept_index].replace(",", " ").replace("\n", "\t")
                            else:
                                bestnl = ""
                                bestcode = ""
                            f.write(f"{method_name},{concept},{this_set},{traintest_string},{i},{gt},{modelaccuracy},{humanaccuracy},{bestnl},{bestcode}\n")

    print("exported to", export)
