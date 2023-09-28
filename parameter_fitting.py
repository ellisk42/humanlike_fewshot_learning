from textfeatures import textfeatures

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import math
import numpy as np
import random
import pickle


# we are using lightning so import that
import pytorch_lightning as pl

def inverse_sigmoid(z):
    if isinstance(z, torch.Tensor):
        return torch.log(z / (1 - z))
    else:
        return math.log(z / (1 - z))

class NumberModel(pl.LightningModule):
    def __init__(self, precomputed_results, evidence, prior, MAP, hidden=0, dropout=0., iterations=100):
        super(NumberModel, self).__init__()

        assert prior in ["uniform", "fixed", "learned"]
        self.prior_type = prior

        # Create learnable parameters for constants:
        # epsilon, alpha, beta, gamma
        # Each of these is a scalar.
        # epsilon is a probability of a binary event, represented as a binary logit.
        self.epsilon = nn.Parameter(torch.tensor(inverse_sigmoid(0.01)))
        
        self.posterior_temperature = nn.Parameter(torch.tensor(1.))

        self.MAP = MAP
        if self.MAP: assert prior == "uniform"

        self.no_latent = any(result.hypotheses == [("no latent", 0, 0)] for result in precomputed_results )
        
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.offset = nn.Parameter(torch.tensor(0.))

        # Remember definition of results
        # Result = namedtuple("Result", "hypotheses log_prior log_weight support log_posterior predictions")

        self.canonically_number_hypotheses(precomputed_results)

        if self.prior_type == "learned":
            if hidden == 0:
                self.prior_network = nn.Sequential(nn.Dropout(dropout), nn.Linear(384, 1))
            else:
                self.prior_network = nn.Sequential(nn.Dropout(dropout), nn.Linear(384, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

            print("precomputing text features")
            features = textfeatures([hypothesis[0].lower() if isinstance(hypothesis, tuple) else hypothesis
                                    for hypothesis in self.every_hypothesis])
            features = {hypothesis: features[i] for i, hypothesis in enumerate(self.every_hypothesis)}
            self.precomputed_features = [ torch.tensor([ features[hypothesis]
                                                           for hypothesis in result.hypotheses ])
                                          for result in precomputed_results ]

        self.iterations = iterations

        self.pN_H = [ torch.tensor(result.support/np.maximum(np.sum(result.support, -1)[:,None], 1e-10)) for result in precomputed_results]
        self.evidence = [torch.tensor([n in E for n in range(101) ]) for E in evidence]
        self.precomputed_results = [self.to_torch(result) for result in precomputed_results]

        self.loss_type = "ll"

    def canonically_number_hypotheses(self, precomputed_results):
        # Go through the results, collect all the hypotheses, and deduplicate them.
        # This is going to be used to create a canonical numbering scheme for the hypotheses.
        every_hypothesis = []
        for result in precomputed_results:
            every_hypothesis.extend(result.hypotheses)
        self.every_hypothesis = list(set(every_hypothesis))
        self.hypothesis_numbering_scheme = {hypothesis: i for i, hypothesis in enumerate(self.every_hypothesis)}
        self.precomputed_hypotheses = [ [self.hypothesis_numbering_scheme[h] for h in result.hypotheses]
                                        for result in precomputed_results ]

    def to_torch(self, result):
        return result._replace(**{k: torch.tensor(v).to(self.device)*1 for k,v in result._asdict().items() if isinstance(v, np.ndarray)})

    def prior(self, d):
        # returns log prior for dataset d
        result = self.precomputed_results[d]

        if self.prior_type == "table":
            prior = self.prior_table[self.precomputed_hypotheses[d]] + result.log_prior
        elif self.prior_type == "fixed":
            prior = result.log_prior
        elif self.prior_type == "uniform":
            prior = torch.zeros_like(result.log_prior)
        elif self.prior_type == "learned":
            prior = self.prior_network(self.precomputed_features[d]).squeeze(-1)

        return prior

    def likelihood(self, d):
        # returns log likelihood for dataset d (across all hypotheses)
        result = self.precomputed_results[d]
        E = self.evidence[d]

        # old code: prone to numerical problems because it does not use logsumexp
        # like = torch.log(self.pN_H[d] * F.sigmoid(-self.epsilon) + 0.01 * F.sigmoid(self.epsilon))
        # new code: uses logsumexp to avoid numerical problems
        like = torch.logaddexp(torch.log(self.pN_H[d]) + F.logsigmoid(-self.epsilon), np.log(0.01) + F.logsigmoid(self.epsilon))
        like = torch.sum(like*(E.unsqueeze(0)), -1)

        return like

    def posterior(self, d):
        # returns log posterior for dataset d (across all hypotheses)
        prior_weight, likelihood_weight, proposal_weight = self.posterior_temperature, self.posterior_temperature, self.posterior_temperature
        if self.prior_type == "learned":
            prior_weight = 1 # putting a coefficient on it is redundant
        unnormalized = prior_weight*self.prior(d) + \
                       likelihood_weight*self.likelihood(d) +\
                       proposal_weight*self.precomputed_results[d].log_weight

        if self.MAP:
            return (unnormalized == torch.max(unnormalized, 0)[0]).float().log()
        else:
            return unnormalized-torch.logsumexp(unnormalized, 0)

    def posterior_predictive(self, d):
        if not self.no_latent:
            log_in_concept = self.precomputed_results[d].support.log().clamp(min=-10000) # clamping fixes numerical issues
            posterior_predictive = torch.logsumexp(log_in_concept + self.posterior(d)[:,None], 0)
            #return posterior_predictive
            posterior_predictive = posterior_predictive.exp()
        else:
            # now we have some "no latent" models, aka raw LLMs
            posterior_predictive = self.precomputed_results[d].support.squeeze(0)

        # models without latent variables and get some extra benefits from this transformation
        # only seems fair to give them more parameters to fit

        # with probability (1-epsilon), the model returns the above
        # with probability epsilon, the model returns a uniform distribution
        posterior_predictive = torch.sigmoid(-self.epsilon) * posterior_predictive + \
                                 torch.sigmoid(self.epsilon) * 0.01

        #return posterior_predictive.log()

        # compute the inverse sigmoid of posterior_predictive
        posterior_predictive = torch.logit(posterior_predictive)

        # linear transform
        posterior_predictive = (posterior_predictive - self.offset) / self.temperature

        # logsigmoid
        posterior_predictive = F.logsigmoid(posterior_predictive)

        return posterior_predictive

    def losses(self, d, human_judgments):
        # human_judgments: vector of length 101. Values are negative when there is missing data. Loss is not computed on missing data.
        # returns both the negative log likelihood and mean squared error

        pp = self.posterior_predictive(d)

        # log probability of returning yes/no for each number being in the concept
        pp_yes = pp 
        pp_no = torch.log1p(-pp.exp()+1e-10)
        
        ll = human_judgments*pp_yes + (1-human_judgments)*pp_no
        ll = ll*(human_judgments >= 0)
        ll = torch.sum(ll) / torch.sum(human_judgments >= 0)

        mse = (pp.exp()-human_judgments)**2
        mse = mse*(human_judgments >= 0)
        mse = torch.sum(mse) / torch.sum(human_judgments >= 0)

        return -ll, mse

    # configuration of optimizers
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        ds, human_accuracies = batch 
        # ds is a list of dataset indices.

        mse_loss, ll_loss = 0, 0
        for d, human_accuracy in zip(ds, human_accuracies):
            ll, mse = self.losses(d, human_accuracy)
            mse_loss += mse
            ll_loss += ll

        if self.loss_type == "mse":
            loss = mse_loss
        elif self.loss_type == "ll":
            loss = ll_loss 

        total_size = len(ds)

        self.log('mse_loss', mse_loss/total_size, on_epoch=True, batch_size=len(ds))
        self.log('ll_loss', ll_loss/total_size, on_epoch=True, batch_size=len(ds))
        self.log('loss', loss/total_size, on_epoch=True, batch_size=len(ds))

        return loss/total_size

    def validation_step(self, batch, batch_idx):
        ds, human_accuracies = batch 
        # ds is a list of dataset indices.

        mse_loss, ll_loss = 0, 0
        human_predictions, machine_predictions = [], []
        for d, human_accuracy in zip(ds, human_accuracies):
            ll, mse = self.losses(d, human_accuracy)
            mse_loss += mse
            ll_loss += ll

            human_accuracy = human_accuracy.detach().cpu().numpy()
            pp = self.posterior_predictive(d).exp().detach().cpu().numpy()
            for i in range(len(pp)):
                if human_accuracy[i] >= 0:
                    human_predictions.append(human_accuracy[i])
                    machine_predictions.append(pp[i])

        if self.loss_type == "mse":
            loss = mse_loss
        elif self.loss_type == "ll":
            loss = ll_loss 

        # compute correlation coefficient
        human_predictions = np.array(human_predictions)
        machine_predictions = np.array(machine_predictions)
        correlation_coefficient = np.corrcoef(human_predictions, machine_predictions)[0,1]

        total_size = len(ds)

        self.log('val_ll_loss', mse_loss/total_size, on_epoch=True, batch_size=len(ds))
        self.log('val_mse_loss', ll_loss/total_size, on_epoch=True, batch_size=len(ds))
        self.log('val_loss', loss/total_size, on_epoch=True, batch_size=len(ds))
        self.log('val_correlation_coefficient', correlation_coefficient, on_epoch=True, batch_size=len(ds))

        return loss/total_size
    



class ShapeModel(pl.LightningModule):
    def __init__(self, precomputed_results, prior, hidden=0, dropout=0., iterations=100, performance=False, load_fixed_prior=False):
        super(ShapeModel, self).__init__()

        assert prior in ["uniform", "fixed", "table", "learned"]
        
        # Create learnable parameters for constants:
        # epsilon_train, epsilon_test, base_rate, memory_decay, likelihood_coefficient, prior_coefficient
        # Each of these is a scalar.
        # Probabilities of binary events are represented as binary logits.
        self.epsilon_train = nn.Parameter(torch.tensor(inverse_sigmoid(0.01)))
        self.epsilon_test = nn.Parameter(torch.tensor(inverse_sigmoid(0.1)))
        self.base_rate = nn.Parameter(torch.tensor(inverse_sigmoid(0.5)))
        self.memory_decay = nn.Parameter(torch.tensor(1.))
        self.likelihood_coefficient = nn.Parameter(torch.tensor(1.0))
        
        self.performance = performance # should we optimize performance instead of fit to humans?

        self.platt_temperature = nn.Parameter(torch.tensor(1.))
        self.platt_offset = nn.Parameter(torch.tensor(0.))

        # torch everything
        self.precomputed_results = [ [ self.to_torch(result) for result in results ]
                                     for results in precomputed_results ]

        self.canonically_number_hypotheses(self.precomputed_results)

        self.prior_type = prior
        if self.prior_type == "table":
            self.prior_table = nn.Parameter(torch.zeros(len(self.hypothesis_numbering_scheme), dtype=torch.float32))
            print("Prior table size:", len(self.hypothesis_numbering_scheme))
        elif self.prior_type == "learned":
            self.load_fixed_prior = load_fixed_prior
            if hidden == 0:
                self.prior_network = nn.Sequential(nn.Dropout(dropout), nn.Linear(384, 1))
                if load_fixed_prior:
                    with open("transferred_prior.pickle", "rb") as handle:
                        transferred_prior_network = pickle.load(handle)
                    self.prior_network = transferred_prior_network
                    for parameter in self.prior_network:
                        parameter.requires_grad_(False)
                    parameter.requires_grad_(False)                    
            else:
                self.prior_network = nn.Sequential(nn.Dropout(dropout), nn.Linear(384, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

            print("precomputing text features")
            def clean_text(text):
                return text.lower().replace("rule: ", "It is ").replace("something is positive if it", "It is").replace("rule for concept #4: ", "").replace(".", "")
            features = textfeatures([clean_text(hypothesis[0] if isinstance(hypothesis, tuple) else hypothesis)
                                    for hypothesis in self.every_hypothesis])
            features = {hypothesis: features[i] for i, hypothesis in enumerate(self.every_hypothesis)}
            self.precomputed_features = [ [ torch.tensor([ features[hypothesis]
                                                           for hypothesis in result.hypotheses ])
                                            for result in results ]
                                        for results in precomputed_results ]

        

        #self.device = torch.device("cpu") #"cuda:0" if torch.cuda.is_available() else 
        self.loss_type = "ll"
        self.iterations = iterations
        
    def canonically_number_hypotheses(self, precomputed_results):
        # Go through the results, collect all the hypotheses, and deduplicate them.
        # This is going to be used to create a canonical numbering scheme for the hypotheses.
        every_hypothesis = []
        for results in precomputed_results:
            for result in results:
                every_hypothesis.extend(result.hypotheses)
        self.every_hypothesis = list(set(every_hypothesis))
        self.hypothesis_numbering_scheme = {hypothesis: i for i, hypothesis in enumerate(self.every_hypothesis)}
        self.precomputed_hypotheses = [ [ [ self.hypothesis_numbering_scheme[h] for h in result.hypotheses ]
                                            for result in results ]
                                           for results in precomputed_results ]

    def to_torch(self, result):
        return result._replace(**{k: torch.tensor(v).to(self.device)*1 for k,v in result._asdict().items() if isinstance(v, np.ndarray)})

    def prior(self, d, e):
        # returns log prior for dataset d on learning episode e
        result = self.precomputed_results[d][e]

        if self.prior_type == "table":
            prior = self.prior_table[self.precomputed_hypotheses[d][e]] + result.log_prior
        elif self.prior_type == "fixed":
            prior = result.log_prior
        elif self.prior_type == "uniform":
            prior = torch.zeros_like(result.log_prior)
        elif self.prior_type == "learned":
            prior = self.prior_network(self.precomputed_features[d][e]).squeeze(-1) #+ result.log_prior

        
        return prior

    def likelihood(self, d, e):
        # returns log likelihood for dataset d on learning episode e
        result = self.precomputed_results[d][e]

        ll = result.num_correct * F.logsigmoid(-self.epsilon_train) +\
                result.num_incorrect * F.logsigmoid(self.epsilon_train) # shape: (num_hypotheses, e)

        if e > 0:
            # compute the likelihood of the training data, taking into account memory decay
            # memory_decay_coefficient = (time_in_past+1)**memory_decay
            memory_decay_coefficient = (e-torch.arange(e))**(-self.memory_decay.abs()) # shape: (e,)

            ll = ll * memory_decay_coefficient[None,:] # shape: (num_hypotheses, e+1)
            ll = ll.sum(1) # shape: (num_hypotheses,)
        else:
            ll = torch.zeros(ll.shape[0]).to(self.device)

        return ll

    def posterior(self, d, e):
        # returns log posterior for dataset d on learning episode e
        result = self.precomputed_results[d][e]

        ll = self.likelihood(d, e)
        prior = self.prior(d, e)
            
        unnormalized_log_posterior = ll * self.likelihood_coefficient.abs() + prior
        log_posterior = unnormalized_log_posterior - torch.logsumexp(unnormalized_log_posterior, 0)

        return log_posterior

    def accuracy(self, d, e):
        # returns accuracy for dataset d on learning episode e
        result = self.precomputed_results[d][e]
        log_posterior = self.posterior(d, e)
        
        # now we recover the ground truth labels for the test instances
        # sort of a hack, we retroactively compute this information:
        # the ground trout label is 1 if the prediction is correct and we are predicting 1, 
        # or if the prediction is incorrect and we are predicting 0.
        test_outputs = result.predictions[0] * result.hypothesis_gives_correct[0]+\
                        (1-result.predictions[0]) * (1-result.hypothesis_gives_correct[0])
        test_outputs = 1*(test_outputs > 0.5)

        # log probability of giving correct answer (for each hypothesis)
        log_correct  = result.hypothesis_gives_correct  *   test_outputs[None,:]  * \
                        torch.log( (1-torch.sigmoid(self.epsilon_test)) + torch.sigmoid(self.epsilon_test)*torch.sigmoid(self.base_rate))
        log_correct += result.hypothesis_gives_correct * (1-test_outputs[None,:]) * \
                        torch.log( (1-torch.sigmoid(self.epsilon_test)) + torch.sigmoid(self.epsilon_test)*(1-torch.sigmoid(self.base_rate)))
        log_correct += (1-result.hypothesis_gives_correct) * test_outputs[None,:] * \
                        ( F.logsigmoid(self.epsilon_test) + F.logsigmoid(self.base_rate))
        log_correct += (1-result.hypothesis_gives_correct) * (1-test_outputs[None,:]) * \
                        ( F.logsigmoid(self.epsilon_test) + F.logsigmoid(-self.base_rate))

        # compute the probability of giving the correct output, averaged over all hypotheses
        log_correct = torch.logsumexp(log_correct + log_posterior[:,None], 0)
        return log_correct

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch, batch_idx):
        ds, es, human_accuracies = batch 
        # ds is a list of dataset indices
        # es is a list of learning episode indices

        if batch_idx==50 and self.prior_type == "learned" and not self.load_fixed_prior:
            with open("transferred_prior.pickle", "wb") as handle:
                pickle.dump(self.prior_network, handle)
            

        mse_loss, ll_loss, accuracy_less_human_accuracy = 0, 0, 0
        for d, e, human_accuracy in zip(ds, es, human_accuracies):
            model_accuracy = self.accuracy(d, e)
            accuracy_less_human_accuracy += (model_accuracy.exp() - human_accuracy).sum()
            
            if self.performance:
                mse_loss += ((model_accuracy.exp() - 1)**2).sum()
                ll_loss += -model_accuracy.sum()
            else:
                mse_loss += ((model_accuracy.exp() - human_accuracy)**2).sum()
                ll_loss += (-(human_accuracy * model_accuracy +\
                            (1-human_accuracy) * torch.log1p(-model_accuracy.exp()))).sum()

        if self.loss_type == "mse":
            loss = mse_loss
        elif self.loss_type == "ll":
            loss = ll_loss 

        total_size = sum( len(human_accuracy) for human_accuracy in human_accuracies)

        self.log('mse_loss', mse_loss/total_size, on_epoch=True, batch_size=len(ds))
        self.log('ll_loss', ll_loss/total_size, on_epoch=True, batch_size=len(ds))
        self.log('loss', loss/total_size, on_epoch=True, batch_size=len(ds))
        self.log('accuracy_less_human_accuracy', accuracy_less_human_accuracy/total_size, on_epoch=True, batch_size=len(ds))

        return loss/total_size

    def validation_step(self, batch, batch_idx):
        ds, es, human_accuracies = batch
        
        mse_loss, ll_loss, accuracy_less_human_accuracy = 0, 0, 0
        xs, ys = [], []

        # for debugging, we are also going to show the highest prior probability hypothesis that (at some point) dominate the posterior, for each dataset
        dominant_hypotheses = {}

        for d, e, human_accuracy in zip(ds, es, human_accuracies):
            model_accuracy = self.accuracy(d, e)

            accuracy_less_human_accuracy += (model_accuracy.exp() - human_accuracy).sum()
                        
            if self.performance:
                mse_loss += ((model_accuracy.exp() - 1)**2).sum()
                ll_loss += -model_accuracy.sum()
            else:
                mse_loss += ((model_accuracy.exp() - human_accuracy)**2).sum()
                ll_loss += (-(human_accuracy * model_accuracy +\
                            (1-human_accuracy) * torch.log1p(-model_accuracy.exp()))).sum()
            
            xs.extend(human_accuracy.tolist())
            ys.extend(model_accuracy.detach().cpu().tolist())

            result = self.precomputed_results[d][e]
            # find the highest prior probability hypothesis that dominates the posterior
            log_posterior = self.posterior(d, e)
            prior = self.prior(d, e)
            
            # which hypothesis dominates the posterior?
            dominant_hypothesis = torch.argmax(log_posterior).item()
            # what is its prior probability?
            dominant_prior = prior[dominant_hypothesis]
            dominant_fixed = self.precomputed_results[d][e].log_prior[dominant_hypothesis]
            dominant_hypothesis = self.precomputed_results[d][e].hypotheses[dominant_hypothesis]

            if d not in dominant_hypotheses:
                dominant_hypotheses[d] = {}
            dominant_hypotheses[d][dominant_hypothesis] = (dominant_prior.item(), dominant_fixed.item())

        # print("finished validation step")

        # print("dominant hypotheses:")
        # for d in sorted(dominant_hypotheses.keys()):
        #     print("dataset", d)
        #     for textual_hypothesis, (numerical_score, fixed) in sorted(dominant_hypotheses[d].items(), key=lambda kv: kv[1][0], reverse=True):
        #         print(f"{textual_hypothesis}:\t{numerical_score}\t{fixed}")
        #     print()

        # compute the correlation using numpy
        xs, ys = np.array(xs), np.array(ys)
        corr = np.corrcoef(xs, ys)[0,1]
        self.log('val_corr', corr, on_epoch=True)

        if self.loss_type == "mse":
            loss = mse_loss
        elif self.loss_type == "ll":
            loss = ll_loss 

        total_size = sum( len(human_accuracy) for human_accuracy in human_accuracies)

        self.log('val_mse_loss', mse_loss/total_size, on_epoch=True, batch_size=len(ds))
        self.log('val_ll_loss', ll_loss/total_size, on_epoch=True, batch_size=len(ds))
        self.log('val_loss', loss/total_size, on_epoch=True, batch_size=len(ds))
        self.log('val_accuracy_less_human_accuracy', accuracy_less_human_accuracy/total_size, on_epoch=True, batch_size=len(ds))

        return loss/total_size
        

    def predict_accuracy_curves(self):
        # compute the final curves
        curves = []
        D = len(self.precomputed_results)
        for d in range(D):
            curves.append(list())
            for e in range(len(self.precomputed_results[d])):
                curves[-1].append(list(self.accuracy(d, e).exp().detach().cpu().numpy()))
        return curves

    def map_expressions(self):
        # compute the best posterior at the final episode
        best_hypotheses = []
        D = len(self.precomputed_results)
        for d in range(D):
            best_index = torch.argmax(self.posterior(d, len(self.precomputed_results[d])-1)).detach().cpu().item()
            best_hypotheses.append( self.precomputed_results[d][-1].hypotheses[best_index] )
        return zip(*best_hypotheses)

    
class ShapeDataset(Dataset):
    def __init__(self, results, human_data, allowed_datasets=None):   

        # by default everything is allowed
        if allowed_datasets is None:
            allowed_datasets = list(range(len(results)))

        self.triples = [ (d, e, torch.tensor(human_data[d][e]))
                         for d in allowed_datasets for e in range(len(results[d])) ]

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return self.triples[idx]

def shape_data_loader(results_train, results_test, 
                      human_train, human_test, batch_size=None):
    train = ShapeDataset(results_train+results_test, human_train+human_test,
                        list(range(len(results_train))))
    test = ShapeDataset(results_train+results_test, human_train+human_test,
                        [d+len(results_train) for d in range(len(results_test))])

    def collate_fn(list_of_things_in_batch):
        first_elements = [thing[0] for thing in list_of_things_in_batch]
        second_elements = [thing[1] for thing in list_of_things_in_batch]
        third_elements = [thing[2] for thing in list_of_things_in_batch]
        return first_elements, second_elements, third_elements
    
    if batch_size is None:
        batch_size = len(train)
    
    train_loader = DataLoader(train, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test, batch_size=len(test), shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader

class NumberDataset(Dataset):
    def __init__(self, results, human_data):
        self.pairs = [ (d, torch.tensor(human_data[d]))
                         for d in range(len(results)) if any(judgment>0 for judgment in human_data[d] )]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

def number_data_loaders(results, human, batch_size=None, folds=4, seed=42):
    """returns a list of tuples of (train loader, test loader), unless folds=0, in which case we return a single loader for all the data"""

    def collate_fn(list_of_things_in_batch):
        first_elements = [thing[0] for thing in list_of_things_in_batch]
        second_elements = [thing[1] for thing in list_of_things_in_batch]
        return first_elements, second_elements
        
    if folds == 0:
        dataset = NumberDataset(results, human)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    random.seed(seed)

    assert len(results) == len(human)

    if batch_size is None:
        batch_size = len(results)

    # d indexes datasets
    # loop over datasets, and split the human judgments into train and test for each fold
    train_dictionary, test_dictionary = {}, {} # map from (d, f) to judgments
    for d in range(len(human)):
        judgments = human[d]
        nonnegative_indices = [i for i in range(len(judgments)) if judgments[i] >= 0]
        random.shuffle(nonnegative_indices)

        for fold in range(folds):
            human_train, human_test = [], []

            test_indices = set(nonnegative_indices[fold::folds])
            train_indices = set(nonnegative_indices) - test_indices

            test_judgments = torch.tensor(judgments)
            train_judgments = torch.tensor(judgments)

            for i in range(101):
                if i in test_indices: train_judgments[i] = -1
                if i in train_indices: test_judgments[i] = -1

            train_dictionary[(d, fold)] = train_judgments
            test_dictionary[(d, fold)] = test_judgments
    

    # now we have the train and test dictionaries
    # we can create the data loaders
    loaders = []  # going to be a list of tuples of (train loader, test loader), one for each fold 
    for fold in range(folds):
        train = NumberDataset(results, [train_dictionary[(d, fold)] for d in range(len(results))])
        test = NumberDataset(results, [test_dictionary[(d, fold)] for d in range(len(results))])
        loaders.append( (DataLoader(train, batch_size=batch_size, shuffle=True),
                         DataLoader(test, batch_size=len(test), shuffle=False)) )


    return loaders
