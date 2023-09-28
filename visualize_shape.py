from textwrap import wrap
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
#import pandas:
import pandas as pd
from human import get_learning_curve


def mcmc_model():
    # the data is stored in set_data/best-predictions.txt
    # the format is:
    # conceptlist, item, decay.position, correct, nyes, nno, alpha, model.ptrue
    # load the data; there is no header in this file though
    data = pd.read_csv("set_data/heldout-best-predictions.txt", sep="\t", header=None)
    data.columns = ["conceptlist", "item", "decay.position", "correct", "nyes", "nno", "alpha", "model.ptrue", "expression"]

    # take the first 15 decay positions 
    data = data[data["decay.position"] <= 15]

    # old R code
    # d$human.ptrue <- d$nyes / (d$nyes+d$nno)
    # d$n.correct   <- ifelse(d$correct, d$nyes, d$nno)
    # d$n.incorrect <- ifelse(d$correct, d$nno,  d$nyes)

    # d$human.pcorrect <- ifelse(d$correct, d$nyes, d$nno) / (d$nyes+d$nno)
    # d$model.pcorrect <- ifelse(d$correct, 
    #                         d$alpha*d$model.ptrue+(1-d$alpha)*0.5, 
    #                         d$alpha*(1-d$model.ptrue)+(1-d$alpha)*0.5)
    # now in python
    data["human.ptrue"] = data["nyes"] / (data["nyes"]+data["nno"])
    data["n.correct"] = np.where(data["correct"], data["nyes"], data["nno"])
    data["n.incorrect"] = np.where(data["correct"], data["nno"], data["nyes"])

    data["human.pcorrect"] = np.where(data["correct"], data["nyes"], data["nno"]) / (data["nyes"]+data["nno"])
    data["model.pcorrect"] = np.where(data["correct"],
                                data["alpha"]*data["model.ptrue"]+(1-data["alpha"])*0.5,
                                data["alpha"]*(1-data["model.ptrue"])+(1-data["alpha"])*0.5)

    # accuracy is the average of model.pcorrect
    accuracy = np.mean(data["model.pcorrect"])
    
    # now we measure the correlation between the human probability of predicting correct,
    # and the model probability of predicting correct
    # this is the correlation between human.pcorrect and model.pcorrect
    # we do this globally over the entire dataframe
    r = np.corrcoef(data["human.pcorrect"], data["model.pcorrect"])[0,1]
    print("accuracy correlation r=",r, "R2=", r**2)

    # now we also calculate the correlation between human.ptrue and model.ptrue
    r = np.corrcoef(data["human.ptrue"], data["model.ptrue"])[0,1]
    print("probability correlation r=",r, "R2=", r**2)

    return r**2, accuracy
    
    #names(d) <- c("conceptlist", "item", "decay.position", "correct", "nyes", "nno", "alpha", "model.ptrue")

def mcmc_special():
    filename = "~/repositories/Fleet/Models/GrammarInference-SetFunctionLearning/output/training-best-predictions.txt"
    # formatted as rows containing:
    # concept_name	response_number	batch_number	N1	N2	N3	alpha	model_prediction	map_program
    # where N1, N2, N3 are the number of yes, no, and unknown responses
    with open(os.path.expanduser(filename), "r") as handle:
        lines = handle.readlines()
    lines = [ line.strip().split("\t") for line in lines[1:] ]

    from human import special_concept
    special_examples={cn: [e[0] for example in special_concept(cn)[-1] for e in example] for cn in [200, 201]}

    curves = {}
    for line in lines:
        concept_name = line[0]
        if "200" in concept_name: concept_name=200
        elif "201" in concept_name: concept_name=201
        else: assert False

        prediction = float(line[7])




        if concept_name not in curves:
            curves[concept_name] = list()
        gt = special_examples[concept_name][int(line[1])]
        if concept_name == 200:print(gt, prediction)
        if gt:
            curves[concept_name].append(prediction)
        else:
            curves[concept_name].append(1-prediction)
    return curves


plt.style.use('seaborn')
plt.rcParams["font.family"] = "DejaVu Sans"


def load_shape_csv(filename):
    data = {}

    # loaded dynamically in case more human subjects take the experiment
    from human import special_concept
    special_shape_data={c:[a for accuracies in special_concept(c)[0] for a in accuracies] for c in [200, 201]}

    # by convention, if the file ends with "_NUMBER.csv", then NUMBER is the number of samples
    # extract it, and fix the number of samples to be None otherwise
    assert filename.endswith(".csv")
    possible_number=filename[:-4].split("_")[-1]
    try: number_of_samples  = int(possible_number)
    except: number_of_samples = None
        
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for row_index, row in enumerate(reader):
            if row_index == 0:
                continue

            # file structure: 
            # method,concept,split,trainortest,index,groundtruth,modelaccuracy,humanaccuracy,maplanguage,mapcode
            method = row[0]
            if "performance" in filename and not "performance=False" in filename:
                method = "performance"
            
            method = (method, number_of_samples)
            concept = int(row[1])
            split = int(row[2])
            trainortest = row[3]

            maplanguage = row[8]
            mapcode = row[9]

            key = (concept, split)
            if key not in data:
                data[key] = {}
            relevant_data = data[key]

            if method not in relevant_data:
                relevant_data[method] = {}
            relevant_data = relevant_data[method]

            if "groundtruth" not in relevant_data:
                relevant_data["groundtruth"] = list()
            relevant_data["groundtruth"].append(float(row[5]))

            if "modelaccuracy" not in relevant_data:
                relevant_data["modelaccuracy"] = list()
            relevant_data["modelaccuracy"].append(float(row[6]))

            if "humanaccuracy" not in relevant_data:
                relevant_data["humanaccuracy"] = list()

            
            if concept in special_shape_data:
                relevant_data["humanaccuracy"].append(special_shape_data[concept][int(row[4])])
            else:
                relevant_data["humanaccuracy"].append(float(row[7]))

            if maplanguage != "":
                relevant_data["maplanguage"] = maplanguage

    return data

def plot_heatmaps(data, response_or_accuracy, export, size):
    if len(response_or_accuracy)==0: return 

    all_concepts = list(sorted({ c for c in data }))
    all_methods = list(sorted({ m for md in data.values() for m in md }))

    n_rows = len(all_methods)
    n_columns = len(response_or_accuracy)

    # produce subfigures for each method
    fig, axs = plt.subplots(n_rows, n_columns, sharex=True, sharey=True, squeeze=False, figsize=(n_columns * size[0], n_rows * size[1]))

    # datasets is a list of dictionaries, mapping from (concept, split) to method to data
    for method_index, method in enumerate(all_methods):
        # get the relevant datasets, for that method
        method_data = {c: data[c][method] for c in all_concepts}

        groundtruth = np.array([ judgment for c in all_concepts for judgment in method_data[c]["groundtruth"] ])
        modelaccuracy = np.array([ judgment for c in all_concepts for judgment in method_data[c]["modelaccuracy"] ])
        humanaccuracy = np.array([ judgment for c in all_concepts for judgment in method_data[c]["humanaccuracy"] ])

        # we also need to compute the responses, not the accuracy
        # if the ground truth is 1, then the accuracy is the percentage of times that it responded 1
        # therefore when ground truth is 1, response is the same as accuracy
        # if the ground truth is 0, then the accuracy is the percentage of times that it responded 0
        # therefore when ground truth is 0, response is 1 - accuracy
        # so we can just do response = groundtruth * accuracy + (1 - groundtruth) * (1 - accuracy)
        modelresponse = groundtruth * modelaccuracy + (1 - groundtruth) * (1 - modelaccuracy)
        humanresponse = groundtruth * humanaccuracy + (1 - groundtruth) * (1 - humanaccuracy)

        for column_index, name in enumerate(response_or_accuracy):
            xs, ys = {"response": (modelresponse, humanresponse), "accuracy": (modelaccuracy, humanaccuracy)}[name]
            correlation = np.corrcoef(xs, ys)[0,1]
            # compute p-value for the correlation
            # p_value =

            # make a heatmap of modelaccuracy,humanaccuracy, but with the intensity on a log scale    
            ax = axs[method_index, column_index]
            heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=10, range=[[0,1],[0,1]], density=True)
            smallest_density = np.min(heatmap[heatmap > 0])
            heatmap[heatmap == 0] = smallest_density
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            log_norm = LogNorm(vmin=heatmap.min().min(), vmax=heatmap.max().max())
            #ax.imshow(np.log(heatmap.T+1), extent=extent, origin='lower', cmap="Blues")
            ax.scatter(xs, ys, #s=7,
             #color="blue",
              alpha=0.03)
            #ax.imshow(heatmap.T, extent=extent, origin='lower', cmap="Blues")
            ax.plot([0,1], [0,1], color="black", linestyle="solid")
            ax.set_aspect("equal")
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            ax.set_xlabel(f"model {name}")
            ax.set_ylabel(f"human {name}")
            # labels at 0, 0.5, 1
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])

            # show the correlation on the plot in the upper left hand corner
            ax.text(0.05, 0.95, f"R²={correlation**2:.2f}", transform=ax.transAxes, ha="left", va="top", color="black")

            pretty_method_name = {"raw LLM":"GPT-4", "latent lang2code":"tuned prior"}.get(method[0],method[0])
            if method[1] is not None: pretty_method_name += f", {method[1]} samples"
            ax.set_title(pretty_method_name)
    plt.tight_layout()
    plt.savefig(export)
    # close the figure
    plt.close(fig)

def plot_comparisons(data, methods, export, size):
    # data : (concept, split), method, seed, key -> value

    models = { m[0] for m in methods }

    artists = []

    # the priority of the models for what order they get drawn
    priority = {"raw LLM":5, "latent lang2code":0, "prior sampling": 2, "latent lang2code (fixed prior)":1,  "L3":3}

    plt.figure(figsize=size)
    for model in sorted(models, key=lambda m: priority.get(m,100)):
        print((model))
        sampling_levels = list(sorted({ m[1] for m in methods if m[0] == model }))

        correlations = [] # a list of lists of correlations, one list per sampling level, and one per seed

        for samples in sampling_levels:
            correlations.append(list())
            try:
                seeds = { sd for concept, split in data for sd in data[(concept, split)][(model,samples)] }
            except:
                import pdb; pdb.set_trace()
            

            print(f"for the method {model} with {samples} samples, we have {len(seeds)} seeds")
            
            print(seeds)
            for seed in seeds:

                # make sure every seed has the same data
                for concept, split in data:
                    assert (model, samples) in data[(concept, split)]
                    assert seed in data[(concept, split)][(model, samples)], f"missing seed {seed} for {(concept, split)} on {(model, samples)}"

                human_accuracy, model_accuracy, ground_truth = [], [], []

                for concept, split in data:
                    assert (model, samples) in data[(concept, split)]
                    human_accuracy.extend(data[(concept, split)][(model, samples)][seed]["humanaccuracy"])
                    model_accuracy.extend(data[(concept, split)][(model, samples)][seed]["modelaccuracy"])
                    ground_truth.extend(data[(concept, split)][(model, samples)][seed]["groundtruth"])

                human_accuracy = np.array(human_accuracy)
                model_accuracy = np.array(model_accuracy)
                ground_truth = np.array(ground_truth)
                human_response = ground_truth * human_accuracy + (1 - ground_truth) * (1 - human_accuracy)
                model_response = ground_truth * model_accuracy + (1 - ground_truth) * (1 - model_accuracy)

                correlation = np.corrcoef(human_response, model_response)[0,1]

                correlations[-1].append(correlation**2)

        pretty_method_name = {"raw LLM":"GPT-4", "prior sampling": "no proposal dist.", "latent lang2code": "tuned prior", "latent lang2code (fixed prior)": "pretrained prior",  'L3':"latent language"}.get(model, model)

        if len(sampling_levels) == 1:
            # solid horizontal line
            if "raw" in model.lower():
                artists.append(plt.axhline(correlations[0][0], label=pretty_method_name, color="black", linestyle="dashed"))
            else:
                artists.append(plt.axhline(correlations[0][0], label=pretty_method_name))
            
        else:
            averages = [np.mean(c) for c in correlations]
            errors = [np.std(c)/((len(c)-1)**0.5) for c in correlations]
            artists.append(plt.errorbar(sampling_levels, averages, yerr=errors, label=pretty_method_name))

    r_mcmc = mcmc_model()[0]
    artists.append(plt.axhline(r_mcmc, label="BPL\n(10⁶-10⁹ samples)", color="black", linestyle="solid"))

    
    plt.xlabel("num samples per batch")
    plt.ylabel("model-human response R²")
    # logarithmic horizontal axis
    plt.xscale("log")
    #plt.ylim([None,1.])

    # put the legend to the right and outside of the plot
    # use the artists list to get the labels
    plt.legend(artists, [a.get_label() for a in artists], bbox_to_anchor=(1.05, 1), loc='upper left')
    

    plt.tight_layout()
    plt.savefig(export)
            
    
def pretty_expression(expression):
    if "∃" in expression or "∀" in expression:
        return expression # probably already pretty
    # parse expression as a lisp s-expression
    import sexpdata
    
    parsed = sexpdata.loads(expression)

    def pp(e):
        if isinstance(e, list):
            f=str(e[0])
            if f == "lambda":
                v = e[1]
                if isinstance(v, list):
                    v = " ".join(v)
                return f"(λ{v}. {pp(e[2])})"
            if f == "and*":
                return "("+ " ∧ ".join(pp(ei) for ei in e[1:]) + ")"
            if f == "or*":
                return "("+" ∨ ".join(pp(ei) for ei in e[1:]) + ")"
            if f == "not*" or f == "not":
                if isinstance(e[1], list) and "eqv" in e[1][0]:
                    return f"{pp(e[1][1])}≠{pp(e[1][2])}"
                return f"¬({pp(e[1])})"
            if f == "implies*":
                return pp(e[1]) + " ⇒ " + pp(e[2])
            if f == "geq*":
                return pp(e[1]) + "≥" + pp(e[2])
            if f == "gt*":
                return pp(e[1]) + " > " + pp(e[2])
            if "eqv" in f:
                return pp(e[1]) + " = " + pp(e[2])
            if f == "color*":
                return pp(e[1]) + ".color"
            if f == "shape*":
                return pp(e[1]) + ".shape"
            if f == "size*":
                return pp(e[1]) + ".size"
            if f == "forall*":
                body = pp(e[1]) # (λ{v}. ...)
                support = pp(e[2])
                v=body[2]
                return "∀" + v + "∈" + support + ". " + body[5:-1]+""
            if f == "exists*":
                body = pp(e[1]) # (λ{v}. ...)
                support = pp(e[2])
                v=body[2]
                return "∃" + v + "∈" + support + ". " + body[5:-1]+""
            if f == "smallest-set*":
                return "smallest(" + pp(e[1]) + ")"
            if f == "largest-set*":
                return "biggest(" + pp(e[1]) + ")"
            if f == "filter*":
                source_set = pp(e[2])
                predicate = pp(e[1]) # (λ{v}. ...)
                v=predicate[2]
                predicate = predicate[5:-1]
                return "{ " + v + "∈" + source_set + " | " + predicate + " }"
            if f == "contains-SFL-OBJECT*":
                return pp(e[2]) + " ∈ " + pp(e[1])
            if f == "is-the-unique*":
                source_set = pp(e[3])
                predicate = pp(e[2]) # (λ{v}. ...)
                v=predicate[2]
                predicate = predicate[5:-1]
                return "{ " + v + "∈" + source_set + " | " + predicate + " } = { "+pp(e[1])+" }"
                return "unique("+pp(e[1]) + ", " + pp(e[2])+", "+pp(e[3])+")"
            if f == "Quoted":
                return pp(e[1])
            if f == "#t":
                return "True"
            if f == "#f":
                return "False"
            if f == "the-unique*":
                return "ι(" + pp(e[1]) + ")"
            return f+"(" + " ".join(pp(ei) for ei in e[1:]) + ")"
        if isinstance(e, sexpdata.Quoted):
            return pp(e.x)
        return str(e)

    while isinstance(parsed, list) and str(parsed[0]) == "lambda":
        parsed = parsed[2]
    pretty=pp(parsed)
    if pretty.endswith(")") and pretty.startswith("("):
        pretty = pretty[1:-1]

    if pretty.replace(" ", "")=="x∈biggest({y∈S|y.shape=x.shape})": # this is an example from the paper so we make it look nicer
        # forall y ∈ S. y.shape = x.shape implies x.size >= y.size
        # but in nice unicode
        pretty = "∀y∈S. y.shape=x.shape ⇒ x.size≥y.size"

    return pretty

def plot_curves(data, methods, curve, export, size):
    # create subfigures
    # one column for each concept_split
    # one row for each method
    f, axs = plt.subplots(len(methods), len(curve), sharex=False, sharey=True, squeeze=False, 
                          figsize=(len(curve)*size[0], len(methods)*size[1]))

    methods = list(methods)

    for concept_index, concept_split in enumerate(curve):
        concept, split = concept_split.split("_")
        concept = int(concept)
        split = int(split)

        if concept == 200:
            expression = "∀c∈colors. |{s∈S : s.color=x.color}| ≥ |{s∈S : s.color=c}|"
            batch_sizes = [len(b) for b in get_learning_curve(concept, "L2")[0] ]
        elif concept == 201:
            expression = "∀c∈colors. |{s∈S : s.color=x.color}| < |{s∈S : s.color=c}|"
            batch_sizes = [len(b) for b in get_learning_curve(concept, "L2")[0] ]
        else:
            with open(f"set_data/concepts/CONCEPT_hg{concept:02}__LIST_L{split}.txt") as handle:
                lines = handle.readlines()
            
            batch_sizes = [ line.count("#") for line in lines[1:]]
            expression = lines[0].strip()

        for method_index, method in enumerate(methods):
            ax = axs[method_index, concept_index]
            maplanguage = ""
            method_data = data[(concept, split)][method]

            groundtruth = np.array(method_data["groundtruth"] )
            modelaccuracy = np.array(method_data["modelaccuracy"])
            humanaccuracy = np.array(method_data["humanaccuracy"])

            # we also need to compute the responses, not the accuracy
            # if the ground truth is 1, then the accuracy is the percentage of times that it responded 1
            # therefore when ground truth is 1, response is the same as accuracy
            # if the ground truth is 0, then the accuracy is the percentage of times that it responded 0
            # therefore when ground truth is 0, response is 1 - accuracy
            # so we can just do response = groundtruth * accuracy + (1 - groundtruth) * (1 - accuracy)
            modelresponse = groundtruth * modelaccuracy + (1 - groundtruth) * (1 - modelaccuracy)
            humanresponse = groundtruth * humanaccuracy + (1 - groundtruth) * (1 - humanaccuracy)

            ax.plot(range(1, 1+len(humanaccuracy)), 
                         humanaccuracy)#, label=f"human") #, color="blue")
            
            correlationA = np.corrcoef(modelaccuracy, humanaccuracy)[0,1]
            correlationP = np.corrcoef(modelresponse, humanresponse)[0,1]

            pretty_method = {"raw LLM":"GPT-4", "latent lang2code":"tuned prior"}

            #color = "red" if method != "raw LLM" else "green"
            ax.plot(range(1, 1+len(method_data["humanaccuracy"])), 
                     method_data["modelaccuracy"])#, label=pretty_method.get(method[0], method[0]))#, color=color)
            # put the correlation in text on the plot somewhere
            #ax.text(0.95, 0.0, f"R²={correlationA**2:.2f}", transform=ax.transAxes, ha="right", va="bottom", color="black", fontweight="bold")


            
            if "maplanguage" in method_data:
                maplanguage = method_data["maplanguage"]            
                maplanguage = maplanguage.replace("An object is positive if it is", "").replace("Rule for Concept #4: ", "").replace("Something is positive if it is a", "").replace("Something is positive if", "").replace("Rule: ", "")
                for boring_suffix in [" in the collection", " in the example"]:
                    if maplanguage.endswith(boring_suffix):
                        maplanguage = maplanguage[:-len(boring_suffix)]
            
            if maplanguage: ax.set_title("\n".join(wrap(pretty_expression(expression), 35)) + "\n" + "\n".join(wrap(maplanguage, 35)))
            else: ax.set_title("\n".join(wrap(pretty_expression(expression), 35)))

            if concept in [200,201]: ax.set_title("\n".join(wrap(maplanguage, 32)))
            print(maplanguage)
            print(expression, pretty_expression(expression))
            
            ax.set_xlabel("response number")
            if concept_index == 0: ax.set_ylabel("accuracy")
            ax.set_ylim(-0.1,1.1)
            ax.set_yticks([0., 0.5, 1.])

            if concept in [200,201]:
                # mcmc baseline for special concepts
                mcmc_curves = mcmc_special()[concept]
                # just surely average because it is super chaotic 

                ax.plot(range(1, 1+len(mcmc_curves)), mcmc_curves, label="BPL", alpha=0.5)

            if concept_index == 0 and not arguments.nolegend: 
                leg = ax.legend(frameon=True)
                for lh in leg.legendHandles: 
                    lh.set_alpha(1)
                leg.get_frame().set_alpha(1)  # Make legend opaque


            # put tiny black marks at each of the batch sizes
            total_examples_so_far =0.5
            for batch_size in batch_sizes[:15]:
                ax.plot([batch_size+total_examples_so_far, batch_size+total_examples_so_far], [0, 0.1], color="black", linestyle="dashed", alpha=0.5)
                total_examples_so_far += batch_size

    plt.tight_layout()
    plt.savefig(export)

def plot_accuracy(data, methods, export, size):
    methods = list(methods)

    mapping_from_method_to_accuracy = {}
    mapping_from_method_to_correlation = {}

    for method_index, method in enumerate(methods):
        method_data = [data[k][method] for k in data]

        accuracy_of_this_method = []
        accuracy_of_the_humans = []
        ground_truth_answers = []

        for data_for_concept in method_data:

            # accuracy_of_this_method.append(np.mean(data_for_concept["modelaccuracy"]))
            # accuracy_of_the_humans.append(np.mean(data_for_concept["humanaccuracy"]))

            accuracy_of_this_method.extend(data_for_concept["modelaccuracy"])
            accuracy_of_the_humans.extend(data_for_concept["humanaccuracy"])
            ground_truth_answers.extend(data_for_concept["groundtruth"])

        mapping_from_method_to_accuracy[method] = np.mean(accuracy_of_this_method)

        # compute correlation between model and human response
        # we also need to compute the responses, not the accuracy
        # if the ground truth is 1, then the accuracy is the percentage of times that it responded 1
        # therefore when ground truth is 1, response is the same as accuracy
        # if the ground truth is 0, then the accuracy is the percentage of times that it responded 0
        # therefore when ground truth is 0, response is 1 - accuracy
        # so we can just do response = groundtruth * accuracy + (1 - groundtruth) * (1 - accuracy)
        ground_truth_answers, accuracy_of_this_method, accuracy_of_the_humans = np.array(ground_truth_answers), np.array(accuracy_of_this_method), np.array(accuracy_of_the_humans)
        modelresponse = ground_truth_answers * accuracy_of_this_method + (1 - ground_truth_answers) * (1 - accuracy_of_this_method)
        humanresponse = ground_truth_answers * accuracy_of_the_humans + (1 - ground_truth_answers) * (1 - accuracy_of_the_humans)
        
        correlationP = np.corrcoef(modelresponse, humanresponse)[0,1]
        mapping_from_method_to_correlation[method] = correlationP**2


    # compute chance level if you guess at base rate
    base_rate = np.mean(ground_truth_answers)
    chance_level = base_rate * base_rate + (1 - base_rate) * (1 - base_rate)

    human_accuracy = np.mean(accuracy_of_the_humans)
    # bar plot of different methods and accuracy
    # also include a bar for the human accuracy
    methods.append(("human", None)) 
    mapping_from_method_to_accuracy[("human", None)] = human_accuracy
    
    plt.plot(figsize=size)
    if "career" in arguments.export:
        pretty_method = {"raw LLM":"GPT-4", "performance": "IS\n100 draws", "latent lang2code":"IS\n(human\nprior)", "latent lang2code (fixed prior)":"pretrain\nprior", "MCMC over\nfirst-order logic":"MCMC\nbaseline\n10⁹ draws",  "human":"human", "L3": "no bayes"}
    else:
        pretty_method = {"raw LLM":"GPT-4", "latent lang2code":"tune\nprior", "latent lang2code (fixed prior)":"pretrain\nprior", "MCMC over\nfirst-order logic":"BPL",  "human":"human", "performance": "tune for\naccuracy"}
    def make_pretty(method):
        if method[0] in pretty_method: return pretty_method[method[0]]
        return method[0]

    methods = list(sorted(methods, key=lambda m: mapping_from_method_to_accuracy[m] if m[0]!="performance" else 0, reverse=True))

    plt.bar(range(len(methods)), [mapping_from_method_to_accuracy[method] for method in methods])
    plt.xticks(range(len(methods)), [make_pretty(m) for m in methods])

    # also for all of the bars, put the accuracy above the bar
    for method_index, method in enumerate(methods):
        plt.text(method_index, mapping_from_method_to_accuracy[method]+0.01, f"{mapping_from_method_to_accuracy[method]:.2f}", ha="center", va="bottom", color="black")

    # check if any of the methods are "performance"
    # if so, then we need to put a note including the R^2 for that bar
    for method_index, method in enumerate(methods):
        if method[0] == "performance":
            r2 = mapping_from_method_to_correlation[method]
            # the note is an arrow pointing to the bar, with the R^2 written above the arrow
            plt.annotate(f"R²={r2:.2f}", 
                          xy=(method_index, mapping_from_method_to_accuracy[method]-0.1), 
                          xytext=(method_index-1., mapping_from_method_to_accuracy[method]-0.035), 
                          arrowprops=dict(arrowstyle="->", color="black", lw=2), ha="center", va="bottom")
     
    
    plt.ylabel("average accuracy")
    plt.ylim(0.5,.85)
    plt.gcf().set_size_inches(*size)

    # make sure that tick marks are multiples of 0.1
    plt.yticks(np.arange(0.5, 0.8, 0.1))

    # put a solid horizontal black line at chance level, but make it go outside of the plot to the right a little bit
    # the graph should not go out passed the black line
    #plt.plot([-1, len(methods)+0.1], [chance_level, chance_level], color="black", linestyle="solid")
    #plt.xlim(-0.5, len(methods)-0.5)
    # above the black line, right outside the plot to the right, and oriented vertically, also put some text saying "chance level" by the black line
    #plt.text(len(methods)-0.45, chance_level, f"chance level", color="black", ha="left", va="bottom", rotation=90)  

    


    plt.tight_layout()
    plt.savefig(export)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("filenames", type=str, nargs="+", help="csv file to plot")
    parser.add_argument("--heatmap", type=str, nargs="+", choices=["accuracy", "response"], default=[])
    parser.add_argument("--curve", type=str, nargs="+", help="concept_split to plot", default=[])
    parser.add_argument("--compare", action="store_true", help="compare the provided methods as the number of samples is varied. also compares with steve's model")
    parser.add_argument("--export", type=str, help="export to this file")
    parser.add_argument("--accuracy", action="store_true", help="make a bar graph showing average accuracy of different methods")
    parser.add_argument("--size", type=str, default="5,5")
    parser.add_argument("--nolegend", action="store_true", help="do not include a legend", default=False)
    arguments = parser.parse_args()
    

    print(arguments.filenames)

    datasets = [load_shape_csv(fn) for fn in arguments.filenames ]
    # merge the datasets, and the random seeds
    data = {} # (concept, split), method, seed, key -> value
    methods = set()
    for seed, dataset in enumerate(datasets): # index in list is going to a proxy for seed
        for concept, split in dataset.keys():
            if (concept, split) not in data:
                data[(concept, split)] = {}
            for method, method_data in dataset[(concept, split)].items():
                methods.add(method)
                if method not in data[(concept, split)]:
                    data[(concept, split)][method] = dict()
                if seed not in data[(concept, split)][method]:
                    data[(concept, split)][method][seed] = dict()                
                for key, value in method_data.items():
                    assert key not in data[(concept, split)][method][seed]
                    data[(concept, split)][method][seed][key] = value
    
    
    assert (len(arguments.heatmap) > 0) + (len(arguments.curve) > 0) + arguments.compare + arguments.accuracy == 1, "specify exactly one kind of plot"

    size = tuple(float(x) for x in arguments.size.split(","))

    if len(arguments.heatmap) > 0:
        # this only works if we had a single seed
        data = { k: { m: list(v.values())[0] for m, v in d.items() } for k, d in data.items() }
        plot_heatmaps(data, arguments.heatmap, arguments.export, size)
    elif arguments.accuracy:
        # this only works if we had a single seed
        # TODO: make this work for multiple seeds
        data = { k: { m: list(v.values())[0] for m, v in d.items() } for k, d in data.items() }
        plot_accuracy(data, methods, arguments.export, size)
    elif arguments.compare:
        plot_comparisons(data, methods, arguments.export, size)
    else:
        # this only works if we had a single seed
        data = { k: { m: list(v.values())[0] for m, v in d.items() } for k, d in data.items() }
        plot_curves(data, methods, arguments.curve, arguments.export, size)
