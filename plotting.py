import os
import csv
from human import get_human_number_data
import numpy as np
import matplotlib.pyplot as plt

#plt.style.use('ggplot')
print(plt.style.available)
plt.style.use('seaborn')

def load_number_csv(filename):
    data = {} # map from examples to list of ratings

    with open(filename, "r") as f:
        reader = csv.reader(f)
        for row_index, row in enumerate(reader):
            if row_index == 0:
                continue
            examples = tuple(map(int, row[1].split("_")))
            probe = int(row[2])
            rating = float(row[3])

            if examples not in data:
                data[examples] = [None for _ in range(101) ]
            
            data[examples][probe] = rating

    return data

def prediction_plots(filenames, examples, export):
    examples = [tuple(map(int, example.split("_"))) for example in examples]

    all_data = []
    all_samples = []
    for filename in filenames:
        data = load_number_csv(filename)
        all_data.append(data)

        # check if we have a sample file
        # this is the same as the file name, but with "_samples" appended, and the ".csv" replaced with ".tsv"
        sample_filename = filename.replace(".csv", "_samples.tsv")
        if os.path.exists(sample_filename):
            print("Found sample file", sample_filename)
            sampledata = {}
            with open(sample_filename, "r") as f:
                reader = csv.reader(f, delimiter="\t")
                for row_index, row in enumerate(reader):
                    if row_index == 0:
                        continue
                    these_examples = tuple(map(int, row[1].split("_")))
                    if these_examples not in examples:
                        continue
                    frequency = int(row[2])
                    concept = row[3]

                    if these_examples not in sampledata:
                        sampledata[these_examples] = list()

                    sampledata[these_examples].append((frequency, concept))

            # draw K=10 samples from each  

            for example, samples in sampledata.items():
                normalizing_constant = sum(f for f, _ in samples)
                distribution = [ f/normalizing_constant for f, t in samples]
                samples = np.random.choice([t for _, t in samples], size=7, p=distribution)
                print("EXAMPLES", example)
                for t in samples:
                    print(t)
                    sampledata[example] = samples
        else:
            print("No sample file found")
            sampledata = {}

        all_samples.append(sampledata)


    # create a subplot for each example
    fig, axs = plt.subplots(len(examples), 1, figsize=figure_size, sharex=False, sharey=True)

    for example, ax in zip(examples, axs):
        human_data = get_human_number_data(example)
        human_data = np.array(human_data)
        human_data[human_data < 0] = 0

                
        ax.bar(np.array(list(range(101)))+0.5, human_data, #color ='blue',
                    width = 1, label="human")
        
        if not arguments.onlyhuman:
            for model_index, (filename, data) in enumerate(zip(filenames, all_data)):
                model_data = data[example]

                model_name = "model"
                if len(filenames) > 1: # need to give them unique names so we can tell them apart
                    if "lang2code" in filename: model_name = "language prior"
                    elif "_code_" in filename: model_name = "code prior"
                    
                ax.stairs(model_data, color=["orange", "green"][model_index], label=model_name, linewidth=1)

        ax.set_title(f"training examples: {', '.join(map(str, example))}", fontsize=12)
        
        if ax == axs[1]:
            ax.set_ylabel("prob. test is in concept", fontsize=11)

        if ax == axs[-1]:
            ax.set_xlabel("test number", fontsize=11)

        ax.margins(x=0.01)

        if example in all_samples[0]:
            def truncate_text(text):
                maxlength = 25
                if len(text) > maxlength:
                    return text[:maxlength-3] + "..."
                else:
                    return text

            posterior_samples = "\n".join(map(truncate_text, sampledata[example][:5]))
            # put the samples to the right of the plot
            ax.text(1.05, 0.5, posterior_samples, transform=ax.transAxes, fontsize=11, verticalalignment="center", horizontalalignment="left", bbox=dict(facecolor='white', alpha=0.5))

        # horizontal axis should have ticks at every 10
        ax.set_xticks(np.arange(0, 101, 10))

    # put the legend in the top right corner
    axs[-1].legend(loc="upper right", framealpha=1)
    plt.tight_layout()
    plt.savefig(export)

def correlation_plots(filenames, export, title=None):
    # don't show fixed prior with code to avoid clutter
    # it doesn't work even if you tune the prior
    filenames = [filename for filename in filenames if not ("fixed" in filename and "number_code" in filename)]
    data = [load_number_csv(filename) for filename in filenames]

    examples = { example_tuple for d in data for example_tuple in d }
    humans = {example_tuple: get_human_number_data(example_tuple) for example_tuple in examples}

    baselines = {}
    different_models = {}

    for filename, data in zip(filenames, data):
        print("processing", filename)
        # discard the directories from the file name
        filename = filename.split("/")[-1]

        # compute the correlation with the human data, across all of the example tuples
        X, Y = [], []
        for example_tuple in examples:
            human_data = humans[example_tuple]
            try: model_data = data[example_tuple]
            except:
                print("Missing data for", example_tuple, "in", filename)
                continue

            for n, (x, y) in enumerate(zip(model_data, human_data)):
                if y >= 0:
                    X.append(x)
                    Y.append(y)

        # compute the correlation
        correlation = np.corrcoef(X, Y)[0,1]**2

        if filename == "number_gpt4.csv":
            baselines["GPT-4"] = (correlation, "--")
        else:
            # the filename should hold the number of samples as the last number in the file name
            samples = int(filename.split("_")[-1].split(".")[0])

            if samples > 100: continue

            
            if "L3" in filename:
                model_name = "latent language"
            elif "pfp" in filename:
                model_name = "no proposal dist."
            elif "fixed" in filename:
                if "lang" in filename:
                    model_name = "pretrained prior"
                else:
                    model_name = "pretrained code prior"
            else:
                if "lang" in filename:
                    model_name = "tuned prior"
                else:
                    model_name = "tuned code prior"

            if "llama2" in filename:
                model_name = model_name+", llama-2 for proposals and likelihood"    
            elif "llama3" in filename:
                model_name = model_name+", llama-2 for likelihood only"       
            elif "llama" in filename:
                model_name = model_name+", llama-2 for proposals only" 


            if model_name not in different_models:
                different_models[model_name] = {}
            if samples not in different_models[model_name]:
                different_models[model_name][samples] = list()
            different_models[model_name][samples].append(correlation)

            print(model_name, correlation)

    # if multiple models are provided, we provide a single plot that shows how the correlation varies as the number of samples varies
    # if just a single file is provided, we generate a scatterplot of model vs human data
    if len(different_models) + len(baselines) == 1:
        plt.figure(figsize=figure_size)
        plt.scatter(X, Y, alpha=0.3)
        plt.xlabel("model prediction")
        plt.ylabel("human rating")
        plt.plot([0, 1], [0, 1], color="black")
        # show the correlation as text on the plot in the upper left hand corner
        plt.text(0.1, 0.75, f"R²={correlation:.2f}", transform=plt.gca().transAxes)
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)
        # ticks at 0, .5, and 1
        plt.xticks([0, 0.5, 1])
        plt.yticks([0, 0.5, 1])
        plt.title(title or f"{model_name}, {samples} samples")
        plt.tight_layout()
        plt.savefig(export)
        
        return

    # plot the results
    # `different_models` is a dictionary from model name to a dictionary from number of samples to correlation
    # those just get plotted as a line plot
    # `baselines` is a dictionary from baseline name to (correlation, style)
    # those are plotted as a solid horizontal line
    plt.figure(figsize=figure_size)
    artists = []
    for model_name, correlations in sorted(different_models.items(), key=lambda zz: zz[0]!="tuned prior"):
        x=list(sorted(correlations))
        y=[np.mean(correlations[k]) for k in x ]
        e=[np.std(correlations[k])/((len(correlations[k])-1)**0.5) for k in x ]
        artists.append(plt.errorbar(x, y, yerr=e, label=model_name))
        print(model_name, list(sorted(correlations)), e)
    #baselines["DreamCoder\n~10,000 test-time samples\n~100,000 train-time examples (dreams)"] = (0.75, "-")
    #baselines["DreamCoder"] = (0.75, "-")
    for baseline_name, (correlation, linestyle) in sorted(baselines.items(),
                                                          key=lambda zz: -zz[1][0]):
        artists.append(plt.axhline(correlation, label=baseline_name, linestyle=linestyle,
                                   color="black"))
    plt.xlabel("num samples")
    plt.ylabel("model-human response R²")
    # xaxis uses log scale because the number of samples is exponential
    plt.xscale("log")
    plt.ylim(0, 1.05)
    # make the legend have 2 columns
    plt.legend(artists, [artist.get_label() for artist in artists], ncol=1, bbox_to_anchor=(1.04, 0.5), loc="center left")
    plt.tight_layout()
    plt.savefig(export)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = "")

    parser.add_argument('--correlation', type=str, nargs="+")

    parser.add_argument('--predictions', type=str, nargs="+")
    parser.add_argument('--examples', type=str, nargs="+")

    parser.add_argument('--export', type=str)
    parser.add_argument('--size', type=str, default="5,5")

    parser.add_argument('--title', type=str, default=None)

    parser.add_argument("--onlyhuman", action="store_true", default=False)    

    arguments = parser.parse_args()

    figure_size = tuple(map(float, arguments.size.split(",")))

    if arguments.correlation:
        correlation_plots(arguments.correlation, arguments.export, arguments.title)

    if arguments.predictions:
        prediction_plots(arguments.predictions, arguments.examples, arguments.export)
