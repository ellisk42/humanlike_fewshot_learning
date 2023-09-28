import pandas as pd
import os 
import numpy as np
import pickle

probability_in_concept={}

def load_number_humans():
    global probability_in_concept

    for root, dirs, files in os.walk("tenenbaum_data"):
        for file in files:
            if file.endswith(".csv"):
                with open(os.path.join(root, file), "r") as f:
                    examples = tuple(sorted(map(int, file.replace(".csv", "").replace("[", "").replace("]", "").split(","))))
                    
                    probability_in_concept[examples] = {int(new_number): float(in_concept) for line in f for new_number, in_concept in [line.strip().split(",")] }


def get_human_number_data(examples):
    if len(probability_in_concept) == 0:
        load_number_humans()
    canonical = tuple(sorted(examples))
    return [probability_in_concept[canonical].get(j, -1)
            for j in range(100+1) ]

def special_concept(concept):
    # load the human responses from special_human_data.tsv
    # this is a tab-separated file with the following columns:
    # base_64_encoding_of_responses, nl_response, concept
    # the base_64_encoding_of_responses is a base 64 encoding of a list of lists
    human_responses = {}
    with open("special_human_data.tsv") as handle:
        import base64
        lines = handle.readlines()
        for line in lines:
            encoded_responses, nl_response, this_concept = line.strip().split("\t")
            this_concept = int(this_concept)
            decoded_responses = eval(base64.b64decode(encoded_responses).decode("utf-8"))
            if this_concept not in human_responses:
                human_responses[this_concept] = [ [ [r] for r in batch ] for batch in decoded_responses]
            else:
                for i, batch in enumerate(decoded_responses):
                    for j in range(len(batch)):
                        human_responses[this_concept][i][j].append(batch[j])

    # compute average human responses
    human_responses = {concept: [ [np.mean(rs) for rs in batch] for batch in batches] for concept, batches in human_responses.items()}


    import random
    random.seed(69420)

    if concept == 200:
        # most popular color
        examples = []
        learning_curve = []
        for i in range(15):
            dominant_color = ["yellow", "green", "blue"][random.randint(0,2) if i>2 else i]
            num_dominant_objects = random.choice([3,4]) if random.random()>0.3 or i == 0 else random.choice([2,3,4])
            num_minority_objects = random.choice(list(range(1, min(3, num_dominant_objects))))

            minority_color = random.choice([c for c in ["yellow", "green", "blue"] if c != dominant_color])

            shapes_that_have_the_dominant_color = [(shape, dominant_color, sz) for shape in ["triangle", "rectangle", "circle"] for sz in [1, 2, 3]]
            shapes_that_have_the_minority_color = [(shape, minority_color, sz) for shape in ["triangle", "rectangle", "circle"] for sz in [1, 2, 3]]

            positive_examples = random.sample(shapes_that_have_the_dominant_color, num_dominant_objects)
            negative_examples = random.sample(shapes_that_have_the_minority_color, num_minority_objects)

            labels = [True]*num_dominant_objects + [False]*num_minority_objects

            examples.append( list(zip(labels, positive_examples+negative_examples)))
            random.shuffle(examples[-1])

            labels = np.array([l for l, _ in examples[-1]])

            hr = np.array(human_responses[200][i])

            learning_curve.append(hr * np.array(labels) + (1-np.array(labels)) * (1-hr) )
        return learning_curve, "(lambda (S) (= (color x)) (most-popular-color S))", examples

    if concept == 202:
        # most popular shape
        examples = []
        learning_curve = []
        import random
        for i in range(15):
            dominant_shape = ["triangle", "rectangle", "circle"][random.randint(0,2) if i>2 else i]
            num_dominant_objects = random.choice([3,4]) if random.random()>0.3 or i ==0 else random.choice([2,3,4])
            num_minority_objects = random.choice(list(range(1, min(3, num_dominant_objects))))
            minority_shape = random.choice([c for c in ["triangle", "rectangle", "circle"] if c != dominant_shape])

            shapes_that_have_the_dominant_shape = [(dominant_shape, color, sz) for color in ["yellow", "green", "blue"] for sz in [1, 2, 3]]
            shapes_that_have_the_minority_shape = [(minority_shape, color, sz) for color in ["yellow", "green", "blue"] for sz in [1, 2, 3]]

            positive_examples = random.sample(shapes_that_have_the_dominant_shape, num_dominant_objects)
            negative_examples = random.sample(shapes_that_have_the_minority_shape, num_minority_objects)

            labels = [True]*num_dominant_objects + [False]*num_minority_objects

            examples.append( list(zip(labels, positive_examples+negative_examples)))
            random.shuffle(examples[-1])

            labels = np.array([l for l, _ in examples[-1]])

            hr = np.array(human_responses[202][i])

            learning_curve.append(hr * np.array(labels) + (1-np.array(labels)) * (1-hr) )

        return learning_curve, "(lambda (S) (= (shape x)) (most-popular-shape S))", examples

    # now we do the least popular color and shape
    # these are the same as the most popular color and shape, but with the labels flipped
    if concept == 201:
        # least popular color
        examples = []
        learning_curve = []
        for i in range(15):
            minority_color = ["yellow", "green", "blue"][random.randint(0,2) if i>2 else i]
            num_dominant_objects = random.choice([3,4]) if random.random()>0.3 or i == 0 else random.choice([2,3,4])
            num_minority_objects = random.choice(list(range(1, min(3, num_dominant_objects))))
            dominant_color = random.choice([c for c in ["yellow", "green", "blue"] if c != minority_color])

            shapes_that_have_the_dominant_color = [(shape, dominant_color, sz) for shape in ["triangle", "rectangle", "circle"] for sz in [1, 2, 3]]
            shapes_that_have_the_minority_color = [(shape, minority_color, sz) for shape in ["triangle", "rectangle", "circle"] for sz in [1, 2, 3]]

            positive_examples = random.sample(shapes_that_have_the_dominant_color, num_dominant_objects)
            negative_examples = random.sample(shapes_that_have_the_minority_color, num_minority_objects)

            labels = [False]*num_dominant_objects + [True]*num_minority_objects

            examples.append( list(zip(labels, positive_examples+negative_examples)))

            random.shuffle(examples[-1])

            labels = np.array([l for l, _ in examples[-1]])

            hr = np.array(human_responses[201][i])

            learning_curve.append(hr * np.array(labels) + (1-np.array(labels)) * (1-hr) )

        return learning_curve, "(lambda (S) (= (color x)) (least-popular-color S))", examples

    if concept == 203:
        # least popular shape
        examples = []
        learning_curve = []
        for i in range(15):
            minority_shape = ["triangle", "rectangle", "circle"][random.randint(0,2) if i>2 else i]
            num_dominant_objects = random.choice([3,4]) if random.random()>0.3 or i == 0 else random.choice([2,3,4])
            num_minority_objects = random.choice(list(range(1, min(3, num_dominant_objects))))
            dominant_shape = random.choice([c for c in ["triangle", "rectangle", "circle"] if c != minority_shape])

            shapes_that_have_the_dominant_shape = [(dominant_shape, color, sz) for color in ["yellow", "green", "blue"] for sz in [1, 2, 3]]
            shapes_that_have_the_minority_shape = [(minority_shape, color, sz) for color in ["yellow", "green", "blue"] for sz in [1, 2, 3]]

            positive_examples = random.sample(shapes_that_have_the_dominant_shape, num_dominant_objects)
            negative_examples = random.sample(shapes_that_have_the_minority_shape, num_minority_objects)

            labels = [False]*num_dominant_objects + [True]*num_minority_objects

            examples.append( list(zip(labels, positive_examples+negative_examples)))
            random.shuffle(examples[-1])

            labels = np.array([l for l, _ in examples[-1]])

            hr = np.array(human_responses[203][i])

            learning_curve.append(hr * np.array(labels) + (1-np.array(labels)) * (1-hr) )

        return learning_curve, "(lambda (S) (= (shape x)) (least-popular-shape S))", examples



def get_learning_curve(concept, ordering):
    
    if concept >= 200:
        # these are special new concepts that Kevin added
        return special_concept(concept)

    os.system(f"mkdir -p set_data/precomputed_pickles")
    pickle_filename = f"set_data/precomputed_pickles/hg{concept:02}_L{ordering}.pickle"
    if os.path.exists(pickle_filename):
        with open(pickle_filename, "rb") as handle:
            return pickle.load(handle)       

    print("get_learning_curve", concept, ordering, "being called for the first time. sorry this is going to take a while but next time it will be cached") 

    data = pd.read_csv("set_data/TurkData-Accuracy.txt", sep="\t", index_col=False, low_memory=False)

    data = data[data["concept"] == f"hg{concept:02}"]
    
    data = data[data["list"] == f"L{ordering}"]

    curve = []

    n_sets = data["set.number"].max()

    subjects = data["subject"].unique()  

    # for each subject, get the average accuracy
    subject_accuracy = {}
    for subject in subjects:
        this_subject = data[data["subject"] == subject]
        subject_accuracy[subject] = (this_subject["response"] == this_subject["right.answer"]).mean()
    
    # remove the subjects with low accuracy, defined as <2 standard deviations below the mean
    import numpy as np
    mean = np.mean(list(subject_accuracy.values()))
    std = np.std(list(subject_accuracy.values()))
    
    bad_accuracies = [accuracy for subject, accuracy in subject_accuracy.items() if accuracy <= mean - 2*std]
    bad_subjects = [subject for subject, accuracy in subject_accuracy.items() if accuracy <= mean - 2*std]

    # remove the subjects with low accuracy, defined as the bottom third in terms of accuracy
    # bad_subjects = [subject for subject, accuracy in subject_accuracy.items() if accuracy <= np.percentile(list(subject_accuracy.values()), 33)]

    # also remove any subjects who completed fewer than 5 sets
    for subject in subjects:
        this_subject = data[data["subject"] == subject]
        if len(this_subject["set.number"].unique()) < 5:
            bad_subjects.append(subject)

    for set_number in range(1, n_sets+1):
        this_set = data[data["set.number"] == set_number]
        this_set = this_set[~this_set["subject"].isin(bad_subjects)]
        # n_responses = this_set["response.number"].max()
        # for response in range(1, n_responses+1):
        #     this_response = this_set[this_set["response.number"] == response]
        
        this_response = this_set

        n_probes = this_response["response.number"].max()
                
        probe_accuracy=[]
        for probe in range(1, n_probes+1):
            this_probe = this_response[this_response["response.number"] == probe]
            probe_accuracy.append((this_probe["response"] == this_probe["right.answer"]).mean())
        # if True:
        #     accuracy = (this_response["response"] == this_response["right.answer"]).mean()
        
        curve.append(probe_accuracy)

    # and now we load the actual examples, and the concept
    with open(f"set_data/concepts/CONCEPT_hg{concept:02}__LIST_L{ordering}.txt") as handle:
        lines = handle.readlines()
    
    expression = lines[0].strip()
    examples = []
    for line_number, line in enumerate(lines[1:]):
        line = line.strip().split("\t")
        truth_values = [c == "t" for c in line[0] if c in "tf" ]
        objects = [ (features[0], features[1], int(features[2]))
                    for object_features in line[1:]
                    for features in [object_features.split(",")]]
        assert len(truth_values) == len(objects)
        assert line_number >= len(curve) or len(truth_values) == len(curve[line_number])
        examples.append(list(zip(truth_values, objects)))
    
    expression = expression.replace("*", "").replace("eqv", "=").replace("'", "").replace("(lambda (x) ", "")[:-1]

    with open(pickle_filename, "wb") as handle:
        pickle.dump((curve, expression, examples), handle)

    return curve, expression, examples

def easy_concepts(N):
    # compute top N easy higher order concepts, according to humans    
    averages = {}
    for n in range(1, 112+1):
        c1, expression, _ = get_learning_curve(n, 1)
        data = c1+get_learning_curve(n, 2)[0]
        # flatten data which is a list of lists
        data = [d for d_ in data for d in d_]
        averages[n] = np.mean(data)
    #sort them by average and print them in descending order
    print(list(sorted(averages.items(), key=lambda x: x[1], reverse=True)))
    easy = []
    for n, average in sorted(averages.items(), key=lambda x: x[1], reverse=True):
        easy.append(n)
        if len(easy) == N: break
    
    return easy

if __name__ == '__main__':
    # dump out the special concepts in this format
    # hg24 L1 10 1 False rectangle blue 2 1 21
    # hg24 L1 11 1 False circle yellow 3 0 22
    # hg24 L1 11 2 False rectangle blue 3 0 22
    # hg24 L1 12 1 False rectangle yellow 1 0 22
    for ll in [1,2]:
        for n in [200,201]:
            human_learning_curve, _, examples  = special_concept(n)
            
            for i, (ex, human_accuracies) in enumerate(zip(examples, human_learning_curve)):
                human_accuracies *= 14 # I know there were fourteen subjects

                for j in range(len(ex)):
                    flag, (shape, color, sz) = ex[j]
                    a = human_accuracies[j]
                    yes = int(a if flag else 14-a)
                    no = int(14-yes)
                    print(f"hg{n:02} L{ll} {i+1} {j+1} {flag} {shape} {color} {sz} {yes} {no}")

    assert False
    special_concepts = {}
    for n in [200,201,202,203]:
        examples = special_concept(n)[-1]
        examples=[ [ [flag, {"color":color, "shape":shape, "size":["small","medium","large"][sz-1]}] for flag, (shape, color, sz) in ex] for ex in examples]
        special_concepts[n]=examples
        print(n, examples)
    serialization = str(special_concepts).replace("True", "true").replace("False", "false").replace("'", '"')
    # dump to human_experiment_webpage/data.json
    with open("human_experiment_webpage/data.json", "w") as handle: 
        handle.write(serialization)
    print(serialization)
    assert False
    # load all of these set concepts
    import numpy as np

    averages = {}
    for n in range(26, 112+1):
        data = get_learning_curve(n, 1)[0]+get_learning_curve(n, 2)[0]
        # flatten data which is a list of lists
        data = [d for d_ in data for d in d_]
        print(f"Concept {n}: {np.mean(data)}")
        averages[n] = np.mean(data)
    # sort them by average and print them in descending order
    for n, average in sorted(averages.items(), key=lambda x: x[1], reverse=True):
        print(f"Concept {n}: {average}")
    assert False


    data = get_human_number_data( [30, 31, 33, 24, 21, 36, 39])
    import pdb; pdb.set_trace()

    y, expression, examples = get_learning_curve(19,1)
    print([len(ex) for ex in examples])
    print([len(_y) for _y in y])
    assert False
    import matplotlib.pyplot as plt
    plt.figure()
    
    # flatten y 
    y = [y__ for y_ in y for y__ in y_]
    plt.plot(range(len(y)), y)
    for j, ex in enumerate(examples[:10]):
        print(f"Example set {j}:")
        for flag, features in ex:
            print(features, "\t", flag)
        print()


    plt.title(expression)
    # set the y-limits to the range 0-1
    plt.ylim(0,1)
    plt.show()

    from proposal import propose_set_hypothesis
    for rule in propose_set_hypothesis(examples[:100], 1000, temperature=1):
        print(rule[0])

    print(expression)
