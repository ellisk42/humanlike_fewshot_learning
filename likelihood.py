import os
import pickle
from subprocess import STDOUT, check_output
import math

from proposal import sample_completions, shape_examples_prompt, higher_order_examples_text
from utilities import logsumexp

def execution_likelihood(h, evidence, epsilon=None):

    support = compute_support(h)

    if not isinstance(evidence, list):
        evidence = [evidence]

    L = 0.
    for n in evidence:
        if n in support:
            if epsilon is None:
                L -= math.log(len(support))
            else:
                L += logsumexp([math.log(epsilon) - math.log(100), 
                                math.log(1-epsilon) - math.log(len(support))])
        else:
            if epsilon is None:
                return float("-inf")
            else:
                L += (math.log(epsilon) - math.log(100))
            
    return L

def save_likelihood_cache():
    global SUPPORT_CACHE
    if SUPPORT_CACHE is None: return 
    with open("support.pickle", "wb") as handle:
        pickle.dump(SUPPORT_CACHE, handle)

        
SUPPORT_CACHE=None
        
def compute_support(h):
    global SUPPORT_CACHE

    if SUPPORT_CACHE is None:
        SUPPORT_CACHE={}
        if os.path.exists("support.pickle"):
            with open("support.pickle", "rb") as handle:
                SUPPORT_CACHE = pickle.load(handle)
            print("Loaded", len(SUPPORT_CACHE), "support calculations")
        else:
            print("cannot find cashed support calculations")

    if h in SUPPORT_CACHE: return SUPPORT_CACHE[h]

    if "#" in h:
        h = h[:h.index("#")]
    try:
        support = check_output(["python", "-c", f"import math; print([{h} for num in range(100+1) ])"], stderr=STDOUT, timeout=0.1)
        support = eval(support)
        support = [n for n, flagged in enumerate(support) if flagged]
    except:
        support = []

    SUPPORT_CACHE[h] = support

    return support



def nl_likelihood(h, n):
    """for each number 1..100, returns a probability [0,1]"""

    if isinstance(n, list):
        total_likelihood=1
        for x in n:
            total_likelihood*=nl_likelihood(h, x)
        return total_likelihood
    
    assert isinstance(n, int)
    
    completions = sample_completions(f"""Question: Regarding the number {n}, is it {h}?
Answer (one word, yes/no): """, 30, max_tokens=2, engine="gpt-3.5-turbo",)
    
    yeses = sum( text.startswith("y") or text.startswith("Y")
                 for text, _, _ in completions )
    nos = sum( text.startswith("n") or text.startswith("N")
                 for text, _, _ in completions )

    if yeses == 0 or (yeses == 0 and nos == 0):
        value=float("-inf")
    else:
        value=math.log(yeses/(nos+yeses))

    return value
        


def marginal_lm_likelihood(examples, probe, n=30, temperature=1, domain="number"):
    if domain == "number":

        # now we do this with gp4
        completions = sample_completions(
            engine="gpt-4",
            prompt=f"""Here are a few example number concepts:
-- The number is even
-- The number is between 30 and 45
-- The number is a power of 3
-- The number is less than 10

Here are some random examples of numbers belonging to a possibly different number concept:
{', '.join(map(str,examples))}

Question: Does the number {probe} belong to the same concept as the above numbers?
Answer (one word, yes/no):""",
            temperature=temperature,   
            n=10,
            stop="\n",
            max_tokens=2)

#         completions = sample_completions(
#         engine="code-cushman-001",
#         prompt=f"""# Python 3
# # Here are a few example number concepts:
# # -- The number is even
# # -- The number is between 30 and 45
# # -- The number is a power of 3
# # -- The number is less than 10
# # 
# # Here are some random examples of numbers belonging to a different number concept:
# # {', '.join(map(str,examples))}
# # Question: Does the number {probe} belong to the same concept as the above numbers?
# # Answer: """,
#         temperature=1,
#         n=30,
#         stop="\n",
#         max_tokens=2
#         )
    elif domain == "shapes":
        def verbalize_features(f):
            shape, color, size = f
            size = ["small", "medium", "large"][size-1]
            return f"{size} {color} {shape}"
        completions = []
        for examples in shape_examples_prompt(examples):
            prompt = f"""Here are some example concepts defined by a logical rule:

Rule: a triangle.
Rule: a green rectangle.
Rule: big or a rectangle (unless that rectangle is blue).
Rule: not both big and green.
Rule: either big or green, but not both.
Rule: either a rectangle or not yellow.
Rule: a circle.

Now please look at the following examples for a new logical rule.

{examples}

Question: Based on the above examples, is a {verbalize_features(probe)} in the concept?
Answer (one word, just write yes/no):"""

            completions.extend(sample_completions(prompt=prompt,
                                    n=5,
                                    engine="gpt-4", #, #"code-davinci-002",#"gpt-3-turbo",
                                    temperature=temperature,
                                    stop="\n",
                                    max_tokens=2))
    elif domain == "higher_order_shapes":
        def verbalize_features(f):
            shape, color, size = f
            size = ["small", "medium", "large"][size-1]
            return f"({size} {color} {shape})"

        all_the_probes = " ".join(map(verbalize_features, probe))
        probe_probabilities = []

        examples = higher_order_examples_text(examples)
        for this_probe in probe:
            prompt = f"""Here are some example concepts defined by a logical rule:

Rule for Concept #1: Something is positive if it is the biggest yellow object in the example
Rule for Concept #2: Something is positive if there is another object with the same color in the example
Rule for Concept #3: Something is positive if it is the same color as the smallest triangle in the example

Now please look at the following examples for a new logical rule.

{examples}

Now we get a new collection of examples for Concept #4:
{all_the_probes}
Question: Based on the above example, is a {verbalize_features(this_probe)} in the concept?
Answer (one word, just write yes/no):"""

            print(prompt)

            completions = sample_completions(prompt=prompt,
                                    n=10,
                                    engine="gpt-4", #, #"code-davinci-002",#"gpt-3-turbo",
                                    temperature=1, 
                                    stop="\n",
                                    max_tokens=2)
            yeses = sum( text.lower().strip().startswith("y") for text, _, _ in completions )
            nos = sum( text.lower().strip().startswith("n") for text, _, _ in completions )
            if yeses == 0 or (yeses == 0 and nos == 0):
                probe_probabilities.append(float("-inf"))
            else:
                probe_probabilities.append(math.log(yeses/(nos+yeses)))
        return probe_probabilities

    yeses = sum( text.strip().startswith("y") or text.strip().startswith("Y")
                 for text, _, _ in completions )
    nos = sum( text.strip().startswith("n") or text.strip().startswith("N")
                 for text, _, _ in completions )

    if yeses == 0 or (yeses == 0 and nos == 0):
        return float("-inf")

    #print(f"P_gpt4({probe} | {examples}) = {yeses/(nos+yeses)})")    

    return math.log(yeses/(nos+yeses))

def propose_next_number(examples, n, temperature=1):
    # https://dspace.mit.edu/bitstream/handle/1721.1/16714/42471842-MIT.pdf?sequence=2&isAllowed=y
    completions = sample_completions(
        engine="code-davinci-002",
        prompt=f"""# Python 3
# Here are a few example number concepts:
# -- The number is even
# -- The number is between 30 and 45
# -- The number is a power of 3
# -- The number is less than 10
# 
# Here are some random examples of numbers belonging to a different number concept:
# {', '.join(map(str,examples))},""",
        temperature=1,
        n=n,
        stop=",",
        max_tokens=5
    )
    histogram = {}
    total=0
    for c,_,_ in completions:
        try:
            v = int(c.strip().split()[0])
        except: continue
        histogram[v] = 1+histogram.get(v,0)
        total+=1

    return [ math.log(histogram[v]/total) if v in histogram else float("-inf")
             for v in range(101) ]
    


def transpiler_likelihood(nl, evidence, epsilon=None):
    completion = sample_completions(
        engine="code-davinci-002",
        prompt=f"""# Write a python function to check if a number is {nl}.
def check_number(num):
    return""",
        temperature=0,
        n=1,
        stop="\n",
        max_tokens=128
    )[0][0]
    
    if isinstance(evidence, list):
        return execution_likelihood(completion, evidence, epsilon=epsilon)
    else:
        return execution_likelihood(completion, [evidence], epsilon=epsilon)
    


    
