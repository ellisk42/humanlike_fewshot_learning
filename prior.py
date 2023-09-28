import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

SIZE = "350M" #"2B"
model = {}
def to_tokens_and_logprobs(input_texts, model_name):
    """https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/17"""

    global model
    if model_name not in model:
        model[model_name] = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    
    input_ids = tokenizer(input_texts, padding=True, return_tensors="pt").input_ids
    outputs = model[model_name](input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

    batch = []
    for input_sentence, input_probs in zip(input_ids, gen_probs):
        text_sequence = []
        for token, p in zip(input_sentence, input_probs):
            if token not in tokenizer.all_special_ids:
                text_sequence.append((tokenizer.decode(token), p.item()))
        batch.append(text_sequence)
    return batch

def simple_prior(texts):

    fn = f"./{SIZE}-prior.pickle"
    if os.path.exists(fn):
        with open(fn, "rb") as handle:
            cache = pickle.load(handle)
    else:
        cache = {}

    jobs = []

    for text in texts:
        if text not in cache:
            jobs.append(text)

    jobs = list(set(jobs))

    if jobs:

        batch_size = 8
        
        while jobs:
            next_jobs = jobs[:batch_size]
            jobs = jobs[batch_size:]
            result = to_tokens_and_logprobs(next_jobs, f"Salesforce/codegen-{SIZE}-mono")
            just_numbers = [ [n for _, n in stuff ] for stuff in result ]
            important_statistics = [ (sum(numbers), len(numbers))
                                     for numbers in just_numbers ]

            for stats, text in zip(important_statistics, next_jobs):
                cache[text] = stats

            
        with open(fn, "wb") as handle:
            pickle.dump(cache, handle)

    return [cache[text] for text in texts ]

nl_prior_cache = None
def nl_prior(texts, domain='number'):
    global nl_prior_cache

    fn = f"./{SIZE}-prior-nl.pickle"
    if nl_prior_cache is None:
        if os.path.exists(fn):
            with open(fn, "rb") as handle:
                nl_prior_cache = pickle.load(handle)
        else:
            nl_prior_cache = {}

    if domain == "shape":
        def remove_shape_prefix(nl):
            stupid_prefixes = ["Rule: ", "Something is positive if ", "Rule for Concept #4: Something is positive if "]
            for stupid_prefix in stupid_prefixes:
                if nl.startswith(stupid_prefix):
                    nl = nl[len(stupid_prefix):]
            return nl
            
        texts = [remove_shape_prefix(text) for text in texts]

    jobs = []

    if domain == 'number':
#         texts = [f"""# Here are a few example number concepts:
# # The number is even
# # The number is between 30 and 45
# # The number is a power of 3
# # The number is less than 10
# # The number is {text}""" for text in texts ]
#         prompt_size=45
        texts = [f"# Here is an example number concept:\n# The number is {text}\n" for text in texts ]
        prompt_size=12
    elif domain == 'shape':
        texts = [f"""# Here are some simple example shape concepts:
# 1. neither a triangle nor a green rectangle
# 2. not blue and large.
# 3. if it is large, then it must be yellow.
# 4. small and blue
# 5. either big or green.
# 6. {text}
""" for text in texts ]
        prompt_size=63

    for text in texts:
        if text not in nl_prior_cache:
            jobs.append(text)

    jobs = list(set(jobs))

    if jobs:

        batch_size = 8
        while jobs:
            print(len(jobs), "prior computations remaining...")
#             prefix=f"""# Here are some simple example shape concepts:
# # 1. neither a triangle nor a green rectangle
# # 2. not blue and large.
# # 3. if it is large, then it must be yellow.
# # 4. small and blue
# # 5. either big or green.
# # 6. """
#             print([j[len(prefix):] for j in jobs])

            next_jobs = jobs[:batch_size]
            jobs = jobs[batch_size:]
            result = to_tokens_and_logprobs(next_jobs, f"Salesforce/codegen-{SIZE}-mono")
            
            # the prompt is some number of tokens

            just_numbers = [ [n for _, n in stuff[prompt_size:] ] for stuff in result ]
            
            important_statistics = [ (sum(numbers), len(numbers))
                                     for numbers in just_numbers ]

            for stats, text in zip(important_statistics, next_jobs):
                nl_prior_cache[text] = stats

            
        with open(fn, "wb") as handle:
            pickle.dump(nl_prior_cache, handle)

    return [nl_prior_cache[text] for text in texts ]

def code_prior(texts):

    fn = f"./{SIZE}-prior-smart.pickle"
    if os.path.exists(fn):
        with open(fn, "rb") as handle:
            cache = pickle.load(handle)
    else:
        cache = {}

    jobs = []

    texts = [f"""# Python 3
# Let's think of a number concept.
# Write a python function that returns true if `num` belongs to this number concept.
def check_if_in_concept(num):
    return {text}\n""" for text in texts ]

    for text in texts:
        if text not in cache:
            jobs.append(text)

    jobs = list(set(jobs))

    if jobs:

        batch_size = 8
        while jobs:
            print(len(jobs), "prior computations remaining...")
            next_jobs = jobs[:batch_size]
            jobs = jobs[batch_size:]
            result = to_tokens_and_logprobs(next_jobs, f"Salesforce/codegen-{SIZE}-mono")

            # the prompt is 47 tokens            
            just_numbers = [ [n for _, n in stuff[47:] ] for stuff in result ]
            
            important_statistics = [ (sum(numbers), len(numbers))
                                     for numbers in just_numbers ]

            for stats, text in zip(important_statistics, next_jobs):
                cache[text] = stats

        with open(fn, "wb") as handle:
            pickle.dump(cache, handle)

    return [cache[text] for text in texts ]
    
    

