from sentence_transformers import SentenceTransformer
import os
import pickle

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def textfeatures(texts):

    texts = [text for text in texts]

    fn = f"textfeatures.pickle"
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
            result = model.encode(next_jobs)
            for stats, text in zip(result, next_jobs):
                cache[text] = stats

            
        with open(fn, "wb") as handle:
            pickle.dump(cache, handle)

    return [cache[text] for text in texts ]