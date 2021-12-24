from typing import Optional
import sys
sys.path.append("..") 
from fastapi import FastAPI
from transformers_encoder import TransformersEncoder
from vectorspace import SensesVSM
import numpy as np


class LMMS:
    def __init__(self) -> None:
        #LMMS initializaiton.
        # NLM/LMMS paths and parameters
        # vecs_path = '/media/dan/ElementsWORK/-xxlarge-v2/albertlmms-sp-wsd.albert-xxlarge-v2.vectors.txt'
        vecs_path = '../data/vectors/lmms-sp-wsd.albert-xxlarge-v2.vectors.txt'
        wsd_encoder_cfg = {
            'model_name_or_path': 'albert-xxlarge-v2',
            'min_seq_len': 0,
            'max_seq_len': 512,
            'layers': [-n for n in range(1, 12 + 1)],  # all layers, with reversed indices
            'layer_op': 'ws',
            'weights_path': '../data/weights/lmms-sp-wsd.albert-xxlarge-v2.weights.txt',
            'subword_op': 'mean'
        }
        
        print('Loading NLM and sense embeddings ...')  # (takes a while)
        self.wsd_encoder = TransformersEncoder(wsd_encoder_cfg)
        self.senses_vsm = SensesVSM(vecs_path, normalize=True)
        print("Loading NLM and sense embeddings successfully!")
    def process_sentence(self, sent, event_target_idxs):
        target_lemma = '_'.join([sent[i].lemma_ for i in event_target_idxs])
        target_pos = [doc[event_target_idxs[index]].pos_ for index in range(0, len(event_target_idxs))]
        tokens = [t.text for t in sent]
        ctx_embeddings = self.wsd_encoder.token_embeddings([tokens])[0]
        target_embedding = np.array([ctx_embeddings[i][1] for i in event_target_idxs]).mean(axis=0)
        target_embedding = target_embedding / np.linalg.norm(target_embedding)
        matches = self.senses_vsm.match_senses(target_embedding, lemma=target_lemma, postag=target_pos, topn=5)
        the_best_match, the_best_sim = matches[0]
        return the_best_match, the_best_sim


app = FastAPI()
LMMS_dis = LMMS()



@app.get("/LMMS/")
def LMMS_disambiguation(sent, event_target_idxs):
        the_best_match, the_best_sim = LMMS_dis.process_sentence(sent, event_target_idxs)
        return {"the_best_match": the_best_match, "the_best_sim": the_best_sim}

# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}