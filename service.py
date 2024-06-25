from __future__ import annotations

import typing as t
import traceback

import numpy as np
import bentoml


EMBEDDER_MODEL_ID = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL_ID = "BAAI/bge-reranker-base"

@bentoml.service(
    traffic={"timeout": 60},
    resources={"gpu": 1},
)
class ServiceEmbedder:

    def __init__(self) -> None:
        import torch
        from sentence_transformers import SentenceTransformer, models
        from FlagEmbedding import FlagModel, FlagReranker
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # self.embedder_model = SentenceTransformer(EMBEDDER_MODEL_ID, device=self.device)
        # print(f"Model '{EMBEDDER_MODEL_ID}' loaded on device: '{self.device}'.")
        
        self.assist_bi_encoder = FlagModel(
            EMBEDDER_MODEL_ID,
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
            use_fp16=False
        ) # Setting use_fp16 to True speeds up computation with a slight performance degradation
        self.assist_cross_encoder = FlagReranker(RERANKER_MODEL_ID, use_fp16=False) # Setting use_fp16 to True speeds up computation with a slight performance degradation
        print(f"Models loaded on device: '{self.device}' 🔥")
        

    @bentoml.api(batchable=True)
    def embedding_batch(
        self,
        sentences: t.List[str]
    ) -> np.ndarray:
        
        # return self.model.encode(sentences, normalize_embeddings=True, device=self.device)
        return self.assist_bi_encoder.encode(sentences)
    
    
    @bentoml.api(batchable=True)
    def rerank(
        self,
        data
    ) -> np.ndarray:
        
        query_doc_list = [[r['query'], r['passage']] for r in data]
        
        # return self.model.encode(sentences, normalize_embeddings=True, device=self.device)
        return self.assist_cross_encoder.compute_score(query_doc_list)