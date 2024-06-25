from __future__ import annotations

import typing as t
import traceback

import numpy as np
import bentoml


NLU_EMBEDDER_MODEL_ID = "bespin-global/klue-sroberta-base-continue-learning-by-mnr"
ASSIST_EMBEDDER_MODEL_ID = "BAAI/bge-base-en-v1.5"
ASSIST_RERANKER_MODEL_ID = "BAAI/bge-reranker-base"

@bentoml.service(
    traffic={"timeout": 60},
    resources={"gpu": 1},
)
class ServiceEmbedder:

    def __init__(self) -> None:
        import torch
        from sentence_transformers import SentenceTransformer
        from FlagEmbedding import FlagModel, FlagReranker
        
        # Check device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load models
        self.assist_embedder = FlagModel(
            ASSIST_EMBEDDER_MODEL_ID,
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
            use_fp16=False
        ) 
        self.assist_reranker = FlagReranker(ASSIST_RERANKER_MODEL_ID, use_fp16=False)
        self.nlu_embedder =  SentenceTransformer(NLU_EMBEDDER_MODEL_ID, device=self.device)
        print(f"Models loaded on device: '{self.device}' 🔥")
        
        
    @bentoml.api(route="/api/nlu/sentence-embedding", batchable=True)
    def nlu_embedd(
        self,
        query: str
    ) -> np.ndarray:
        
        return self.nlu_embedder.encode(query, device=self.device)


    @bentoml.api(route="/api/nlu/sentence-embedding-batch", batchable=True)
    def nlu_embedd_batch(
        self,
        data
    ) -> np.ndarray:
        query_list = [r['text'] for r in data]
        
        return self.nlu_embedder.encode(query_list, device=self.device)
    
    
    @bentoml.api(route="/api/assist/sentence-embedding", batchable=True)
    def assist_embedd(
        self,
        query: str
    ) -> np.ndarray:
        
        return self.assist_embedder.encode_queries(query) # query_instruction_for_retrieval + query


    @bentoml.api(route="/api/assist/sentence-embedding-batch", batchable=True)
    def assist_embedd_batch(
        self,
        data: t.List[str]
    ) -> np.ndarray:
        
        return self.assist_embedder.encode(data) 
    
    
    @bentoml.api(route="/api/assist/cross-encoder/similarity-scores", batchable=True)
    def assist_rerank(
        self,
        data
    ) -> np.ndarray:
        
        query_doc_list = [[r['query'], r['passage']] for r in data]
        
        return self.assist_reranker.compute_score(query_doc_list)