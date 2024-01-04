import argparse
import logging

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from FlagEmbedding import FlagModel, FlagReranker

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

from _config import logger

logger.info("🔥 bb8-embedder by Triton Inferece Server")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
nlu_embedder = SentenceTransformer('bespin-global/klue-sroberta-base-continue-learning-by-mnr', device=device)
assist_bi_encoder = FlagModel('BAAI/bge-base-en-v1.5', 
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
            use_fp16=False) # Setting use_fp16 to True speeds up computation with a slight performance degradation
assist_cross_encoder = FlagReranker('BAAI/bge-reranker-base', use_fp16=False) # Setting use_fp16 to True speeds up computation with a slight performance degradation


@batch
def _infer_fn_nlu(sequence: np.ndarray):
    sequence = np.char.decode(sequence.astype("bytes"), "utf-8")  # need to convert dtype=object to bytes first
    sequence = sum(sequence.tolist(), [])

    embed_vectors = nlu_embedder.encode(sequence)

    return {'embed_vectors': embed_vectors}

@batch
def _infer_fn_assist_biencoder_query(sequence: np.ndarray):
    sequence = np.char.decode(sequence.astype("bytes"), "utf-8")  # need to convert dtype=object to bytes first
    sequence = sum(sequence.tolist(), [])

    embed_vectors = assist_bi_encoder.encode_queries(sequence)

    return {'embed_vectors': embed_vectors}


@batch
def _infer_fn_assist_biencoder_passage(sequence: np.ndarray):
    sequence = np.char.decode(sequence.astype("bytes"), "utf-8")  # need to convert dtype=object to bytes first
    sequence = sum(sequence.tolist(), [])

    embed_vectors = assist_bi_encoder.encode(sequence)

    return {'embed_vectors': embed_vectors}

@batch
def _infer_fn_assist_crossencoder(queries: np.ndarray, passages:np.ndarray):
    queries = np.char.decode(queries.astype("bytes"), "utf-8")  # need to convert dtype=object to bytes first
    passages = np.char.decode(passages.astype("bytes"), "utf-8")  # need to convert dtype=object to bytes first
    queries = sum(queries.tolist(), [])
    passages = sum(passages.tolist(), [])

    query_passage_list = [[query, passage] for query, passage in zip(queries, passages)]
    similarity_scores = assist_cross_encoder.compute_score(query_passage_list)

    return {'similarity_scores': similarity_scores}



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=10000,
        help="Batch size of request.",
        required=False,
    )
    args = parser.parse_args()


    with Triton() as triton:
        logger.info("Loading embedding model.")
        triton.bind(
            model_name="bb8-embedder-nlu",
            infer_func=_infer_fn_nlu,
            inputs=[
                Tensor(name="sequence", dtype=bytes, shape=(1,)),
            ],
            outputs=[
                Tensor(name="embed_vectors", dtype=bytes, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=args.max_batch_size),
        )
        triton.bind(
            model_name="bb8-embedder-assist-biencoder-query",
            infer_func=_infer_fn_assist_biencoder_query,
            inputs=[
                Tensor(name="sequence", dtype=bytes, shape=(1,)),
            ],
            outputs=[
                Tensor(name="embed_vectors", dtype=bytes, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=args.max_batch_size),
        )
        triton.bind(
            model_name="bb8-embedder-assist-biencoder-passage",
            infer_func=_infer_fn_assist_biencoder_passage,
            inputs=[
                Tensor(name="sequence", dtype=bytes, shape=(1,)),
            ],
            outputs=[
                Tensor(name="embed_vectors", dtype=bytes, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=args.max_batch_size),
        )
        triton.bind(
            model_name="bb8-embedder-assist-crossencoder",
            infer_func=_infer_fn_assist_crossencoder,
            inputs=[
                Tensor(name="queries", dtype=bytes, shape=(1,)),
                Tensor(name="passages", dtype=bytes, shape=(1,)),
            ],
            outputs=[
                Tensor(name="similarity_scores", dtype=bytes, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=args.max_batch_size),
        )
        logger.info("Serving inference")
        triton.serve()


if __name__ == "__main__":
    main()