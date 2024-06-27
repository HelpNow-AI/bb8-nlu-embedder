import argparse
import logging
from pathlib import Path
import gc

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, CrossEncoder
# from FlagEmbedding import FlagModel, FlagReranker

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor, DeviceKind
from pytriton.model_config.triton_model_config import TritonModelConfig
from pytriton.model_config.parser import ModelConfigParser
from pytriton.triton import Triton, TritonConfig

from _config import logger

import gzip
import base64

logger.info("🔥 bb8-embedder by Triton Inferece Server")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    gpu_properties = torch.cuda.get_device_properties(0)

def check_cuda_memory():
    if device.type == "cuda":
        current_memory = round(torch.cuda.memory_allocated() / (1024**3), 4)
        total_memory = round(gpu_properties.total_memory / (1024**3), 4)
        print(f'>> Usage of Current Memory: {current_memory} GB / {total_memory} GB')
    
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print('>> Not using CUDA.')
check_cuda_memory()


# Load models
nlu_embedder = SentenceTransformer('bespin-global/klue-sroberta-base-continue-learning-by-mnr', device=device)
# assist_bi_encoder = FlagModel('BAAI/bge-base-en-v1.5', 
#             query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
#             use_fp16=False) # Setting use_fp16 to True speeds up computation with a slight performance degradation
# assist_cross_encoder = FlagReranker('BAAI/bge-reranker-base', use_fp16=False) # Setting use_fp16 to True speeds up computation with a slight performance degradation

assist_bi_encoder = SentenceTransformer('BAAI/bge-base-en-v1.5', device=device)

assist_cross_encoder_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-base')
assist_cross_encoder = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base')
assist_cross_encoder.to(device)
assist_cross_encoder.eval()



@batch
def _infer_fn_nlu(sequence: np.ndarray):
    print(sequence)
    sequence = [gzip.decompress(s).decode("utf-8") for s in sequence]
    print(sequence)
    sequence = np.char.decode(sequence.astype("bytes"), "utf-8")  # need to convert dtype=object to bytes first
    sequence = sum(sequence.tolist(), [])

    embed_vectors = nlu_embedder.encode(sequence, device=device)

    return {'embed_vectors': embed_vectors}

@batch
def _infer_fn_assist_biencoder_query(sequence: np.ndarray):
    sequence = np.char.decode(sequence.astype("bytes"), "utf-8")  # need to convert dtype=object to bytes first
    sequence = sum(sequence.tolist(), [])

    new_sequence = []
    for s in sequence:
        s = base64.b64decode(s.encode("utf-8"))
        s = gzip.decompress(s)
        new_sequence.append(s.decode("utf-8"))

    # embed_vectors = assist_bi_encoder.encode_queries(sequence)
    instruction = "Represent this sentence for searching relevant passages: "
    embed_vectors = assist_bi_encoder.encode([instruction+q for q in new_sequence], normalize_embeddings=True, device=device)

    return {'embed_vectors': embed_vectors}


# @batch
# def _infer_fn_assist_biencoder_passage(sequence: np.ndarray):
#     sequence = np.char.decode(sequence.astype("bytes"), "utf-8")  # need to convert dtype=object to bytes first
#     sequence = sum(sequence.tolist(), [])
#
#     # embed_vectors = assist_bi_encoder.encode(sequence)
#     embed_vectors = assist_bi_encoder.encode(sequence, normalize_embeddings=True, device=device)
    

    # return {'embed_vectors': embed_vectors}

@batch
def _infer_fn_assist_crossencoder(queries: np.ndarray, passages:np.ndarray):
    queries = np.char.decode(queries.astype("bytes"), "utf-8")  # need to convert dtype=object to bytes first
    passages = np.char.decode(passages.astype("bytes"), "utf-8")  # need to convert dtype=object to bytes first
    queries = sum(queries.tolist(), [])
    passages = sum(passages.tolist(), [])

    query_passage_list = [[query, passage] for query, passage in zip(queries, passages)]
    # similarity_scores = assist_cross_encoder.compute_score(query_passage_list)
    
    with torch.no_grad():
        inputs = assist_cross_encoder_tokenizer(query_passage_list, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = inputs.to(device)
        similarity_scores = assist_cross_encoder(**inputs, return_dict=True).logits

    return {'similarity_scores': similarity_scores.cpu().numpy()}



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=128,
        help="Batch size of request.",
        required=False,
    )
    args = parser.parse_args()

    from typing import Dict, Optional
    from pytriton.model_config.common import QueuePolicy


    # class DynamicBatcher:
    #     max_queue_delay_microseconds: int = 100
    #     preferred_batch_size: Optional[list] = [128,512,1024]
    #     preserve_ordering: bool = False
    #     priority_levels: int = 0
    #     default_priority_level: int = 0
    #     default_queue_policy: Optional[QueuePolicy] = None
    #     priority_queue_policy: Optional[Dict[int, QueuePolicy]] = None
    #
    # class ModelConfig:
    #     batching: bool = True
    #     max_batch_size: int = 1024
    #     batcher: DynamicBatcher = DynamicBatcher()
    #     response_cache: bool = False


    with Triton(config= TritonConfig(allow_gpu_metrics=True)) as triton:
        logger.info("Loading embedding model.")
        triton.bind(
            model_name="bb8-embedder-nlu",
            infer_func=_infer_fn_nlu,
            inputs=[
                Tensor(name="sequence", dtype=bytes, shape=(-1,)),
            ],
            outputs=[
                Tensor(name="embed_vectors", dtype=np.float32, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=args.max_batch_size),
            #config=ModelConfigParser.from_file(config_path=Path('./model_config/bb8-embedder-nlu.pbtxt')),
            strict=True
        )
        triton.bind(
            model_name="bb8-embedder-nlu-batch",
            infer_func=_infer_fn_nlu,
            inputs=[
                Tensor(name="sequence", dtype=bytes, shape=(-1,)),
            ],
            outputs=[
                Tensor(name="embed_vectors", dtype=np.float32, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=args.max_batch_size),
            # config=ModelConfigParser.from_file(config_path=Path('./model_config/bb8-embedder-nlu.pbtxt')),
            strict=True
        )
        triton.bind(
            model_name="bb8-embedder-assist-biencoder",
            infer_func=_infer_fn_assist_biencoder_query,
            inputs=[
                
                Tensor(name="sequence", dtype=bytes, shape=(-1,)),
            ],
            outputs=[
                Tensor(name="embed_vectors", dtype=np.float32, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=args.max_batch_size),
            #config=ModelConfigParser.from_file(config_path=Path('./model_config/bb8-embedder-assist-biencoder-query.pbtxt')),
            strict=True
        )

        triton.bind(
            model_name="bb8-embedder-assist-biencoder-batch",
            infer_func=_infer_fn_assist_biencoder_query,
            inputs=[
                Tensor(name="sequence", dtype=bytes, shape=(-1,)),
            ],
            outputs=[
                Tensor(name="embed_vectors", dtype=np.float32, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=args.max_batch_size),
            #config=ModelConfigParser.from_file(config_path=Path('./model_config/bb8-embedder-assist-biencoder-passage.pbtxt')),
            strict=True
        )
        triton.bind(
            model_name="bb8-embedder-assist-crossencoder",
            infer_func=_infer_fn_assist_crossencoder,
            inputs=[
                Tensor(name="queries", dtype=bytes, shape=(-1,)),
                Tensor(name="passages", dtype=bytes, shape=(-1,)),
            ],
            outputs=[
                Tensor(name="similarity_scores", dtype=np.float32, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=args.max_batch_size),
            #config=ModelConfigParser.from_file(config_path=Path('./model_config/bb8-embedder-assist-crossencoder.pbtxt')),
            strict=True
        )
        logger.info("Serving inference")
        triton.serve()


if __name__ == "__main__":
    main()