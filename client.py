import numpy as np
from pytriton.client import ModelClient

sequence = np.array(
    [
        ["안녕하세요"]
    ]
)
sequence = np.char.encode(sequence, "utf-8")
print(f"Sequence: {sequence}")

with ModelClient("localhost", "bb8-embedder-nlu", init_timeout_s=600.0) as nlu_client:
    result_dict = nlu_client.infer_batch(sequence)
    embed_vector = [row.astype(float) for row in result_dict['embed_vectors']]
    print(embed_vector)






sequence = np.array(
    [
        ["john is my friends"],["hello"]
    ]
)
sequence = np.char.encode(sequence, "utf-8")
print(f"Sequence: {sequence}")


with ModelClient("34.64.251.218:8000", "bb8-embedder-assist", init_timeout_s=600.0) as assist_client:
    result_dict = assist_client.infer_batch(sequence)
    embed_vector = np.array([row.astype(float) for row in result_dict['embed_vectors']])
    print(embed_vector)


