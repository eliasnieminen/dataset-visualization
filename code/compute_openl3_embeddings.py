from dynamic_percussion_dataset import DynamicPercussionDataset
from dpd_environment import DPDEnvironment

import time

env = DPDEnvironment("dataset_envs/env.json")

settings = [
    {
        'input_repr': 'mel256',
        'content_type': 'env',
        'embedding_size': 6144,
        'center': True,
        'hop_size': 0.1,
    },
]

counter = 1
len_set = len(settings)
for setting in settings:
    print(f"Processing {counter} / {len_set} settings.")

    dataset = DynamicPercussionDataset(audio_directory=env.get_audio_dir(),
                                       serialization_directory=env.get_serialization_dir())
    time_before = time.time()
    dataset.calculate_embeddings(input_repr=setting['input_repr'],
                                 content_type=setting['content_type'],
                                 embedding_size=setting['embedding_size'],
                                 center=setting['center'],
                                 hop_size=setting['hop_size'],
                                 batch_size=4)
    time_after = time.time()
    print(f"Elapsed time: {time_after - time_before} seconds")
    dataset.serialize_embeddings()
    counter += 1


