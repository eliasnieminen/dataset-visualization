from pathlib import Path
from typing import Union, Tuple, Dict, List, Optional
import pickle

import numpy as np
import librosa as lbr


class OpenL3EmbeddingPackage:

    def __init__(self,
                 embeddings: Optional[np.ndarray] = None,
                 timestamps: Optional[np.ndarray] = None,
                 metadata: Optional[Dict] = None):

        self.embeddings = embeddings
        self.embedding_mean = None
        self.timestamps = timestamps
        self.metadata = metadata

        self.sample_class = None
        self.sample_id = None
        self.raw_audio_path = None
        self.original_sr = None

        if metadata is not None:
            self.set_metadata(metadata)

    def set_embeddings(self, embeddings: np.ndarray):
        self.embeddings = embeddings

    def set_embedding_mean(self, embedding_mean: np.ndarray):
        self.embedding_mean = embedding_mean

    def set_timestamps(self, timestamps: np.ndarray):
        self.timestamps = timestamps

    def set_metadata(self, metadata):
        self.sample_class = metadata['class']
        self.sample_id = metadata['sample_id']
        self.raw_audio_path = metadata['raw_audio_path']
        self.original_sr = metadata['original_sr']

    def set_sample_class(self, sample_class: str):
        self.sample_class = sample_class

    def set_sample_id(self, sample_id: str):
        self.sample_id = sample_id

    def set_raw_audio_path(self, raw_audio_path: Union[str, Path]):
        self.raw_audio_path = raw_audio_path

    def set_original_sr(self, original_sr: int):
        self.original_sr = original_sr

    def get_audio(self) -> Union[Tuple[np.ndarray, int], None]:
        if self.raw_audio_path is not None:
            return lbr.load(self.raw_audio_path, sr=self.original_sr)
        else:
            raise ValueError("raw_audio_path not specified")

    def serialize(self, serialization_directory: Union[str, Path]):
        ser_dir = Path(serialization_directory).resolve()
        to_serialize = {
            "embeddings": self.embeddings,
            "embedding_mean": self.embedding_mean,
            "timestamps": self.timestamps,
            "metadata": self.metadata,
        }

        filename = "openl3_emb_" + self.sample_id + ".pickle"

        full_serialization_path = str(Path(ser_dir/filename).resolve())
        serialization_file = open(full_serialization_path, 'wb')
        pickle.dump(to_serialize, serialization_file,
                    pickle.HIGHEST_PROTOCOL)

        serialization_file.close()


class OpenL3EmbeddingPackageWrapper:

    def __init__(self,
                 embedding_packages: Optional[Dict[
                     str, List[OpenL3EmbeddingPackage]]] = None):
        self.embedding_packages = embedding_packages

    def add_package(self, package: OpenL3EmbeddingPackage):
        sample_class = package.sample_class

        if self.embedding_packages is None:
            self.embedding_packages = {}

        if sample_class in list(self.embedding_packages.keys()):
            self.embedding_packages[sample_class].append(package)
        else:
            self.embedding_packages[sample_class] = [package]

    def get_class_labels(self) -> List:
        return list(self.embedding_packages.keys())

    def __len__(self) -> int:
        if self.embedding_packages is not None:
            return len(self.embedding_packages.keys())
        else:
            return 0

    def __getitem__(self,
                    item: Union[str, int]) \
                    -> Union[List[OpenL3EmbeddingPackage],
                                  OpenL3EmbeddingPackage]:

        if isinstance(item, str):
            return self.embedding_packages[item]
        elif isinstance(item, int):
            sample_class = list(self.embedding_packages.keys())[item]
            return self.embedding_packages[sample_class]
        else:
            raise ValueError("The type of 'item' must be either string or "
                             "integer")

    def get_package_by_id(self, sample_id: str) -> OpenL3EmbeddingPackage:
        sample_class = "_".join(sample_id.split("_")[0:-1])
        for sample in self.embedding_packages[sample_class]:
            if sample.sample_id == sample_id:
                return sample

    def serialize_embeddings(self, serialization_directory: str):
        ser_dir = Path(serialization_directory).resolve()
        for sample_class in list(self.embedding_packages.keys()):
            packages = self.embedding_packages[sample_class]
            for package in packages:
                package.serialize(ser_dir)
