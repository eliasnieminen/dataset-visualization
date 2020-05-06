import pathlib
from pathlib import Path
import pickle
import os
from typing import *
from dataobjects import OpenL3EmbeddingPackage, OpenL3EmbeddingPackageWrapper

def get_root_directory(location: str) -> Union[None, Path]:
    """
    Returns the root directory of the project in the current work environment.
    This is useful when working from multiple locations.
    :param location: Location of the work station
    :return: The root directory of the project as Path object, and None
             if the location is not recognized
    """
    if location == 'hervanta':
        return Path('P:/StudentDocuments/Documents/kandi2020/')
    elif location == 'linux-desktop':
        return Path('/~/code/kandi2020/')
    elif location == 'nlk':
        return Path('D:/kandi2020/')
    else:
        return None


def get_files_from_directory(directory: str) -> List[Path]:
    """
    :param directory: The directory of the files that need to be listed
    :return: List of Path objects of the files in the directory
    """
    return list(Path(directory).iterdir())


def get_sample_files_and_classes(playlist_path: Path,
                                 root_dir: Path) -> Dict:
    """
    :param playlist_path: The path to the playlist file that contains paths to
           sample files
    :return: A dictionary with class names as keys and lists of full paths to
             sample files as payloads
    """

    sample_files = {}

    # The playlist file has one path per row, each row pointing to a single
    # sample file
    playlist_file = open(str(playlist_path), 'r')

    for row in playlist_file:
        row = row.strip()
        file_name = row.split(os.sep)[-1]
        full_path = (root_dir/row).resolve()
        sample_class = file_name.split('_')[1]
        if sample_class in sample_files.keys():
            sample_files[sample_class].append(str(full_path))
        else:
            sample_files[sample_class] = [str(full_path)]

    playlist_file.close()

    return sample_files


def serialize_features(embeddings_timestamps: Dict,
                       serialization_dir: Path) -> None:

    for sample_class in embeddings_timestamps.keys():

        print(f'Serializing {sample_class} samples...')
        counter = 0

        embeddings, timestamps, metadata = embeddings_timestamps[sample_class]

        num_samples = len(embeddings)

        for i in range(num_samples):

            if counter % 10 == 0:
                print(f'{counter} / {num_samples}')

            to_serialize = {'embeddings': embeddings[i],
                            'timestamps': timestamps[i],
                            'metadata': metadata[i]}

            serialization_filename = 'emb_openl3_' + str(sample_class) +\
                                     '_' + str(counter) + '.pickle'

            serialization_path = (serialization_dir/
                                  serialization_filename).resolve()

            serialization_file = open(str(serialization_path), 'wb')

            pickle.dump(to_serialize, serialization_file,
                        pickle.HIGHEST_PROTOCOL)

            serialization_file.close()
            counter += 1


def unserialize_embeddings(serialization_directory: Union[str, Path],
                           class_filter: Optional[List] = None) \
                           -> OpenL3EmbeddingPackageWrapper:
    """

    :param serialization_directory:
    :return:
    """

    serialization_directory = Path(serialization_directory).resolve()
    openl3_package_wrapper = OpenL3EmbeddingPackageWrapper()

    for file in serialization_directory.iterdir():
        if file.is_file():

            pickle_file = open(str(file), 'rb')
            sample_embeddings_timestamps = pickle.load(pickle_file)
            pickle_file.close()

            embeddings = sample_embeddings_timestamps['embeddings']
            embedding_mean = sample_embeddings_timestamps['embedding_mean']
            timestamps = sample_embeddings_timestamps['timestamps']
            metadata = sample_embeddings_timestamps['metadata']

            sample_class = metadata["class"]

            add = False

            if class_filter is not None and sample_class in class_filter:
                add = True
            elif class_filter is None:
                add = True
            else:
                add = False

            if add:
                openl3_package = OpenL3EmbeddingPackage(embeddings=embeddings,
                                                        timestamps=timestamps,
                                                        metadata=metadata)
                openl3_package.set_embedding_mean(embedding_mean)
                openl3_package_wrapper.add_package(package=openl3_package)

    return openl3_package_wrapper

