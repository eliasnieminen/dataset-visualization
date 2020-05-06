import os
from pathlib import Path
from typing import *

import numpy as np
import librosa as lbr
import openl3
from tensorflow.keras import Model

from dataobjects import OpenL3EmbeddingPackage, OpenL3EmbeddingPackageWrapper
import file_io


class DynamicPercussionDataset:

    def __init__(self,
                 audio_directory: Optional[Union[Path, str, None]] = None,
                 serialization_directory: Optional[Union[Path, str, None]] = None):
        """
        Initialization of the dataset.
        :param audio_directory: The base directory of the Dynamic Percussion
               Dataset.
        :param serialization_directory: In the case new embeddings are
               calculated, they will be serialized here.
        """

        self.all_classes = {
            "BD": {
                "BD_REG",
                "BD_DAMP"
            },
            "SN": {
                "SN_ON_REG",
                "SN_ON_DAMP",
                "SN_ON_DOUBLE",
                "SN_ON_BUZZ",
                "SN_ON_CROSS",
                "SN_OFF_REG"
            },
            "TOM": {
                "TOM_HI",
                "TOM_MED",
                "TOM_LO"
            },
            "HH": {
                "HH_CLOSED",
                "HH_SEMI",
                "HH_OPEN",
                "HH_BELL"
            },
            "CYM": {
                "CYM_CRASH_REG",
                "CYM_CHINA",
                "CYM_RIDE_REG",
                "CYM_RIDE_BELL",
                "CYM_FX_STACK",
                "CYM_FX_SPLASH",
            },
            "PERC": {
                "PERC_TAMB",
                "PERC_TRIANGLE",
                "PERC_CABASA",
                "PERC_WOODBLOCK",
                "PERC_SHAKER",
                "PERC_COWBELL",
                "PERC_VIBRASLAP"
            }
        }

        self.superclasses = {
            "BD": "Bass drum",
            "SN": "Snare drum",
            "TOM": "Tom tom",
            "HH": "Hi-hat",
            "CYM": "Cymbal",
            "PERC": "Percussion",
        }

        self.subclasses = {
            "BD_REG": "Bass drum, regular",
            "BD_DAMP": "Bass drum, dampened",
            "SN_ON_REG": "Snare drum, regular",
            "SN_ON_DAMP": "Snare drum, dampened",
            "SN_ON_DOUBLE": "Snare drum, double stroke",
            "SN_ON_BUZZ": "Snare drum, buzz roll",
            "SN_ON_CROSS": "Snare drum, cross-sticking",
            "SN_OFF_REG": "Snare drum, snares off, regular",
            "TOM_HI": "Tom tom, high tuning",
            "TOM_MED": "Tom tom, medium tuning",
            "TOM_LO": "Tom tom, low tuning",
            "HH_CLOSED": "Hi-hat, closed",
            "HH_SEMI": "Hi-hat, semi-opened",
            "HH_OPEN": "Hi-hat, fully opened",
            "HH_BELL": "Hi-hat, bell hit",
            "CYM_CRASH_REG": "Crash cymbal, regular",
            "CYM_RIDE_REG": "Ride cymbal, regular",
            "CYM_RIDE_BELL": "Ride cymbal's bell",
            "CYM_CHINA": "Chinese crash cymbal",
            "CYM_FX_STACK": "Two stacked cymbals",
            "CYM_FX_SPLASH": "Splash cymbal",
            "PERC_TAMB": "Percussion, tambourine",
            "PERC_TRIANGLE": "Percussion, triangle",
            "PERC_CABASA": "Percussion, cabasa",
            "PERC_WOODBLOCK": "Percussion, woodblock",
            "PERC_SHAKER": "Percussion, shaker",
            "PERC_COWBELL": "Percussion, cowbell",
            "PERC_VIBRASLAP": "Percussion, vibraslap",
        }

        self.dynamics = {
            "pp": "Very quiet",
            "p": "Quiet",
            "mf": "Regular",
            "f": "Loud",
            "ff": "Very loud",
        }

        self.audio_paths_by_class = {}
        self.audio_paths_by_dynamics = {}

        for dynamic in list(self.dynamics.keys()):
            self.audio_paths_by_dynamics[dynamic] = []

        self.audio_directory = None
        self.serialization_directory = None
        self.set_audio_directory(audio_directory)
        self.serialization_directory = serialization_directory
        self.embedding_wrapper = None
        self.openl3settings: Union[List, None] = None

        if self.serialization_directory is not None:
            self.serialization_directory = Path(serialization_directory).resolve()

        if self.audio_directory is not None:
            self.load_audio()

    def print_classes(self, with_explanation: Optional[bool] = True) -> None:
        """
        Prints the available percussion classes.
        :param with_explanation: If true, the class explanations will
               be printed.
        :return: None
        """

        for key in self.subclasses:
            if with_explanation:
                print(f"{key}: {self.subclasses[key]}")
            else:
                print(key)

    def set_audio_directory(self, audio_directory: Union[Path, str, None]) -> None:
        """
        Sets the current audio directory.
        :param audio_directory: The base directory of the Dynamic Percussion
               Dataset.
        :return: None
        """

        if audio_directory is not None:
            self.audio_directory = Path(audio_directory).resolve()

    def set_serialization_directory(self,
                                    serialization_directory: Union[Path, str]) -> None:
        """
        Sets the current serialization directory, where the calculated embeddings
        will be saved.
        :param serialization_directory: In the case new embeddings are
               calculated, they will be serialized here.
        :return: None
        """

        if serialization_directory is not None:
            self.serialization_directory \
                = Path(serialization_directory).resolve()
        else:
            raise AttributeError("The serialization directory must not be None")

    def load_audio(self) -> bool:
        """
        Loads the audio paths for calculating the embeddings.
        :return: True, if the audio paths were collected succesfully.
                 False, otherwise.
        """
        if self.audio_directory is None:
            return False

        # Walk data directory
        for folder, subfolders, filenames in os.walk(str(self.audio_directory)):

            audio_paths = []
            folder_name = Path(folder).parts[-1]

            if folder_name == self.audio_directory.parts[-1]:
                continue  # Skip dataset's parent folder
            elif folder_name in list(self.subclasses.keys()):  # Found subclass
                for filename in filenames:
                    filename_as_path = Path(filename)
                    if filename_as_path.suffix == '.wav':
                        full_path = os.path.join(folder, filename)
                        audio_paths.append(full_path)
                        dynamic = filename_as_path.stem.split('_')[-1]
                        self.audio_paths_by_dynamics[dynamic].append(full_path)
                    self.audio_paths_by_class[folder_name] = audio_paths

        return True

    def load_serialized_data(self, class_filter: Optional[List] = None) -> None:
        """
        Loads serialized pre-calculated embeddings from the file system.
        :param class_filter: If not None, only classes found in the given
               filter list will be loaded. The class filter can contain
               superclass or subclass labels.
        :return: None
        """
        if self.serialization_directory is None:
            return
        else:
            allowed_classes = self.get_classes(class_filter=class_filter)
            self.embedding_wrapper \
                = file_io.unserialize_embeddings(str(self.serialization_directory),
                                                 class_filter=allowed_classes)

    def get_embedding_wrapper(self) -> OpenL3EmbeddingPackageWrapper:
        """
        Returns the constructed OpenL3EmbeddingPackageWrapper.
        :return: OpenL3EmbeddingPackageWrapper containing all the samples that
                 were loaded from the file system.
        """
        return self.embedding_wrapper

    def calculate_embeddings(self,
                             max_items_per_class: Optional[int] = 1000,
                             max_classes: Optional[int] = 1000,
                             class_filter: Optional[Union[List, None]] = None,
                             model: Optional[Union[Model, None]] = None,
                             input_repr: Optional[str] = 'mel256',
                             content_type: Optional[str] = 'music',
                             embedding_size: Optional[int] = 6144,
                             center: Optional[bool] = True,
                             hop_size: Optional[float] = 0.1,
                             batch_size: Optional[int] = 32,
                             verbose: Optional[bool] = True) -> None:
        """
        Initializes the calculation process of the OpenL3 embeddings.
        :param max_items_per_class:
               To speed up the calculation process, the amount of processed
               samples can be restricted. If None, all the samples in
               self.audio_paths_by_class will be processed.

        :param max_classes:
               To speed up the calculation process, the amount of classes can
               be restricted. If None, all the found classes will be processed.

        :param class_filter:
               If not None, only classes found in the given filter list will
               be loaded. The class filter can contain superclass or subclass
               labels.

        :param model:
               A custom model for calculating the embeddings. More information
               can be found in the OpenL3 docs. (get_audio_embeddings)

        :param input_repr:
               The input representation of the sample. Can be linear, mel128
               or mel256. More information can be found in the OpenL3 docs.
               (get_audio_embeddings)

        :param content_type:
               The content type of the samples to be processed. Can be music or
               env. More information can be found in the OpenL3 docs.
               (get_audio_embeddings)

        :param embedding_size:
               The size of the calculated embeddings. Can be 512 or 6144.
               More information can be found in the OpenL3 docs.
               (get_audio_embeddings)

        :param center:
               The location of the returned timestamps. More information can be
               found in the OpenL3 docs. (get_audio_embeddings)

        :param hop_size:
               The hop size used to calculate the embeddings. More information
               can be found in the OpenL3 docs. (get_audio_embeddings)

        :param batch_size:
               The number of samples that are fed to the model at once. More
               information can be found in the OpenL3 docs.
               (get_audio_embeddings)

        :param verbose:
               The amount of information printed on the screen during the
               calculation procedure.

        :return: None
        """

        if self.openl3settings is None:
            self.openl3settings = {
                'input_repr': input_repr,
                'content_type': content_type,
                'embedding_size': embedding_size,
                'center': center,
                'hop_size': hop_size
            }

        # Initialize embedding container
        all_embeddings = OpenL3EmbeddingPackageWrapper()

        # Class counter keeps track of how many classes have been processed.
        class_counter = 0
        for class_label in list(self.audio_paths_by_class.keys()):
            if class_filter is not None and class_label not in class_filter:
                self.log(f"Skipping class {class_label}")
                continue
            self.log(f"Processing class {class_label}")

            # Openl3 will process these lists to get the embeddings.
            audio_list = []
            sr_list = []

            # The package list keeps track of all the packages before being
            # added to the all_embeddings container.
            package_list = []

            # Load audio samples and respective sample rates to lists
            self.log("Loading audio...")
            counter = 0
            for audio_path in self.audio_paths_by_class[class_label]:
                audio, sr = lbr.load(audio_path, sr=None)

                # Important metadata needed to play the audio later on when
                # clicked on the plot.
                metadata = {
                    'class': class_label,
                    'sample_id': class_label + '_' + str(counter),
                    'raw_audio_path': audio_path,
                    'original_sr': sr,
                    'openl3settings': self.openl3settings
                }

                # A container package is initialized for each sample
                package = OpenL3EmbeddingPackage(embeddings=None,
                                                 timestamps=None,
                                                 metadata=metadata)

                audio_list.append(audio)
                sr_list.append(sr)
                package_list.append(package)

                if max_items_per_class is not None and \
                   counter >= max_items_per_class:
                    break

                counter += 1

            self.log("Computing embeddings...")

            # Here the embeddings are calculated with the OpenL3 model specified
            # in the arguments.
            emb_list, ts_list \
                = openl3.get_audio_embedding(audio_list, sr_list,
                                             model=model,
                                             input_repr=input_repr,
                                             content_type=content_type,
                                             embedding_size=embedding_size,
                                             center=center,
                                             hop_size=hop_size,
                                             batch_size=batch_size,
                                             verbose=verbose)

            counter = 0
            for embeddings in emb_list:
                package_list[counter].set_embeddings(embeddings)
                all_embeddings.add_package(package_list[counter])
                counter += 1

            if max_classes is not None and class_counter >= max_classes:
                break

            class_counter += 1

        # The container holds all the computed embeddings.
        self.embedding_wrapper = all_embeddings

    def serialize_embeddings(self) -> None:
        """
        Serializes the calculated embeddings to the file system, so that they
        can be used later.
        :return: None
        """
        if self.serialization_directory is None:
            raise AttributeError("The serialization directory was not set.")
        else:

            settings = []
            for key in list(self.openl3settings.keys()):
                settings.append(str(self.openl3settings[key]))

            final_serdir = Path(self.serialization_directory /
                                ('openl3_' + "_".join(settings))).resolve()

            final_serdir.mkdir(parents=True)

            self.log(f"Serializing to {self.serialization_directory} ...")
            self.embedding_wrapper.serialize_embeddings(final_serdir)

    def log(self, msg) -> None:
        """
        Degub messages.
        :param msg: The message to show.
        :return: None
        """
        print(msg)

    def get_classes(self, class_filter: Union[List, None]) -> Union[List, None]:
        """
        A helper class for constructing the class filter.
        :param class_filter: If not None, only classes found in the given
               filter list will be loaded. The class filter can contain
               superclass or subclass labels.
        :return: A list of subclass labels.
        """
        classes = []
        if class_filter is not None:
            for item in class_filter:
                if item in self.all_classes.keys():
                    for subitem in self.all_classes[item]:
                        classes.append(subitem)
                elif item in self.subclasses.keys():
                    classes.append(item)
            if len(classes) == 0:
                classes = None
        else:
            classes = None

        return classes
