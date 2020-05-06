# Audio Dataset Visualization

This repository contains the source code for visualizing the [Dynamic Percussion Dataset](https://zenodo.org/record/3780109#.XrL4x2gzaUk)
with a method that utilizes deep audio embeddings produced by the [OpenL3](https://openl3.readthedocs.io/en/latest/) and the t-SNE
dimensionality reduction algorithm. The method is described in my Bachelor's thesis (link will appear here
after the publication of the thesis). The structure of the dataset is also described in the same paper.

## Requirements

The following packages need to be installed in order for the code to work:

- [NumPy](https://numpy.org/)
- [OpenL3](https://openl3.readthedocs.io/en/latest/)
- [Scikit-learn](https://scikit-learn.org/stable/index.html) 0.22.2
- [Tensorflow](https://www.tensorflow.org/) 1.14 (GPU version recommended)
- [LibROSA](https://librosa.github.io/librosa/)
- [matplotlib](https://matplotlib.org/) (for plotting)
- [sounddevice](https://python-sounddevice.readthedocs.io/en/0.3.15/) (for playing the sounds interactively)
- [tikzplotlib](https://github.com/nschloe/tikzplotlib) (for exporting the plots in the tikz format)

In addition, the [Jupyter Notebook](https://jupyter.org/) is required for using the interactive notebook.

## Installation

Install the Python dependencies.
Make sure to download the DPD dataset and extract the files somewhere in system. After cloning/downloading the repository,
fill the `code/dataset_envs/env.json` with appropriate paths to your project and data locations. When the requirements
have been installed correctly, the `code/OpenL3 and TSNE visualization.ipynb` notebook should run correctly.

Warning: the installation of Tensorflow GPU can be tricky. The OpenL3 API is only compatible with the Tensorflow GPU
version 1.14 and lower, and installing the Nvidia CUDA environment according to that is required for the GPU calculations
to run correctly. Therefore, the dataset contains pre-computed embeddings of the audio samples produced with many variations of
the OpenL3 model, so the installation of the Tensorflow GPU is not mandatory to construct the visualization.

## Usage

Open the `code/OpenL3 and TSNE visualization.ipynb` notebook and run each of the code cells. Click the points in the plot
to listen to the sounds. Some sounds can be very quiet and some very loud.
