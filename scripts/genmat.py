#!/usr/bin/env python3

import sys
sys.path.insert(0, '.')  # Add the current directory to the system path for importing custom modules

import yaml  # For parsing YAML configuration files
import json  # For pretty-printing dataset summaries
import pathlib  # For handling file paths
from functools import lru_cache  # For caching the parse_raw_csi function to avoid redundant parsing
import pandas as pd  # For reading and processing CSV data
import numpy as np  # For numerical operations on CSI data
import scipy.io  # For saving the dataset in MATLAB format

from HAR.constants import CSI_COL_NAMES as CSI_COLUMN_NAMES  # Import predefined CSI column names

rng = np.random.default_rng(seed=42)  # Initialize a random number generator with a fixed seed for reproducibility

@lru_cache(maxsize=None)
def parse_raw_csi(infile):
    # Parse raw CSI data from a file and compute amplitude values
    # Args: infile (str): Path to the input CSV file containing raw CSI data
    # Returns: np.ndarray: Transposed CSI amplitude array of shape (54, N), where N is the number of samples
    print(f'[INFO] Loading CSI data from: {infile}')
    df = pd.read_csv(infile, header=None, names=CSI_COLUMN_NAMES)  # Read the CSV file into a DataFrame

    n_nonHT = df.loc[df['sig_mode'] == 0].shape[0]  # Count non-HT (non-High Throughput) packets
    n_HT = df.loc[df['sig_mode'] == 1].shape[0]  # Count HT packets
    n_VHT = df.loc[df['sig_mode'] == 3].shape[0]  # Count VHT (Very High Throughput) packets

    print(f'[INFO] HT packets: {n_HT}, Dropped non-HT: {n_nonHT}, Dropped VHT: {n_VHT}')  # Log packet counts

    fs = (1000000.0 / df['local_timestamp'].diff()[1:]).mean()  # Estimate sampling frequency from timestamp differences
    print(f'[INFO] Estimated CSI sampling frequency: {fs:.2f} Hz')  # Log estimated frequency

    df = df.loc[df['sig_mode'] == 1]  # Keep only HT packets for processing
    csi_raw = df['data'].copy()  # Extract the raw CSI data column

    csi_data = np.array([
        np.fromstring(record.strip('[]'), dtype=int, sep=',')  # Parse each CSI record into an array
        for record in csi_raw
    ])

    # Keep the first 54 complex subcarriers (108 values: real and imaginary pairs)
    csi_data = csi_data[:, :108]

    # Calculate amplitude from real and imaginary components
    csi_amp = np.array([
        np.sqrt(record[::2] ** 2 + record[1::2] ** 2)  # Compute amplitude using Pythagorean theorem
        for record in csi_data
    ])

    return csi_amp.T  # Transpose to shape (54, N), where 54 is the number of subcarriers


class GenMat:
    DAQ_HZ = 100  # Data acquisition frequency in Hz (100 samples per second)

    def __init__(self, winsize=256, max_samples_per_class=-1):
        # Initialize the GenMat class for generating a structured CSI dataset
        # Args: winsize (int, optional): Size of the time window for each sample (default: 256)
        #       max_samples_per_class (int, optional): Maximum number of samples per class (default: -1 for unlimited)
        self.winsize = winsize  # Window size for each sample
        self.max_samples_per_class = max_samples_per_class  # Limit on samples per class
        self.classnames = []  # List to store activity class names
        self.nsamples = []  # List to store the number of samples per class
        self.X = None  # Matrix to store the CSI data
        self.dim = None  # Dimensions of each sample (subcarriers, window size)

    def add_class(self, classname, sources):
        # Add data for a specific activity class to the dataset
        # Args: classname (str): Name of the activity class (e.g., 'walking')
        #       sources (list): List of [infile, strip_sec_begin, strip_sec_end] for each source file
        cmat = None  # Initialize the class matrix
        for infile, strip_sec_begin, strip_sec_end in sources:
            amp = parse_raw_csi(infile)  # Load and parse the CSI amplitude data
            n_sc, duration = amp.shape  # Get number of subcarriers and duration

            isamples = range(
                self.DAQ_HZ * strip_sec_begin,  # Start after stripping initial seconds
                duration - self.DAQ_HZ * strip_sec_end - self.winsize,  # End before stripping final seconds
                self.winsize  # Step by window size
            )

            mat = np.zeros((n_sc * self.winsize, len(isamples)))  # Initialize matrix for this source
            for i, idx in enumerate(isamples):
                mat[:, i] = amp[:, idx:idx + self.winsize].reshape(-1)  # Flatten the CSI data window

            self.dim = (n_sc, self.winsize)  # Set dimensions (subcarriers, window size)
            cmat = mat if cmat is None else np.hstack((cmat, mat))  # Stack horizontally with existing data

        rng.shuffle(cmat.T)  # Shuffle the samples for this class
        if self.max_samples_per_class > 0:
            cmat = cmat[:, :self.max_samples_per_class]  # Limit the number of samples if specified

        self.X = cmat if self.X is None else np.hstack((self.X, cmat))  # Stack with overall dataset
        self.nsamples.append(cmat.shape[1])  # Record the number of samples
        self.classnames.append(classname)  # Record the class name

    def dump(self, outfile):
        # Save the dataset to a MATLAB file
        # Args: outfile (str): Path to the output MATLAB file
        scipy.io.savemat(outfile, {
            'dim': self.dim,  # Dimensions (subcarriers, window size)
            'nsamples': self.nsamples,  # Number of samples per class
            'classnames': self.classnames,  # List of class names
            'csi': self.X  # CSI data matrix
        })
        print(f'[SUCCESS] CSI dataset has been saved to: {outfile}')

    def summary(self):
        # Print a summary of the generated dataset
        print('[SUMMARY] Dataset overview:')
        print(json.dumps({
            'Signal Dimensions': self.dim,
            'Sample Counts': self.nsamples,
            'Activity Classes': self.classnames,
            'CSI Matrix Shape': self.X.shape
        }, indent=4))  # Print a JSON-formatted summary

def main(args):
    # Main function to generate datasets based on a YAML recipe
    # Args: args: Command-line arguments parsed by argparse
    with open(args.recipe, 'r') as f:
        cfg = yaml.safe_load(f)  # Load the YAML configuration file

    data_dir = pathlib.Path(cfg['data_dir'])  # Define path for data directory
    dest_dir = pathlib.Path(cfg['dest_dir'])  # Define path for destination directory

    for target, recipe in cfg['targets'].items():
        target = dest_dir / target  # Construct the full target path
        g = GenMat(
            winsize=recipe['winsize'],  # Set window size from recipe
            max_samples_per_class=recipe['max_samples_per_class']  # Set max samples from recipe
        )

        for clsname, sources in recipe['classes'].items():
            formatted_sources = [
                [data_dir / src[0], src[1], src[2]] for src in sources  # Format source paths and parameters
            ]
            g.add_class(clsname, sources=formatted_sources)  # Add data for each class

        print('\n[INFO] ----------------------------')
        print(f'[INFO] Generating dataset for output: {target}')
        g.summary()  # Print summary of the generated dataset

        if args.dry_run:
            print('[INFO] Dry-run mode enabled. Skipping dataset file creation.')  # Skip if dry-run
            continue

        g.dump(target)  # Save the dataset to the target file

if __name__ == '__main__':
    # Entry point for the script, parsing command-line arguments and running the main function
    import argparse

    parser = argparse.ArgumentParser(description='Generate structured CSI dataset from YAML recipe')
    parser.add_argument('--recipe', help='Path to recipe file (YAML)', required=True, type=str)
    parser.add_argument('--dry-run', help='Dry run, do not generate dataset but just show final summary', action='store_true')

    args = parser.parse_args()

    try:
        main(args)  # Run the main function with parsed arguments
    except KeyboardInterrupt:
        print('[WARNING] Operation interrupted by user.')
        
        # Example command to run the script with a specific recipe file
        # python ./scripts/genmat.py --recipe C:\Users\Admin\Desktop\ravicsi\Recordedcsi\recipes.yaml
