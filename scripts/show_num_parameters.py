from src.utils.count_parameters import count_parameters
from pprint import pformat
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path")
args = parser.parse_args()
path = args.path

num_parameters = count_parameters(path)
params = pformat(num_parameters)
print(params)