
from collections import OrderedDict
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("OLD")
parser.add_argument("NEW")
args = parser.parse_args()

parameters = torch.load(args.OLD, map_location=torch.device('cpu'))

new_parameters = OrderedDict(
    [(f"_genbert.{k}", v) for k, v in parameters.items()]
)

torch.save(new_parameters, args.NEW)
