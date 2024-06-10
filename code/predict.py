import os
import time
import stat
import pickle
import argparse
import numpy as np
from mindspore import context
from mindsponge import PipeLine
from mindsponge.common.protein import from_pdb_string

parser = argparse.ArgumentParser(description='Inputs for eval.py')
parser.add_argument('--pkl_path', type=str, default="../data/predict/111L_A_1_mini.pkl", help='pkl path')
parser.add_argument('--pdb_path', type=str, default="../data/predict/111L_A_1_renum.pdb", help='pdb_path')
parser.add_argument('--ckpt_path', type=str, default="../output/MEGA_Assessment.ckpt", help='ckpt_path')
parser.add_argument('--device_id', type=int, default=0, help='device id')
parser.add_argument('--device_target', type=str, default="GPU", help='device target')

args = parser.parse_args()
if args.device_target == "Ascend":
    context.set_context(device_target="Ascend", ascend_config={"precision_mode": "must_keep_origin_dtype"})
elif args.device_target == "GPU":
    context.set_context(device_target="GPU")

pipe = PipeLine(name="MEGAAssessment")
pipe.set_device_id(args.device_id)

# from mindsponge.pipeline.pipeline import download_config
# download_config(pipe.config["predict_256"], pipe.config_path + "predict_256.yaml")
# conf = load_config(pipe.config_path + "predict_256.yaml")

pipe.initialize("predict_256")
pipe.model.from_pretrained(args.ckpt_path)

# load raw feature
f = open(args.pkl_path, "rb")
raw_feature = pickle.load(f)
f.close()

# load decoy pdb
with open(args.pdb_path, 'r') as f:
    decoy_prot_pdb = from_pdb_string(f.read())
    f.close()
raw_feature['decoy_aatype'] = decoy_prot_pdb.aatype
raw_feature['decoy_atom_positions'] = decoy_prot_pdb.atom_positions
raw_feature['decoy_atom_mask'] = decoy_prot_pdb.atom_mask

res = protein_assessment.predict(raw_feature)
print("score is:", np.mean(res))
