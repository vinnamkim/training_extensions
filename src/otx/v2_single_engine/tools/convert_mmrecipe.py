import os
from otx.v2_single_engine.utils.config import mmconfig_dict_to_dict
from argparse import ArgumentParser
from mmengine.config import Config
from omegaconf import OmegaConf, DictConfig

parser = ArgumentParser()
parser.add_argument("-n", "--recipe-name", type=str, required=True)
parser.add_argument("-o", "--output-dir", type=str, required=True)
parser.add_argument("-i", "--input-path", type=str, required=True)

override = parser.add_argument_group("override")
override.add_argument("--base", type=str, default="detection")
override.add_argument("--data", type=str, default="mmdet")
override.add_argument("--model", type=str, default="mmdet")


if __name__ == "__main__":
    args = parser.parse_args()

    config = Config.fromfile(args.input_path)
    config = mmconfig_dict_to_dict(config)

    omega_conf = OmegaConf.create(
        {
            "defaults": [
                {"override /base": args.base},
                {"override /data": args.data},
                {"override /model": args.model},
            ],
            "data": {
                "subsets": {
                    "train": {
                        "batch_size": config["train_dataloader"]["batch_size"],
                        "transforms": config["train_dataloader"]["dataset"]["pipeline"],
                    },
                    "val": {
                        "batch_size": config["val_dataloader"]["batch_size"],
                        "transforms": config["val_dataloader"]["dataset"]["pipeline"],
                    },
                },
            },
            "model": {"otx_model": {"config": config["model"]}},
        }
    )

    print(omega_conf)
    output_path = os.path.join(args.output_dir, args.recipe_name + ".yaml")
    with open(output_path, "w") as fp:
        fp.write("# @package _global_\n")
        OmegaConf.save(omega_conf, fp)
