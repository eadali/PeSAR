from .yacs import CfgNode

cfg = CfgNode(new_allowed=True)
cfg.save_dir = "./"

# Pipeline configuration
cfg.pipeline = CfgNode(new_allowed=True)
cfg.pipeline.detector = CfgNode(new_allowed=True)
cfg.pipeline.detector.thresholds = CfgNode(new_allowed=True)
cfg.pipeline.detector.slicing = CfgNode(new_allowed=True)
# Tracker configuration
cfg.pipeline.tracker = CfgNode(new_allowed=True)


def load_config(cfg, args_cfg):
    cfg.defrost()
    cfg.merge_from_file(args_cfg)
    cfg.freeze()


if __name__ == "__main__":
    import sys

    with open(sys.argv[1], "w") as f:
        print(cfg, file=f)
