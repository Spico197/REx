from omegaconf import OmegaConf

from rex.utils.config import get_config_from_cmd


def test_cmd_args(tmp_path):
    base_config = OmegaConf.from_dotlist(
        ["train.model=???", "train.lr=2e-5", "data.train_path=???", "train.epoch=100"]
    )
    custom_config = OmegaConf.from_dotlist(
        ["train.model=A", "data.train_path=path/to/data", "train.batchsize=64"]
    )
    add_args = [
        "-a",
        "data.train_path=path/to/data_train.jsonl",
        "train.warmup_step=200",
    ]

    base_path = tmp_path / "base_config.yaml"
    OmegaConf.save(base_config, base_path)
    custom_path = tmp_path / "custom_config.yaml"
    OmegaConf.save(custom_config, custom_path)

    args = [f"--base-config-filepath={base_path}", f"-c={custom_path}", *add_args]
    config = get_config_from_cmd(args)

    assert config.train == {
        "model": "A",
        "lr": 2e-05,
        "epoch": 100,
        "batchsize": 64,
        "warmup_step": 200,
    }
    assert config.data == {"train_path": "path/to/data_train.jsonl"}
