import pytest
from omegaconf import OmegaConf

from rex.utils.config import ConfigArgument, ConfigParser


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
    config = ConfigParser.parse_cmd(cmd_args=args)

    assert config.train == {
        "model": "A",
        "lr": 2e-05,
        "epoch": 100,
        "batchsize": 64,
        "warmup_step": 200,
    }
    assert config.data == {"train_path": "path/to/data_train.jsonl"}


def test_config_argument():
    config = ConfigParser.parse_cmd(
        ConfigArgument("-d", "--task-dir", type=str, help="task dir"),
        cmd_args=["-d", "/path/to/123"],
    )
    assert config.task_dir == "/path/to/123"

    with pytest.raises(AssertionError):
        config = ConfigParser(("-d", "--task-dir"))


def test_resolve_update(tmp_path):
    base_config = OmegaConf.from_dotlist(["name=${data_type}", "data_type=dev"])
    assert base_config.name == "dev"
    base_path = tmp_path / "base_config.yaml"
    OmegaConf.save(base_config, base_path)

    custom_config = OmegaConf.from_dotlist(["data_type=test"])
    custom_path = tmp_path / "custom_config.yaml"
    OmegaConf.save(custom_config, custom_path)

    args = [f"--base-config-filepath={base_path}", f"-c={custom_path}"]
    config = ConfigParser.parse_cmd(cmd_args=args)
    assert config.data_type == "test"

    config.data_type = "new"
    assert config.name == "new"
