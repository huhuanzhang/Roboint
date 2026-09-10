"""Hydra 配置组合测试：默认配置可组合、factory 可构建。"""

from omegaconf import OmegaConf

from robo_int.infra.factory import build_algo, build_model, build_task


def _compose(overrides: list[str] | None = None):
    from hydra import compose, initialize_config_dir
    import pathlib

    config_dir = str(pathlib.Path("configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="config", overrides=overrides or [])


def test_default_config_builds_everything(tmp_path):
    cfg = _compose()
    assert cfg.model.name == "mock"
    assert cfg.algo.name == "sft"
    assert cfg.infra.regularization.dropout == 0.0

    task = build_task(cfg)
    model = build_model(cfg)
    algo = build_algo(cfg, model)
    assert algo.cfg.device == cfg.infra.device


def test_dropout_option_injection(tmp_path):
    """infra 的 dropout 选项必须注入模型构造（用户点名的基础设施能力）。"""
    cfg = _compose(["infra.regularization.dropout=0.25"])
    model = build_model(cfg)
    dropouts = [m.p for m in model.modules() if isinstance(m, __import__("torch").nn.Dropout)]
    assert any(abs(p - 0.25) < 1e-9 for p in dropouts)


def test_model_group_swaps():
    for model_name in ["mock", "pi05", "gr00t_n17", "fast_wam"]:
        cfg = _compose([f"model={model_name}"])
        assert cfg.model.name == model_name
        # 未接入的基模构建时给出带指引的 NotImplementedError
        if model_name != "mock":
            import pytest
            with pytest.raises(NotImplementedError):
                build_model(cfg)


def test_algo_group_swaps():
    """algo=rlt 组合可构建（RLT 已是真实现，需传真实基模）。"""
    cfg = _compose(["algo=rlt"])
    assert cfg.algo.name == "rlt"
    model = build_model(cfg)
    from robo_int.algos.rlt import RLT

    algo = build_algo(cfg, model)
    assert isinstance(algo, RLT)
    assert algo.action_limit == cfg.task.action_limit
    assert all(not p.requires_grad for p in algo.model.parameters())


def test_sarm2_algo_and_task_swaps():
    """algo=sarm2（奖励模型，不消费 VLA）与 task=mock_pick_place 组合可构建。"""
    cfg = _compose(["algo=sarm2", "task=mock_pick_place"])
    model = build_model(cfg)
    from robo_int.algos.sarm2 import SARM2
    from robo_int.tasks.mock_pick_place import MockPickPlaceTask

    task = build_task(cfg)
    assert isinstance(task, MockPickPlaceTask)

    algo = build_algo(cfg, model)
    assert isinstance(algo, SARM2)
    assert algo.num_primitives == len(cfg.algo.primitives)
    assert algo.phase == "stage"
