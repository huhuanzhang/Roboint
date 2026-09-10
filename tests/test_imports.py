"""import 冒烟测试：核心模块在无 GPU/真机/重依赖环境下必须可导入。"""


def test_import_core():
    import robo_int  # noqa: F401
    import robo_int.robots  # noqa: F401
    import robo_int.tasks  # noqa: F401
    import robo_int.models  # noqa: F401
    import robo_int.arch  # noqa: F401
    import robo_int.algos  # noqa: F401
    import robo_int.data.cleaning  # noqa: F401
    import robo_int.data.dataset  # noqa: F401
    import robo_int.data.episodes  # noqa: F401
    import robo_int.data.scoring  # noqa: F401
    import robo_int.infra.factory  # noqa: F401
    import robo_int.infra.rollout  # noqa: F401
    import robo_int.infra.train_loop  # noqa: F401


def test_import_stubs():
    """stub 子包可导入（不实例化就不会触发 NotImplementedError）。"""
    import robo_int.algos.rlt  # noqa: F401
    import robo_int.algos.dsrl  # noqa: F401
    import robo_int.algos.sarm2  # noqa: F401
    import robo_int.models.pi05  # noqa: F401
    import robo_int.models.gr00t_n17  # noqa: F401
    import robo_int.models.fast_wam  # noqa: F401
    import robo_int.arch.mem  # noqa: F401
    import robo_int.arch.ttt  # noqa: F401
    import robo_int.robots.unitree  # noqa: F401
    import robo_int.robots.brainco_revo2  # noqa: F401
