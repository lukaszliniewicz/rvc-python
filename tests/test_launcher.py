from pathlib import Path


def test_linux_launcher_preserves_its_supervised_root_process():
    launcher = (Path(__file__).parents[1] / "run.sh").read_text(encoding="utf-8")

    assert 'exec "$PIXI_EXE" run' not in launcher
    assert '"$PIXI_EXE" run --environment "$PIXI_ENV" python run.py' in launcher
