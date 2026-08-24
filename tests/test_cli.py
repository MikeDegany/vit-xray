"""CLI wiring. No models are loaded -- these only check that Typer is set up right."""

import pytest
from typer.testing import CliRunner

from vit_xray import __version__
from vit_xray.cli import app

runner = CliRunner()


def test_help_renders():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "vit-xray" in result.output
    # The rich_help_panel groups are what make the help readable; if Typer stops
    # honouring them the panels silently collapse into one flat Options list.
    for panel in ("Model & device", "Panels", "Output"):
        assert panel in result.output


def test_short_help_flag():
    assert runner.invoke(app, ["-h"]).exit_code == 0


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_missing_image_is_rejected_before_any_model_loads():
    # exists=True on the argument means we fail in argument parsing, not after spending
    # several minutes pulling a 4.6 GB checkpoint.
    result = runner.invoke(app, ["definitely_not_here.png"])
    assert result.exit_code != 0


def test_cli_import_stays_light():
    """Importing the CLI must not drag in torch, timm or numpy.

    This is what keeps `--help` at ~0.2 s instead of ~7 s. It regresses silently the
    moment someone adds a convenient top-level import, so it gets a test rather than a
    comment. Run in a subprocess because the rest of the suite has already imported
    everything into this one.
    """
    import subprocess
    import sys

    code = (
        "import sys; import vit_xray.cli; "
        "heavy = [m for m in ('torch', 'timm', 'numpy', 'matplotlib', 'sklearn') "
        "if m in sys.modules]; "
        "print(','.join(heavy))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"cli.py now imports {out.stdout.strip()} at module level"


def test_lazy_package_attributes_still_resolve():
    # __init__ resolves inspect/XrayResult through __getattr__; make sure that plumbing
    # works and did not just quietly turn the public API into AttributeErrors.
    import vit_xray

    assert callable(vit_xray.inspect)
    assert vit_xray.XrayResult.__name__ == "XrayResult"
    assert vit_xray.DEFAULT_MODEL.startswith("vit_")
    with pytest.raises(AttributeError):
        vit_xray.no_such_thing


def test_showcase_hint_only_when_default_was_used():
    # The hint exists for someone who just ran `vit-xray photo.jpg` and does not know
    # there is a better model. Anyone who passed --model has already made that choice.
    from vit_xray.cli import showcase_hint
    from vit_xray.defaults import DEFAULT_MODEL, SHOWCASE_MODEL

    hint = showcase_hint([DEFAULT_MODEL])
    assert hint is not None
    assert SHOWCASE_MODEL in hint
    # It must state the cost -- this points a laptop user at a 4.3 GB download.
    assert "4.3 GB" in hint

    assert showcase_hint([SHOWCASE_MODEL]) is None
    assert showcase_hint(["vit_base_patch16_224.dino"]) is None
    assert showcase_hint([DEFAULT_MODEL, SHOWCASE_MODEL]) is None


def test_default_model_is_small_enough_to_be_a_default():
    # The whole point of the default is that it fits on a laptop. ViT-g does not.
    from vit_xray.defaults import DEFAULT_MODEL, SHOWCASE_MODEL

    assert DEFAULT_MODEL != SHOWCASE_MODEL
    assert "giant" not in DEFAULT_MODEL
    assert "base" in DEFAULT_MODEL


def test_both_models_are_real_timm_names():
    import timm

    from vit_xray.defaults import DEFAULT_MODEL, SHOWCASE_MODEL

    known = set(timm.list_models(pretrained=True))
    assert DEFAULT_MODEL in known
    assert SHOWCASE_MODEL in known
