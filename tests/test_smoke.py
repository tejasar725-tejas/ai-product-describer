"""Fast smoke tests for the project entry points."""

def test_cli_hello(capsys):
    from app.cli import main
    main(["hello"])
    assert "ready" in capsys.readouterr().out


def test_cli_help_is_available(capsys):
    from app.cli import main
    import pytest
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "train" in capsys.readouterr().out
