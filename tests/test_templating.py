"""Placeholder resolution rules."""

from app.models import DefinitionInput
from app.templating import build_values, find_unresolved, render, secret_keys


def test_supplied_argument_wins_over_default():
    inputs = [DefinitionInput(key="service", type="text", defaultValue="apache2")]

    assert build_values(inputs, {"service": "nginx"}) == {"service": "nginx"}


def test_default_fills_an_omitted_argument():
    inputs = [DefinitionInput(key="service", type="text", defaultValue="apache2")]

    assert build_values(inputs, {}) == {"service": "apache2"}


def test_booleans_render_the_way_a_shell_reads_them():
    inputs = [DefinitionInput(key="force", type="boolean")]

    assert build_values(inputs, {"force": True}) == {"force": "true"}


def test_render_substitutes_known_keys():
    assert render("systemctl restart {{svc}}", {"svc": "apache2"}) == (
        "systemctl restart apache2"
    )


def test_render_masks_secret_keys():
    rendered = render("ssh -i {{key}} host", {"key": "PRIVATE"}, mask={"key"})

    assert "PRIVATE" not in rendered
    assert "••••••••" in rendered


def test_render_leaves_unknown_keys_untouched():
    # Rendering stays total; reporting the gap is find_unresolved's job.
    assert render("echo {{missing}}", {}) == "echo {{missing}}"


def test_find_unresolved_reports_only_missing_keys():
    missing = find_unresolved(["a {{one}}", "b {{two}}", None], {"one": "1"})

    assert missing == ["two"]


def test_secret_keys_picks_password_inputs():
    inputs = [
        DefinitionInput(key="token", type="password"),
        DefinitionInput(key="name", type="text"),
    ]

    assert secret_keys(inputs) == {"token"}
