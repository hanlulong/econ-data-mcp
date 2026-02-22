from __future__ import annotations

from backend.providers.bis import BISProvider


def test_bis_display_name_uses_human_readable_dataflow_label():
    provider = BISProvider()
    display = provider._build_indicator_display_name(  # pylint: disable=protected-access
        indicator_code="WS_DSR",
        indicator_label=None,
        series_key=None,
        series_dimensions=[],
    )
    assert display != "WS_DSR"
    assert "debt service" in display.lower()

