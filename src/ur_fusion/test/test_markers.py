from ur_fusion.uncertainty_markers_node import FILTERS, label_text


def test_labels_render_in_rviz():
    """Text markers must avoid what RViz garbles: spaces and non-ASCII."""
    for _, _, label in FILTERS.values():
        text = label_text(label, 0.0094)
        assert text.isascii()
        assert ' ' not in text
    assert label_text('Fused', 0.0094) == 'Fused:+/-9.4mm'
