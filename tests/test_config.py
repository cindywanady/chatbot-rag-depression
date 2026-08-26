"""Config loading + sweep expansion."""


def test_page_mapping_and_scope(config):
    assert config.source.printed_offset == 2
    mi4 = config.source.unit("MI.4")
    assert (mi4.phys_start, mi4.phys_end) == (38, 59)  # printed 36-57
    assert mi4.printed_start(config.source.printed_offset) == 36
    assert mi4.printed_end(config.source.printed_offset) == 57
    # default case-management scope
    assert set(config.source.scope) == {"MI.1", "MI.2", "MI.4", "MI.7", "MI.8"}
    for name in config.source.scope:
        config.source.unit(name)  # every scoped unit must exist


def test_sweep_expansion(config):
    ids = {c.config_id for c in config.chunking.configs}
    assert "fixed-128-0" in ids            # common (near-)no-truncation condition
    assert "fixed-256-64" in ids
    assert "recursive-512-0" in ids
    assert "structure-512-0" in ids
    assert "structure-512-0-ctx" in ids    # context-enriched ablation
    for c in config.chunking.configs:
        assert 0 <= c.overlap < c.chunk_size
        assert c.strategy in {"fixed", "recursive", "structure"}


def test_glyph_map_resolved(config):
    # hex keys resolved to real characters; PUA bullet -> the bullet marker
    assert config.cleaning.glyph_map[chr(0xF0B7)] == config.cleaning.bullet_marker
