from app.retrieval.registry import (
    WORKFLOWS,
    UnknownWorkflowError,
    build_flow,
    spec_for,
)


def test_registry_has_the_three_seed_flows():
    assert set(WORKFLOWS) == {"embedding", "keyword", "hybrid"}


def test_every_flow_instantiates():
    for name, spec in WORKFLOWS.items():
        flow = build_flow(spec.class_key)
        assert flow.name == name
        assert flow.description.strip()
        assert callable(flow.retrieve)


def test_embed_needing_flows_are_flagged():
    assert spec_for("embedding").needs_embed_model is True
    assert spec_for("hybrid").needs_embed_model is True
    assert spec_for("keyword").needs_embed_model is False


def test_unknown_class_key_raises():
    try:
        build_flow("does-not-exist")
    except UnknownWorkflowError as exc:
        assert "does-not-exist" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected UnknownWorkflowError")


def test_class_key_matches_registry_name():
    for name, spec in WORKFLOWS.items():
        assert spec.factory.name == name
        assert spec.class_key == name