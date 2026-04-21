"""Import smoke-tests.

These are deliberately lightweight: they verify that every module in the
package imports cleanly and that public symbols are resolvable. They do
*not* require the source datasets and run in a few seconds. They are the
primary sanity check CI performs on every push.
"""

import importlib

import pytest

MODULES = [
    "fraud_fl",
    "fraud_fl.utils",
    "fraud_fl.plotting",
    "fraud_fl.data",
    "fraud_fl.models",
    "fraud_fl.ctgan_dp",
    "fraud_fl.federated",
    "fraud_fl.attacks",
    "fraud_fl.metrics",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    """Each module in the package must import without error."""
    importlib.import_module(name)


def test_version_string():
    import fraud_fl

    assert isinstance(fraud_fl.__version__, str)
    assert fraud_fl.__version__.count(".") >= 1


def test_public_api_surface():
    """A subset of the public API must be importable from its module."""
    from fraud_fl.data import (  # noqa: F401
        FEATURES,
        TARGET,
        harmonise_and_merge,
        stratified_temporal_partition,
    )
    from fraud_fl.federated import evaluate, fedavg_round, rdp_epsilon  # noqa: F401
    from fraud_fl.models import FraudMLP, to_tensor  # noqa: F401
    from fraud_fl.utils import load_config, resolve_device, set_seed  # noqa: F401

    assert TARGET == "Class"
    assert len(FEATURES) == 29
