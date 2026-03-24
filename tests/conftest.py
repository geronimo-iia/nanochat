import pytest
import torch


@pytest.fixture(autouse=True, scope="module")
def reset_dynamo():
    """Reset torch.compile cache between test modules.

    torch.compile(dynamic=False) compiles one kernel per unique tensor shape.
    Running many test modules in the same process exhausts the dynamo cache
    (default limit=8), causing FailOnRecompileLimitHit in later modules.
    Resetting per module keeps each module isolated without affecting production.
    """
    torch._dynamo.reset()
