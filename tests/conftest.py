print("Loading conftest.py")

import jax
import gc


def pytest_runtest_teardown(item, nextitem):
    """Clean up JAX caches and force garbage collection after each test."""
    print(f"Running teardown for {item.name}")
    jax.clear_caches()
    gc.collect()
