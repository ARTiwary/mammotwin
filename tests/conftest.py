"""
Windows/Python 3.12 workaround: torch's import path calls
platform.machine(), which in Python 3.12 falls back to a WMI query ONLY
when PROCESSOR_ARCHITECTURE (and PROCESSOR_ARCHITEW6432) aren't visible
in os.environ. On some locked-down Windows machines (corporate security
policy, restricted process environments, some IDE/test-runner launch
contexts) that variable can be stripped even though it's set at the OS
level, and the WMI fallback query can hang or crash the process outright
(observed: "Windows fatal exception: code 0x8007000e" during pytest
collection, when test_api.py's `pytest.importorskip("torch")` triggers
the very first torch import in the process).

Setting it explicitly here, before any test module is collected/imported,
is a no-op if it was already set correctly, and fixes the hang if it
wasn't -- either way this is safe and does not affect Linux/Mac.
"""
import os
import platform

if platform.system() == "Windows" and not os.environ.get("PROCESSOR_ARCHITECTURE"):
    os.environ["PROCESSOR_ARCHITECTURE"] = "AMD64"
