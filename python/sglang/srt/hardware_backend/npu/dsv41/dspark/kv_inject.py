"""Skip draft injection for globally empty CED chunks."""

from functools import wraps

from ..policy import enabled


def adapt_inject(original):
    if not enabled():
        return original

    @wraps(original)
    def inject(injector, *args, **kwargs):
        backend = injector.model_runner.attn_backend
        if getattr(injector, "_dsv41_in_prefill", False) and getattr(
            backend, "_dsv41_skip_prefill_injection", False
        ):
            return
        return original(injector, *args, **kwargs)

    return inject
