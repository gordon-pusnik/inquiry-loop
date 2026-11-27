"""Own the backends loaded by one interactive CLI process."""

from contextlib import ExitStack


SWITCHABLE = ("openai", "mlx", "llama")
_LOCAL = ("mlx", "llama", "airllm")


def close_model(model):
    close = getattr(model, "close", None)
    if callable(close):
        close()


class InteractiveModels:
    def __init__(self, create_model, backend, model, options=None):
        self._create_model = create_model
        overrides = tuple(sorted((name, value) for name, value in (options or {}).items()
                                 if name != "stub" and value is not None))
        self._active = (backend, overrides)
        self._loaded = {self._active: model}
        self._retiring = None

    @property
    def backend(self):
        return self._active[0] if self._active else None

    @property
    def model(self):
        return self._loaded.get(self._active)

    def _load(self, key):
        backend, overrides = key
        self._loaded[key] = self._create_model(backend, **dict(overrides))

    def switch(self, backend):
        key = (backend, ())
        if key in self._loaded and key != self._retiring:
            self._active = key
            return

        previous = self._active
        local = next((entry for entry in self._loaded if entry[0] in _LOCAL), None)
        released_local = False
        try:
            if backend == "llama":
                from inquiry_loop.backends.llama_eng import resolve_server

                resolve_server()
            if backend in _LOCAL and local is not None:
                if previous == local:
                    self._active = None
                self._retiring = local
                close_model(self._loaded[local])
                del self._loaded[local]
                self._retiring = None
                released_local = True
            self._load(key)
        except Exception as exc:
            if released_local and previous == local:
                try:
                    self._load(previous)
                    self._active = previous
                except Exception as restore_error:
                    raise ValueError(
                        f"Could not load {backend}: {exc}. Could not restore {previous[0]}: "
                        f"{restore_error}. Choose an engine with /model."
                    ) from exc
                raise ValueError(f"Could not load {backend}: {exc}. Restored {previous[0]}.") from exc
            raise ValueError(f"Could not load {backend}: {exc}") from exc
        self._active = key

    def close(self):
        with ExitStack() as stack:
            for model in self._loaded.values():
                stack.callback(close_model, model)
            self._loaded.clear()
            self._active = None
            self._retiring = None
