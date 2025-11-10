"""Module shim so `python -m src.inference` calls the runtime entrypoint."""

from .runtime import main


if __name__ == "__main__":  # pragma: no cover
    main()
