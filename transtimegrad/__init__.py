from importlib.metadata import PackageNotFoundError, version
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # type: ignore

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0-unknown"
