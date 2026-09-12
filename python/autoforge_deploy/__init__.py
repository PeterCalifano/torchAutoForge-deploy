"""Public Python package entrypoint for autoforge_deploy."""

from __future__ import annotations

from importlib import import_module, util
from pathlib import Path
import os
import sys
from types import ModuleType

HAS_WRAPPER = False
WRAPPER_IMPORT_ERROR: ImportError | None = None
_DLL_DIRECTORY_HANDLES_: list[object] = []

# Python 3.8+ requires explicit DLL search directories on Windows. Retain each
# handle for the package lifetime so later delay-loaded runtime dependencies
# remain resolvable.
if os.name == "nt":
    _DLL_DIRECTORY_HANDLES_.append(
        os.add_dll_directory(str(Path(__file__).resolve().parent))
    )


def _export_wrapper_module(module_: ModuleType) -> None:
    """Re-export the compiled wrapper's public names at package scope."""
    public_names_ = getattr(module_, "__all__", None)
    if public_names_ is None:
        public_names_ = [name_ for name_ in dir(module_) if not name_.startswith("_")]
    for name_ in public_names_:
        globals()[name_] = getattr(module_, name_)


def _import_build_linked_wrapper() -> ModuleType:
    """Import the exact build-tree wrapper recorded by generated metadata.

    Returns:
        Loaded build-tree extension module.

    Raises:
        ImportError: If metadata, native artifacts, or the module spec are
            unavailable.
    """
    try:
        from . import _wrapper_build
    except ImportError as exc:
        raise ImportError("No build-linked wrapper metadata is available.") from exc

    module_path_ = Path(_wrapper_build.WRAPPER_MODULE_PATH)
    if not module_path_.is_file():
        raise ImportError(f"Build-linked wrapper module was not found at '{module_path_}'.")

    if os.name == "nt":
        runtime_paths_ = getattr(
            _wrapper_build,
            "WRAPPER_RUNTIME_LIBRARY_PATHS",
            [],
        )
        for runtime_path_ in runtime_paths_:
            dll_path_ = Path(runtime_path_).parent
            if dll_path_.is_dir():
                _DLL_DIRECTORY_HANDLES_.append(
                    os.add_dll_directory(str(dll_path_))
                )

    package_name_ = __name__.split(".")[-1]
    module_name_ = f"{__name__}.{package_name_}"
    spec_ = util.spec_from_file_location(module_name_, module_path_)
    if spec_ is None or spec_.loader is None:
        raise ImportError(f"Could not load build-linked wrapper spec from '{module_path_}'.")

    module_ = util.module_from_spec(spec_)
    sys.modules[module_name_] = module_
    spec_.loader.exec_module(module_)
    return module_


package_name_ = __name__.split(".")[-1]
wrapper_module_: ModuleType | None = None

try:
    wrapper_module_ = import_module(f".{package_name_}", __name__)
except ImportError:
    try:
        wrapper_module_ = _import_build_linked_wrapper()
    except ImportError as exc:
        WRAPPER_IMPORT_ERROR = exc

if wrapper_module_ is not None:
    _export_wrapper_module(wrapper_module_)
    HAS_WRAPPER = True
