                                      
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata


project_root = Path(SPEC).resolve().parent


def _data_dir(path_name: str, target_name: str) -> tuple[str, str]:
    path = (project_root / path_name).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return (str(path), target_name)


datas = [
    _data_dir("src", "src"),
    _data_dir("static", "static"),
    _data_dir("logs", "logs"),
    _data_dir("assets", "assets"),
]

datas += collect_data_files("streamlit")
datas += copy_metadata("streamlit")

icon_path = (project_root / "assets" / "app.ico").resolve()
icon_arg = str(icon_path) if icon_path.exists() else None

hiddenimports = [
    "streamlit",
    "webview",
    "uvicorn",
    "requests",
    "sentence_transformers",
    "transformers",
    "faiss",
    "tqdm",
    "aiohttp",
    "anyio",
    "fastapi",
    "fastapi.middleware.cors",
    "fastapi.responses",
    "pydantic",
    "pydantic_core",
    "starlette",
    "starlette.middleware",
    "starlette.responses",
    "numpy",
    "torch",
    "rank_bm25",
    "langchain_text_splitters",
    "jinja2",
    "markdown_it",
    "pygments",
    "watchdog",
    "watchdog.observers",
    "watchdog.events",
    "importlib_metadata",
    "streamlit.runtime.scriptrunner.magic_funcs",
]

hiddenimports += collect_submodules("src")


a = Analysis(
    ["run_app_launcher.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="MochiChatbot",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_arg,
)

if False:
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name="MochiChatbot",
    )
