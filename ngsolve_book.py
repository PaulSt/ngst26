import base64
import hashlib
import io
import os
import subprocess
import tempfile
from html import escape
from numbers import Real
from pathlib import Path

from IPython.display import HTML, display
from ngsolve.webgui import Draw as _ngsolve_draw


def _css_size(value):
    if isinstance(value, Real):
        return f"{value}px"
    return str(value)


def _scene_dir():
    configured = os.environ.get("WEBGUI_SCENE_DIR")
    if not configured:
        return None

    path = Path(configured)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _scene_url(name):
    base = os.environ.get("WEBGUI_BASE", "").strip().rstrip("/")
    if base == "/":
        base = ""
    return f"{base}/webgui_scenes/{name}" if base else f"/webgui_scenes/{name}"


def _iframe(srcdoc, width, height, extra=""):
    return f"""
<iframe
  class="ngsolve-webgui-frame"
  srcdoc="{escape(srcdoc, quote=True)}"
  style="width: {_css_size(width)}; height: {_css_size(height)}; max-width: 100%; border: 1px solid #d6d3d1; border-radius: 6px; background: white; display: block;"
  {extra}
  allowfullscreen
></iframe>
"""


def _screenshot(scene_html):
    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = Path(tmpdir) / "scene.html"
        image_path = Path(tmpdir) / "scene.png"
        html_path.write_text(scene_html, encoding="utf-8")

        for executable in ("chromium", "chromium-browser", "google-chrome"):
            try:
                subprocess.run(
                    [
                        executable,
                        "--headless=new",
                        "--disable-gpu",
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--use-gl=angle",
                        "--use-angle=swiftshader-webgl",
                        "--enable-unsafe-swiftshader",
                        "--hide-scrollbars",
                        "--window-size=1000,540",
                        "--virtual-time-budget=6000",
                        f"--screenshot={image_path}",
                        f"file://{html_path}",
                    ],
                    timeout=90,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except (FileNotFoundError, subprocess.SubprocessError):
                continue

            if not image_path.exists():
                continue

            try:
                from PIL import Image

                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    image.thumbnail((760, 760))
                    buffer = io.BytesIO()
                    image.save(buffer, "JPEG", quality=70, optimize=True)
                payload = buffer.getvalue()
                return "data:image/jpeg;base64," + base64.b64encode(payload).decode()
            except Exception:
                payload = image_path.read_bytes()
                return "data:image/png;base64," + base64.b64encode(payload).decode()

    return None


def _static_scene_frame(scene_html, width, height):
    scene_dir = _scene_dir()
    if scene_dir is None:
        return _iframe(scene_html, width, height, 'loading="lazy"')

    name = hashlib.sha1(scene_html.encode("utf-8")).hexdigest() + ".html"
    (scene_dir / name).write_text(scene_html, encoding="utf-8")
    url = _scene_url(name)
    url_html = escape(url, quote=True)

    preview = _screenshot(scene_html)
    if preview:
        srcdoc = (
            "<!doctype html><html><head><meta charset=\"utf-8\"><style>"
            "html,body{margin:0;height:100%;overflow:hidden;font-family:system-ui,sans-serif}"
            "#preview{width:100%;height:100%;object-fit:contain;background:#f3f4f6;display:block;filter:grayscale(35%)}"
            "#overlay{position:absolute;inset:0;border:0;cursor:pointer;background:rgba(100,110,130,.32);"
            "display:flex;align-items:center;justify-content:center;text-decoration:none}"
            "#overlay span{background:rgba(20,20,20,.62);color:white;padding:9px 18px;border-radius:24px;font-size:15px}"
            "</style></head><body>"
            f"<img id=\"preview\" src=\"{preview}\">"
            f"<a id=\"overlay\" href=\"{url_html}\" target=\"_self\">"
            "<span>click to load interactive 3D</span></a>"
            "</body></html>"
        )
        return _iframe(srcdoc, width, height)

    srcdoc = (
        "<!doctype html><html><head><meta charset=\"utf-8\"><style>"
        "html,body{margin:0;height:100%;font-family:system-ui,sans-serif}"
        "#overlay{position:absolute;inset:0;border:0;cursor:pointer;background:#eef0f4;"
        "display:flex;align-items:center;justify-content:center;text-decoration:none}"
        "#overlay span{background:rgba(20,20,20,.62);color:white;padding:9px 18px;border-radius:24px;font-size:15px}"
        "</style></head><body>"
        f"<a id=\"overlay\" href=\"{url_html}\" target=\"_self\">"
        "<span>click to load interactive 3D</span></a>"
        "</body></html>"
    )
    return _iframe(srcdoc, width, height)


def Draw(obj, *args, show=True, width="100%", height="520px", **kwargs):
    scene = _ngsolve_draw( obj, *args, show=False, width=width, height=height, **kwargs,)

    if show:
        html = scene.GenerateHTML()
        display(HTML(_static_scene_frame(html, width, height)))

    return scene
