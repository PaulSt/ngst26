from html import escape
from numbers import Real

from IPython.display import HTML, display
from ngsolve.webgui import Draw as _ngsolve_draw


def _css_size(value):
    if isinstance(value, Real):
        return f"{value}px"
    return str(value)

def Draw(obj, *args, show=True, width="100%", height="520px", **kwargs):
    scene = _ngsolve_draw( obj, *args, show=False, width=width, height=height, **kwargs,)

    if show:
        html = scene.GenerateHTML()
        iframe = f"""
<iframe
  class="ngsolve-webgui-frame"
  srcdoc="{escape(html, quote=True)}"
  style="width: {_css_size(width)}; height: {_css_size(height)}; max-width: 100%; border: 1px solid #d6d3d1; border-radius: 6px; background: white; display: block;"
  allowfullscreen
></iframe>
"""
        display(HTML(iframe))

    return scene
