"""Turning the notes into a page that shows the formulas.

The notes are markdown with LaTeX in them. A markdown viewer shows the
text, but most of them do not draw the formulas, they only show the
LaTeX as it is. The notes are turned into a page that draws them, so the
formulas can be read without any extra program.
"""

import base64
import html
import mimetypes
import re
from pathlib import Path
from typing import Any, Optional

# A page that works when it is opened in a browser. The formulas are
# drawn by MathJax, which is loaded from a CDN.
PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{
    color-scheme: light dark;
    --background: #ffffff;
    --foreground: #1b1b1f;
    --muted: #5b5b66;
    --border: #e2e2e8;
    --code: #f4f4f7;
    --accent: #b5312c;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --background: #16161a;
      --foreground: #e8e8ee;
      --muted: #a0a0ae;
      --border: #2c2c34;
      --code: #22222a;
    }}
  }}
  body {{
    margin: 0 auto;
    padding: 2.5rem 1.25rem 6rem;
    max-width: 52rem;
    background: var(--background);
    color: var(--foreground);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    font-size: 17px;
    line-height: 1.65;
  }}
  h1, h2, h3, h4 {{ line-height: 1.25; margin-top: 2.2em; }}
  h1 {{ font-size: 1.9rem; border-bottom: 1px solid var(--border); padding-bottom: .35em; }}
  h2 {{ font-size: 1.45rem; border-bottom: 1px solid var(--border); padding-bottom: .25em; }}
  h3 {{ font-size: 1.15rem; }}
  a {{ color: var(--accent); }}
  code {{ background: var(--code); padding: .15em .35em; border-radius: 4px; font-size: .9em; }}
  pre {{ background: var(--code); padding: 1rem; border-radius: 8px; overflow-x: auto; }}
  pre code {{ background: none; padding: 0; }}
  blockquote {{
    margin: 1.2em 0; padding: .1em 1em; border-left: 4px solid var(--accent);
    color: var(--muted);
  }}
  table {{ border-collapse: collapse; width: 100%; margin: 1.2em 0; display: block; overflow-x: auto; }}
  th, td {{ border: 1px solid var(--border); padding: .5em .75em; text-align: left; }}
  th {{ background: var(--code); }}
  hr {{ border: none; border-top: 1px solid var(--border); margin: 2.5em 0; }}
  img {{ max-width: 100%; height: auto; border-radius: 8px; margin: 1.2em 0; display: block; }}
  p > em:only-child {{ display: block; color: var(--muted); font-size: .9em; margin-top: -.6em; }}
  .time-mark {{
    display: inline-block; margin-top: 2.5em; padding: .2em .6em;
    border: 1px solid var(--border); border-radius: 999px;
    color: var(--muted); font-size: .85em; letter-spacing: .04em;
  }}
  mjx-container[display="true"] {{ margin: 1.4em 0; overflow-x: auto; overflow-y: hidden; }}
</style>
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['\\\\(', '\\\\)']],
    displayMath: [['\\\\[', '\\\\]']],
    processEscapes: true
  }},
  options: {{
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
  }},
  startup: {{
    typeset: true
  }}
}};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body>
{body}
</body>
</html>
"""

TIME_MARK_PATTERN = re.compile(
    r'<h3>\s*(\d\d:\d\d:\d\d)\s*</h3>', re.IGNORECASE
)


def _mathjax_renderer(tex: str, options: dict[str, Any]) -> str:
    """Turn a formula into the html that MathJax draws.

    The delimiters are the ones that MathJax is told to look for in the
    page.
    """
    if options.get("displayMode"):
        return f'<div class="math block">\\[{tex}\\]</div>'
    return f'<span class="math inline">\\({tex}\\)</span>'


def render_notes_html(
    markdown_text: str,
    title: str = "Lecture Notes",
    base_dir: Optional[Path] = None
) -> str:
    """Turn the notes into a page that shows the formulas.

    When the folder of the notes is given, the pictures of the diagrams
    are put inside the page. The page then stands on its own: it can be
    sent to someone, or opened from anywhere, and the pictures are still
    there.
    """
    body = _render_markdown(markdown_text)
    body = _mark_time_stamps(body)
    if base_dir is not None:
        body = _inline_images(body, base_dir)
    safe_title = html.escape(title)
    return PAGE_TEMPLATE.format(title=safe_title, body=body)


IMAGE_PATTERN = re.compile(r'<img\s+[^>]*?src="([^"]+)"', re.IGNORECASE)


def _inline_images(body: str, base_dir: Path) -> str:
    """Put the pictures of the diagrams inside the page.

    A picture that is linked is a picture that is lost the moment the
    notes are moved. The bytes of the picture go into the page itself.
    """
    folder = Path(base_dir)

    def replace(match: "re.Match[str]") -> str:
        source = match.group(1)
        data_uri = _as_data_uri(source, folder)
        if data_uri is None:
            return match.group(0)
        return match.group(0).replace(f'src="{source}"', f'src="{data_uri}"')

    return IMAGE_PATTERN.sub(replace, body)


def _as_data_uri(source: str, folder: Path) -> Optional[str]:
    """The picture as text that a browser can draw."""
    if source.startswith(("data:", "http://", "https://", "//")):
        return None
    candidate = Path(source)
    if candidate.is_absolute():
        path = candidate
    else:
        path = folder / candidate
    try:
        resolved_folder = folder.resolve()
        resolved_path = path.resolve()
        # A picture is inlined only when it lives inside the folder of the
        # notes. An absolute path or a "../" that points elsewhere would
        # otherwise copy bytes from anywhere on the machine into the page.
        resolved_path.relative_to(resolved_folder)
    except (OSError, ValueError):
        return None
    try:
        if not resolved_path.is_file():
            return None
        payload = resolved_path.read_bytes()
    except OSError:
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _render_markdown(markdown_text: str) -> str:
    try:
        from markdown_it import MarkdownIt
        from mdit_py_plugins.dollarmath import dollarmath_plugin
    except ImportError:
        # Without the renderer the text is still shown, the formulas are
        # only not drawn.
        return "<pre>" + html.escape(markdown_text) + "</pre>"

    # Raw HTML in the notes is not rendered: the notes are written by a
    # model, and a page that draws them must not run markup that came from
    # the lecture. Links are still turned into links.
    parser = MarkdownIt("commonmark", {"html": False, "linkify": True})
    parser.enable("table")
    try:
        parser.enable("strikethrough")
    except Exception:
        pass
    parser = parser.use(dollarmath_plugin, renderer=_mathjax_renderer)
    return parser.render(markdown_text)


def _mark_time_stamps(body: str) -> str:
    """Show the time mark of a section as a label.

    The time mark is the heading of the section in the markdown, so it
    becomes the label of the section in the page.
    """
    return TIME_MARK_PATTERN.sub(
        lambda match: f'<p class="time-mark">{match.group(1)}</p>', body
    )