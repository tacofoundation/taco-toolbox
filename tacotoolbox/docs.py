"""
TACO Documentation Generator

Generate clean HTML and Markdown documentation from TACOLLECTION.json files.

Optional dependencies (required together):
    - jinja2: Template rendering
    - markdown: Rich text formatting

Install with: pip install jinja2 markdown
"""

import html
import json
from pathlib import Path

from tacotoolbox._constants import (
    DOCS_CSS_FILE,
    DOCS_JS_MAP,
    DOCS_JS_PIT,
    DOCS_JS_UI,
    DOCS_TEMPLATE_HTML,
    DOCS_TEMPLATE_MD,
)
from tacotoolbox._exceptions import TacoDocumentationError
from tacotoolbox._logging import get_logger

try:
    import markdown  # type: ignore[import-untyped]
    from jinja2 import Environment, FileSystemLoader

    DOCS_AVAILABLE = True
except ImportError:
    DOCS_AVAILABLE = False

logger = get_logger(__name__)


def _check_dependencies() -> None:
    """Verify required dependencies are installed."""
    if not DOCS_AVAILABLE:
        raise ImportError(
            "Documentation generation requires jinja2 and markdown.\n"
            "Install with: pip install jinja2 markdown"
        )


def _get_template_env() -> Environment:
    """Initialize Jinja2 environment for template rendering."""
    templates_dir = Path(__file__).parent / "templates"

    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=True,
    )

    env.globals["zip"] = zip

    return env


def _load_vendor_files(templates_dir: Path) -> dict:
    """Load vendor JavaScript and CSS libraries from disk."""
    vendor_dir = templates_dir / "vendor"
    vendor_files = {}

    try:
        vendor_files["d3"] = (vendor_dir / "d3.min.js").read_text(encoding="utf-8")
        vendor_files["leaflet_js"] = (vendor_dir / "leaflet.min.js").read_text(
            encoding="utf-8"
        )
        vendor_files["leaflet_css"] = (vendor_dir / "leaflet.min.css").read_text(
            encoding="utf-8"
        )
        vendor_files["highlight"] = (vendor_dir / "highlight.min.js").read_text(
            encoding="utf-8"
        )
        vendor_files["highlight_css"] = (
            vendor_dir / "atom-one-dark.min.css"
        ).read_text(encoding="utf-8")
        vendor_files["highlight_python"] = (
            vendor_dir / "languages" / "python.min.js"
        ).read_text(encoding="utf-8")
        vendor_files["highlight_r"] = (vendor_dir / "languages" / "r.min.js").read_text(
            encoding="utf-8"
        )
        vendor_files["highlight_julia"] = (
            vendor_dir / "languages" / "julia.min.js"
        ).read_text(encoding="utf-8")

        logger.debug(f"Loaded {len(vendor_files)} vendor files")

    except FileNotFoundError as e:
        raise TacoDocumentationError(f"Vendor file not found: {e}") from e
    except Exception as e:
        raise TacoDocumentationError(f"Failed to load vendor files: {e}") from e

    return vendor_files


def generate_markdown(
    input: str | Path,
    output: str | Path = "README.md",
) -> None:
    """
    Generate Markdown documentation from TACOLLECTION.json.

    Args:
        input: Path to TACOLLECTION.json file
        output: Output path for generated markdown

    Raises:
        ImportError: If jinja2 or markdown packages are not installed
        FileNotFoundError: If input file does not exist
        TacoDocumentationError: If documentation generation fails
    """
    _check_dependencies()

    input = Path(input)
    output = Path(output)

    logger.debug(f"Loading collection from: {input}")

    if not input.exists():
        raise FileNotFoundError(f"Collection not found: {input}")

    try:
        with open(input, encoding="utf-8") as f:
            collection = json.load(f)
    except json.JSONDecodeError as e:
        raise TacoDocumentationError(f"Invalid JSON in {input}: {e}") from e

    logger.debug("Rendering markdown template")

    try:
        env = _get_template_env()
        template = env.get_template(DOCS_TEMPLATE_MD)

        md_content = template.render(
            collection=collection,
            id=collection.get("id", "TACO Dataset"),
            description=collection.get("description", ""),
            keywords=collection.get("keywords", []),
        )

    except Exception as e:
        raise TacoDocumentationError(f"Failed to render markdown: {e}") from e

    output.write_text(md_content, encoding="utf-8")
    logger.info(f"Generated Markdown: {output.absolute()}")


def generate_html(
    input: str | Path,
    output: str | Path = "index.html",
    inline_deps: bool = True,
    catalogue_url: str = "https://tacofoundation.github.io/catalogue",
    download_base_url: str | None = None,
) -> None:
    """
    Generate interactive HTML documentation from TACOLLECTION.json.

    Args:
        input: Path to TACOLLECTION.json file
        output: Output path for generated HTML
        inline_deps: Embed JS libraries inline for offline usage
        catalogue_url: URL to link back to catalogue (None to hide)
        download_base_url: Base URL for file downloads (appends filename from extents)

    Raises:
        ImportError: If jinja2 or markdown packages are not installed
        FileNotFoundError: If input file does not exist
        TacoDocumentationError: If documentation generation fails
    """
    _check_dependencies()

    input = Path(input)
    output = Path(output)

    logger.debug(f"Loading collection from: {input}")

    if not input.exists():
        raise FileNotFoundError(f"Collection not found: {input}")

    try:
        with open(input, encoding="utf-8") as f:
            collection = json.load(f)
    except json.JSONDecodeError as e:
        raise TacoDocumentationError(f"Invalid JSON in {input}: {e}") from e

    pit_schema = collection.get("taco:pit_schema", {})
    pit_schema_json = json.dumps(pit_schema, indent=2)

    extents = collection.get("taco:sources", {}).get("extents", [])
    extents_json = json.dumps(extents, indent=2)

    logger.debug(f"Found {len(extents)} spatial extents")

    description_html = _description_to_html(collection.get("description", ""))

    templates_dir = Path(__file__).parent / "templates"

    try:
        css_content = (templates_dir / DOCS_CSS_FILE).read_text(encoding="utf-8")
        logger.debug(f"Loaded {DOCS_CSS_FILE} ({len(css_content)} chars)")

        js_pit = (templates_dir / DOCS_JS_PIT).read_text(encoding="utf-8")
        js_map = (templates_dir / DOCS_JS_MAP).read_text(encoding="utf-8")
        js_ui = (templates_dir / DOCS_JS_UI).read_text(encoding="utf-8")

        logger.debug(
            f"Loaded JS: {DOCS_JS_PIT} ({len(js_pit)} chars), "
            f"{DOCS_JS_MAP} ({len(js_map)} chars), "
            f"{DOCS_JS_UI} ({len(js_ui)} chars)"
        )

        vendor = {}
        if inline_deps:
            vendor = _load_vendor_files(templates_dir)
            logger.info("Using inline dependencies (offline-ready)")
        else:
            logger.info("Using CDN dependencies (requires internet)")

    except FileNotFoundError as e:
        raise TacoDocumentationError(f"Asset loading failed: {e}") from e
    except Exception as e:
        raise TacoDocumentationError(f"Failed to read assets: {e}") from e

    logger.debug("Rendering HTML template")

    try:
        env = _get_template_env()
        template = env.get_template(DOCS_TEMPLATE_HTML)

        html_content = template.render(
            collection=collection,
            description_html=description_html,
            css=css_content,
            js_pit=js_pit,
            js_map=js_map,
            js_ui=js_ui,
            pit_schema_json=pit_schema_json,
            extents_json=extents_json,
            inline_deps=inline_deps,
            vendor=vendor,
            catalogue_url=catalogue_url,
            download_base_url=download_base_url,
        )

    except Exception as e:
        raise TacoDocumentationError(f"Failed to render HTML: {e}") from e

    output.write_text(html_content, encoding="utf-8")

    size_mb = len(html_content) / (1024 * 1024)
    logger.info(f"Generated HTML: {output.absolute()} ({size_mb:.2f} MB)")


def _description_to_html(description: str) -> str:
    """Convert description to HTML using markdown library."""
    import re

    if not description:
        return "<p>No description provided.</p>"

    # Normalize whitespace: multiple newlines -> paragraph breaks, single newlines -> spaces
    description = re.sub(r"\n\n+", "\n\n", description)
    description = re.sub(r"(?<!\n)\n(?!\n)", " ", description)
    description = description.strip()

    try:
        return markdown.markdown(
            description, extensions=["extra"], output_format="html5"
        )
    except Exception as e:
        logger.warning(f"Markdown rendering failed, using fallback: {e}")
        return _simple_markdown_to_html(description)


def _simple_markdown_to_html(description: str) -> str:
    """Simple markdown parser supporting headers, lists, and paragraphs."""
    lines = description.strip().split("\n")
    html_parts: list[str] = []
    in_list = False
    paragraph_buffer: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_buffer:
            text = " ".join(paragraph_buffer)
            html_parts.append(
                f'<p class="description-paragraph">{html.escape(text)}</p>'
            )
            paragraph_buffer.clear()

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            html_parts.append("</ul>")
            in_list = False

    for line in lines:
        line = line.strip()

        if not line:
            flush_paragraph()
            close_list()
            continue

        if line.startswith("#"):
            flush_paragraph()
            close_list()
            header = html.escape(line.lstrip("#").strip())
            html_parts.append(f'<h4 class="description-header">{header}</h4>')
            continue

        if line.startswith(("-", "•")):
            flush_paragraph()
            if not in_list:
                html_parts.append('<ul class="description-list">')
                in_list = True
            item = html.escape(line.lstrip("-•").strip())
            html_parts.append(f"<li>{item}</li>")
            continue

        close_list()
        paragraph_buffer.append(line)

    flush_paragraph()
    close_list()

    return "\n".join(html_parts)
