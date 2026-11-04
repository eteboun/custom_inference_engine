import argparse
import html
import json
from pathlib import Path


STYLE = """
body { font-family: system-ui, sans-serif; margin: 2rem; color: #222; }
h1 { font-size: 1.3rem; }
h2 { font-size: 1.05rem; margin-top: 2rem; }
h3 { font-size: 0.95rem; margin-top: 1.2rem; }
section { margin-top: 2.5rem; border-top: 2px solid #999; }
h4 { font-size: 0.9rem; margin-top: 1rem; }
.columns { display: flex; gap: 2rem; align-items: flex-start; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { border: 1px solid #ccc; padding: 0.25rem 0.6rem; text-align: right; }
th { background: #f2f2f2; }
td:first-child, th:first-child { text-align: left; }
"""


def format_value(value) -> str:
    if value is None:
        return "&ndash;"
    if isinstance(value, float):
        return f"{value:.4g}"
    return html.escape(str(value))


def render_table(columns: list, rows: list[list]) -> str:
    head = "".join(f"<th>{format_value(c)}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{format_value(v)}</td>" for v in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_sections(report: dict, heading: str) -> list[str]:
    details = {
        "benchmark": report["benchmark"],
        "timestamp": report["timestamp"],
        "commit": report["git"]["commit"],
        "dirty": report["git"]["dirty"],
        **report["env"],
        **report["settings"],
    }
    sections = [render_table(["setting", "value"], [[k, v] for k, v in details.items()])]
    for table in report["tables"]:
        sections.append(f"<{heading}>{html.escape(table['title'])}</{heading}>")
        sections.append(render_table(table["columns"], table["rows"]))

    return sections


def render_page(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(title)}</title><style>{STYLE}</style></head>"
        f"<body><h1>{html.escape(title)}</h1>{body}</body></html>"
    )


def render_report(report: dict) -> str:
    return render_page(report["title"], "".join(render_sections(report, "h2")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()

    for path in args.inputs:
        with open(path, encoding="utf-8") as f:
            report = json.load(f)

        output = path.with_suffix(".html")
        with open(output, "w", encoding="utf-8") as f:
            f.write(render_report(report))
        print(f"saved: {output}")


if __name__ == "__main__":
    main()
