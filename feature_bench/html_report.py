import argparse
import html
import json
from pathlib import Path


STYLE = """
body { font-family: system-ui, sans-serif; margin: 2rem; color: #222; }
h1 { font-size: 1.3rem; }
h2 { font-size: 1.05rem; margin-top: 2rem; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { border: 1px solid #ccc; padding: 0.25rem 0.6rem; text-align: right; }
th { background: #f2f2f2; }
td:first-child, th:first-child { text-align: left; }
"""


def format_value(value) -> str:
    if value is None:
        return "&ndash;"
    if isinstance(value, float):
        return f"{value:.3f}"
    return html.escape(str(value))


def render_table(columns: list, rows: list[list]) -> str:
    head = "".join(f"<th>{format_value(c)}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{format_value(v)}</td>" for v in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_report(report: dict) -> str:
    details = {
        "benchmark": report["benchmark"],
        "timestamp": report["timestamp"],
        "commit": report["git"]["commit"],
        "dirty": report["git"]["dirty"],
        **report["env"],
        **report["settings"],
    }
    sections = [
        f"<h1>{html.escape(report['title'])}</h1>",
        render_table(["setting", "value"], [[k, v] for k, v in details.items()]),
    ]
    for table in report["tables"]:
        sections.append(f"<h2>{html.escape(table['title'])}</h2>")
        sections.append(render_table(table["columns"], table["rows"]))

    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(report['title'])}</title><style>{STYLE}</style></head>"
        f"<body>{''.join(sections)}</body></html>"
    )


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
