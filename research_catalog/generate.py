"""Generate a validated research index for the repository README."""

import json
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "research_catalog" / "research.json"
README = ROOT / "README.md"

START = "<!-- RESEARCH_INDEX_START -->"
END = "<!-- RESEARCH_INDEX_END -->"


def main():
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))

    if catalog["repository"] != ROOT.name:
        raise ValueError("Repository name does not match the catalog.")

    entries = catalog["research"]
    if not entries:
        raise ValueError("Research catalog cannot be empty.")

    identifiers = set()
    lines = [
        "## Research Programs & Scientific Implementations",
        "",
        "The following research programs have corresponding scientific implementations in this repository.",
        "",
    ]

    for entry in entries:
        identifier = entry["id"]
        title = entry["title"]

        if identifier in identifiers:
            raise ValueError(f"Duplicate research identifier: {identifier}")
        identifiers.add(identifier)

        lines.extend(
            [
                f"### Research {identifier} — {title}",
                "",
                entry["description"],
                "",
            ]
        )

        for field, heading in (
            ("implementation", "Scientific implementation"),
            ("tests", "Automated validation"),
            ("experiments", "Research experiments"),
        ):
            paths = entry[field]
            if not paths:
                continue

            lines.append(f"**{heading}**")
            lines.append("")

            for relative_path in paths:
                path = Path(relative_path)

                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Unsafe path: {relative_path}")

                target = ROOT / path
                if not target.is_file():
                    raise FileNotFoundError(relative_path)

                url = quote(path.as_posix(), safe="/")
                lines.append(f"- [`{path.name}`]({url})")

            lines.append("")

    generated = "\n".join(lines).rstrip()
    content = README.read_text(encoding="utf-8")

    if START in content or END in content:
        if content.count(START) != 1 or content.count(END) != 1:
            raise ValueError("Invalid research index markers.")

        before, remainder = content.split(START, 1)
        _, after = remainder.split(END, 1)
        updated = before + START + "\n" + generated + "\n" + END + after
    else:
        heading = "## Research Domains"
        if content.count(heading) != 1:
            raise ValueError("Expected Research Domains heading not found.")

        section = START + "\n" + generated + "\n" + END + "\n\n"
        updated = content.replace(heading, section + heading, 1)

    if updated != content:
        README.write_text(updated, encoding="utf-8")
        print("RESEARCH INDEX UPDATED")
    else:
        print("RESEARCH INDEX ALREADY CURRENT")


if __name__ == "__main__":
    main()
