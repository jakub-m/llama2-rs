from pathlib import Path
from string import Template
import sys


def main():
    readme_tpl_content = Path("./README.md.tpl").open().read()
    template_values = {}
    for mermaid_path in Path(".").glob("*.mermaid"):
        value_name = mermaid_path.stem
        value_body = mermaid_path.open().read()
        template_values[value_name] = value_body
        print(
            f"{mermaid_path} -> {value_name} {len(value_body) // 1024}kB",
            file=sys.stderr,
        )
    rendered = Template(readme_tpl_content).substitute(**template_values)
    print(rendered)


main()
