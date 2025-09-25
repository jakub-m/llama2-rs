from pathlib import Path
from string import Template


def main():
    readme_tpl_content = Path("./README.md.tpl").open().read()
    detailed_mermaid = Path("./diagram.mermaid").open().read()
    rendered = Template(readme_tpl_content).substitute(
        detailed_mermaid=detailed_mermaid
    )
    print(rendered)


main()
