#!/usr/bin/env python3
"""Bundle local CSS, JS, images and citation into one portable HTML preview."""
import argparse
import base64
import html
import mimetypes
from pathlib import Path
import re


def build(output):
    root = Path(__file__).resolve().parent.parent
    document = (root / 'index.html').read_text(encoding='utf-8')

    def asset(path):
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError('Asset is outside the project: ' + path)
        return resolved

    def stylesheet(match):
        path = match.group(1)
        return '<style data-source="' + path + '">\n' + asset(path).read_text(encoding='utf-8') + '\n</style>'

    def javascript(match):
        path = match.group(1)
        code = asset(path).read_text(encoding='utf-8')
        if '</script' in code.lower():
            raise ValueError('Inline script terminator in ' + path)
        return '<script data-source="' + path + '">\n' + code + '\n</script>'

    def inline_asset(match):
        attribute, path = match.group(1), match.group(2)
        mime = mimetypes.guess_type(path)[0] or 'application/octet-stream'
        if path.endswith('.bib'):
            mime = 'application/x-bibtex'
        encoded = base64.b64encode(asset(path).read_bytes()).decode('ascii')
        return attribute + '="data:' + mime + ';base64,' + encoded + '"'

    document = re.sub(r'<link rel="stylesheet" href="([^"]+)">', stylesheet, document)
    document = re.sub(r'<script defer src="([^"]+)"></script>', javascript, document)
    document = re.sub(r'(src|href)="(static/(?:images|files)/[^"]+)"', inline_asset, document)
    # Preserve a useful filename after embedding the citation as a data URI.
    document = document.replace(' download>Download .bib', ' download="koneobench.bib">Download .bib')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding='utf-8')
    print(f'Created {output} ({output.stat().st_size:,} bytes)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('KoNeoBench_preview.html'))
    args = parser.parse_args()
    build(args.output.resolve())
