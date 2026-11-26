#!/usr/bin/env python3
"""Resolve code anchors in a markdown file: [text](../path#L{{pattern}}) or [text](../path#L{{after-pattern>>pattern}})
becomes [text](../path#L<line>), the first line containing `pattern` (after the first line containing `after-pattern`).
    tools/doc_anchors.py SRC.md OUT.md POC_DIR"""
import re
import sys


def main(src, out, root):
    text = open(src).read()
    cache = {}

    def lines(path):
        if path not in cache:
            cache[path] = open(f"{root}/{path}").read().splitlines()
        return cache[path]

    def resolve(m):
        path, spec = m.group(1), m.group(2)
        after, pat = spec.split(">>", 1) if ">>" in spec else (None, spec)
        ls = lines(path)
        i0 = 0
        if after is not None:
            i0 = next(i for i, l in enumerate(ls) if after in l)
        for i in range(i0, len(ls)):
            if pat in ls[i]:
                return f"](../{path}#L{i + 1})"
        raise SystemExit(f"anchor not found: {path}: {spec}")
    text = re.sub(r"\]\(\.\./([^)#]+)#L\{\{(.+?)\}\}\)", resolve, text)
    open(out, "w").write(text)


if __name__ == "__main__":
    main(*sys.argv[1:4])
