#!/usr/bin/env python
"""Assert document[start:end]==answer for all snippets; print mismatch count."""
from lib_common import load_config, load_benchmark, subset_tests, read_contract


def main():
    cfg = load_config()
    tests = subset_tests(load_benchmark(cfg), cfg["subset_size"])
    mismatches = 0
    total = 0
    for t in tests:
        for s in t["snippets"]:
            total += 1
            doc = read_contract(s["file_path"], cfg)
            start, end = s["span"]
            extracted = doc[start:end]
            if extracted != s["answer"]:
                mismatches += 1
    print(f"Mismatches: {mismatches}/{total}")


if __name__ == "__main__":
    main()

