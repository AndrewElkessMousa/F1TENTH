#!/usr/bin/env python3
import csv
import sys


def main():
    if len(sys.argv) != 3:
        print('Usage: convert_centerline_format.py input.txt output.csv')
        return 1

    rows = []
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().replace(';', ',').replace('\t', ',')
            if not line:
                continue
            parts = [p for p in line.split(',') if p]
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                rows.append((x, y))
            except ValueError:
                continue

    with open(sys.argv[2], 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f'Wrote {len(rows)} points to {sys.argv[2]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
