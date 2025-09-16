import hashlib, os, pathlib

def md5_of(path: str) -> str:
    md5 = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    return md5.hexdigest()

root = pathlib.Path(__file__).parent
lines = []
for p in sorted(root.rglob('*')):
    if p.is_file() and p.name not in ('checksums.md',):
        h = md5_of(str(p))
        rel = p.relative_to(root).as_posix()
        line = f"{h}  {rel}"
        print(line)
        lines.append(line)

(root/'checksums.md').write_text("\n".join(lines), encoding='utf-8')
print("\nMD5 list saved to checksums.md")
