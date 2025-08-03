import os

OUTPUT_FILE = 'tree.txt'
ROOT_DIR = '.'  # 当前目录，也可以改成绝对路径
LIMITED_PATH = os.path.join('data', 'ShapeNetCore.v2.PC15k')
MAX_ITEMS = 2
IGNORED_DIRS = {'.git', 'outs', '__pycache__', '.ipynb_checkpoints', 'logs', 'docs'}  # 可添加更多要忽略的目录名


def write_tree(path, f, prefix=''):
    try:
        entries = sorted(os.listdir(path))
    except PermissionError:
        return

    entries = [e for e in entries if e not in IGNORED_DIRS]
    is_limited = os.path.relpath(path, ROOT_DIR).startswith(LIMITED_PATH)
    if is_limited:
        entries = entries[:MAX_ITEMS]

    for i, entry in enumerate(entries):
        full_path = os.path.join(path, entry)
        connector = '└── ' if i == len(entries) - 1 else '├── '
        f.write(f'{prefix}{connector}{entry}\n')
        if os.path.isdir(full_path):
            extension = '    ' if i == len(entries) - 1 else '│   '
            write_tree(full_path, f, prefix + extension)


def main():
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(f'{ROOT_DIR}/\n')
        write_tree(ROOT_DIR, f)


if __name__ == '__main__':
    main()
