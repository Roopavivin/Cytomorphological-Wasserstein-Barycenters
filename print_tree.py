import os

def print_tree(dir_path, level=0, max_level=3, ignore=None):
    if ignore is None:
        ignore = []
    
    if level >= max_level:
        return
        
    items = sorted(os.listdir(dir_path))
    items = [item for item in items if item not in ignore]
    
    for i, item in enumerate(items):
        is_last = (i == len(items) - 1)
        prefix = '    ' * level + ('\\--- ' if is_last else '+--- ')
        print(f"{prefix}{item}")
        
        item_path = os.path.join(dir_path, item)
        if os.path.isdir(item_path):
            print_tree(item_path, level + 1, max_level, ignore)

if __name__ == "__main__":
    print("wcb-ot")
    print_tree("wcb-ot", max_level=3, ignore=['__pycache__', '.venv', '.git'])
