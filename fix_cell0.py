"""Fix Cell 0 to remove auto-execution"""
import json
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

notebook_path = r'c:\Users\kule9\hybrid_framework\hybrid_framework_complete.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Get Cell 0 source
src = ''.join(nb['cells'][0]['source']) if isinstance(nb['cells'][0]['source'], list) else nb['cells'][0]['source']

# Find where the class ends (line 326 has "return static_df...")
lines = src.split('\n')

# Keep only lines up to and including the return statement in generate_datasets method (line 326)
class_only_lines = []
for i, line in enumerate(lines):
    class_only_lines.append(line)
    if i == 326:  # This is the last line of the generate_datasets method
        break

class_only = '\n'.join(class_only_lines)

# Update Cell 0
nb['cells'][0]['source'] = class_only

# Save
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('✓ Removed auto-execution from Cell 0')
print(f'✓ Cell 0 now contains only the SyntheticStudentDataGenerator class')
print(f'✓ Original size: {len(src)} chars, New size: {len(class_only)} chars')
