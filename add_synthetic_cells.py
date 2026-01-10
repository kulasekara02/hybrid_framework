"""
Script to add synthetic data generation cells to the notebook
"""

import json
import shutil
import sys
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Backup the notebook first
notebook_path = r'c:\Users\kule9\hybrid_framework\hybrid_framework_complete.ipynb'
backup_path = f'{notebook_path}.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
shutil.copy(notebook_path, backup_path)
print(f'✓ Backup created: {backup_path}')

# Load notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f'✓ Loaded notebook with {len(nb["cells"])} cells')

# Configuration cell to insert after cell 5 (data loading)
config_cell_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 2.5 SYNTHETIC DATA CONFIGURATION\n",
        "**Toggle between real and synthetic data for experiments**"
    ]
}

config_cell_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# SYNTHETIC DATA CONFIGURATION\n",
        "# Set USE_SYNTHETIC_DATA = True to generate and use synthetic data\n",
        "# Set USE_SYNTHETIC_DATA = False to use real data (default)\n",
        "# ============================================================================\n",
        "\n",
        "USE_SYNTHETIC_DATA = False  # Toggle this to switch between real and synthetic data\n",
        "NUM_SYNTHETIC_STUDENTS = 1000  # Number of synthetic students to generate\n",
        "SYNTHETIC_RANDOM_SEED = 42  # Change this for different random datasets\n",
        "\n",
        "print(f'Data mode: {\"SYNTHETIC\" if USE_SYNTHETIC_DATA else \"REAL\"}')\n",
        "if USE_SYNTHETIC_DATA:\n",
        "    print(f'Will generate {NUM_SYNTHETIC_STUDENTS} synthetic students with seed {SYNTHETIC_RANDOM_SEED}')"
    ]
}

generator_cell_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# GENERATE SYNTHETIC DATA (if enabled)\n",
        "# ============================================================================\n",
        "\n",
        "if USE_SYNTHETIC_DATA:\n",
        "    print('\\n' + '='*80)\n",
        "    print('GENERATING SYNTHETIC DATA')\n",
        "    print('='*80 + '\\n')\n",
        "    \n",
        "    # Initialize the generator from Cell 0\n",
        "    generator = SyntheticStudentDataGenerator(\n",
        "        num_students=NUM_SYNTHETIC_STUDENTS,\n",
        "        num_weeks=32,\n",
        "        random_seed=SYNTHETIC_RANDOM_SEED\n",
        "    )\n",
        "    \n",
        "    # Generate the datasets\n",
        "    print('Generating students with complex, non-linear relationships...')\n",
        "    df_static_synth, df_temporal_synth, _, _ = generator.generate_datasets(output_dir='synthetic_data')\n",
        "    \n",
        "    # Split into Latvia and Global (50-50 split)\n",
        "    n_latvia = len(df_static_synth) // 2\n",
        "    \n",
        "    # Latvia subset\n",
        "    df_latvia_static = df_static_synth.iloc[:n_latvia].copy()\n",
        "    latvia_student_ids = df_latvia_static['student_id'].tolist()\n",
        "    df_latvia_temporal = df_temporal_synth[df_temporal_synth['student_id'].isin(latvia_student_ids)].copy()\n",
        "    \n",
        "    # Global subset  \n",
        "    df_global_static = df_static_synth.iloc[n_latvia:].copy()\n",
        "    global_student_ids = df_global_static['student_id'].tolist()\n",
        "    df_global_temporal = df_temporal_synth[df_temporal_synth['student_id'].isin(global_student_ids)].copy()\n",
        "    \n",
        "    # Combine (same as real data loading)\n",
        "    df_static = pd.concat([df_latvia_static, df_global_static], ignore_index=True)\n",
        "    df_temporal = pd.concat([df_latvia_temporal, df_global_temporal], ignore_index=True)\n",
        "    \n",
        "    print('\\n' + '='*80)\n",
        "    print('✓ SYNTHETIC DATA GENERATED SUCCESSFULLY')\n",
        "    print('='*80)\n",
        "    print(f'\\nTotal students (static): {len(df_static)}')\n",
        "    print(f'Total temporal records: {len(df_temporal)}')\n",
        "    print(f'Unique students in temporal: {df_temporal[\"student_id\"].nunique()}')\n",
        "    print(f'\\nSuccess rate: {df_static[\"success_label\"].mean():.1%}')\n",
        "    print(f'Average GPA: {df_static[\"gpa_prev\"].mean():.2f}')\n",
        "    print(f'Average engagement: {df_static[\"mean_weekly_engagement\"].mean():.3f}')\n",
        "    print('\\nRisk category distribution:')\n",
        "    print(df_static['risk_category'].value_counts())\n",
        "    print('\\n' + '='*80)\n",
        "else:\n",
        "    print('Using real data (already loaded in previous cell)')"
    ]
}

# Insert cells after cell 5 (data loading)
insert_position = 6

nb['cells'].insert(insert_position, config_cell_markdown)
nb['cells'].insert(insert_position + 1, config_cell_code)
nb['cells'].insert(insert_position + 2, generator_cell_code)

print(f'✓ Inserted 3 new cells at position {insert_position}')
print(f'  - Cell {insert_position}: Markdown header')
print(f'  - Cell {insert_position + 1}: Configuration toggle')
print(f'  - Cell {insert_position + 2}: Synthetic data generation')

# Save notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'\n✓ Notebook updated successfully!')
print(f'✓ Total cells now: {len(nb["cells"])}')
print(f'\n📝 To use synthetic data:')
print(f'   1. Run Cell 0 (Synthetic Data Generator class)')
print(f'   2. Run Cell {insert_position + 1} and set USE_SYNTHETIC_DATA = True')
print(f'   3. Run Cell {insert_position + 2} to generate synthetic data')
print(f'   4. Continue with the rest of the notebook normally')
