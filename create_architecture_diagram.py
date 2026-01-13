"""
Create Architecture Diagram for Hybrid Model Methodology
Shows: LSTM + RF → Meta-Features → XGBoost → Prediction
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
plt.title('Late Fusion Pipeline: Hybrid Student Success Prediction Framework', 
          fontsize=20, fontweight='bold', pad=20)

# Define colors
color_lstm = '#3498db'  # Blue
color_rf = '#2ecc71'    # Green
color_meta = '#e74c3c'  # Red
color_xgb = '#9b59b6'   # Purple
color_output = '#f39c12' # Orange

# ============= INPUT DATA =============
# Temporal Data
temporal_box = FancyBboxPatch((0.5, 6.5), 1.5, 1.5, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='black', facecolor='#ecf0f1', linewidth=2)
ax.add_patch(temporal_box)
ax.text(1.25, 7.25, 'Temporal\nData\n(32 weeks)', 
        ha='center', va='center', fontsize=11, fontweight='bold')

# Static Data
static_box = FancyBboxPatch((0.5, 4.5), 1.5, 1.5, 
                            boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor='#ecf0f1', linewidth=2)
ax.add_patch(static_box)
ax.text(1.25, 5.25, 'Static\nData\n(Background)', 
        ha='center', va='center', fontsize=11, fontweight='bold')

# ============= BASE MODELS =============
# LSTM Model
lstm_box = FancyBboxPatch((2.5, 6.5), 2, 1.5, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=color_lstm, linewidth=3)
ax.add_patch(lstm_box)
ax.text(3.5, 7.5, 'LSTM', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='white')
ax.text(3.5, 7.0, 'Sequential\nPatterns', ha='center', va='center', 
        fontsize=10, color='white')

# Random Forest Model
rf_box = FancyBboxPatch((2.5, 4.5), 2, 1.5, 
                        boxstyle="round,pad=0.1", 
                        edgecolor='black', facecolor=color_rf, linewidth=3)
ax.add_patch(rf_box)
ax.text(3.5, 5.5, 'Random Forest', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='white')
ax.text(3.5, 5.0, 'Static\nFeatures', ha='center', va='center', 
        fontsize=10, color='white')

# ============= ARROWS TO BASE MODELS =============
# Temporal → LSTM
arrow1 = FancyArrowPatch((2.0, 7.25), (2.5, 7.25),
                        arrowstyle='->', mutation_scale=30, 
                        linewidth=2, color='black')
ax.add_patch(arrow1)

# Static → RF
arrow2 = FancyArrowPatch((2.0, 5.25), (2.5, 5.25),
                        arrowstyle='->', mutation_scale=30, 
                        linewidth=2, color='black')
ax.add_patch(arrow2)

# ============= META-FEATURES =============
meta_box = FancyBboxPatch((5.0, 5.5), 2, 2, 
                         boxstyle="round,pad=0.1", 
                         edgecolor='black', facecolor=color_meta, linewidth=3)
ax.add_patch(meta_box)
ax.text(6.0, 6.8, 'Meta-Features', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='white')
ax.text(6.0, 6.2, 'LSTM Probs', ha='center', va='center', 
        fontsize=10, color='white')
ax.text(6.0, 5.9, 'RF Probs', ha='center', va='center', 
        fontsize=10, color='white')

# ============= ARROWS TO META-FEATURES =============
# LSTM → Meta
arrow3 = FancyArrowPatch((4.5, 7.25), (5.0, 6.7),
                        arrowstyle='->', mutation_scale=30, 
                        linewidth=2, color='black')
ax.add_patch(arrow3)

# RF → Meta
arrow4 = FancyArrowPatch((4.5, 5.25), (5.0, 5.8),
                        arrowstyle='->', mutation_scale=30, 
                        linewidth=2, color='black')
ax.add_patch(arrow4)

# ============= XGBOOST META-LEARNER =============
xgb_box = FancyBboxPatch((7.5, 5.5), 1.8, 2, 
                        boxstyle="round,pad=0.1", 
                        edgecolor='black', facecolor=color_xgb, linewidth=3)
ax.add_patch(xgb_box)
ax.text(8.4, 6.8, 'XGBoost', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='white')
ax.text(8.4, 6.3, 'Meta-\nLearner', ha='center', va='center', 
        fontsize=11, color='white')

# Arrow: Meta → XGBoost
arrow5 = FancyArrowPatch((7.0, 6.5), (7.5, 6.5),
                        arrowstyle='->', mutation_scale=30, 
                        linewidth=2, color='black')
ax.add_patch(arrow5)

# ============= FINAL PREDICTION =============
pred_box = FancyBboxPatch((7.5, 3.5), 1.8, 1.5, 
                         boxstyle="round,pad=0.1", 
                         edgecolor='black', facecolor=color_output, linewidth=3)
ax.add_patch(pred_box)
ax.text(8.4, 4.5, 'Prediction', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='white')
ax.text(8.4, 4.0, 'Pass/Fail\n+ Risk Level', ha='center', va='center', 
        fontsize=10, color='white')

# Arrow: XGBoost → Prediction
arrow6 = FancyArrowPatch((8.4, 5.5), (8.4, 5.0),
                        arrowstyle='->', mutation_scale=30, 
                        linewidth=2, color='black')
ax.add_patch(arrow6)

# ============= ANNOTATIONS =============
# Add "Late Fusion" label
ax.text(6.0, 8.5, 'LATE FUSION ARCHITECTURE', 
        ha='center', va='center', fontsize=16, 
        fontweight='bold', bbox=dict(boxstyle='round', 
        facecolor='yellow', alpha=0.7))

# Add methodology notes
note_text = (
    "• Temporal features (32 weeks) → LSTM\n"
    "• Static features (40+ variables) → Random Forest\n"
    "• Predictions combined as meta-features\n"
    "• XGBoost learns optimal fusion strategy"
)
ax.text(0.5, 2.5, note_text, fontsize=10, 
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

# Add performance indicator
perf_text = "✓ Improved accuracy over individual models"
ax.text(8.4, 2.8, perf_text, ha='center', fontsize=11, 
        fontweight='bold', color='green')

plt.tight_layout()

# Save the figure
output_path = r'c:\Users\kule9\Videos\hybrid_framework\outputs\methodology_architecture.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Architecture diagram saved to: {output_path}")

# Also save as high-res for presentation
output_path_highres = r'c:\Users\kule9\Videos\hybrid_framework\outputs\methodology_architecture_highres.png'
plt.savefig(output_path_highres, dpi=600, bbox_inches='tight', facecolor='white')
print(f"✅ High-resolution diagram saved to: {output_path_highres}")

plt.show()
