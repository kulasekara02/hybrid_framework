"""
View and Export LSTM Probability Scores
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# This script assumes you have already run test2.ipynb and have these variables:
# - lstm_val_pred (validation probabilities)
# - y_val_bin (actual labels)
# - df_val (validation dataframe with student info)

# If running standalone, you need to load the model and predict
print("=" * 70)
print(" " * 15 + "📊 LSTM PROBABILITY SCORES VIEWER")
print("=" * 70)

# Check if running in notebook context
try:
    # If variables exist from notebook
    lstm_probabilities = lstm_val_pred
    actual_labels = y_val_bin
    print(f"\n✅ Found {len(lstm_probabilities)} LSTM predictions")
    
    # Create DataFrame with results
    results_df = pd.DataFrame({
        'Student_ID': range(len(lstm_probabilities)),
        'LSTM_Probability': lstm_probabilities,
        'LSTM_Prediction': (lstm_probabilities > 0.5).astype(int),
        'Actual_Label': actual_labels,
        'Correct': (lstm_probabilities > 0.5).astype(int) == actual_labels
    })
    
    # Add risk categories
    results_df['Risk_Level'] = pd.cut(
        results_df['LSTM_Probability'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['High Risk (Fail)', 'Medium Risk', 'Low Risk (Pass)']
    )
    
    # Display statistics
    print("\n" + "=" * 70)
    print("📈 LSTM PROBABILITY SCORE STATISTICS")
    print("=" * 70)
    print(f"\nMinimum Probability: {results_df['LSTM_Probability'].min():.4f}")
    print(f"Maximum Probability: {results_df['LSTM_Probability'].max():.4f}")
    print(f"Mean Probability:    {results_df['LSTM_Probability'].mean():.4f}")
    print(f"Median Probability:  {results_df['LSTM_Probability'].median():.4f}")
    print(f"Std Deviation:       {results_df['LSTM_Probability'].std():.4f}")
    
    # Risk distribution
    print("\n" + "=" * 70)
    print("🎯 RISK DISTRIBUTION")
    print("=" * 70)
    print(results_df['Risk_Level'].value_counts().to_string())
    
    # Show sample predictions
    print("\n" + "=" * 70)
    print("📋 SAMPLE LSTM PREDICTIONS (First 10 students)")
    print("=" * 70)
    print(results_df.head(10).to_string(index=False))
    
    # Save to CSV
    output_path = r'c:\Users\kule9\Videos\hybrid_framework\outputs\lstm_probability_scores.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Full results saved to: {output_path}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.patch.set_facecolor('white')
    
    # 1. Histogram of probabilities
    axes[0, 0].hist(results_df['LSTM_Probability'], bins=50, 
                    color='#3498db', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(x=0.5, color='red', linestyle='--', 
                      linewidth=2, label='Decision Threshold')
    axes[0, 0].set_xlabel('LSTM Probability Score', fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontweight='bold')
    axes[0, 0].set_title('📊 Distribution of LSTM Probability Scores', 
                        fontweight='bold', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Box plot by risk level
    risk_order = ['High Risk (Fail)', 'Medium Risk', 'Low Risk (Pass)']
    results_df['Risk_Level'] = pd.Categorical(results_df['Risk_Level'], 
                                              categories=risk_order, ordered=True)
    sns.boxplot(data=results_df, x='Risk_Level', y='LSTM_Probability', 
                ax=axes[0, 1], palette='Set2')
    axes[0, 1].set_xlabel('Risk Category', fontweight='bold')
    axes[0, 1].set_ylabel('LSTM Probability', fontweight='bold')
    axes[0, 1].set_title('📦 Probability Distribution by Risk Level', 
                        fontweight='bold', fontsize=14)
    axes[0, 1].tick_params(axis='x', rotation=15)
    axes[0, 1].grid(alpha=0.3, axis='y')
    
    # 3. Correct vs Incorrect predictions
    correct_probs = results_df[results_df['Correct']]['LSTM_Probability']
    incorrect_probs = results_df[~results_df['Correct']]['LSTM_Probability']
    
    axes[1, 0].hist([correct_probs, incorrect_probs], bins=30, 
                    label=['Correct', 'Incorrect'],
                    color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0.5, color='black', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('LSTM Probability Score', fontweight='bold')
    axes[1, 0].set_ylabel('Count', fontweight='bold')
    axes[1, 0].set_title('✅ Prediction Accuracy Analysis', 
                        fontweight='bold', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Risk level pie chart
    risk_counts = results_df['Risk_Level'].value_counts()
    colors_pie = ['#e74c3c', '#f39c12', '#2ecc71']
    axes[1, 1].pie(risk_counts.values, labels=risk_counts.index, 
                   autopct='%1.1f%%', colors=colors_pie, startangle=90,
                   textprops={'fontweight': 'bold', 'fontsize': 11})
    axes[1, 1].set_title('🎯 Student Risk Distribution', 
                        fontweight='bold', fontsize=14)
    
    plt.suptitle('LSTM Model: Probability Score Analysis', 
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save visualization
    viz_path = r'c:\Users\kule9\Videos\hybrid_framework\outputs\lstm_probability_visualization.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Visualization saved to: {viz_path}")
    
    plt.show()
    
except NameError:
    print("\n⚠️  Variables not found. Please run test2.ipynb first!")
    print("After running the notebook, the following variables will be available:")
    print("   - lstm_val_pred: LSTM probability scores")
    print("   - y_val_bin: Actual labels")
    print("\nOr load a trained model to generate predictions.")

print("\n" + "=" * 70)
print("✅ LSTM Probability Analysis Complete!")
print("=" * 70)
