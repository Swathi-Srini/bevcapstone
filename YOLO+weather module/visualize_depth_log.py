"""
Visualize YOLO + Depth Estimation Results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

def visualize_depth_log(csv_path='../../manual_drive_output/yolo_depth_log.csv', output_dir='./depth_visualizations'):
    """
    Create visualizations from depth log CSV.
    
    Args:
        csv_path: Path to yolo_depth_log.csv or yolo_depth_realtime.csv
        output_dir: Directory to save plots
    """
    # Load data
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"✗ CSV not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"✓ Loaded {len(df)} detections from {csv_path}")
    print("\n" + "="*60)
    print("📊 DEPTH STATISTICS")
    print("="*60)
    print(df[['class', 'confidence', 'depth_m']].describe())
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('YOLO + Depth Estimation Analysis', fontsize=16, fontweight='bold')
    
    # 1. Depth distribution
    ax = axes[0, 0]
    ax.hist(df['depth_m'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Depth (meters)')
    ax.set_ylabel('Frequency')
    ax.set_title('Depth Distribution')
    ax.grid(True, alpha=0.3)
    
    # 2. Depth by class
    ax = axes[0, 1]
    df.boxplot(column='depth_m', by='class', ax=ax)
    ax.set_xlabel('Class')
    ax.set_ylabel('Depth (meters)')
    ax.set_title('Depth by Object Class')
    plt.sca(ax)
    plt.xticks(rotation=45)
    
    # 3. Confidence vs Depth
    ax = axes[0, 2]
    classes = df['class'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    for i, cls in enumerate(sorted(classes)):
        cls_data = df[df['class'] == cls]
        ax.scatter(cls_data['confidence'], cls_data['depth_m'], 
                  label=cls, alpha=0.6, s=50, color=colors[i])
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Depth (meters)')
    ax.set_title('Confidence vs Depth')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 4. Detection count by class
    ax = axes[1, 0]
    class_counts = df['class'].value_counts()
    ax.barh(class_counts.index, class_counts.values, color='coral', edgecolor='black')
    ax.set_xlabel('Number of Detections')
    ax.set_title('Detections by Class')
    for i, v in enumerate(class_counts.values):
        ax.text(v + 1, i, str(v), va='center')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 5. Detections over time
    ax = axes[1, 1]
    step_counts = df.groupby('step').size()
    ax.plot(step_counts.index, step_counts.values, color='green', linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Step (Frame)')
    ax.set_ylabel('Detections per Frame')
    ax.set_title('Detections Over Time')
    ax.grid(True, alpha=0.3)
    
    # 6. Mean depth by class
    ax = axes[1, 2]
    mean_depth = df.groupby('class')['depth_m'].mean().sort_values()
    ax.barh(mean_depth.index, mean_depth.values, color='skyblue', edgecolor='black')
    ax.set_xlabel('Mean Depth (meters)')
    ax.set_title('Average Depth by Class')
    for i, v in enumerate(mean_depth.values):
        ax.text(v + 0.2, i, f'{v:.1f}m', va='center')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'depth_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {plot_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("📈 DETECTION STATISTICS")
    print("="*60)
    print(f"\nTotal Detections: {len(df)}")
    print(f"Frames with Detections: {df['step'].nunique()}")
    print(f"\nDepth Range: {df['depth_m'].min():.2f}m to {df['depth_m'].max():.2f}m")
    print(f"Mean Depth: {df['depth_m'].mean():.2f}m")
    print(f"Median Depth: {df['depth_m'].median():.2f}m")
    print(f"Std Dev: {df['depth_m'].std():.2f}m")
    
    print(f"\nConfidence Range: {df['confidence'].min():.3f} to {df['confidence'].max():.3f}")
    print(f"Mean Confidence: {df['confidence'].mean():.3f}")
    
    print("\nDetections by Class:")
    for cls, count in df['class'].value_counts().items():
        avg_conf = df[df['class'] == cls]['confidence'].mean()
        avg_depth = df[df['class'] == cls]['depth_m'].mean()
        print(f"  {cls:15} : {count:3} detections | Avg Conf: {avg_conf:.3f} | Avg Depth: {avg_depth:.2f}m")
    
    print("\n" + "="*60)
    
    # Show plot
    plt.show()


if __name__ == '__main__':
    import sys
    
    # Get CSV path from command line or use default
    csv_path = sys.argv[1] if len(sys.argv) > 1 else '../../manual_drive_output/yolo_depth_log.csv'
    
    visualize_depth_log(csv_path)
