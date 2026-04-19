#!/usr/bin/env python3
"""
Generate PDF comparison report of controller performance metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import os

def generate_comparison_pdf():
    base_path = os.path.expanduser('~/sim_ws/src/neural_network_control/neural_network_control/')
    metrics_file = os.path.join(base_path, 'performance_metrics.csv')
    
    if not os.path.exists(metrics_file):
        print(f"❌ Metrics file not found: {metrics_file}")
        return
    
    # Load data
    df = pd.read_csv(metrics_file)
    
    if len(df) == 0:
        print("❌ No metrics data found")
        return
    
    # Rename PINN_v2 to NN for display
    df['controller'] = df['controller'].replace('PINN_v2', 'NN')
    
    print(f"📊 Loaded {len(df)} metric records")
    
    # Create PDF
    pdf_path = os.path.join(base_path, 'NN_Controller_Comparison_Report.pdf')
    
    with PdfPages(pdf_path) as pdf:
        # ===== PAGE 1: TITLE & SUMMARY TABLE =====
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('F1TENTH Controller Performance Comparison Report', fontsize=20, fontweight='bold')
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Add timestamp
        timestamp_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        fig.text(0.5, 0.92, timestamp_text, ha='center', fontsize=10, style='italic')
        
        # Group by controller and compute statistics
        controllers = df['controller'].unique()
        summary_data = []
        
        for controller in controllers:
            controller_data = df[df['controller'] == controller]
            n_laps = len(controller_data)
            
            summary_data.append({
                'Controller': controller,
                'Laps': n_laps,
                'Avg Lap Time\n(s)': f"{controller_data['lap_time_s'].mean():.3f}",
                'Avg RMS X\nError (m)': f"{controller_data['rms_x_error_m'].mean():.4f}",
                'Avg RMS Y\nError (m)': f"{controller_data['rms_y_error_m'].mean():.4f}",
                'Avg RMS Pos\nError (m)': f"{controller_data['rms_positional_error_m'].mean():.4f}",
                'Avg CTE\n(m)': f"{controller_data['mean_cte_m'].mean():.4f}",
                'Max CTE\n(m)': f"{controller_data['max_cte_m'].mean():.4f}",
                'Steering\nSmooth': f"{controller_data['steering_smoothness_rad_per_sample'].mean():.4f}",
                'Avg\nCorrelation': f"{controller_data['trajectory_correlation'].mean():.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create table with better layout
        table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, 
                         cellLoc='center', loc='center', bbox=[0, 0.1, 1, 0.75])
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 2.5)
        
        # Style header
        for i in range(len(summary_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(summary_df) + 1):
            for j in range(len(summary_df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ===== PAGE 2: POSITIONAL ERROR COMPARISON =====
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Positional Error Analysis', fontsize=16, fontweight='bold')
        
        metrics_to_plot = [
            ('rms_x_error_m', 'RMS X Error (m)'),
            ('rms_y_error_m', 'RMS Y Error (m)'),
            ('rms_positional_error_m', 'RMS Positional Error (m)'),
            ('mean_cte_m', 'Mean CTE (m)')
        ]
        
        for ax, (metric, title) in zip(axes.flat, metrics_to_plot):
            for controller in controllers:
                controller_data = df[df['controller'] == controller]
                ax.plot(controller_data['lap'].values, controller_data[metric].values, 
                       marker='o', label=controller, linewidth=2)
            ax.set_xlabel('Lap')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ===== PAGE 3: CTE & SMOOTHNESS COMPARISON =====
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Cross-Track Error & Smoothness Analysis', fontsize=16, fontweight='bold')
        
        metrics_to_plot = [
            ('rms_cte_m', 'RMS CTE (m)'),
            ('max_cte_m', 'Max CTE (m)'),
            ('steering_smoothness_rad_per_sample', 'Steering Smoothness (rad/sample)'),
            ('trajectory_correlation', 'Trajectory Correlation')
        ]
        
        for ax, (metric, title) in zip(axes.flat, metrics_to_plot):
            for controller in controllers:
                controller_data = df[df['controller'] == controller]
                ax.plot(controller_data['lap'].values, controller_data[metric].values, 
                       marker='s', label=controller, linewidth=2)
            ax.set_xlabel('Lap')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ===== PAGE 4: LAP TIME & DETAILED COMPARISON =====
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Lap Time & Performance Summary', fontsize=16, fontweight='bold')
        
        # Lap time comparison
        ax = axes[0, 0]
        for controller in controllers:
            controller_data = df[df['controller'] == controller]
            ax.plot(controller_data['lap'].values, controller_data['lap_time_s'].values, 
                   marker='^', label=controller, linewidth=2, markersize=8)
        ax.set_xlabel('Lap')
        ax.set_ylabel('Lap Time (s)')
        ax.set_title('Lap Time Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Box plot: RMS Positional Error
        ax = axes[0, 1]
        data_to_box = [df[df['controller'] == c]['rms_positional_error_m'].values for c in controllers]
        ax.boxplot(data_to_box, labels=controllers)
        ax.set_ylabel('RMS Positional Error (m)')
        ax.set_title('RMS Positional Error Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Box plot: CTE
        ax = axes[1, 0]
        data_to_box = [df[df['controller'] == c]['mean_cte_m'].values for c in controllers]
        ax.boxplot(data_to_box, labels=controllers)
        ax.set_ylabel('Mean CTE (m)')
        ax.set_title('Cross-Track Error Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Average metrics bar chart
        ax = axes[1, 1]
        avg_rms = [df[df['controller'] == c]['rms_positional_error_m'].mean() for c in controllers]
        x_pos = np.arange(len(controllers))
        bars = ax.bar(x_pos, avg_rms, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax.set_xlabel('Controller')
        ax.set_ylabel('Avg RMS Positional Error (m)')
        ax.set_title('Average Performance Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(controllers, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, avg_rms)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ===== PAGE 5: DETAILED METRICS TABLE =====
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Detailed Per-Lap Metrics', fontsize=16, fontweight='bold')
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Create detailed table
        display_df = df[['controller', 'lap', 'lap_time_s', 'rms_x_error_m', 'rms_y_error_m', 
                        'rms_positional_error_m', 'rms_cte_m', 'steering_smoothness_rad_per_sample']].copy()
        display_df.columns = ['Controller', 'Lap', 'Time (s)', 'RMS X (m)', 'RMS Y (m)', 
                             'RMS Pos (m)', 'RMS CTE (m)', 'Steer Smooth']
        
        # Round numeric columns
        for col in display_df.columns[2:]:
            display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}')
        
        table = ax.table(cellText=display_df.values, colLabels=display_df.columns,
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 0.95])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        # Style header
        for i in range(len(display_df.columns)):
            table[(0, i)].set_facecolor('#2196F3')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(display_df) + 1):
            for j in range(len(display_df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f9f9f9')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"✅ PDF Report generated: {pdf_path}")
    return pdf_path

if __name__ == '__main__':
    generate_comparison_pdf()
