import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.patheffects as path_effects

def get_loss_from_log(log_file, target_epoch):
    """从日志文件中提取指定epoch的损失值"""
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # 寻找目标epoch的日志行
    target_line = None
    for line in lines:
        if f'epoch: {target_epoch:4}' in line:
            target_line = line
            break
    
    if target_line:
        # 提取kl_loss和semantic_consistency_loss
        # 由于这两个损失是total_loss的组成部分
        # 我们可以从total_loss中提取它们的贡献
        parts = target_line.split(',')
        kl_loss = 0.0
        semantic_loss = 0.0
        for part in parts:
            if 'kl_loss' in part:
                kl_loss = float(part.split(':')[1].strip())
            if 'semantic_consistency_loss' in part:
                semantic_loss = float(part.split(':')[1].strip())
        
        return kl_loss + semantic_loss  # 返回总的对齐损失
    
    return None

def get_losses_from_checkpoint(checkpoint_dir, best_midpoint):
    """从checkpoint中获取所有损失值"""
    model_epoch = (int(best_midpoint) // 10) * 10
    checkpoint_path = Path(checkpoint_dir) / f'gdan_{model_epoch}.pkl'
    
    try:
        checkpoint_data = torch.load(checkpoint_path)
        if 'loss_values' in checkpoint_data:
            loss_values = checkpoint_data['loss_values']
            return {
                'kl_loss': loss_values['kl_loss'],
                'semantic_consistency_loss': loss_values['semantic_consistency_loss'],
                'semantic_alignment_loss': loss_values['semantic_alignment_loss']
            }
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
    return None

def plot_comprehensive_analysis(eval_results_file, agent_analysis_file, checkpoint_dir, log_file):
    # 加载评估结果和agent分析数据
    eval_results = torch.load(eval_results_file)
    agent_data = torch.load(agent_analysis_file)
    
    # 创建图表布局
    fig = plt.figure(figsize=(15, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    
    epochs = eval_results['model_epochs']
    test_acc = eval_results['test_acc']
    val_acc = eval_results['val_acc']
    
    # 将epoch调整到每个区间的中点
    interval_midpoints = np.array(epochs) + 5
    
    # 添加垂直网格线
    for e in range(0, int(max(epochs)) + 11, 10):
        ax1.axvline(x=e, color='gray', linestyle='--', alpha=0.3)
    
    # 绘制主要曲线
    ax1.plot(interval_midpoints, test_acc, 'b-', label='Test Accuracy', linewidth=2)
    ax1.plot(interval_midpoints, val_acc, 'r--', label='Validation Accuracy', linewidth=2)
    
    # 找到最佳性能点
    best_idx = np.argmax(test_acc)
    best_midpoint = interval_midpoints[best_idx]
    best_acc = test_acc[best_idx]
    interval_start = best_midpoint - 5
    interval_end = best_midpoint + 5
    
    # 获取所有损失值
    losses = get_losses_from_checkpoint(checkpoint_dir, best_midpoint)
    
    # 标注最佳区间（黄色阴影）
    ax1.axvspan(interval_start, interval_end, 
                color='yellow', alpha=0.2, label='Best Interval')
    
    # 标注最佳点（红点）和损失值说明
    ax1.scatter(best_midpoint, best_acc, c='red', s=100, zorder=5)
    
    if losses is not None:
        # 创建一个圆圈标注
        circle = plt.Circle((best_midpoint + 15, best_acc), radius=0.02, 
                          fill=False, color='red', linestyle='--')
        ax1.add_patch(circle)
        
        # 添加损失值和相关代码的说明（移到右下角）
        ax1.annotate(
            f'$\\mathbf{{Cross-domain\\ Semantic\\ Alignment\\ Loss}}$: {losses["semantic_alignment_loss"]:.4f}\n\n'
            'Composed of:\n'
            '1. KL Distribution Loss:\n'
            '   kl_loss = torch.mean((μ_source - μ_target)²)\n'
            '2. Semantic Consistency Loss:\n'
            '   for agent in agents:\n'
            '       s_loss += MSE(agent.semantic_score(source),\n'
            '                     agent.semantic_score(target))',
            xy=(best_midpoint, best_acc),
            xytext=(best_midpoint + 20, best_acc - 0.15),
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='arc3,rad=-0.2'
            ),
            fontsize=10,
            weight='normal',
            color='black',
            path_effects=[
                path_effects.withStroke(linewidth=2, foreground='blue', alpha=0.2)
            ]
        )
        
        annotation_text = (f'Best Interval: {int(interval_start)}-{int(interval_end)}\n'
                         f'Best Model at Epoch {int(best_midpoint)}\n'
                         f'Accuracy: {best_acc:.4f}')
    else:
        annotation_text = (f'Best Interval: {int(interval_start)}-{int(interval_end)}\n'
                         f'Best Model at Epoch {int(best_midpoint)}\n'
                         f'Accuracy: {best_acc:.4f}')
    
    # 添加基本注释
    ax1.annotate(annotation_text,
                xy=(best_midpoint, best_acc),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 设置主图属性
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Performance on Test Set')
    ax1.legend()
    ax1.grid(True, axis='y')
    
    # Agent权重分布图（保持不变）
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    weights_data = np.array(agent_data['agent_weights_history'])
    agent_epochs = agent_data['epochs']
    
    for e in range(0, int(max(agent_epochs)) + 11, 10):
        ax2.axvline(x=e, color='gray', linestyle='--', alpha=0.3)
    
    for i in range(weights_data.shape[1]):
        ax2.plot(agent_epochs, weights_data[:, i], 
                label=f'Agent {i+1}', linewidth=2)
    
    ax2.axvspan(interval_start, interval_end, color='yellow', alpha=0.2)
    
    mask = (np.array(agent_epochs) >= interval_start) & \
           (np.array(agent_epochs) < interval_end)
    if np.any(mask):
        best_interval_weights = weights_data[mask]
        avg_weights = np.mean(best_interval_weights, axis=0)
        entropy = -(avg_weights * np.log(avg_weights + 1e-6)).sum()
        
        ax2.text(0.02, 0.98, 
                f'Best Interval ({int(interval_start)}-{int(interval_end)}):\n' + 
                '\n'.join([f'Agent {i+1}: {w:.3f}' for i, w in enumerate(avg_weights)]) +
                f'\nWeight Entropy: {entropy:.4f}',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Agent Weights')
    ax2.set_title('Agent Weight Distribution')
    ax2.legend()
    ax2.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('comprehensive_analysis_with_loss.pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    plot_comprehensive_analysis(
        'result/cross_domain_apy_to_awa2/full_evaluation_results.pt',
        'data/checkpoints/gdan_cross_domain/agent_analysis.pt',
        'data/checkpoints/gdan_cross_domain',
        'data/checkpoints/gdan_cross_domain/gdan_log.txt'
    ) 