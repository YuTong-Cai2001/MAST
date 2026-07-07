import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_agent_analysis(analysis_file):
    # 加载分析数据
    data = torch.load(analysis_file)
    
    # 创建一个包含两个子图的图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # 绘制Agent权重变化
    weights_data = np.array(data['agent_weights_history'])
    epochs = data['epochs']  # 现在这个是实际的训练轮数
    
    for i in range(weights_data.shape[1]):
        ax1.plot(epochs, weights_data[:, i], 
                label=f'Agent {i+1}', 
                linewidth=2)
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Agent Weights')
    ax1.set_title('Agent Weight Distribution Over Training')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制Agent性能对比
    performance_data = np.array(data['agent_performance_history'])
    
    for i in range(performance_data.shape[1]):
        ax2.plot(epochs, performance_data[:, i],
                label=f'Agent {i+1}',
                linewidth=2)
    
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Reconstruction Loss')
    ax2.set_title('Agent Performance Over Training')
    ax2.legend()
    ax2.grid(True)
    
    # 添加最终epoch标注
    final_epoch = epochs[-1]
    ax1.text(0.02, 0.98, f'Final Epoch: {final_epoch}', 
             transform=ax1.transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('agent_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    plot_agent_analysis('checkpoints/agent_analysis.pt') 