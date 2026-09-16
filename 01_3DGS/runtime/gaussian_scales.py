# 运行结束后的高斯存储地址：/data/output/point_cloud/iteration_30000/point_cloud.ply
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData

def analyze_scales(ply_path, output_img):
    # 读取PLY文件
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    
    # 提取缩放参数（假设存储为对数尺度，需要指数转换）
    # 提取缩放参数（正确的手动指数转换实现）
    scales = np.exp(np.vstack([vertices['scale_0'], 
                            vertices['scale_1'],
                            vertices['scale_2']]).T)
    
    # 统计指标计算
    stats = {
        'scale_0': {
            'min': np.min(scales[:,0]),
            'max': np.max(scales[:,0]),
            'mean': np.mean(scales[:,0]),
            'median': np.median(scales[:,0]),
            'std': np.std(scales[:,0])
        },
        'scale_1': {
            'min': np.min(scales[:,1]),
            'max': np.max(scales[:,1]),
            'mean': np.mean(scales[:,1]),
            'median': np.median(scales[:,1]),
            'std': np.std(scales[:,1])
        },
        'scale_2': {
            'min': np.min(scales[:,2]),
            'max': np.max(scales[:,2]),
            'mean': np.mean(scales[:,2]),
            'median': np.median(scales[:,2]),
            'std': np.std(scales[:,2])
        }
    }

    # 绘制分布图
    plt.figure(figsize=(15,5))
    bin_counts = []
    
    for i in range(3):
        ax = plt.subplot(1,3,i+1)
        n, bins, patches = ax.hist(scales[:,i], bins=50, alpha=0.7)
        
        # 标注统计信息
        info = f"Min: {stats[f'scale_{i}']['min']:.3f}\n" \
               f"Max: {stats[f'scale_{i}']['max']:.3f}\n" \
               f"Mean: {stats[f'scale_{i}']['mean']:.3f}\n" \
               f"Median: {stats[f'scale_{i}']['median']:.3f}\n" \
               f"Std: {stats[f'scale_{i}']['std']:.3f}"
        
        ax.text(0.65, 0.95, info, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.2))
        
        ax.set_xlabel(f'Scale {i} (m)')
        ax.set_ylabel('Count')
        ax.set_title(f'Scale {i} Distribution')
        ax.grid(True)

        # 记录区间分布数据
        interval_stats = []
        for left, right, count in zip(bins[:-1], bins[1:], n):
            interval_stats.append({
                'interval': f"{left:.3f}-{right:.3f}",
                'count': int(count),
                'percentage': f"{count/len(scales)*100:.1f}%"
            })
        bin_counts.append(interval_stats)
        stats[f'scale_{i}']['interval_distribution'] = interval_stats

    # 保存量化统计结果
    output_txt = output_img.replace('.png', '_stats.txt')
    with open(output_txt, 'w') as f:
        for i in range(3):
            f.write(f"=== Scale {i} 区间分布 ===\n")
            f.write("区间范围(m)\t\t数量\t占比\n")
            for entry in bin_counts[i]:
                f.write(f"{entry['interval']}\t{entry['count']}\t{entry['percentage']}\n")
            f.write("\n")

    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"可视化结果已保存至: {output_img}")
    print(f"量化统计已保存至: {output_txt}")

if __name__ == "__main__":

    input = 'data/output/point_cloud/iteration_30000/point_cloud.ply'
    output = 'data/output/gaussian_scale_stats.png'

    analyze_scales(input, output)