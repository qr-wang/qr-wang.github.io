import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.font_manager import FontProperties
from scipy.special import comb  # 用于计算组合数
import random
import platform
import math
import os

# 根据操作系统设置字体路径
if platform.system() == "Windows":
    font_path = r"c:\windows\fonts\simsun.ttc"  # Windows字体路径
elif platform.system() == "Darwin":  # macOS
    font_path = "/System/Library/Fonts/STHeiti Medium.ttc"  # macOS字体路径
else:
    font_path = None  # 其他系统，默认不设置

if font_path:
    font = FontProperties(fname=font_path, size=14)
    plt.rcParams['font.sans-serif'] = [font.get_name()]
    plt.rcParams['axes.unicode_minus'] = False
else:
    print("警告：未找到适合的中文字体路径，中文显示可能失败。")

# 初始化网格和粒子位置（按列优先填充）
def initialize_particles(L, N):
    particles = []
    for col in range(L):  # 从左到右遍历列
        for row in range(L):  # 从上到下遍历行
            if len(particles) >= N:  # 如果已经放置了N个粒子，则停止
                return particles
            particles.append((row, col))  # 添加粒子位置（行，列）
    return particles

# 粒子移动规则
def move_particle(particles, L):
    # 随机选择一个粒子
    idx = random.randint(0, len(particles) - 1)
    x, y = particles[idx]

    # 随机选择一个方向
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 右、下、左、上
    dx, dy = random.choice(directions)

    # 计算新的位置
    new_x, new_y = x + dx, y + dy

    # 检查新位置是否在网格内且没有被其他粒子占据
    if 0 <= new_x < L and 0 <= new_y < L and (new_x, new_y) not in particles:
        particles[idx] = (new_x, new_y)  # 更新粒子位置

    return particles

# 计算状态数Ω和熵S
def calculate_omega_and_entropy(particles, L):
    # 统计每一列的粒子数
    column_counts = [0] * L
    for x, y in particles:
        column_counts[y] += 1

    # 计算状态数Ω
    omega = 1
    for n_i in column_counts:
        omega *= comb(L, n_i)  # 组合数C(L, n_i)

    # 计算熵S（假设k_b = 1）
    if omega > 0:
        entropy = math.log(omega)
    else:
        entropy = 0

    return omega, entropy

# 动画更新函数
def update(frame, particles, L, circles, ax_left, ax_right, entropy_list, y_line, max_updates):
    # 显示当前进度
    progress = f"进度: {frame+1}/{max_updates}"
    if hasattr(update, 'progress_text'):
        update.progress_text.remove()
    update.progress_text = ax_left.text(0.5, -0.1, progress, 
                                       transform=ax_left.transAxes, fontsize=12, 
                                       ha="center", color="green")
    
    # 每100步更新一次熵值显示（如果总步数较少，改为每10步更新）
    update_interval = max(1, max_updates // 10)  # 动态调整更新间隔
    if frame % update_interval == 0:  
        # 计算状态数Ω和熵S
        omega, entropy = calculate_omega_and_entropy(particles, L)

        # 清除旧文本
        for txt in ax_left.texts:
            if 'Ω' in txt.get_text() or 'S' in txt.get_text():
                txt.remove()

        # 显示新文本
        ax_left.text(0.5, 1.05, f"Ω = {omega:.2e}, S = {entropy:.2f}", 
                     transform=ax_left.transAxes, fontsize=14, ha="center", color="red")

        # 记录熵S
        entropy_list.append(entropy)

        # 更新右侧子图
        ax_right.clear()
        ax_right.plot(range(len(entropy_list)), entropy_list, color='blue', linewidth=2)
        ax_right.axhline(y=y_line, color='red', linestyle='--', linewidth=1.5)  # 绘制水平直线
        ax_right.text(0, y_line, f"{y_line:.2f}", color='red', va='center', ha='right')  # 在左侧标注值
        ax_right.set_xlabel("时间步数", fontproperties=font)
        ax_right.set_ylabel("熵 S", fontproperties=font)
        ax_right.set_title("熵 S 随时间的变化", fontproperties=font)
        ax_right.grid(True, alpha=0.3)

    # 移动粒子
    particles = move_particle(particles, L)

    # 更新每个圆形的位置
    for i, circle in enumerate(circles):
        x, y = particles[i]
        circle.center = (y + 0.5, x + 0.5)  # 转换为绘图坐标

    return circles

# 创建动画
def animate(L, N, interval, max_updates=100, save_gif=False, gif_filename="particle_diffusion.gif"):
    particles = initialize_particles(L, N)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1]})
    
    # 设置图形标题
    fig.suptitle(f"粒子扩散模拟 - {L}x{L}网格, {N}个粒子", fontproperties=font, fontsize=16, y=0.95)

    # 设置左侧子图（粒子扩散动画）
    ax_left.set_aspect('equal')  # 确保网格是正方形
    ax_left.set_xlim(0, L)
    ax_left.set_ylim(0, L)
    ax_left.set_xticks(np.arange(L))
    ax_left.set_yticks(np.arange(L))
    ax_left.set_xticklabels([])
    ax_left.set_yticklabels([])
    ax_left.grid(True, alpha=0.5)
    ax_left.set_title(f"粒子位置分布", fontproperties=font, y=1.02)

    # 创建圆形表示粒子
    circles = []
    for x, y in particles:
        circle = Circle((y + 0.5, x + 0.5), 0.4, color='blue', alpha=0.7)  # 圆形半径为0.4
        ax_left.add_patch(circle)
        circles.append(circle)

    # 初始化右侧子图（熵 S 随时间的变化）
    entropy_list = []
    ax_right.set_xlabel("时间步数", fontproperties=font)
    ax_right.set_ylabel("熵 S", fontproperties=font)
    ax_right.set_title("熵 S 随时间的变化", fontproperties=font)
    ax_right.grid(True, alpha=0.3)

    # 计算水平直线的y坐标（理论最大熵）
    n_avg = N / L  # 每列平均粒子数
    binom = comb(L, n_avg)  # 组合数C(L, n_avg)
    y_line = L * math.log(binom)  # y = L * ln(C(L, n_avg))

    # 创建动画
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=max_updates,  # 限制为100次更新
        fargs=(particles, L, circles, ax_left, ax_right, entropy_list, y_line, max_updates), 
        interval=interval,
        blit=False,
        repeat=False
    )
    
    # 保存GIF
    if save_gif:
        print(f"正在保存GIF到: {gif_filename}")
        try:
            # 设置GIF写入器参数
            writer = animation.PillowWriter(fps=max(1, 1000//interval) if interval > 0 else 10,
                                          metadata=dict(artist='Particle Diffusion Simulation'),
                                          bitrate=1800)
            ani.save(gif_filename, writer=writer)
            print(f"GIF保存成功: {gif_filename}")
        except Exception as e:
            print(f"保存GIF时出错: {e}")
    
    plt.show()
    
    return ani

# 主程序
if __name__ == "__main__":
    print("=" * 50)
    print("粒子扩散模拟程序")
    print("=" * 50)
    
    # 获取用户输入，如果输入为空则使用默认值
    L_input = input("请输入网格大小 L（默认 10）: ")
    NL_input = input("请输入粒子列数 N/L（默认 2）: ")
    interval_input = input("请输入多少毫秒更新一帧（默认 2）: ")
    max_updates_input = input("请输入最大更新次数（默认 100）: ")
    
    # GIF保存选项
    save_gif_input = input("是否保存为GIF文件？(y/n, 默认 n): ")
    gif_filename = None
    if save_gif_input.lower() in ['y', 'yes']:
        gif_filename_input = input("请输入GIF文件名（默认 particle_diffusion.gif）: ")
        gif_filename = gif_filename_input if gif_filename_input else "particle_diffusion.gif"

    # 设置参数
    L = int(L_input) if L_input else 10
    NL = int(NL_input) if NL_input else 2
    N = NL * L
    interval = int(interval_input) if interval_input else 2
    max_updates = int(max_updates_input) if max_updates_input else 100
    save_gif = save_gif_input.lower() in ['y', 'yes']

    # 参数验证
    if N > L * L:
        print("错误：粒子数目不能超过网格大小！")
        print(f"网格大小: {L}x{L} = {L*L}, 粒子数目: {N}")
    else:
        print(f"\n模拟参数:")
        print(f"- 网格大小: {L}x{L}")
        print(f"- 粒子数目: {N}")
        print(f"- 更新间隔: {interval} 毫秒")
        print(f"- 总更新次数: {max_updates}")
        if save_gif:
            print(f"- 将保存为: {gif_filename}")
        
        print("\n开始模拟...")
        animate(L, N, interval, max_updates, save_gif, gif_filename)
