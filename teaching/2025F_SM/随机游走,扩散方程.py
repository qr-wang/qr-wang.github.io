import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 获取用户输入（改进版）
user_input = input("请输入粒子数（默认500，推荐100-2000，直接回车使用默认值）：")
try:
    n = int(user_input)
    if n <= 0:  # 处理非正整数输入
        print(f"输入值{n}无效，已启用默认值500")
        n = 500
except ValueError:  # 处理非数字输入
    print("输入无效，已启用默认值500")
    n = 500

# 模拟参数
#n = 500       # 粒子数
D = 0.1       # 扩散系数
dt = 0.01     # 时间步长
num_steps = 100  # 总步数
total_time = num_steps * dt  # 总时长
x_lim = 8     # 显示范围

# 初始化粒子位置
positions = np.zeros((n, 2))

# 创建图形和子图
fig = plt.figure(figsize=(12, 6))
ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)  # 左侧主图
ax2 = plt.subplot2grid((2, 2), (0, 1))             # 右上方平均绝对位移
ax3 = plt.subplot2grid((2, 2), (1, 1))             # 右下方均方位移

# 设置坐标轴范围
ax1.set_xlim(-x_lim, x_lim)
ax1.set_ylim(-x_lim, x_lim)
ax1.set_title('粒子扩散模拟')
ax1.set_xlabel('X 位置')
ax1.set_ylabel('Y 位置')

# 添加实时时间显示文本
time_text = ax1.text(0.05, 0.90, '', transform=ax1.transAxes, 
                    fontsize=12, backgroundcolor='white', 
                    bbox=dict(facecolor='white', edgecolor='black'))

# 添加粒子数显示文本
n_text = ax1.text(0.05, 0.95, f'粒子数: {n}', transform=ax1.transAxes,
                 fontsize=12, backgroundcolor='white',
                 bbox=dict(facecolor='white', edgecolor='black'))

# 固定右侧子图横坐标
for ax in [ax2, ax3]:
    ax.set_xlim(0, total_time)
    ax.grid(True, alpha=0.3)

# 初始化图形元素
scatter = ax1.scatter([], [], s=8, alpha=0.6)
line_mean, = ax2.plot([], [], 'r-', lw=1.5, label='模拟值')
line_msd,  = ax3.plot([], [], 'b-', lw=1.5, label='模拟值')
t_values, mean_abs_disp_values, msd_values = [], [], []

# 理论曲线（验证正确性）
t_theory = np.linspace(0, total_time, 100)
# 均方位移理论值（二维扩散：4Dt）
ax3.plot(t_theory, 4*D*t_theory, 'k--', lw=1.2, label=r'理论值 $4Dt$') 
# 平均绝对位移理论值（二维：sqrt(πDt)）
ax2.plot(t_theory, np.sqrt(np.pi*D*t_theory), 'k--', lw=1.2, label=r'理论值 $\sqrt{\pi D t}$')
ax2.legend()
ax3.legend()

# 子图标签设置
ax2.set_title('平均绝对位移随时间变化')
ax2.set_xlabel('时间 t')
# ax2.set_ylabel('$\langle |r| \\rangle$')
ax2.set(ylabel=r'$\langle |r| \rangle$')  # 添加 r 前缀


ax3.set_title('均方位移随时间变化')
ax3.set_xlabel('时间 t')
# ax3.set_ylabel('$\langle r^2 \\rangle$')
ax3.set(ylabel=r'$\langle r^2 \rangle$')  # 添加 r 前缀


def init():
    """初始化动画"""
    scatter.set_offsets(np.zeros((n, 2)))
    line_mean.set_data([], [])
    line_msd.set_data([], [])
    time_text.set_text('')
    n_text.set_text(f'粒子数: {n}')  # 新增
    return scatter, line_mean, line_msd, time_text, n_text  # 新增返回元素
    
def update(frame):
    """动画更新函数"""
    global positions
    
    # 生成随机位移（二维正态分布）
    displacements = np.random.normal(0, np.sqrt(2*D*dt), (n, 2))
    positions += displacements
    
    # 更新散点图
    scatter.set_offsets(positions)
    
    # 计算当前时间
    current_time = (frame + 1) * dt
    time_text.set_text(f'时间: {current_time:.2f}s')
    
    # 计算统计量
    displacements = np.linalg.norm(positions, axis=1)
    mean_abs_disp = np.mean(displacements)
    msd = np.mean(positions[:,0]**2 + positions[:,1]**2)
    
    # 记录数据
    t_values.append(current_time)
    mean_abs_disp_values.append(mean_abs_disp)
    msd_values.append(msd)
    
    # 更新曲线
    line_mean.set_data(t_values, mean_abs_disp_values)
    line_msd.set_data(t_values, msd_values)
    
    # 自动调整纵坐标范围
    ax2.relim(visible_only=True)
    ax2.autoscale_view(scaley=True, scalex=False)
    ax3.relim(visible_only=True)
    ax3.autoscale_view(scaley=True, scalex=False)
    
    return scatter, line_mean, line_msd, time_text, n_text  # 新增返回元素

# 创建动画
ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=num_steps,
    init_func=init, 
    blit=True, 
    interval=30,
    repeat=False
)


# # To save the animation using Pillow as a gif
# writer = animation.PillowWriter(fps=10,
#                                 metadata=dict(artist='Me'),
#                                 bitrate=180)
# ani.save('随机游走,扩散方程.gif', writer=writer)

plt.tight_layout()
plt.show()
