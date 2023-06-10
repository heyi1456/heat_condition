from __future__ import print_function
import getopt
import sys
import numpy as np
import time
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import cuda
from mpi4py import MPI
import os

# CUDA内核函数
@cuda.jit
def evolve_kernel(u, u_previous, a, dt, dx2, dy2):
    i, j = cuda.grid(2)
    if 1 <= i < u.shape[0] - 1 and 1 <= j < u.shape[1] - 1:
        u[i, j] = u_previous[i, j] + a * dt * (
                (u_previous[i + 1, j] - 2 * u_previous[i, j] + u_previous[i - 1, j]) / dx2 +
                (u_previous[i, j + 1] - 2 * u_previous[i, j] + u_previous[i, j - 1]) / dy2)


# 主要的计算函数
def evolve(u, u_previous, a, dt, dx2, dy2):
    d_u = cuda.to_device(u)
    d_u_previous = cuda.to_device(u_previous)
    block_size = (16, 16)
    grid_size = (u.shape[0] // block_size[0] + 1, u.shape[1] // block_size[1] + 1)
    evolve_kernel[grid_size, block_size](d_u, d_u_previous, a, dt, dx2, dy2)
    return d_u.copy_to_host()  # 返回u


# 初始化numpy矩阵字段
def init_fields(lenX, lenY, Tguess, Ttop, Tbottom, Tleft, Tright):
    # 初始化
    field = np.empty((lenX, lenY), dtype=np.float64)
    field.fill(Tguess)
    field[(lenY - 1):, :] = Ttop
    field[:1, :] = Tbottom
    field[:, (lenX - 1):] = Tright
    field[:, :1] = Tleft
    field0 = field.copy()  # 上一步的温度场数组
    return field, field0


# 保存图像
def write_field(X, Y, field, step, size, device, lenX, timesteps):
    plt.gca().clear()
    # 配置等高线
    plt.title("Temperature")
    plt.contourf(X, Y, field, levels=50, cmap=plt.cm.jet)
    if step == 0:
        plt.colorbar()

    # 获取绝对路径
    img_dir = os.path.abspath('img')

    # 保存图像
    plt.savefig(
        f'{img_dir}/heat_{size}_{device}_{lenX}_{timesteps}_{step}.png')


# 主函数
def main(lenX, lenY, timesteps, image_interval, size, device):
    # 基本参数设置
    a = 0.5  # 扩散系数
    delta = 1

    # 边界条件
    Ttop = 100
    Tbottom = 0
    Tleft = 0
    Tright = 0

    # 初始猜测的内部温度场
    Tguess = 30

    # 网格间距
    dx = 0.01
    dy = 0.01
    dx2 = dx ** 2
    dy2 = dy ** 2

    # 时间步长
    dt = dx2 * dy2 / (2 * a * (dx2 + dy2))

    # 设置颜色插值和颜色映射
    colorinterpolation = 50
    colourMap = plt.cm.jet

    # 设置网格
    X, Y = np.meshgrid(np.arange(0, lenX), np.arange(0, lenY))

    # MPI全局变量
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # 上下相邻的MPI线程
    up = rank - 1
    if up < 0:
        up = MPI.PROC_NULL
    down = rank + 1
    if down > size - 1:
        down = MPI.PROC_NULL

    # MPI线程之间的通信（上下）
    def exchange(field):
        # 向下发送，从上接收
        sbuf = field[-2, :]
        rbuf = field[0, :]
        comm.Sendrecv(sbuf, dest=down, recvbuf=rbuf, source=up)
        # 向上发送，从下接收
        sbuf = field[1, :]
        rbuf = field[-1, :]
        comm.Sendrecv(sbuf, dest=up, recvbuf=rbuf, source=down)

    # 迭代
    def iterate(field, local_field, local_field0, timesteps, image_interval, X, Y):
        for m in tqdm(range(1, timesteps + 1)):
            exchange(local_field0)
            comm.Barrier()
            local_field = evolve(local_field, local_field0, a, dt, dx2, dy2)  # 接收evolve函数的返回值
            local_field0 = local_field.copy()  # 将local_field0设置为local_field的副本
            if m % image_interval == 0:
                comm.Gather(local_field[1:-1, :], field, root=0)
                comm.Barrier()
                if rank == 0:
                    write_field(X, Y, field, m, size, device, lenX, timesteps)

        # 迭代结束后再进行一次evolve操作
        local_field = evolve(local_field, local_field0, a, dt, dx2, dy2)
        comm.Gather(local_field[1:-1, :], field, root=0)
        if rank == 0:
            write_field(X, Y, field, timesteps, size, device, lenX, timesteps)

    # 主程序
    # 读取和分发初始温度场
    if rank == 0:
        field, field0 = init_fields(lenX, lenY, Tguess, Ttop, Tbottom, Tleft, Tright)
        shape = field.shape
        dtype = field.dtype
    else:
        field = None
        field0 = None
        shape = None
        dtype = None

    shape = comm.bcast(shape, root=0)  # 广播维度
    dtype = comm.bcast(dtype, root=0)  # 广播数据类型

    if shape[0] % size:
        raise ValueError('温度场的行数（' + str(shape[0]) + '）必须能够被MPI任务数（' + str(size) + '）整除。')

    n = shape[0] // size  # 每个MPI任务的行数
    m = shape[1]  # 温度场的列数
    buff = np.zeros((n, m), dtype)
    comm.Scatter(field, buff, root=0)  # 分发数据
    local_field = np.zeros((n + 2, m), dtype)  # 需要两行边界
    local_field[1:-1, :] = buff

    # 将数据复制到非边界行
    local_field0 = np.zeros_like(local_field)  # 用于上一步的温度场数组

    # 修正边界边界行以考虑非周期性
    if True:
        if rank == 0:
            local_field[0, :] = local_field[1, :]
        if rank == size - 1:
            local_field[-1, :] = local_field[-2, :]

    local_field0[:] = local_field[:]

    # 绘制/保存初始温度场
    write_field(X, Y, field, 0, size, device, lenX, timesteps)

    # 迭代
    t0 = time.time()
    iterate(field, local_field, local_field0, timesteps, image_interval, X, Y)
    t1 = time.time()

    # 绘制/保存最终温度场
    comm.Gather(local_field[1:-1, :], field, root=0)
    if rank == 0:
        write_field(X, Y, field, timesteps, size, device, lenX, timesteps)
        print("运行时间: {0}".format(t1 - t0))


if __name__ == '__main__':
    lenX = 100
    lenY = 100
    timesteps = 10000
    image_interval = 1000
    size = 1
    device = 'gpu'
    main(lenX, lenY, timesteps, image_interval, size, device)
