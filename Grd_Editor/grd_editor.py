import sys
import xarray as xr
import yaml
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 后端
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
import numpy as np
from scipy.spatial import ConvexHull

selected_polygons = []

def onselect(verts):
    """多边形选择回调，将多边形顶点保存到列表并绘制视觉反馈"""
    print("多边形顶点：", verts)
    selected_polygons.append(verts)
    print(f"已选择多边形 {len(selected_polygons)} 个。")
    # 绘制红色线条显示当前多边形（不干扰选择器）
    ax.plot([v[0] for v in verts] + [verts[0][0]], [v[1] for v in verts] + [verts[0][1]], 'r-', linewidth=1)
    fig.canvas.draw()

def apply_mask(z, mask):
    """将所有选择的多边形应用到 z.values 中"""
    x, y = np.meshgrid(z.lon, z.lat)
    coords = np.vstack([x.flatten(), y.flatten()]).T
    for verts in selected_polygons:
        path = Path(verts)
        inside = path.contains_points(coords).reshape(z.shape)
        mask &= ~inside
    z.values[~mask] = np.nan
    return z

def on_key(event):
    """按键事件：按 'q' 退出并保存"""
    if event.key == 'q':
        plt.close(fig)

def quadtree(xy, val, thresh, maxlevel, startlevel = 1, ax = None, cmap=plt.cm.terrain, norm=  None):
    """四叉树降采样函数（Python实现，基于提供的MATLAB代码）"""
    if xy.shape[0] < 3:
        return 0, 0, 0, np.empty((0,2)), np.array([]), [], np.empty((0,5)), np.empty((0,5))

    if startlevel == 1:
        if ax is None:
            fig, ax = plt.subplots(figsize=(7,7))

    if startlevel > maxlevel:
        try:
            hull = ConvexHull(xy[:,1:3])
            ax.plot(xy[hull.vertices,1], xy[hull.vertices,2], color=[0.8,0.8,0.8])
        except:
            pass
        return 0, 0, 0, np.empty((0,2)), np.array([]), [], np.empty((0,5)), np.empty((0,5))

    x = xy[:,1]
    y = xy[:,2]

    dx = np.max(x) - np.min(x)
    dy = np.max(y) - np.min(y)

    cx = np.min(x) + dx / 2
    cy = np.min(y) + dy / 2

    lim = [np.min(x), np.max(x), np.min(y), np.max(y)]
    xlims = np.array([lim[0], lim[0], lim[1], lim[1], lim[0]])
    ylims = np.array([lim[2], lim[3], lim[3], lim[2], lim[2]])

    mpoly = np.nanmean(val)
    spoly = np.nanvar(val)

    if spoly < thresh:
        if startlevel == 1:
            raise ValueError(f'Quadtree threshold variance ({thresh}) is larger than data variance ({spoly}). Please, lower it.')
        try:
            hull = ConvexHull(xy[:,1:3])
            vertices = hull.vertices
            hull_x = xy[vertices,1]
            hull_y = xy[vertices,2]
            poly = Polygon(np.column_stack((hull_x, hull_y)), closed=True, facecolor=cmap(norm(mpoly)) if norm is not None else 'gray', edgecolor='k', alpha=0.5)
            ax.add_patch(poly)
            polys_out = [xy[vertices,:]]
        except:
            polys_out = []
        nb = 1
        err = np.nansum(np.abs(val - mpoly))  # 处理NaN
        npoints = len(val)
        centers = np.array([cx, cy])
        values = mpoly
        return nb, err, npoints, centers, values, polys_out, xlims, ylims

    nb = 0
    err = 0
    npoints = np.array([], dtype=int)
    centers = np.empty((0,2))
    values = np.array([])
    polys = []
    xlims_all = np.empty((0,5))
    ylims_all = np.empty((0,5))

    for i in range(1,5):
        if i == 1:
            xyv = np.array([
                [cx-dx, cy-dy],
                [cx-dx, cy],
                [cx, cy],
                [cx, cy-dy]
            ])
        elif i == 2:
            xyv = np.array([
                [cx, cy-dy],
                [cx, cy],
                [cx+dx, cy],
                [cx+dx, cy-dy]
            ])
        elif i == 3:
            xyv = np.array([
                [cx-dx, cy],
                [cx-dx, cy+dy],
                [cx, cy+dy],
                [cx, cy]
            ])
        elif i == 4:
            xyv = np.array([
                [cx, cy],
                [cx, cy+dy],
                [cx+dx, cy+dy],
                [cx+dx, cy]
            ])
        if xyv.shape[0] > 0:
            in_idx = np.where(
                (x <= xyv[2,0]) & (x >= xyv[0,0]) & (y <= xyv[1,1]) & (y >= xyv[0,1])
            )[0]
            if len(in_idx) > 0:
                pnb, perr, pnpoints, pcenters, pvalues, ppolys, pxlims, pylims = quadtree(xy[in_idx], val[in_idx], thresh, maxlevel, startlevel+1, ax=ax, cmap=cmap, norm=norm)
                if pnb > 0:
                    nb += pnb
                    err += perr
                    npoints = np.append(npoints, pnpoints)
                    if centers.shape[0] == 0:
                        centers = pcenters if pcenters.ndim == 2 else pcenters.reshape(1, -1)
                    else:
                        centers = np.vstack((centers, pcenters if pcenters.ndim == 2 else pcenters.reshape(1, -1)))
                    values = np.append(values, pvalues)
                    polys.extend(ppolys)
                    if xlims_all.shape[0] == 0:
                        xlims_all = pxlims if pxlims.ndim == 2 else pxlims.reshape(1, -1)
                    else:
                        xlims_all = np.vstack((xlims_all, pxlims if pxlims.ndim == 2 else pxlims.reshape(1, -1)))
                    if ylims_all.shape[0] == 0:
                        ylims_all = pylims if pylims.ndim == 2 else pylims.reshape(1, -1)
                    else:
                        ylims_all = np.vstack((ylims_all, pylims if pylims.ndim == 2 else pylims.reshape(1, -1)))

    return nb, err, npoints, centers, values, polys, xlims_all, ylims_all

if __name__ == "__main__":
    yaml_path = sys.argv[1]
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    grd_path = config["grd_path"]
    output_path = config["output_path"]
    # 从YAML读取bounding_box（可选）
    bounding_box = config.get("bounding_box", None)  # 如果未设置，则None，全域处理

    data = xr.open_dataset(grd_path)
    z = data['z']
    mask = np.ones(z.shape, dtype=bool)

    # 调试数据信息
    print("z.lon:", z.lon.values)
    print("z.lat:", z.lat.values)
    print("z.shape:", z.shape)
    print("NaN count:", np.isnan(z.values).sum())

    # 准备四叉树输入数据
    lon_grid, lat_grid = np.meshgrid(z.lon, z.lat)
    xy = np.column_stack((np.arange(z.size), lon_grid.flatten(), lat_grid.flatten()))
    val = z.values.flatten()
    vmin = np.nanmin(val)
    vmax = np.nanmax(val)
    norm = Normalize(vmin, vmax)

    # 如果有bounding_box，只过滤该区域内的点
    if bounding_box is not None:
        lon1, lat1, lon2, lat2 = bounding_box
        lon_min = min(lon1, lon2)
        lon_max = max(lon1, lon2)
        lat_min = min(lat1, lat2)
        lat_max = max(lat1, lat2)
        mask_boundary = (lon_grid >= lon_min) & (lon_grid <= lon_max) & (lat_grid >= lat_min) & (lat_grid <= lat_max)
        xy = xy[mask_boundary.flatten()]
        val = val[mask_boundary.flatten()]
        print(f"应用四叉树于边界区域：lon [{lon_min}, {lon_max}], lat [{lat_min}, {lat_max}]，点数：{len(val)}")
        if len(val) == 0:
            print("边界区域内无数据点，跳过四叉树。")

    # 创建绘图窗口
    fig, ax = plt.subplots()
    img = ax.imshow(z.values, extent=[z.lon.min(), z.lon.max(), z.lat.min(), z.lat.max()],
                    origin="lower", cmap="terrain")
    plt.colorbar(img, ax=ax, label="Elevation")

    # 应用四叉树降采样并叠加到绘图中（调整thresh和maxlevel以适合您的数据）
    if len(val) > 0:
        nb, err, npoints, centers, values, polys, xlims, ylims = quadtree(xy, val, thresh=1, maxlevel=10, ax=ax, cmap=img.cmap, norm=norm)
        print(f"四叉树降采样结果：多边形数量 {nb}，总误差 {err}")
    else:
        print("无有效数据进行四叉树。")

    # 创建 PolygonSelector
    selector = PolygonSelector(ax, onselect, useblit=True)
    print("请在图像窗口绘制多边形：左键单击添加顶点，右键删除顶点，双击或右键闭合多边形。")
    print("绘制完一个多边形后，按 'esc' 键开始绘制新多边形。按 'q' 保存并退出。")

    # 添加按键事件处理
    fig.canvas.mpl_connect('key_press_event', on_key)

    # 使用 plt.show() 处理事件循环
    plt.show()

    # 应用遮罩并保存
    if selected_polygons:
        z = apply_mask(z, mask)
        z.to_netcdf(output_path)
        print(f"已保存修改后的 grd 文件：{output_path}")
    else:
        print("未选择任何多边形，未保存文件。")