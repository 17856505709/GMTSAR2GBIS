import sys
import numpy as np
import yaml
from scipy.io import savemat
import pygmt as gmt
import os

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python gmtsar2gbis.py <yaml_path>")
        print("yaml_path: the absolute path of the yaml file")
        sys.exit()
    yaml_path = sys.argv[1]

    # 读取配置文件信息
    with open(yaml_path, "r") as f:
        config = yaml.load(f, Loader = yaml.FullLoader)
    unwrap_ll_path = config["unwrap_ll"]
    Heading = config["Heading"]
    mat_file_output = config["mat_file_output"]
    incidence = config["incidence"]
    print(f"unwrap_ll_path:\t{unwrap_ll_path}")
    print(f"Incidence:\t\t{incidence}")
    print(f"Heading Angle:\t{Heading}")
    print(f"mat_file_path:\t{mat_file_output}")
    unwrap_ll_dir_path, unwrap_ll_filename = os.path.split(unwrap_ll_path)
    print(f"unwrap_ll_dir_path:\t{unwrap_ll_dir_path}")

    # 合并unwrap_ll.grd 和 入射角grd 文件
    i = 0
    lon_filter, lat_filter, unwrap_filter, inc_filter = [], [], [], []
    gmt.grd2xyz(unwrap_ll_path, output_type = "file", outfile = f"{unwrap_ll_dir_path}/unwrap_ll.xyz")
    with open(f"{unwrap_ll_dir_path}/unwrap_ll.xyz", "r") as f1:
        for line1 in f1:
            i = i + 1
            line1_list = line1.split()
            lon = float(line1_list[0])
            lat = float(line1_list[1])
            unwrap = float(line1_list[2])
            if lon < -180:
                lon += 360
            if np.isnan(lon) or np.isnan(lat) or np.isnan(unwrap):
                continue
            # if i % 5 != 0:
            #     continue
            lon_filter.append(float(lon))
            lat_filter.append(float(lat))
            unwrap_filter.append(float(unwrap))
    print(len(unwrap_filter))
    print(len(lon_filter))
    print(len(lat_filter))
    heading_filter = np.full(len(lon_filter), Heading)
    inc_filter = np.full(len(lon_filter), incidence)

    lon_filter = np.array(lon_filter)
    lat_filter = np.array(lat_filter)
    unwrap_filter = np.array(unwrap_filter)

    # 保存至.mat文件中
    data_dict = {
        "Heading": heading_filter.reshape(-1, 1),
        "Inc": inc_filter.reshape(-1, 1),
        "Lat": lat_filter.reshape(-1, 1),
        "Lon": lon_filter.reshape(-1, 1),
        "Phase": unwrap_filter.reshape(-1, 1)
    }
    savemat(mat_file_output, data_dict)