# GMTSAR2GBIS
通过Python脚本处理将GMTSAR处理的结果转换成GBIS可用的数据格式
## Grd_editor
该程序通过读取config.yaml文件，将对应的GMTSAR生成的Grd文件进行可视化编辑，为便于与GBIS更好接轨，在可视化中进行相同算法的四叉树降采样，可据此框选需要去除的区域，尤其是由于掩膜导致的区域零散采样点过多的区域。
## GMTSAR2GBIS
该程序通过读取config.yaml文件，将对应的Grd文件，与影像对应的入射角数据、航向角数据转换成GBIS可读的.mat数据。（注：未进行5：1降采样）
