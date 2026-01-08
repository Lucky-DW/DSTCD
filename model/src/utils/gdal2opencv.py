"""
gdal,opencv互换
"""
from osgeo import gdal
import cv2
import numpy as np


# 读取tif数据集
def readTif(fileName):
    in_ds = gdal.Open(fileName)
    if in_ds == None:
        print(fileName + "文件无法打开")
    rows = in_ds.RasterYSize  # 获取数据高度
    cols = in_ds.RasterXSize  # 获取数据宽度
    bands = in_ds.RasterCount  # 获取数据波段数
    # 这个数据类型没搞清楚，1，2，3代表什么https://blog.csdn.net/u010579736/article/details/84594742
    # GDT_Byte = 1, GDT_UInt16 = 2, GDT_UInt32 = 4, GDT_Int32 = 5, GDT_Float32 = 6
    datatype = in_ds.GetRasterBand(1).DataType
    # print("数据类型：", datatype)

    array_data = in_ds.ReadAsArray()  # 将数据写成数组，读取全部数据，numpy数组,array_data.shape  (4, 36786, 37239) ,波段，行，列
    del in_ds
    if bands == 1:
        array_data = np.expand_dims(array_data, axis=0)
    if bands >= 4:
        array_data = array_data[0:3, :, :]
    array_data = array_data.transpose((1, 2, 0))

    array_data = cv2.cvtColor(array_data, cv2.COLOR_RGB2BGR)
    # print(array_data.shape)
    return array_data


# 获取投影信息
def Getproj(fileName):
    dataset = gdal.Open(fileName)
    return dataset.GetProjection()


# 获取仿射矩阵信息
def Getgeotrans(fileName):
    dataset = gdal.Open(fileName)
    return dataset.GetGeoTransform()


# opencv数据转gdal
def OpencvData2GdalData(OpencvImg_data):
    # 若为二维，格式相同
    if (len(OpencvImg_data.shape) == 2):
        GdalImg_data = OpencvImg_data
    else:
        if 'int8' in OpencvImg_data.dtype.name:
            GdalImg_data = np.zeros((OpencvImg_data.shape[2], OpencvImg_data.shape[0], OpencvImg_data.shape[1]),
                                    np.uint8)
        elif 'int16' in OpencvImg_data.dtype.name:
            GdalImg_data = np.zeros((OpencvImg_data.shape[2], OpencvImg_data.shape[0], OpencvImg_data.shape[1]),
                                    np.uint16)
        else:
            GdalImg_data = np.zeros((OpencvImg_data.shape[2], OpencvImg_data.shape[0], OpencvImg_data.shape[1]),
                                    np.float32)
        for i in range(OpencvImg_data.shape[2]):
            # 注意，opencv为BGR
            data = OpencvImg_data[:, :, OpencvImg_data.shape[2] - i - 1]
            data = np.reshape(data, (OpencvImg_data.shape[0], OpencvImg_data.shape[1]))
            GdalImg_data[i] = data
    return GdalImg_data


# gdal数据转opencv
def GdalData2OpencvData(GdalImg_data):
    if 'int8' in GdalImg_data.dtype.name:
        OpencvImg_data = np.zeros((GdalImg_data.shape[1], GdalImg_data.shape[2], GdalImg_data.shape[0]), np.uint8)
    elif 'int16' in GdalImg_data.dtype.name:
        OpencvImg_data = np.zeros((GdalImg_data.shape[1], GdalImg_data.shape[2], GdalImg_data.shape[0]), np.uint16)
    else:
        OpencvImg_data = np.zeros((GdalImg_data.shape[1], GdalImg_data.shape[2], GdalImg_data.shape[0]), np.float32)
    for i in range(GdalImg_data.shape[0]):
        OpencvImg_data[:, :, i] = GdalImg_data[GdalImg_data.shape[0] - i - 1, :, :]
    return OpencvImg_data


# 保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_height, im_width, im_bands = im_data.shape

    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape

    print(im_data.shape)

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)

    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[:, :, i])
        # dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

# 测试OpenCV转gdal
# proj = Getproj("D:\\cnn_matching-shenyang\\test_data\\ka_E1.tif")
# geotrans = Getgeotrans("D:\\cnn_matching-shenyang\\test_data\\ka_E1.tif")
# OpencvImg_data = cv2.imread("D:\\cnn_matching-shenyang\\test_data\\ka_E1.tif")
# GdalImg_data = OpencvData2GdalData(OpencvImg_data)
# writeTiff(GdalImg_data, geotrans, proj, "2.tif")

# 测试gdal转OpenCV
# dataset = readTif("2.tif")
# # dataset = gdal.Open("D:\\cnn_matching-shenyang\\test_data\\ka1_E1.tif")
# width = dataset.RasterXSize  # 栅格矩阵的列数
# height = dataset.RasterYSize  # 栅格矩阵的行数
# GdalImg_data = dataset.ReadAsArray(0, 0, width, height)  # 获取数据
# OpencvImg_data = GdalData2OpencvData(GdalImg_data)
# # cv2.imwrite("3.tif", OpencvImg_data)
# print('done')
