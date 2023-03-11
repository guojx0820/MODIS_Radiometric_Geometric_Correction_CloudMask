import numpy as np
from osgeo import gdal, osr
import os
from pyhdf.SD import SD
import cv2 as cv
import time


class MODIS_Radiometric_Geometric_Correction:
    def __init__(self, l1b_file, cloud_file, out_name):
        self.l1b_file = l1b_file
        self.cloud_file = cloud_file
        self.out_name = out_name
        self.geo_resolution = 0.01

    def _read_modis_data_(self):
        modis_l1b = SD(self.l1b_file)
        modis_cloud = SD(self.cloud_file)
        qkm_rad = self._radical_calibration_(modis_l1b, 'EV_250_Aggr1km_RefSB', 'radiance_scales',
                                             'radiance_offsets')
        hkm_rad = self._radical_calibration_(modis_l1b, 'EV_500_Aggr1km_RefSB', 'radiance_scales',
                                             'radiance_offsets')
        cloud_data = modis_cloud.select('Cloud_Mask').get()
        lon = modis_l1b.select('Longitude').get()
        lat = modis_l1b.select('Latitude').get()
        return qkm_rad, hkm_rad, cloud_data, lon, lat

    def _radical_calibration_(self, modis_l1b, dataset_name, scales, offsets):
        object = modis_l1b.select(dataset_name)
        data = object.get()
        scales = object.attributes()[scales]
        offsets = object.attributes()[offsets]
        data_rad = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.float64)
        for i_layer in range(data.shape[0]):
            data_rad[i_layer, :, :] = scales[i_layer] * (data[i_layer, :, :] - offsets[i_layer])
        return data_rad

    def _cloud_mask_(self, cloud_data):
        cloud_0 = cloud_data[0, :, :]
        cloud_0 = (np.int64(cloud_0 < 0) * (256 + cloud_0)) + (np.int64(cloud_0 >= 0) * cloud_0)
        cloud_binary = np.zeros((cloud_0.shape[0], cloud_0.shape[1], 8), dtype=np.int64)
        for i_cloud in range(8):
            cloud_binary[:, :, i_cloud] = cloud_0 % 2
            cloud_0 //= 2
        clear_result = np.int64(cloud_binary[:, :, 0] == 1) & np.int64(cloud_binary[:, :, 1] == 1) \
                       & np.int64(cloud_binary[:, :, 2] == 1)
        ocean_result = np.int64(cloud_binary[:, :, 6] == 0) & np.int64(cloud_binary[:, :, 7] == 0)
        cloud_result = np.int64(clear_result == 0) | np.int64(ocean_result == 0)
        return clear_result

    def _band_extract_(self, lon, lat, qkm_rad, hkm_rad, clear_result):
        blue_band = hkm_rad[0] * clear_result
        green_band = hkm_rad[1] * clear_result
        red_band = qkm_rad[0] * clear_result
        nir_band = qkm_rad[1] * clear_result
        com_bands = np.array([blue_band, green_band, red_band, nir_band])
        band_list = []
        for i_band in com_bands:
            geo_band, lon_min, lat_max = self._georeference_(lon, lat, i_band)
            filter_band = self._average_filtering_(geo_band)
            band_list.append(filter_band)
        band_arr = np.array(band_list)
        return band_arr, lon_min, lat_max

    def _georeference_(self, lon, lat, data):
        lon_interp = cv.resize(lon, (data.shape[1], data.shape[0]), interpolation=cv.INTER_LINEAR)
        lat_interp = cv.resize(lat, (data.shape[1], data.shape[0]), interpolation=cv.INTER_LINEAR)
        lon_min = np.min(lon_interp)
        lon_max = np.max(lon_interp)
        lat_min = np.min(lat_interp)
        lat_max = np.max(lat_interp)

        geo_box_col = np.int64(np.ceil((lon_max - lon_min) / self.geo_resolution))
        geo_box_row = np.int64(np.ceil((lat_max - lat_min) / self.geo_resolution))
        geo_box = np.zeros((geo_box_row, geo_box_col), dtype=np.float64)
        geo_box_col_pos = np.int64(np.floor((lon_interp - lon_min) / self.geo_resolution))
        geo_box_row_pos = np.int64(np.floor((lat_max - lat_interp) / self.geo_resolution))
        geo_box[geo_box_row_pos, geo_box_col_pos] = data

        return geo_box, lon_min, lat_max

    def _average_filtering_(self, geo_box):
        geo_box_plus = np.zeros((geo_box.shape[0] + 2, geo_box.shape[1] + 2), dtype=np.float64) - 9999.0
        geo_box_plus[1:geo_box.shape[0] + 1, 1:geo_box.shape[1] + 1] = geo_box
        geo_box_out = np.zeros((geo_box.shape[0], geo_box.shape[1]), dtype=np.float64)
        for i_geo_box_row in range(1, geo_box.shape[0] + 1):
            for i_geo_box_col in range(1, geo_box.shape[1] + 1):
                if geo_box_plus[i_geo_box_row, i_geo_box_col] == 0.0:
                    temp_window = geo_box_plus[i_geo_box_row - 1:i_geo_box_row + 2,
                                  i_geo_box_col - 1:i_geo_box_col + 2]
                    temp_window = temp_window[temp_window > 0]
                    temp_window_sum = np.sum(temp_window)
                    temp_window_num = np.sum(np.int64(temp_window > 0.0))
                    if temp_window_num > 3:
                        geo_box_out[i_geo_box_row - 1, i_geo_box_col - 1] = temp_window_sum / temp_window_num
                    else:
                        geo_box_out[i_geo_box_row - 1, i_geo_box_col - 1] = 0.0
                else:
                    geo_box_out[i_geo_box_row - 1, i_geo_box_col - 1] = geo_box_plus[
                        i_geo_box_row, i_geo_box_col]
        return geo_box_out

    def _write_tiff_(self, data, lon_min, lat_max):
        cols = data.shape[2]
        rows = data.shape[1]
        band_count = data.shape[0]
        driver = gdal.GetDriverByName('GTiff')
        out_raster = driver.Create(self.out_name, cols, rows, band_count, gdal.GDT_Float64)

        out_raster.SetGeoTransform((lon_min, self.geo_resolution, 0, lat_max, 0, self.geo_resolution))
        out_raster_SRS = osr.SpatialReference()
        # 代码4326表示WGS84坐标
        out_raster_SRS.ImportFromEPSG(4326)
        out_raster.SetProjection(out_raster_SRS.ExportToWkt())

        # 获取数据集第一个波段，是从1开始，不是从0开始
        for i_band_count in range(band_count):
            out_raster.GetRasterBand(i_band_count + 1).WriteArray(data[i_band_count])
        out_raster.FlushCache()
        out_raster = None


if __name__ == '__main__':
    start_time = time.time()
    input_directory = '/mnt/e/Experiments/AOD_Retrieval/DATA/MOD021KM_MOD35Cloud_202205/'
    output_directory = '/mnt/e/Experiments/AOD_Retrieval/DATA/Results/Results_MOD021KM_Rad_Geo_Cal/'
    if os.path.exists(output_directory) == False:
        os.makedirs(output_directory)
    for root, dirs, files in os.walk(input_directory):
        l1b_file_list = [input_directory + i_hdf for i_hdf in files if
                         i_hdf.endswith('.hdf') and i_hdf.startswith('MOD02')]
        cloud_file_list = [input_directory + i_hdf for i_hdf in files if
                           i_hdf.endswith('.hdf') and i_hdf.startswith('MOD35')]
    for i_l1b in l1b_file_list:
        for i_cloud in cloud_file_list:
            if os.path.basename(i_l1b)[10:22] == os.path.basename(i_cloud)[10:22]:
                start_time_each = time.time()
                out_name = output_directory + os.path.basename(i_l1b[:-4]) + '_Rad_Geo_Cor.tiff'
                modis_rad_geo_cor = MODIS_Radiometric_Geometric_Correction(i_l1b, i_cloud, out_name)
                qkm_rad, hkm_rad, cloud_data, lon, lat = modis_rad_geo_cor._read_modis_data_()
                clear_result = modis_rad_geo_cor._cloud_mask_(cloud_data)
                com_band, lon_min, lat_max = modis_rad_geo_cor._band_extract_(lon, lat, qkm_rad, hkm_rad, clear_result)
                modis_rad_geo_cor._write_tiff_(com_band, lon_min, lat_max)

                end_time_each = time.time()
                run_time_each = round(end_time_each - start_time_each, 3)
                print('The image of ' + os.path.basename(i_l1b)[10:22] + ' is saved! The time consuming is ' + str(
                    run_time_each) + ' s.')
    end_time = time.time()
    run_time = round(end_time - start_time, 3)
    print('The total time consuming is ' + str(run_time) + ' s.')
