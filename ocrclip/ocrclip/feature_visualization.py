# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Email: yuwenwen62@gmail.com
# @Created Time: 5/14/22 6:41 PM

import cv2
import mmcv
import numpy as np
import os
import torch
import matplotlib.pyplot as plt


def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmaps.append(heatmap)

    return heatmaps

def draw_feature_map(features,title = 'title', save_dir = 'feature_map',img_name = None):
    i=0
    if isinstance(features,torch.Tensor):
        for heat_maps in features:
            # img_path = f'/home/wwyu/dataset/mmocr_det_data/icdar2015/imgs/test/{img_name}'
            # save_dir = '/home/wwyu/dataset/mmocr_det_data/icdar2015/vis_feat_ic15_1'
            # img_path = f'/home/wwyu/dataset/mmocr_det_data/ctw1500/imgs/test/{img_name}'
            # save_dir = '/home/wwyu/dataset/mmocr_det_data/ctw1500/vis_feat_ctw'
            # img_path = f'/home/wwyu/dataset/mmocr_det_data/total_text/imgs/test/{img_name}'
            # save_dir = '/home/wwyu/dataset/mmocr_det_data/total_text/vis_feat_tt'
            img_path = f'/home/wwyu/dataset/mmocr_det_data/td_tr/td500/test_images/{img_name}'
            save_dir = '/home/wwyu/dataset/mmocr_det_data/td_tr/vis_feat_td'

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            img = mmcv.imread(img_path)
            heat_maps=heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            # heatmap = cv2.resize(heatmap, (h, w))
            for idx, heatmap in enumerate(heatmaps):
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
                heatmap = np.uint8(255 * heatmap) # 将热力图转换为RGB格式
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # # 将热力图应用于原始图像
                # superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
                superimposed_img = heatmap * 0.5 + img*0.3
                # superimposed_img = heatmap
                # plt.imshow(superimposed_img,cmap='gray') # need BGR2RGB
                # plt.imshow(superimposed_img,cmap='jet')
                # plt.imshow(img,cmap='jet')
                # plt.title(title)
                # plt.show()
                save_file_name = os.path.join(save_dir, f'{os.path.basename(img_path).split(".")[0]}_{title}_'+str(idx)+'.png')
                # cv2.imwrite(save_file_name, superimposed_img)  # 将图像保存到硬盘
                # cv2.imshow("1",superimposed_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite(save_file_name, superimposed_img)

    else:
        for featuremap in features:
            img_path = '/home/wwyu/dataset/mmocr_det_data/ctw1500/imgs/test/1125.jpg'
            save_dir = '/home/wwyu/dataset/mmocr_det_data/ctw1500/vis_feat'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            img = mmcv.imread(img_path)
            heat_maps=heat_maps.unsqueeze(0)

            heatmaps = featuremap_2_heatmap(featuremap)
            # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            for idx, heatmap in enumerate(heatmaps):
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # 将热力图应用于原始图像
                superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
                # superimposed_img = heatmap * 0.5 + img*0.3
                # superimposed_img = heatmap
                plt.imshow(superimposed_img,cmap='gray')
                # plt.imshow(superimposed_img,cmap='jet')
                plt.title(title)
                plt.show()
                # 下面这些是对特征图进行保存，使用时取消注释
                # cv2.imshow("1",superimposed_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # cv2.imwrite(os.path.join(save_dir,name +str(i)+'.png'), superimposed_img)
                # i=i+1
