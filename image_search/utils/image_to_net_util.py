# -*- coding: utf-8 -*-
# from image_search.config.logging_config import logger
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#  抽取其中图片转换向量部分内容
import datetime
import base64
import time

#  · 以下版本固定使用以下处理过程
#  -- 特征抽取：ResNet50
#  · 说明：以下为特征提取、相似度计算过程相关算法及函数选择
import tensorflow as tf
from numpy import linalg as LA
from rembg import remove
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from tensorflow.keras.preprocessing import image

"""图片处理相关（获取、转换、裁剪等操作）"""
import numpy as np
import pandas as pd
from io import BytesIO
import cv2 as cv
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False
from skimage import io

#  · 不打印警告  Try using .loc[row_indexer,col_indexer] = value instead
import warnings
warnings.filterwarnings('ignore')

# 禁用TensorFlow的某些警告和信息打印
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import sys
# sys.path.append('./u2net')
# sys.path.append('/home/wu.tt/project_image_v3_2/image_search_api_inner/')

#  -- clip_cn
import torch
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
print("Available models:", available_models())
# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:",device)
# device='cpu'

#  -- 定义想要识别的类
lab=["鞋","服装","箱包","其它"]
text = clip.tokenize(lab).to(device)




class Net(object):
    def __init__(self):
        '''clip_cn'''
        self.clip_cn_model, self.clip_cn_preprocess = load_from_name("ViT-H-14", device=device, download_root='./')  # 3.57G
        self.clip_cn_model.eval()

        '''resnet参数'''
        '''
        # include_top：是否保留顶层的3个全连接网络
        # weights：None代表随机初始化，即不加载预训练权重。'imagenet'代表加载预训练权重
        # input_tensor：可填入Keras tensor作为模型的图像输出tensor
        # input_shape：可选，仅当include_top=False有效，应为长为3的tuple，指明输入图片的shape，图片的宽高必须大于48，如(200,200,3)
        # pooling：当include_top = False时，该参数指定了池化方式。None代表不池化，最后一个卷积层的输出为4D张量。‘avg’代表全局平均池化，‘max’代表全局最大值池化。
        # classes：可选，图片分类的类别数，仅当include_top = True并且不加载预训练权重时可用。
        self.imgsz = 448
        self.weight = 'imagenet'  # weights: 'imagenet'
        self.pooling = 'max'  # pooling: 'max' or 'avg'
        self.input_shape = (self.imgsz, self.imgsz, 3)  # (width, height, 3), width and height should >= 48
        self.model_resnet = ResNet50(weights=self.weight,
                                     input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                                     pooling=self.pooling, include_top=False)
        # self.model_resnet.predict(np.zeros((1, self.imgsz, self.imgsz, 3)))

        self.dir = "./net_data"
        self.query = None
        self.img_dict = {}
        self.selection = None
        self.model_list = ["resnet"]  # ,"resnet_shoes2", "resnet_shoes"
        # logger.info("图片向量化各模块参数设置完成.")
        '''

        self.imgsz = 448

    #  · 图片处理为统一tensor  · 参考 imagedata_trans 函数：图片数据转换&归一化
    def image_to_tf(self, img):
        img = tf.image.resize(img, [self.imgsz, self.imgsz])
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        return img

    def image_to_tf_byimagestyle(self, img_pad, byimagestyle):
        if byimagestyle in ("pad", "remove"):
            img_pad = cv.resize(self.margin_add(img_pad), (self.imgsz, self.imgsz))
            self.img_searchby = img_pad  # selection截图后边缘裁剪并填充补齐后
            queryImg = self.image_to_tf(img_pad)
        else:
            #  -- 灰度图
            gray0 = cv.cvtColor(img_pad, cv.COLOR_RGB2GRAY)
            gray = cv.merge((gray0, gray0, gray0))
            if byimagestyle == "gray":
                queryImg = cv.resize(self.margin_add(gray), (self.imgsz, self.imgsz))
                self.img_searchby = queryImg  # selection截图后边缘裁剪并填充补齐置灰
                queryImg = self.image_to_tf(queryImg)
            else:
                #  -- 轮廓图
                #     高斯去噪
                blurred = cv.GaussianBlur(gray0, (9, 9), 0)
                #     索比尔算子来计算x、y方向梯度
                gradX = cv.Sobel(blurred, ddepth=cv.CV_32F, dx=1, dy=0)
                gradY = cv.Sobel(blurred, ddepth=cv.CV_32F, dx=0, dy=1)
                gradient = cv.subtract(gradX, gradY)
                gradient = cv.convertScaleAbs(gradient)

                #  -- 轮廓图背景换为白色
                # 生成同等大小白色背景
                blank = np.full((gradient.shape[0], gradient.shape[1]), (255), gradient.dtype)
                gradient = Image.fromarray(gradient)
                blank = Image.fromarray(blank)
                width = (blank.width - gradient.width) // 2
                height = (blank.height - gradient.height) // 2
                blank.paste(gradient, (width, height), gradient)

                gradient = np.array(blank)
                gradient = cv.merge((gradient, gradient, gradient))
                queryImg = cv.resize(self.margin_add(gradient), (self.imgsz, self.imgsz))
                self.img_searchby = queryImg  # selection截图后边缘裁剪并填充补齐置灰
                queryImg = self.image_to_tf(queryImg)

        return queryImg

    #  · 边界裁剪
    def corp_margin(self, imagePath, limit=0.05, marginpad=5):
        if isinstance(imagePath, str):
            image = Image.open(imagePath)  # 打开tiff图像
            img = np.array(image)
        else:
            img = imagePath
        if len(img.shape) > 2:
            img2 = img.sum(axis=2)
            r = img.shape[-1]
            val0 = img[0, 0, 0]
        else:
            # 兼容灰度图
            img2 = img
            r = 1
            val0 = img[0, 0]
        (row, col) = img2.shape
        # print(row, col)
        t = time.time()

        #  如果边框为黑色
        if val0 <= limit:
            axis1_sum = list(img2.sum(axis=1))
            '''
            import matplotlib.pyplot as plt
            pd.DataFrame({"axis1_sum": axis1_sum}).plot()
            plt.show()
            print(axis1_sum)
            '''
            row_top = axis1_sum.index(list(filter(lambda i: i > limit * col, axis1_sum))[0])
            axis1_sum_ = axis1_sum[::-1]
            row_down = row - 1 - axis1_sum_.index(list(filter(lambda i: i > limit * col, axis1_sum_))[0])

            axis0_sum = list(img2.sum(axis=0))
            col_top = axis0_sum.index(list(filter(lambda i: i > limit * row, axis0_sum))[0])
            axis0_sum_ = axis0_sum[::-1]
            col_down = col - 1 - axis0_sum_.index(list(filter(lambda i: i > limit * row, axis0_sum_))[0])

        #  如果边框为白色
        else:
            axis1_sum = list(img2.sum(axis=1))
            '''
            import matplotlib.pyplot as plt
            pd.DataFrame({"axis1_sum": axis1_sum}).plot()
            plt.show()
            print(axis1_sum)
            '''
            row_top = axis1_sum.index(list(filter(lambda i: i < (255 - limit) * r * col, axis1_sum))[0])
            axis1_sum_ = axis1_sum[::-1]
            row_down = row - 1 - axis1_sum_.index(list(filter(lambda i: i < (255 - limit) * r * col, axis1_sum_))[0])

            axis0_sum = list(img2.sum(axis=0))
            col_top = axis0_sum.index(list(filter(lambda i: i < (255 - limit) * r * row, axis0_sum))[0])
            axis0_sum_ = axis0_sum[::-1]
            col_down = col - 1 - axis0_sum_.index(list(filter(lambda i: i < (255 - limit) * r * row, axis0_sum_))[0])

        # print(col_top, row_top, col_down, row_down)

        row_top = max(0, row_top - marginpad)
        row_down = min(row, row_down + marginpad)
        col_top = max(0, col_top - marginpad)
        col_down = min(col, col_down + marginpad)

        # print(col_top, row_top, col_down, row_down)
        # print("用时：", time.time() - t)

        try:
            new_img = img[row_top:row_down, col_top:col_down, 0:3]
        except Exception as e:
            print(e)
            new_img = img[row_top:row_down, col_top:col_down]
        return new_img, (col_top, row_top, col_down, row_down), (row, col)

    #  · 图片裁剪   -- 增加基于行列像素差值的裁剪  保留差值非零且占据面积最大的部分
    def corp_image(self, img, diff_min=5, ax=1):
        img2 = img.sum(axis=2)
        #  -- 先按高度方向裁剪再按宽度方向裁剪
        try:
            df = pd.DataFrame({"H_max_min": img2.max(axis=ax) - img2.min(axis=ax), "index": list(range(img2.shape[np.abs(ax - 1)]))})
            df_up0 = df[df.H_max_min > diff_min]
            df_up0["index_diff"] = df_up0["index"].diff()
            df_up0["index_last"] = df_up0["index"] - df_up0["index_diff"]
            # df_up0["index_last"] = df_up0[["index", "index_diff"]].apply(lambda x: x[0] - x[1], axis=1)
            cut_top_list = [df_up0.index.values[0]] + list(df_up0[df_up0.index_diff > 1].index.values)
            cut_down_list = list(df_up0[df_up0.index_diff > 1].index_last.values) + [df_up0.index.values[-1]]

            diff = list(map(lambda x: x[0] - x[1], zip(cut_down_list, cut_top_list)))
            index_target = diff.index(max(diff))

            cut_top = max(0, cut_top_list[index_target] - 5)
            cut_down = min(cut_down_list[index_target] + 5, img2.shape[np.abs(ax - 1)])

            if ax == 1:
                img_cut = img[int(cut_top):int(cut_down), :, :]
            else:
                img_cut = img[:, int(cut_top):int(cut_down), :]

            return img_cut
        except Exception as e:
            print(e)
            return img

    #  · 图片裁剪（20240325修改：基于轮廓图截取，兼容更多图片）   -- 增加基于行列像素差值的裁剪  保留差值非零且占据面积最大的部分
    def corp_image_new(self,img,diff_min=5,ax=1):
        img2 = img.sum(axis=2)

        #  -- 先按高度方向裁剪再按宽度方向裁剪
        try:
            df=pd.DataFrame({"H_max_min":img2.max(axis=ax)-img2.min(axis=ax),"index":list(range(img2.shape[np.abs(ax-1)]))})
            df_up0=df[df.H_max_min>diff_min]
            df_up0["index_diff"]=df_up0["index"].diff()
            df_up0["index_last"]=df_up0["index"]-df_up0["index_diff"]
            # df_up0["index_last"] = df_up0[["index", "index_diff"]].apply(lambda x: x[0] - x[1], axis=1)
            cut_top_list=[df_up0.index.values[0]]+list(df_up0[df_up0.index_diff>1].index.values)
            cut_down_list=list(df_up0[df_up0.index_diff>1].index_last.values)+[df_up0.index.values[-1]]

            diff= list(map(lambda x: x[0]-x[1], zip(cut_down_list, cut_top_list)))
            index_target=diff.index(max(diff))

            cut_top=max(0,cut_top_list[index_target]-5)
            cut_down=min(cut_down_list[index_target]+5,img2.shape[np.abs(ax-1)])

            if ax==1:
                img_cut=img[int(cut_top):int(cut_down),:,:]
            else:
                img_cut=img[:,int(cut_top):int(cut_down),:]

            return img_cut,(cut_top,cut_down)
        except Exception as e:
            print(e)
            return img,(None,None)

    def img_cut_mix(self,img_cut):
        #  -- 灰度图
        Conv_hsv_Gray = cv.cvtColor(img_cut, cv.COLOR_RGB2GRAY)

        #  -- 轮廓图
        mask=cv.Canny(Conv_hsv_Gray,
                      threshold1=20,   # threshold1：第一个阈值，用于边缘连接；
                      threshold2=150,  # threshold2：第二个阈值，用于边缘检测；
                      apertureSize=3   # apertureSize：Sobel 算子的大小，可选值为 3、5、7，默认值为 3；
                     )
        # Image.fromarray(mask).show()

        #  -- 轮廓图进一步处理
        #     高斯去噪
        blurred = cv.GaussianBlur(mask, (9, 9), 0)
        # blurred = cv.GaussianBlur(mask, (3, 3), 0)
        #     索比尔算子来计算x、y方向梯度
        gradX = cv.Sobel(blurred, ddepth=cv.CV_32F, dx=1, dy=0)
        gradY = cv.Sobel(blurred, ddepth=cv.CV_32F, dx=0, dy=1)
        gradient = cv.subtract(gradX, gradY)
        gradient = cv.convertScaleAbs(gradient)

        #  -- 轮廓图背景换为白色
        # 生成同等大小白色背景
        blank = np.full((gradient.shape[0], gradient.shape[1]), (255), gradient.dtype)
        gradient = Image.fromarray(gradient)
        blank = Image.fromarray(blank)
        width = (blank.width - gradient.width) // 2
        height = (blank.height - gradient.height) // 2
        blank.paste(gradient, (width, height), gradient)

        gradient = np.array(blank)

        gradient = cv.merge((gradient, gradient, gradient))

        # Image.fromarray(gradient).show()

        #  -- 基于轮廓图计算截取位置
        img_cut0, _01 = self.corp_image_new(gradient, diff_min=5, ax=1)
        img_cut0, _00 = self.corp_image_new(img_cut0, diff_min=5, ax=0)
        img_cut0, _11 = self.corp_image_new(img_cut0, diff_min=20, ax=1)
        img_cut0, _10 = self.corp_image_new(img_cut0, diff_min=20, ax=0)
        cut_top, cut_down = _01
        if None not in [cut_top, cut_down]:
            img_cut = img_cut[int(cut_top):int(cut_down), :, :]
        cut_top, cut_down = _00
        if None not in [cut_top, cut_down]:
            img_cut = img_cut[:, int(cut_top):int(cut_down), :]
        cut_top, cut_down = _11
        if None not in [cut_top, cut_down]:
            img_cut = img_cut[int(cut_top):int(cut_down), :, :]
        cut_top, cut_down = _10
        if None not in [cut_top, cut_down]:
            img_cut = img_cut[:, int(cut_top):int(cut_down), :]
        return img_cut

    #  · 边界填充
    def margin_add(self, img1, is_show=False):
        '''
        函数原型：cv2.copyMakeBorder(src,top, bottom, left, right ,borderType,value)
            src：需要填充的图像
            top：图像上面填充边界的长度
            bottom：图像上下面填充边界的长度
            left：图像左面填充边界的长度
            right ：图像右面填充边界的长度
            borderType：边界的类型
                BORDER_REPLICATE：复制法，即复制最边缘的像素。例如：aaaa|abcdefg|ggggg
                BORDER_REFLECT：反射法,即以最边缘的像素为对称轴。例如：fedcba|abcdefg|gfedec
                BORDER_REFLECT_101：反射法,也是最边缘的像素为对称轴，但与BORDER_REFLECT有区别。例如：fedcb|abcdefg|fedec
                BORDER_WRAP：外包装法，即以图像的左边界与右边界相连，上下边界相连。例如：cdefgh|abcdefgh|abcdefg
                BORDER_CONSTANT：常量法。
            value：填充的边界颜色，通常用于常量法填充中。
        '''
        imgsz = self.imgsz
        try:
            h, w, r = img1.shape
        except Exception as e:
            print(e)
            h, w = img1.shape
            r = None
        # print("h,w,r:", h, w, r)
        if is_show:
            io.imshow(img1)
            io.show()

        #  -- 设最大边界为imgsz，长宽等比变化
        rs_times = max(h, w) / imgsz
        h = int(h / rs_times)
        w = int(w / rs_times)
        img1 = cv.resize(img1, (w, h))
        # print("img1:", img1.shape)

        if h > w:
            left_size = int((h - w) / 2)
            right_size = int((h - w) / 2)
            top_size = 0
            bottom_size = 0
        elif h < w:
            left_size = 0
            right_size = 0
            top_size = int((w - h) / 2)
            bottom_size = int((w - h) / 2)
        else:
            left_size = 0
            right_size = 0
            top_size = 0
            bottom_size = 0

        if is_show:
            io.imshow(img1)
            io.show()

        if max(left_size, right_size, top_size, bottom_size) > 0:
            import numpy as np
            if r is not None:
                value = [
                    np.nanmedian([img1[0, 0, 0], img1[h - 1, 0, 0], img1[0, w - 1, 0], img1[h - 1, w - 1, 0]]),
                    np.nanmedian([img1[0, 0, 1], img1[h - 1, 0, 1], img1[0, w - 1, 1], img1[h - 1, w - 1, 1]]),
                    np.nanmedian([img1[0, 0, 2], img1[h - 1, 0, 2], img1[0, w - 1, 2], img1[h - 1, w - 1, 2]])
                ]
            else:
                value = [np.nanmedian([img1[0, 0], img1[h - 1, 0], img1[0, w - 1], img1[h - 1, w - 1]])]
            img1 = cv.copyMakeBorder(img1, top_size, bottom_size, left_size, right_size, cv.BORDER_CONSTANT, value=value)


        #  -- 为图像边缘添加一圈白色空白包装
        # img1 = cv.copyMakeBorder(img1, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=[255,255,255])     # 白色
        # img1 = cv.copyMakeBorder(img1, 10, 10, 10, 10, cv.BORDER_REPLICATE)                         # 复制边缘颜色

        if is_show:
            io.imshow(img1)
            io.show()

        # print("img1:",img1.shape)
        return img1

    #  · 图片旋转angel度
    def img_rotate(self, img, angel):
        """逆时针旋转图像任意角度

        Args:
            img (np.array): [原始图像]
            angel (int): [逆时针旋转的角度]

        Returns:
            [array]: [旋转后的图像]
        """
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, angel, 1.0)
        # 调整旋转后的图像长宽
        rotated_h = int((w * np.abs(M[0, 1]) + (h * np.abs(M[0, 0]))))
        rotated_w = int((h * np.abs(M[0, 1]) + (w * np.abs(M[0, 0]))))
        M[0, 2] += (rotated_w - w) // 2
        M[1, 2] += (rotated_h - h) // 2
        # print(M)
        # print(rotated_h)
        # print(rotated_w)
        # 旋转图像
        value = (255, 255, 255)  # 白色背景填充
        # value = [
        #     np.nanmedian([img[0, 0, 0], img[-1, 0, 0], img[0,  -1, 0], img[-1,  -1, 0]]),
        #     np.nanmedian([img[0, 0, 1], img[-1, 0, 1], img[0,  -1, 1], img[-1, -1, 1]]),
        #     np.nanmedian([img[0, 0, 2], img[-1, 0, 2], img[0, -1, 2], img[-1,  -1, 2]])
        # ]
        rotated_img = cv.warpAffine(img, M, (rotated_w, rotated_h), borderValue=value)

        return rotated_img

    #  · 移除背景
    def remove_background_new(self, img):
        # gc.collect()   # 若被调用时不包含参数，则启动完全的垃圾回收。可选的参数 generation 可以是一个整数，指明需要回收哪一代（从 0 到 2 ）的垃圾。
        # self.print_virtual_memory()

        #    生成同等大小白色背景
        blank = np.full((img.shape[0], img.shape[1], 3), (255, 255, 255), img.dtype)

        #    移除背景
        img_rm = remove(img)

        #    改为白色背景
        frontImage = Image.fromarray(img_rm)
        background = Image.fromarray(blank)

        frontImage = frontImage.convert("RGBA")  # Convert image to RGBA
        background = background.convert("RGBA")  # Convert image to RGBA
        width = (background.width - frontImage.width) // 2  # Calculate width to be at the center
        height = (background.height - frontImage.height) // 2  # Paste the frontImage at (width, height)
        background.paste(frontImage, (width, height), frontImage)  # Save this image

        #    读取白色背景图片
        img = np.array(background)[:, :, 0:3]
        return img


    #  · 读取原图，并裁剪去图片边缘空白，生成灰度图，同时输出原图、灰度图
    def image_return(self, img, log=None, **kwargs):
        #  -- 原图
        self.img = img  # 原图

        #  -- 前端返回框选位置信息
        selection = kwargs.get("selection", None)
        byimagestyle = kwargs.get("byimagestyle", "pad")
        # byretate = int(kwargs.get("byretate", 0))
        byretate = kwargs.get("byretate", [])
        bg_remove = kwargs.get("bg_remove", 'null')


        #  -- 按条件进行图片特征转
        if byimagestyle == "org":  # 原图
            self.img_cut = self.img  # 无selection截图
            img = cv.resize(self.margin_add(img), (self.imgsz, self.imgsz))
            self.img_searchby = img
            queryImg = [self.image_to_tf(img)]
        else:
            #  -- 基于对query的框选内容进行截图操作
            if selection is not None:
                print("selection:", selection)
                if "range" in selection.keys():
                    range_x = selection["range"]["x"]
                    range_y = selection["range"]["y"]
                else:
                    range_x = selection["x"]
                    range_y = selection["y"]
                img_cut = img[int(range_y[0]):int(range_y[1]), int(range_x[0]):int(range_x[1]), :]

                # -- 以下会修改原图呈现
                # cv2.copyMakeBorder(src, top, bottom, left, right, borderType, value)
                # self.img = cv.copyMakeBorder(img_cut, int(range_y[0]), int(img.shape[0]-range_y[1]), int(range_x[0]), int(img.shape[0]-range_x[1]), cv.BORDER_CONSTANT, value=[255,255,255])     # 截图以外的区域使用白色填充，方便使用截图后的图片对原图片库进行检索
            else:
                img_cut = img

            self.img_cut = img_cut  # selection截图

            #  -- 移除背景
            if bg_remove == "remove":
                # img = self.remove_background(img)
                # img_cut = self.remove_background(img_cut)  # 对裁剪后的图片进行背景移除
                t0 = datetime.datetime.now()
                img_cut = self.remove_background_new(img_cut)  # 对裁剪后的图片进行背景移除
                print("移除背景耗时：", datetime.datetime.now() - t0)

            #  -- 移除黑白边框前先裁剪掉边缘log   -- 先按高度方向裁剪再按宽度方向裁剪
            # img_cut0 = self.corp_image(img_cut.copy(), diff_min=5, ax=1)
            # img_cut0 = self.corp_image(img_cut0, diff_min=5, ax=0)
            # img_cut0 = self.corp_image(img_cut0, diff_min=20,ax=1)  # url="https://img.alicdn.com/imgextra/i1/2074690906/O1CN01B9uP9U1IYzWwTnJPQ_!!2074690906.jpg"
            # img_cut0 = self.corp_image(img_cut0, diff_min=20, ax=0)
            # io.imshow(img_cut0)
            # io.show()

            img_cut0=self.img_cut_mix(img_cut)
            # io.imshow(img_cut0)
            # io.show()

            if min(img_cut.shape[0], img_cut.shape[1]) > (min(img.shape[0], img.shape[1]) / 5):
                img_pad = img_cut0
            else:
                #  -- 移除空白边界
                img_pad, (col_top, row_top, col_down, row_down), (row, col) = self.corp_margin(img_cut, limit=0.5,  marginpad=5)

            # print("原图尺寸：",img.shape,"裁剪后尺寸：",img_pad.shape)

            #  -- 图片尺寸
            img_shape_org = img.shape
            img_shape_cut = img_pad.shape

            #   -- 旋转裁剪后的图片
            if isinstance(byretate, int):
                if byretate != 0:
                    img_pad = self.img_rotate(img_pad, byretate)
                if byimagestyle!='clip_cn':
                    queryImg = self.image_to_tf_byimagestyle(img_pad, byimagestyle)
                else:
                    queryImg=img_pad
            else:
                if byimagestyle != 'clip_cn':
                    queryImg = [self.image_to_tf_byimagestyle(self.img_rotate(img_pad, byretate_i), byimagestyle) for   byretate_i in byretate]
                else:
                    queryImg=[self.img_rotate(img_pad, byretate_i) for byretate_i in byretate]
                    self.img_searchby = img_pad

        return queryImg


    def Xnet_extract_feat(self, img, model_name="resnet"):
        #  -- 图片转换
        '''try:
            img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        except Exception as e:
            print(e)
            img = io.imread(img_path)
            img = tf.image.resize(img, [224, 224])'''

        #  -- 特征抽取
        img = preprocess_input_resnet(img.copy())
        feat = self.model_resnet.predict(img,workers=8)

        # print("feat:",feat.shape)
        norm_feat = feat[0] / LA.norm(feat[0])
        return norm_feat

    #  · 使用clip_cn提取图片特征
    def Clip_cn_extract_feat(self, img):
        t0 = datetime.datetime.now()
        image = self.clip_cn_preprocess(Image.fromarray(img)).unsqueeze(0).to(device)  # 剪切图
        with torch.no_grad():
            image_features = self.clip_cn_model.encode_image(image)
            # text_features = self.clip_cn_model.encode_text(text)
            print("image_features：", image_features.size())
            # 对特征进行归一化，请使用归一化后的图文特征用于下游任务   (torch.Size([1, 512]), torch.Size([4, 512]))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # text_features /= text_features.norm(dim=-1, keepdim=True)
        print("抽取特征用时：", datetime.datetime.now() - t0)

        return image_features.cpu().numpy()[0]


    #  · 使用clip_cn提取文本特征
    def Clip_cn_text_feat(self, text):
        t0 = datetime.datetime.now()
        txt = clip.tokenize(text).to(device)
        with torch.no_grad():
            text_features = self.clip_cn_model.encode_text(txt)
            print("text_features：", text_features.size())
            # 对特征进行归一化，请使用归一化后的图文特征用于下游任务   (torch.Size([1, 512]), torch.Size([4, 512]))
            # text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=1, keepdim=True)
        print("抽取特征用时：", datetime.datetime.now() - t0)
        return text_features.cpu().numpy()[0]


    #  · 定义相似度计算函数 可以删除
    def similarity_mix_5(self, queryVec_list, byretate):
        t0 = datetime.datetime.now()
        #  -- 特征归一化
        # queryVec_ = [list(queryVec / LA.norm(queryVec)) for queryVec in queryVec_list]
        queryVec_ = queryVec_list  # 抽取时已经归一化
        # 等效为：
        # from sklearn.preprocessing import normalize
        # queryVec_ = normalize(np.array(queryVec_list).reshape(len(queryVec_list), -1))

        self.images_milvus = self.images_milvus_pad
        print("相似度计算【待查特征归一化】用时：", datetime.datetime.now() - t0)
        t0 = datetime.datetime.now()

        #  -- 搜索特征组 基于模型、图片确定
        #  -- 加载数据
        self.images_milvus.load()
        top = self.images_milvus.search(queryVec_, self.byField, self.search_params,
                                        limit=self.limit,
                                        output_fields=self.output_fields,
                                        consistency_level="Strong",
                                        # expr="category1_name = ''".format('女鞋')),
                                        partition_names=self.query_partition_names  # 指定分区
                                        )

        print("相似度计算【检索过程】用时：", datetime.datetime.now() - t0)

        t0 = datetime.datetime.now()
        num = len(top)
        df_score = pd.DataFrame({"ids": sum([list(top[num].ids) for num in range(num)], []),
                                 "distances": sum([list(top[num].distances) for num in range(num)], []),
                                 "nid": sum([[item.entity.get("nid") for item in top[num]] for num in range(num)], []),
                                 "obj_name": sum( [[item.entity.get("obj_name") for item in top[num]] for num in range(num)], []),
                                 "queryVec_index": sum([[num] * len(top[num].ids) for num in range(num)],   [])})  # ids对应h5文件中的index

        # print("单版本返回数据量:", df_score.groupby(["queryVec_index"])["ids"].count())

        if self.search_params["metric_type"] == "L2":
            # df_score["scores"] = 1 / (1 + df_score["distances"])  # 1/(1+d(x,y))
            df_score["scores"] = 100 - df_score["distances"]
        else:
            #  · 方案1：直接通过阈值过滤旋转后的图片
            #  · 添加方位旋转后的相似度阈值限定  （20221206添加；背景：以下箱包检索出现旋转后的连凉鞋高度相似  https://inner-oss.bellecdn.com/files/analy/98563612/O1CN01tQ4PuR1cYLNNLgxv7_!!98563612.jpg）
            # df_score=df_score[((df_score.queryVec_index==byretate.index(0))|(df_score.distances>=0.825))]

            #  · 方案2：对旋转后的图片添加权重
            df_score["distances"] = df_score[["queryVec_index", "distances"]].apply( lambda x: x[-1] if x[0] == byretate.index(0) else 0.98 * x[-1], axis=1)

            df_score["scores"] = df_score["distances"]  # 归一化后存入milvis的向量，IP检索结果的distances与手算余弦相似度np.dot(feat_img1,feat_img2.T)一致

        print("相似度计算【get商品信息&整理数据】用时：", datetime.datetime.now() - t0)

        return df_score


    #  · base64转换
    def image_to_base64(self, image_np):
        image = cv.imencode('.png', image_np)[1]
        image_code = str(base64.b64encode(image), 'utf-8')
        print(cv.imencode('.png', image_np))
        print(image_code[:20])
        return image_code

    def base64_to_image(self, base64_code):
        # base64解码
        img_data = base64.b64decode(base64_code)
        # 转换为np数组
        img_array = np.fromstring(img_data, np.uint8)
        # 转换成opencv可用格式
        img = cv.imdecode(img_array, cv.IMREAD_COLOR)
        # img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
        # img = cv.imdecode(img, cv.COLOR_BGR2RGB)

        return img

    #  · 按条件搜索
    """选一张测试图片测试检索效果  -- 相似度采用余弦相似度或ssim度量"""

    def image_to_netvector(self, image,
                           bymodel="resnet",
                           # byimagestyle="pad",
                           byimagestyle="clip_cn",
                           byretate=[0]):
        """
        :param image_url:       待抽取特征的图片链接              （二选一）
        :param image_data:      待抽取特征的图片数据 base64格式    （二选一）
        :param bymodel:         抽取特征的模型，不用改
        :param byimagestyle:    图片格式，不用改
        :param byretate:        抽取哪些方位的特征，特征库准备过程用默认的就好
        :param kwargs:
        :return:
        """
        self.byimagestyle = byimagestyle

        img = self.request_img(image)
        #  不在进行图片旋转
        queryImg_list = self.image_return(img, byimagestyle=byimagestyle, byretate=byretate)

        #  · 提取模型特征
        # queryVec_list = [self.Xnet_extract_feat(queryImg, model_name=bymodel) for queryImg in  queryImg_list]  # 修改此处改变提取特征的网络
        queryVec_list = [self.Clip_cn_extract_feat(queryImg) for queryImg in queryImg_list]  # 修改此处改变提取特征的网络

        photo_emb = np.array(queryVec_list[0], dtype=np.float32)
        return photo_emb

    #  获取图片
    def request_img(self, image):
        img = np.asarray(Image.open(BytesIO(image)).convert('RGB'))
        #  将尺寸大于800×800的图片resize为800×800
        if max(img.shape)>800:
            img = cv.resize(img, (800, 800))
        self.query_img = img.copy()
        return img




# # """以下为测试,任务中可删掉"""
# import requests
# if __name__ == "__main__":
#     #  · 基于图片链接得到图片特征
#     image_url = 'https://blec-img.bellecdn.com/pics//staccato/2012/99823964/99823964_07_l.jpg?1'
#     image_url = 'https://retailp2.bellecdn.com/2021/MDM/FM/F4ZUFG02ZU1DC7.jpg?v=1612685555956'
#     Net=Net()
#     headers={
#             "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36",
#             'Content-Type': 'image/jpeg'
#             }
#     img=requests.get(image_url, headers=headers).content
#     print(img)
#     print("img:",Net.request_img(img).shape)

    # result_records=Net.image_to_netvector(img)

    # l=[0.0019848614, 0.07424236, 0.017753666, 0.004741608, 0.024053115, 0.004880625, 0.019280387, 0.03475055, 0.0013856066, 0.0067046112, 0.009860957, 0.057557207, 0.0054480946, 0.011905481, 0.03572496, 0.01581337, 0.011842515, 0.0035733352, 0.002570166, 0.028182324, 0.004564361, 0.0064992956, 0.025421022, 0.01639419, 0.017330455, 0.020582829, 0.011894304, 0.0033511557, 0.055298645, 0.001690626, 0.0, 0.016079718, 0.024951445, 0.0, 0.018516446, 0.004128952, 0.010906872, 0.021648247, 0.0063088844, 0.004855605, 0.0043451716, 0.018347984, 0.0005593361, 0.008312555, 0.016224047, 0.048505273, 0.02962239, 0.100011826, 0.006475939, 0.012522459, 0.015434254, 0.016794428, 0.044070065, 0.031741485, 0.0010724909, 0.0072110416, 0.0059195096, 0.024133455, 0.013673242, 0.013074425, 0.014162431, 0.013185828, 0.0, 0.01892016, 0.030281164, 0.0053096903, 0.03436647, 0.022971354, 0.0, 0.01642357, 0.0020948728, 0.008533833, 0.03118366, 0.013783182, 0.0008162353, 0.016056499, 0.03480649, 0.009638559, 0.030626308, 0.03624506, 0.07832687, 0.0129400985, 0.022980759, 0.0017903922, 0.002514562, 0.015170798, 0.0, 0.033208262, 0.007974131, 0.003933246, 0.030248908, 0.011331526, 0.041798007, 0.0077680526, 0.031875215, 0.003906104, 0.011791215, 0.0036090931, 0.010075108, 0.01573651, 0.016322644, 0.045694556, 0.0013836846, 0.019275945, 0.028867668, 0.024335638, 0.0020542552, 7.0964714e-05, 0.03878285, 0.0059676967, 0.034304693, 0.032865804, 0.016127558, 0.004537168, 0.0, 0.015208499, 0.078894176, 0.010745298, 0.006325814, 0.009393075, 0.012072131, 0.005682736, 0.011890673, 0.011780966, 0.022729084, 0.028879074, 0.01914034, 0.007736846, 0.028404566, 0.008842835, 0.004660872, 0.020324308, 0.0017951628, 0.013400862, 0.0023980113, 0.0002253898, 0.032713305, 0.0044218777, 0.0051314253, 0.008571156, 0.014575177, 0.001844904, 0.0030111943, 0.03361107, 0.004708051, 0.0, 0.0076224413, 0.0, 0.014433456, 0.0017209843, 0.018367706, 0.021161381, 0.0040403022, 0.014924418, 0.04404808, 0.06514986, 0.038048502, 0.03401363, 0.0018311958, 0.015154963, 0.042037144, 0.016673123, 0.00027243057, 0.01060913, 0.007731057, 0.003118972, 0.019834846, 0.0, 0.050229337, 0.030244792, 0.010979653, 0.0445941, 0.07733743, 0.014592815, 0.0041700867, 0.012988919, 0.029823577, 0.031583045, 0.007294452, 0.032593533, 0.02179659, 0.016211024, 0.0077517503, 0.011609576, 0.04733116, 0.008231464, 0.011486114, 0.013907112, 0.035184346, 0.004986072, 0.0, 0.043772295, 0.033069383, 0.030256629, 0.012168676, 0.02167929, 0.030173196, 0.033651926, 0.010873929, 0.004591289, 0.018643778, 0.012113119, 0.015562141, 0.047303047, 0.021810574, 0.009192142, 0.017439697, 0.0074961274, 0.0044849617, 0.0037091295, 0.0058401, 0.016889824, 0.0032384922, 0.016179144, 0.013933984, 0.0019108263, 0.030048123, 0.006149624, 0.01760447, 0.0019064167, 0.014394659, 0.011192553, 0.02102285, 0.037571274, 0.0032296462, 0.06373826, 0.05668541, 0.0061059445, 0.009177549, 0.01748633, 0.02043837, 0.0, 0.010216727, 0.0077636894, 0.0024421527, 0.0042826068, 0.0059492406, 0.02921821, 0.012869478, 0.0039009016, 0.0067446274, 0.008768106, 0.0076819626, 0.0, 0.0072465963, 0.018790781, 0.05418668, 0.011901571, 0.005642463, 0.06278767, 0.008174468, 0.008579046, 0.015975857, 0.013490092, 0.046655953, 0.017901726, 0.007931726, 0.010706303, 0.009590808, 0.016225498, 0.047662463, 0.004714687, 0.00069850095, 0.011762828, 0.019374391, 0.029468255, 0.005059448, 0.00930894, 0.017939784, 0.007456756, 0.009017112, 0.017629478, 0.001455289, 0.0050287494, 0.012579858, 0.017556136, 0.014433449, 0.010456317, 0.018659936, 0.0031130956, 0.01776882, 0.016235314, 0.023605933, 0.0075775827, 0.00036840767, 0.045173228, 0.029578978, 0.024423182, 0.038276695, 0.026036205, 0.05894318, 0.006366209, 0.004306078, 0.005541043, 0.012445027, 0.0020240785, 0.00013628512, 0.022670694, 0.016973723, 0.018553166, 0.01443972, 0.003945134, 0.06466044, 0.03290697, 0.012484628, 0.016827615, 0.03981765, 0.01877146, 0.002318661, 0.019248186, 0.026129095, 0.021050077, 0.012393868, 0.0070513193, 0.012152886, 0.020273989, 0.0062369267, 0.024262635, 0.02590729, 0.017647764, 0.019286158, 0.009966897, 0.0038299733, 0.013854857, 0.03328324, 0.02501085, 0.008782679, 0.00016773505, 0.0028326747, 0.038811583, 0.027210353, 0.0028530005, 0.013211154, 0.020970445, 0.013625676, 0.02253057, 0.01840123, 0.050598934, 0.017054288, 0.014095903, 0.0, 0.06644158, 0.007008488, 0.0009119318, 0.041442394, 0.0016912669, 0.020787105, 0.057090264, 0.038532685, 0.0025976829, 0.0035412689, 0.032290712, 0.025471762, 0.009952942, 0.012464755, 0.012895599, 0.015894279, 0.0051926686, 0.0050431164, 0.014935071, 0.009090292, 0.005786734, 0.011259179, 0.008211053, 0.02307734, 0.029699046, 0.010412323, 0.0, 0.033314418, 0.0, 0.0, 0.01230992, 0.0058906353, 0.034960493, 0.002187148, 0.021117702, 0.0011576492, 0.010549796, 0.0056551057, 0.015439856, 0.0048235073, 0.0053953985, 0.0, 0.015771499, 0.000943378, 0.017608123, 0.005855159, 0.0, 0.0, 0.010467639, 0.0, 0.008986382, 0.039829742, 0.01718826, 0.015019368, 0.043504138, 0.0028058817, 0.040064722, 0.0, 0.011098116, 0.051945083, 0.011732401, 0.027492277, 0.021609262, 0.0041751596, 0.022963222, 0.015972694, 0.0029159638, 0.014985266, 0.0025692596, 0.053802606, 0.031992327, 0.04890282, 0.006488886, 0.0, 0.015790235, 0.0017449564, 0.01034496, 0.0046298504, 0.016754322, 0.004398812, 0.008370911, 0.0017738143, 0.0033560644, 0.044747934, 0.004248132, 0.008620291, 0.07087907, 0.010136386, 0.0014662284, 0.014926411, 0.034506008, 0.012502107, 0.0025778878, 0.03280511, 0.007182298, 0.018976728, 0.0, 0.024422195, 0.014962748, 0.020125207, 0.019351106, 0.0021711744, 0.026428906, 0.0037866386, 0.012302855, 0.018138502, 0.016660674, 0.026786946, 0.0051314193, 0.028828632, 0.007521351, 0.00019853811, 0.0054359226, 0.015649466, 0.008511573, 0.035516456, 0.015953276, 0.023944099, 0.06296955, 0.05988587, 0.0047423425, 0.0051672123, 0.0051030563, 0.025323523, 0.0417651, 0.01658497, 0.015696611, 0.016647639, 0.00043810488, 0.025525583, 0.005946832, 0.004086948, 0.008383032, 0.020780701, 0.026165834, 0.01686194, 0.0011904653, 0.00669909, 0.024460573, 0.0023684087, 0.015129131, 0.02611059, 0.018584762, 0.0017250112, 0.01430834, 0.021703463, 0.02054002, 0.017749902, 0.0, 0.014720863, 0.014351051, 0.0025346945, 0.008099235, 0.006986182, 0.05977827, 0.0017595784, 0.0110821435, 0.022483535, 0.0033674075, 0.006791247, 0.015201211, 0.008587136, 0.0038900159, 0.0051149866, 0.040464647, 0.003856716, 0.004655395, 0.010307478, 0.0152685335, 0.0043238597, 0.001444195, 0.014528602, 0.0, 0.0014851785, 0.005310742, 0.008046024, 0.0032245535, 0.0, 0.0053368346, 0.0, 0.008212475, 0.017002415, 0.0078021395, 0.0069281817, 0.025673201, 0.0024947599, 0.04015204, 0.04319855, 0.02046788, 0.01598252, 0.0, 0.0073071294, 0.025559885, 0.010837053, 0.024772046, 0.028396022, 0.005623004, 0.01732392, 0.021980282, 0.013860672, 0.02534064, 0.0034639626, 0.0070997565, 0.007054921, 0.03092458, 0.008366658, 0.015878279, 0.0032043299, 0.0073260074, 0.0018949431, 0.035824146, 0.015606747, 0.039597422, 0.0070237927, 0.0055172555, 0.0, 0.041313216, 0.01938651, 0.035664763, 0.005114985, 0.020635959, 0.039634142, 0.014932702, 0.014960887, 0.01906834, 0.012543739, 0.0007538991, 0.015300364, 0.0079434635, 0.017644228, 0.042351704, 0.023108274, 0.006305893, 0.01870008, 0.015076034, 0.009495913, 0.022339785, 0.01161018, 0.026941733, 0.020541154, 0.009905898, 0.017197853, 0.011703506, 0.0041711815, 0.0019215685, 0.007006912, 0.038615752, 0.014300934, 0.01546516, 0.005996525, 0.007720058, 0.0022637765, 0.01068174, 0.0026317327, 0.0063693505, 0.004719849, 0.04290375, 0.02854314, 0.016342174, 0.012195866, 0.013974682, 0.0069356, 0.000290918, 0.0035805693, 0.017581979, 0.004279639, 0.002456095, 0.025472922, 0.039039496, 0.003867771, 0.025114845, 0.07703892, 0.012461624, 0.0118013015, 0.001572478, 0.005898872, 0.010057032, 0.0041870326, 0.010111376, 0.001486748, 0.0, 0.0065647364, 0.011049937, 0.0073764855, 0.005896408, 0.03252632, 0.0032808157, 0.008794034, 0.023814403, 0.0048483186, 0.00827528, 0.01461208, 0.04227036, 0.021566141, 0.018825086, 0.015330587, 0.011884017, 0.0066043236, 0.044814955, 0.002708475, 0.02520339, 0.04389396, 0.005319634, 0.015722938, 0.030288754, 0.038337436, 0.007652332, 0.02183504, 0.021623755, 0.016275862, 0.011328679, 0.010747961, 0.0043601645, 0.0, 0.009167067, 0.024380252, 0.00157573, 0.007194408, 0.004072481, 0.0011521481, 0.0, 0.015077334, 0.021776216, 0.045810025, 0.009279734, 0.021872269, 0.00093830796, 0.034021653, 0.0070437435, 0.0034143392, 0.00379984, 0.0029323169, 0.03630142, 0.0025202148, 0.023556773, 0.026072286, 0.025880583, 0.022751858, 0.03449191, 0.0, 0.0013459116, 0.014275284, 0.041504037, 0.004048575, 0.005681385, 0.005255804, 0.0, 0.0036712652, 0.011977773, 0.012677652, 0.0022068983, 0.016602509, 0.0146665955, 0.03683127, 0.0, 0.101171605, 0.030253176, 0.04122986, 0.016635755, 0.0063032117, 0.006215355, 0.04379102, 0.025820853, 0.0045634396, 0.02356009, 0.017651265, 0.007993033, 0.005732152, 0.012663994, 0.02507886, 0.0, 0.011836261, 0.019798722, 0.00932081, 0.009700317, 0.00029049086, 0.0137095, 0.0013084828, 0.03825588, 0.011581751, 0.0060425755, 0.01793035, 0.033366382, 0.021409523, 0.06078644, 0.0041391533, 0.0, 0.012267058, 0.0062784324, 0.004857632, 0.024340665, 0.013261212, 0.01879878, 0.021773797, 0.007886363, 0.007362912, 0.009658423, 0.007765235, 0.042624027, 0.0058299573, 0.058792524, 0.0037193245, 0.010663591, 0.013609621, 0.016431464, 0.023319928, 0.012260358, 0.040365886, 0.006026485, 0.021318305, 0.01375718, 0.019004645, 0.005011913, 0.0012178819, 0.019279292, 0.03724471, 0.0, 0.002009611, 0.0025125488, 0.025856186, 0.01584046, 0.009956345, 0.0020031952, 0.016472694, 0.07109227, 0.033069488, 0.03142321, 0.0, 0.02395438, 0.0025907252, 0.008818043, 0.0012829036, 0.0059453836, 0.013994872, 0.011877202, 0.013501999, 0.011053686, 0.0064844713, 0.013071783, 0.009473671, 0.019303955, 0.039649855, 0.016179608, 0.01746941, 0.052698664, 0.003974155, 0.0026995966, 0.012942516, 0.014228955, 0.021554448, 0.008853089, 0.017379675, 0.011400086, 0.009071073, 0.044128213, 0.004302254, 0.008418057, 0.0094356015, 0.029411364, 0.022814728, 0.0140918335, 0.0008690399, 0.0055221184, 0.0059434734, 0.00019649754, 0.0, 0.01066995, 0.010333633, 0.0, 0.0069020977, 0.007155048, 0.0036840818, 0.007514254, 0.030058347, 0.03141317, 0.02307621, 0.012180443, 0.0093476605, 0.04140385, 0.027754297, 0.011202694, 0.009051703, 0.018593485, 0.0019666154, 0.0007139636, 0.027760739, 0.020036953, 0.026930878, 0.009308747, 0.0051341653, 0.02462053, 0.012253811, 0.039437935, 0.015653187, 0.022609428, 0.016852021, 0.036132738, 0.0048937034, 0.012690869, 0.026116403, 0.009744678, 0.0016353204, 0.017508503, 0.0047083506, 0.0029533904, 0.008266401, 0.003923071, 0.01616094, 0.008387851, 0.0, 0.013055255, 0.020540852, 0.015830042, 0.008806165, 0.019911205, 0.0024930069, 0.018924814, 0.035181034, 0.00339439, 0.010677137, 0.0030815962, 0.013580269, 0.0, 0.013205022, 0.0010549214, 0.0042924318, 0.034197155, 0.024526127, 0.005871858, 0.059493262, 0.011801243, 0.0034305893, 0.010315832, 0.020484999, 0.007536267, 0.0069619897, 0.008331249, 0.02646047, 0.0017731221, 0.0100424085, 0.014294723, 0.0021566595, 0.023371061, 0.025086692, 0.022263138, 0.023078863, 0.007854341, 0.0019552219, 0.045328744, 0.0022151163, 0.0, 0.0035315696, 0.0065867715, 0.026956882, 0.043250043, 0.0057059145, 0.0, 0.006870326, 0.008327721, 0.009411286, 0.016911022, 0.02052841, 0.0065046996, 0.024931228, 0.03896206, 0.01360031, 0.045165908, 0.0009537648, 0.00261491, 0.027210485, 0.008312734, 0.02417714, 0.0075353864, 0.01663253, 0.011593385, 0.031002317, 0.0039281375, 0.015680026, 0.023309441, 0.05483253, 0.032764163, 0.003965175, 0.0032297813, 0.06178846, 0.010221074, 0.00015926159, 0.013375634, 0.0023198677, 0.011088563, 0.007555756, 0.0032861733, 0.0758857, 0.0, 0.008816202, 0.00031122085, 0.008738034, 0.016330793, 0.0051087574, 0.016222462, 0.07419473, 0.023087261, 0.0013551355, 0.02401894, 0.014154641, 0.0, 0.014143945, 0.02121211, 0.020927101, 0.038824216, 0.051676814, 0.0058472515, 0.0, 0.0070091044, 0.018730035, 0.00938859, 0.01960719, 0.0018224126, 1.8714481e-05, 0.021090355, 0.012954735, 0.015450329, 0.010505019, 0.058466915, 0.02759101, 0.00015333307, 0.016481396, 0.004746344, 0.006863445, 0.016475724, 0.0070769475, 0.0011386687, 0.004448579, 0.007655179, 0.026807519, 0.0021317396, 0.013775955, 0.011895478, 0.002174656, 0.028166527, 0.017286774, 0.017206347, 0.0005435961, 0.014837806, 0.011440376, 0.0042357687, 0.0091062775, 0.039346524, 0.004122288, 0.013745498, 0.007484489, 0.012224316, 0.014110982, 0.004492344, 0.019532168, 0.009791829, 0.021977054, 0.046423763, 0.017906178, 0.001061757, 0.009020232, 0.0066094077, 0.0056018764, 0.027903857, 0.0019221173, 0.0014432492, 0.02347183, 0.009413021, 0.01830391, 0.0070326924, 0.0001891563, 0.019526776, 0.002200582, 0.016653463, 0.0057198172, 0.019232424, 0.0052826125, 0.013077473, 0.0157693, 0.0007581599, 0.028784255, 0.0009901515, 0.022212105, 0.011175929, 0.004511551, 0.018395394, 0.007338626, 0.029970642, 0.006413814, 0.0046880143, 0.020966446, 0.003165814, 0.002778063, 0.018738205, 0.040060934, 0.022643287, 0.021874122, 0.089640014, 0.012142974, 0.021968191, 0.0, 0.03621058, 0.01390046, 0.0111102965, 0.0016432991, 0.016345315, 0.0067203143, 0.006219344, 0.052537363, 0.0023635316, 0.014333841, 0.017760798, 0.02701952, 0.0, 0.0088196, 0.0074347123, 0.0, 0.0131827565, 0.03252099, 0.006181106, 0.038628776, 0.0, 0.0, 0.0029313057, 0.0, 0.0017751777, 0.02951526, 0.01986804, 0.0, 0.005988693, 0.012943621, 0.03742388, 0.016034745, 0.0023368313, 0.01179969, 0.011712975, 0.019528402, 0.01164547, 0.007512708, 0.035819992, 0.0049451343, 0.0035193954, 0.02506737, 0.0046023643, 0.0050578574, 0.0039939918, 0.0005021902, 0.0072004735, 0.009199126, 0.011998198, 0.016580096, 0.009845028, 0.029758424, 0.026322711, 0.010377561, 0.021861933, 0.0023796982, 0.0016236765, 0.012453265, 0.040154736, 0.019030549, 0.016212704, 0.01733981, 0.0070637492, 0.046478536, 0.012898774, 0.02372154, 0.024148585, 0.01854878, 0.019698866, 0.02289363, 0.010374088, 0.01987028, 0.064950675, 0.0216892, 0.07788982, 0.018586313, 0.02317421, 0.014322295, 0.016811166, 0.0, 0.008847084, 0.005828879, 0.01953109, 0.0066665392, 0.0, 0.0030169566, 0.003139152, 0.012863959, 0.005938242, 0.044041183, 0.0025190392, 0.014469381, 0.017489314, 0.013207263, 0.0047980947, 0.059547342, 0.014192259, 0.0060217786, 0.023303133, 0.036184043, 0.02385096, 0.051293153, 0.04241112, 0.052922856, 0.008088814, 0.031130977, 0.011989535, 0.009904902, 0.0016776829, 0.0, 0.010811362, 0.0130613195, 0.025873555, 0.019965015, 0.009575058, 0.010957818, 0.088497095, 0.011425303, 0.047653165, 0.02150044, 0.00957601, 0.030471636, 0.0071420637, 0.027186843, 0.01133283, 0.031225068, 0.0068155196, 0.0102331815, 0.04500852, 0.017340751, 0.016667658, 0.025807144, 0.013328034, 0.0, 0.012653405, 0.010056298, 0.019885551, 0.0, 0.031793885, 0.0050626867, 0.0061815064, 0.025631703, 0.008907699, 0.009027054, 0.012803829, 0.011921087, 3.9437182e-05, 0.018337965, 0.026050402, 0.028974026, 0.009894973, 0.008630221, 0.031159947, 0.0001822601, 0.0050537116, 0.0, 0.014093472, 0.002528206, 0.027438302, 0.021411166, 0.040680572, 0.010294954, 0.011885005, 0.0060576773, 0.03395894, 0.005918824, 0.0070412406, 0.0015409766, 0.011633496, 0.0057357503, 0.017848575, 0.032879632, 0.013312346, 0.034804862, 0.014348987, 0.004680851, 0.0043587903, 0.009298888, 0.030633004, 0.009658002, 0.0024262401, 0.0, 0.010293253, 0.013515125, 0.016087245, 0.005202105, 0.009036314, 0.023186123, 0.01591438, 0.0067429864, 0.045444556, 0.0036656517, 0.0072346414, 0.012632064, 0.013727142, 0.044025745, 0.0, 0.009863006, 0.015937759, 0.0044103246, 0.022812862, 0.0133354785, 0.010694597, 0.0017180663, 0.027020479, 0.0034576005, 0.014598494, 0.006876607, 0.02273157, 0.004975833, 0.020204313, 0.007909092, 0.006538176, 0.0014415866, 0.009727578, 0.068982586, 0.02134677, 0.022474399, 0.0068808096, 0.021488179, 0.011258734, 0.014710387, 0.012225986, 0.009444547, 0.056963354, 0.0454103, 0.000120758086, 0.009691157, 0.021995945, 0.022617385, 0.0, 0.032509085, 0.009034151, 0.00880938, 0.016257143, 0.053176984, 0.0020196335, 0.016637769, 0.036091816, 0.018706309, 0.006074979, 0.019685192, 0.015921652, 0.0015020318, 0.014601477, 0.0018369636, 0.03808756, 0.026469488, 0.007609764, 0.024744496, 0.023665352, 0.0045848903, 0.006003217, 0.0009770998, 0.006539536, 0.0065220348, 0.011203503, 0.001177218, 0.009256767, 0.0103492895, 0.00975762, 0.035983115, 0.074598484, 0.019964939, 0.031120507, 0.032887813, 0.014830683, 0.024643702, 0.00975369, 0.026019067, 0.0017891384, 0.030856635, 0.0004003555, 0.0138683515, 0.0, 0.0014833608, 0.005669755, 0.010589631, 0.022409273, 0.03030789, 0.031867005, 0.009028473, 0.06801113, 0.013955327, 0.030108046, 0.008822344, 0.011713417, 0.027439741, 0.0, 0.018090906, 0.0229306, 0.0077900994, 0.007173149, 0.01999833, 0.0045695193, 0.02344491, 0.027642982, 0.0020058984, 0.0127039505, 0.04101739, 0.024442421, 0.010790183, 0.0034691405, 0.03038893, 0.0026175617, 0.024151115, 0.007633186, 0.010776977, 0.0073154224, 0.042751797, 0.08225708, 0.0080808485, 0.012949864, 0.01989858, 0.0056508332, 0.014023666, 0.00010527942, 0.0010895288, 0.008326229, 0.022499256, 0.004536016, 0.011990072, 0.0008916625, 0.002561468, 0.0016508591, 0.008151844, 0.0, 0.021527397, 0.03266029, 0.0, 0.010467263, 0.073262386, 0.026624339, 0.009956811, 0.010091769, 0.015527102, 0.01661237, 0.010680884, 0.07184954, 0.0054684416, 0.003004866, 0.047714025, 0.0069744983, 0.0, 0.0147243375, 0.009833672, 0.034071054, 0.0070602954, 0.011076717, 0.0029963376, 0.04030247, 0.017863678, 0.0, 0.0048423433, 0.018801043, 0.016218003, 0.0040202118, 0.0069280877, 0.007451945, 0.0, 0.0, 0.005034101, 0.008420034, 0.0056064595, 0.0050337673, 0.022390425, 0.018219339, 0.009839233, 0.00409593, 0.0062141684, 0.00441182, 0.017176632, 0.007318759, 0.052759655, 0.019827573, 0.00043135273, 0.012357992, 0.0, 0.004884121, 0.00094821723, 0.03218502, 0.024535356, 0.0025433442, 0.01056087, 0.00014456293, 0.008309328, 0.029202908, 0.0, 0.067467436, 0.026764048, 0.007905099, 0.017391955, 0.030781647, 0.02126665, 0.007129885, 0.034672223, 0.008868991, 0.011070337, 0.026744962, 0.00686398, 0.101542406, 0.028441854, 0.004293208, 0.033781294, 0.022059504, 0.002759442, 0.012942688, 0.013996699, 0.008503214, 0.032073144, 0.08544194, 0.008211124, 0.021955857, 0.009477297, 0.0044373856, 0.0, 0.025764022, 0.0, 0.012157969, 0.008431114, 0.021503193, 0.02243716, 0.0028915724, 0.0211328, 0.0048621143, 0.010129843, 0.005582327, 0.034105357, 0.0069633042, 0.019529058, 0.06796992, 0.0023366082, 0.013281847, 0.045337565, 0.039976202, 0.014998086, 0.03359306, 0.04132431, 0.027954869, 0.056645922, 0.0057588783, 0.006260881, 0.015238369, 0.036884677, 0.0, 0.0017341345, 0.025840381, 0.012386614, 0.008119656, 0.0, 0.005163354, 0.07589743, 0.01047576, 0.014087329, 0.0058825286, 0.0, 0.0049403803, 0.0013232187, 0.009254066, 0.0033633923, 0.031696346, 0.013205931, 0.018969866, 0.019031007, 0.008013673, 0.0023613242, 0.015140303, 0.0018193949, 0.033190425, 0.004418786, 0.046000294, 0.00857795, 0.025870945, 0.026448628, 0.030595474, 0.010613063, 0.031655747, 0.0018965312, 0.013248067, 0.0037824037, 0.0, 0.003994569, 0.022200953, 0.012540893, 0.0, 0.002919112, 0.0035281226, 0.010483935, 0.024549674, 0.010178587, 0.030324308, 0.01701173, 0.02420762, 0.0033234276, 0.0, 0.039238065, 0.02445087, 0.024219044, 0.012555866, 0.0060947016, 0.0, 0.0, 0.016969813, 0.02640839, 0.041641593, 0.017960588, 0.056648783, 0.018213084, 0.028816462, 0.014680203, 0.015582826, 0.020515425, 0.02753386, 0.022272378, 0.007757169, 0.0015895191, 0.028127575, 0.032582138, 0.015176393, 0.0040481626, 0.05633416, 0.014254188, 0.02836194, 0.0025375914, 0.030616364, 0.035260994, 7.423011e-05, 0.03961877, 0.0064325808, 0.008281026, 0.01520596, 0.016990935, 0.01711566, 0.008508593, 0.0055560316, 0.00777999, 0.00867651, 0.047745727, 0.0014092646, 0.011897573, 0.0, 0.0015592928, 0.007005539, 8.5430125e-05, 0.018960733, 0.010173542, 0.043372106, 0.021901134, 0.002224863, 0.027944392, 0.01987346, 0.003937727, 0.0, 0.04264037, 0.011345528, 0.023903081, 0.0, 0.0025013126, 0.015468114, 0.022714552, 0.03210656, 0.03380617, 0.02983114, 0.008386545, 0.0, 0.020322824, 0.010756837, 0.010798745, 0.033533886, 0.02331632, 0.004886114, 0.019336665, 0.024421502, 0.050405454, 0.00595455, 0.0, 0.029608704, 0.006908501, 0.024667643, 0.007828428, 0.028509583, 0.030782878, 0.0, 0.017626213, 0.0035063452, 0.0072708502, 0.014985071, 0.03628372, 0.033228915, 0.021897579, 0.002248917, 0.0057161488, 0.034964588, 0.0, 0.024473337, 0.0, 0.012365322, 0.0037020834, 0.01909097, 0.020738637, 0.032838162, 0.01740155, 0.025932562, 0.0032288728, 0.018451849, 0.033063672, 0.017596023, 0.017240236, 0.03794881, 0.03185432, 0.015588752, 0.0105095925, 0.008515864, 0.0225048, 0.0, 0.011240041, 0.0, 0.0146402465, 0.0016392065, 0.0162618, 0.0065599387, 0.015582848, 0.012595225, 0.007353704, 0.021017058, 0.045528058, 0.019690447, 0.006773103, 0.025545053, 0.0013096231, 0.048372373, 0.002321536, 0.0027383536, 0.008302387, 0.006961149, 0.004067166, 0.0066524018, 0.00907581, 0.0026629684, 0.008456604, 0.011047094, 0.009607586, 0.008892308, 0.0, 0.0054118577, 0.007846234, 0.031835902, 0.015720537, 0.0023259649, 0.005177856, 0.014929201, 0.01251241, 0.005825621, 0.0076951166, 0.01820884, 0.073257335, 0.029358381, 0.012391651, 0.093513876, 0.024283305, 0.046580043, 0.009036154, 0.027527882, 0.0061753127, 0.014705659, 0.005075064, 0.013833835, 0.066609405, 0.005660587, 0.0040701656, 0.026329815, 0.025763746, 0.047175325, 0.054335784, 0.04892511, 0.04731282, 0.0076281684, 0.041146807, 0.013294441, 0.019842448, 0.007378812, 0.0038779988, 0.003506385, 0.0046828124, 0.009102593, 0.0055171424, 0.0026536612, 0.0053354967, 0.00644426, 0.015714444, 0.006011227, 0.005165076, 0.0070866174, 0.0013353991, 0.030436248, 0.0038683224, 0.016543584, 0.009901192, 0.011945586, 0.019688876, 0.00928308, 0.013725233, 0.026887605, 0.007936888, 0.014468478, 0.0, 0.006937692, 0.01899098, 0.024155717, 0.012982521, 0.016608445, 0.030571306, 0.015813906, 0.009762095, 0.019502362, 0.0040654275, 0.02226989, 0.008214974, 0.018300908, 0.0319148, 0.00079303485, 0.0, 0.0034950834, 0.0077224704, 0.021974023, 0.017594364, 0.018104864, 0.015402923, 0.009239024, 0.026594646, 0.0023833294, 0.04329142, 0.0109017715, 0.033207655, 0.0039921473, 0.0025701108, 0.0, 0.018752035, 0.0047878795, 0.008593264, 0.01560844, 0.025082693, 0.04371652, 0.016580913, 0.0029377933, 0.022528753, 0.022978818, 0.015574248, 0.022247417, 0.009410281, 0.0034421312, 0.033204082, 0.01060115, 0.0007464529, 0.033813685, 0.011352983, 0.0012723866, 0.025260892, 0.011264265, 0.0047571636, 0.01838475, 0.00802242, 0.0070507955, 0.00031820577, 0.0059529277, 0.0022162546, 0.0, 0.025304826, 0.00081746664, 0.034998003, 0.011913744, 0.020156408, 0.0020790952, 0.020292617, 0.016316196, 0.015037534, 0.023768686, 0.0067117037, 0.022181448, 0.011639093, 0.013544151, 0.031892408, 0.004570389, 0.058814883, 0.0878965, 0.011299304, 0.0020485953, 0.004671092, 0.01169458, 0.00017257304, 0.035506826, 0.047031384, 0.0, 0.029251592, 0.018948307, 0.079607405, 0.038109872, 0.0043990845, 0.013884165, 0.008091643, 0.03464616, 0.04793128, 0.0016226795, 0.014038117, 0.035182938, 0.004866101, 0.001945859, 0.021876512, 0.0, 0.0057825483, 0.004092175, 0.02606383, 0.039520234, 0.032453097, 0.01151926, 0.015359027, 0.027622972, 0.003993069, 0.027517065, 0.005779117, 0.00969421, 0.0072031743, 0.010147739, 0.03460345, 0.015834767, 0.021071972, 0.011551551, 0.009554612, 0.0026650294, 0.034966193, 0.010684007, 0.0019440048, 0.045442987, 0.021292888, 0.012143184, 0.012998796, 0.019404579, 0.017531939, 0.0055601886, 0.012700625, 0.0040788795, 0.05991779, 0.028035846, 0.010756913, 0.02359438, 0.028021649, 0.0077878395, 0.00046929234, 0.011045213, 0.035050176, 0.014551859, 0.031599738, 0.0057143597, 0.0, 0.03907937, 0.03199569, 0.015419578, 0.02631404, 7.8528705e-05, 0.01489404, 0.016386315, 0.005043727, 0.0, 0.0, 0.015872084, 0.008661539, 0.023499789, 0.0056992774, 0.008440582, 0.007422981, 0.014476489, 0.006323396, 0.010608723, 0.007770435, 0.012491849, 0.0029317716, 0.022008007, 0.019010535, 0.0059874845, 0.037211798, 0.0044088853, 0.0026496395, 0.027415177, 0.011552363, 0.035553746, 0.0068258275, 0.010580445, 0.0024162605, 0.018160505, 0.028994799, 0.028284628, 0.02803792, 0.0071415026, 0.03330317, 0.013321919, 0.019096224, 0.009289129, 0.019667355, 0.005797746, 0.0098592825, 0.017838452, 0.016357431, 0.028692761, 0.01744125, 0.007208719, 0.0257724, 0.010295314, 0.02964122, 0.004987979, 0.005967405, 0.032678157, 0.008098576, 0.006759401, 0.0, 0.017681668, 0.012262869, 0.03487173, 0.00592548, 0.0010236221, 0.017875321, 0.010850325, 0.0018719422, 0.013883252, 0.02486701, 0.0106668, 0.0077529764, 0.012084811, 0.0031820617, 0.010038575, 0.022264795, 0.008534978, 0.029279873, 0.020807512, 0.009737711, 0.0018697625, 0.055297825, 0.029801426, 0.01273716, 0.013848642, 0.0076226844, 0.007993398, 0.006271636, 0.0026576042, 0.011392474, 0.026876085, 0.01859412, 0.020309078, 0.0136991935, 0.011241865, 0.027553217, 0.015061411, 0.018123683, 0.0, 0.005319827, 0.0077856714, 0.031288456, 0.008274419, 0.020811655, 0.047469355, 0.0142214615, 0.0008322875, 0.014368817, 0.021842446, 0.009771148, 0.040500056, 0.034609355, 0.00038983728, 0.004408594, 0.0030750956, 0.020125184, 0.023243885, 0.0, 0.0088811945, 0.0, 0.020321064, 0.007844811, 0.0, 0.0, 0.010046125, 0.010421852, 0.02296403, 0.0068354914, 0.00022682984, 0.010233043, 0.021839188, 0.00018340533, 0.0053538415, 0.00062872167, 0.035615824, 0.0014348334, 0.013038041, 0.01104238, 0.029114198, 0.017922176, 0.026845502, 0.007572309, 0.04633379, 0.0045642247, 0.0, 0.028761743, 0.010077032, 0.025055919, 0.0061194114, 0.028479004, 0.017047914, 0.0026722276, 0.005147476]
    # print(list(result_records)==l)

    # print(list(result_records))


