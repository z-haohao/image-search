# -*- coding: utf-8 -*-
import os
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
from skimage import io





class Net(object):
    def __init__(self):
        '''resnet参数'''
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

        self.model_resnet.predict(np.zeros((1, self.imgsz, self.imgsz, 3)))
        self.dir = "./net_data"
        self.query = None
        self.img_dict = {}
        self.selection = None
        self.model_list = ["resnet"]  # ,"resnet_shoes2", "resnet_shoes"
        # logger.info("图片向量化各模块参数设置完成.")

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
            df = pd.DataFrame(
                {"H_max_min": img2.max(axis=ax) - img2.min(axis=ax), "index": list(range(img2.shape[np.abs(ax - 1)]))})
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
        # print(img2.shape)
        # pd.DataFrame({"W":img2.sum(axis=0),"H:":img2.sum(axis=1)}).plot()
        # plt.show()
        # pd.DataFrame({"W_max-min":img2.max(axis=0)-img2.min(axis=0),"H_max-min":img2.max(axis=1)-img2.min(axis=1)}).plot()
        # plt.show()
        # io.imshow(img)
        # io.show()
        # # print(list(img2.sum(axis=1)))
        # # print(list(img2.max(axis=0)-img2.min(axis=0)))
        # print(list(img2.max(axis=1)-img2.min(axis=1)))

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
            img1 = cv.copyMakeBorder(img1, top_size, bottom_size, left_size, right_size, cv.BORDER_CONSTANT,
                                     value=value)

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
                img_pad, (col_top, row_top, col_down, row_down), (row, col) = self.corp_margin(img_cut, limit=0.5,
                                                                                               marginpad=5)

            # print("原图尺寸：",img.shape,"裁剪后尺寸：",img_pad.shape)

            #  -- 图片尺寸
            img_shape_org = img.shape
            img_shape_cut = img_pad.shape

            #   -- 旋转裁剪后的图片
            if isinstance(byretate, int):
                if byretate != 0:
                    img_pad = self.img_rotate(img_pad, byretate)
                    queryImg = self.image_to_tf_byimagestyle(img_pad, byimagestyle)
            else:
                queryImg = [self.image_to_tf_byimagestyle(self.img_rotate(img_pad, byretate_i), byimagestyle) for
                            byretate_i in byretate]
                # queryImg=[]
                # for byretate_i in byretate:
                #     img_pad = self.img_rotate(img_pad, byretate_i)
                #     queryImg.append(self.image_to_tf_byimagestyle(img_pad, byimagestyle))

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
        feat = self.model_resnet.predict(img)
        # print("feat:",feat.shape)
        norm_feat = feat[0] / LA.norm(feat[0])
        return norm_feat

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
                                 "obj_name": sum(
                                     [[item.entity.get("obj_name") for item in top[num]] for num in range(num)], []),
                                 "queryVec_index": sum([[num] * len(top[num].ids) for num in range(num)],
                                                       [])})  # ids对应h5文件中的index

        # print("单版本返回数据量:", df_score.groupby(["queryVec_index"])["ids"].count())

        if self.search_params["metric_type"] == "L2":
            # df_score["scores"] = 1 / (1 + df_score["distances"])  # 1/(1+d(x,y))
            df_score["scores"] = 100 - df_score["distances"]
        else:
            #  · 方案1：直接通过阈值过滤旋转后的图片
            #  · 添加方位旋转后的相似度阈值限定  （20221206添加；背景：以下箱包检索出现旋转后的连凉鞋高度相似  https://inner-oss.bellecdn.com/files/analy/98563612/O1CN01tQ4PuR1cYLNNLgxv7_!!98563612.jpg）
            # df_score=df_score[((df_score.queryVec_index==byretate.index(0))|(df_score.distances>=0.825))]

            #  · 方案2：对旋转后的图片添加权重
            df_score["distances"] = df_score[["queryVec_index", "distances"]].apply(
                lambda x: x[-1] if x[0] == byretate.index(0) else 0.98 * x[-1], axis=1)

            df_score["scores"] = df_score[
                "distances"]  # 归一化后存入milvis的向量，IP检索结果的distances与手算余弦相似度np.dot(feat_img1,feat_img2.T)一致

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
                           byimagestyle="pad",
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
        queryVec_list = [self.Xnet_extract_feat(queryImg, model_name=bymodel) for queryImg in
                         queryImg_list]  # 修改此处改变提取特征的网络

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



"""以下为测试,任务中可删掉"""
import requests
if __name__ == "__main__":
    # Check for TensorFlow GPU access
    print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")

    # See TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    #  · 基于图片链接得到图片特征
    image_url = 'https://blec-img.bellecdn.com/pics//staccato/2012/99823964/99823964_07_l.jpg?1'
    image_url = 'https://retailp2.bellecdn.com/2021/MDM/FM/F4ZUFG02ZU1DC7.jpg?v=1612685555956'
    Net=Net()
    headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36",
            'Content-Type': 'image/jpeg'
            }
    img=requests.get(image_url, headers=headers).content
    print("img:",Net.request_img(img).shape)

    result_records=Net.image_to_netvector(img)
    # print(list(result_records))
    if list(result_records) is not None:
        print('训练集下载完成。')