import json
import math
import os
import time

import cv2
import faiss
import numpy as np
import torch
import torchvision

from .base_module import BaseModule
from .deep_sort import build_tracker



class BaseRecLabelDecode(object):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False):
        self.beg_str = 'sos'
        self.end_str = 'eos'
        self.character_str = []
        self.reverse = False

        if character_dict_path is None:
            self.character_str = '0123456789abcdefghijklmnopqrstuvwxyz'
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, 'rb') as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip('\n').strip('\r\n')
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(' ')
            dict_character = list(self.character_str)
            if 'arabic' in character_dict_path:
                self.reverse = True

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[
                    batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]
            text = ''.join(char_list)
            if self.reverse:  # for arabic rec
                text = self.pred_reverse(text)

            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path, use_space_char, **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path,
                                             use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character


class trafficSignBroken():

    def __init__(self, config):
        self.threshold = config['broken_threshold']

    def set_color(self, img):
        """
        对输入图像进行处理,现在只添加了色域转换,后面还要添加调节图片的饱和度/对比度等信息,方便进行mask的提取.
        :param img: 原始图片
        :return:  处理后的图片
        """
        kernel_size = 5
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        return img_hsv

    def find_mask_by_color(self, img_hsv, color):
        """
        :param img_hsv:  颜色通道转换成 HSV 之后的图片
        :param color: 需要提取的 sign 的颜色
        :return: 满足 颜色要求的 mask
        """
        if color == 'red':
            # 设置红色标志的阈
            redLower01 = np.array([0, 43, 46], dtype=np.uint8)  # 部分红
            redUpper01 = np.array([10, 255, 255], dtype=np.uint8)
            redLower02 = np.array(
                [125, 21, 31],
                dtype=np.uint8)  # 部分红  包含了紫色 棕色 # OG:125, 43, 46
            redUpper02 = np.array([180, 255, 255], dtype=np.uint8)
            red_mask01 = cv2.inRange(img_hsv, redLower01, redUpper01)
            red_mask02 = cv2.inRange(img_hsv, redLower02, redUpper02)
            mask = red_mask01 + red_mask02
        elif color == 'blue':
            Lower = np.array(
                [100, 43, 17],
                dtype=np.uint8)  # [100, 100, 46]  # og:100, 43, 46
            Upper = np.array([130, 255, 255],
                             dtype=np.uint8)  # [130, 255, 255] [124, 255, 255]
            mask = cv2.inRange(img_hsv, Lower, Upper)
        elif color == 'green':  # green should be a little lower
            # Lower = np.array([75, 43, 22])  # [37, 43, 46] [25, 52, 72] og_now  # [65,6,22]
            # Upper = np.array([100, 255, 255])  # [102, 255, 255]                # [100,225,200]
            Lower = np.array(
                [65, 6, 22])  # [37, 43, 46] [25, 52, 72] og_now  # [65,6,22]
            Upper = np.array(
                [100, 225,
                 200])  # [102, 255, 255]                # [100,225,200]
            mask = cv2.inRange(img_hsv, Lower, Upper)
        elif color == 'gray':  # Don't change anymore
            Lower = np.array([34, 0, 0], dtype=np.uint8)
            Upper = np.array([78, 43, 30], dtype=np.uint8)  # 180, 200, 255
            mask = cv2.inRange(img_hsv, Lower, Upper)
        elif color == 'orange':  # 黄色的图片会 偏橙色 经过实验验证upper中v的分量小于200即可
            Lower = np.array([11, 15, 46], dtype=np.uint8)
            Upper = np.array([25, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(img_hsv, Lower, Upper)
        # elif color == 'yellow':
        #     #Lower = np.array([26, 100, 46], dtype=np.uint8)  # 11, 50, 50   26 #20,15,46
        #     #Upper = np.array([34, 255, 255], dtype=np.uint8)  # 50, 255, 255   #34,255,255
        #     Lower = np.array([11,15,46], dtype=np.uint8)  # 11, 50, 50   26 #20,15,46
        #     Upper = np.array([34, 255, 255], dtype=np.uint8)  # 50, 255, 255   #34,255,255
        #     mask = cv2.inRange(img_hsv, Lower, Upper)
        else:
            mask = None
        return mask

    def get_sign_area(self, img, color):
        """
        :param img:  原始的输入图片
        :param color:  需要提取的sign 的颜色
        :return: 可能是sign 的区域的坐标列表，[x, y, w, h]
        """
        img_hsv = self.set_color(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if color == 'white':
            mask = cv2.threshold(gray, 220, 255,
                                 cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        elif color == 'black':
            B, G, R = cv2.split(img)
            mask1 = cv2.threshold(B, 20, 255, cv2.THRESH_BINARY_INV)[1]
            mask2 = cv2.threshold(G, 30, 255, cv2.THRESH_BINARY_INV)[1]
            mask3 = cv2.threshold(R, 40, 255, cv2.THRESH_BINARY_INV)[1]
            temp = cv2.bitwise_and(mask1, mask2)
            mask = cv2.bitwise_and(temp, mask3)
        else:
            mask = self.find_mask_by_color(
                img_hsv, color=color)  # find the mask of a color
        return mask / 255.0

    def get_area(self, img):
        """
        得到交通标志颜色范围内的mask面积
        """
        greenMask = self.get_sign_area(img, color='green')
        redMask = self.get_sign_area(img, color='red')
        whiteMask = self.get_sign_area(img, color='white')
        blueMask = self.get_sign_area(img, color='blue')
        # yellowMask = self.get_sign_area(img, color='yellow')
        orangeMask = self.get_sign_area(img, color='orange')
        blackMask = self.get_sign_area(img, color='black')
        grayMask = self.get_sign_area(img, color='gray')

        # mask = cv2.bitwise_or(yellowMask, whiteMask)
        mask = cv2.bitwise_or(whiteMask, greenMask)
        mask = cv2.bitwise_or(mask, redMask)
        mask = cv2.bitwise_or(mask, blueMask)
        mask = cv2.bitwise_or(mask, orangeMask)
        mask = cv2.bitwise_or(mask, blackMask)
        mask = cv2.bitwise_or(mask, grayMask)

        return mask.sum()

    def traffic_detect(self, label, image, x1, y1, x2, y2, h, w):
        '''
        Judge whether the sign is covered.
        '''
        flag = False
        # 计算不同交通标志中未遮挡的颜色占整体的比例
        if label == 'sanjiaobiaozhi':
            th = self.threshold * (1.00 / 2.00)
        elif label == 'yuanbiaozhi':
            th = self.threshold * (3.14 / 4.00)  # π/4
        else:
            th = self.threshold
        area = image.shape[0] * image.shape[1]

        if x1 < 10 or y1 < 10 or x2 > w - 10 or y2 > h - 10:
            # breakpoint()
            print('the object is at coner')
            return flag
        if area < 6400:
            print('ignore sign area: ', area)
            return flag
        maskArea = self.get_area(image)
        ratio = round(maskArea / area, 3)
        print(f'th:{th} and {ratio}')
        eps = 0.00001
        if ratio < th - eps:
            flag = True
        return flag

    def preprocess_filter(self, img_path):
        '''
            filter the traffic signs from two rules:
            1. avg_r > 200 avg_g > 200 avg_b > 200 or ang_v > 150
            2. traffic sign is not captured completly
        '''
        image_bgr = cv2.imread(img_path)
        # houchuli 过滤
        b, g, r = cv2.split(image_bgr)
        b_list = [m for i in b for m in i]
        g_list = [m for i in g for m in i]
        r_list = [m for i in r for m in i]
        avg_b = sum(b_list) / len(b_list)
        avg_g = sum(g_list) / len(g_list)
        avg_r = sum(r_list) / len(r_list)
        if avg_b > 200 and avg_g > 200 and avg_r > 200:
            print('the avg of rgb are bigger than 200')
            return None
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image_hsv)
        v_list = [m for i in v for m in i]
        avg_v = sum(v_list) / len(v_list)
        if avg_v > 150:
            print('the avg of v is bigger than 150')
            return None
        return image_bgr


class TextRecognizer(BaseModule):

    def __init__(self, config):

        self.img_size = [3, 32, 320]
        self.ctc_decode = CTCLabelDecode(
            use_space_char=True, character_dict_path=config['ocr_dict'])
        self.model_path = config['ocr_model_path']
        self.build_session(self.model_path)

    def resize_norm_img(self, img):
        imgC, imgH, imgW = self.img_size

        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image

        return padding_im

    def run(self, image, is_down=False):
        st = time.time()
        image = self.resize_norm_img(image)
        preds = self.session.run(
            None, {self.session.get_inputs()[0].name: image[None]})[0]

        rec_result, _ = self.ctc_decode(preds)[0]
        rec_result = self.postprocess(rec_result, is_down)
        return rec_result, time.time() - st

    def postprocess(self, rec_result, is_down):
        '''
        逻辑纠正公里桩检测结果：
            上半部分：均为数字
            下半部分：字母+数字
            不包含特殊字符

        '''
        res_final = 'None'
        if len(rec_result) > 0:
            if is_down:
                alpha = rec_result[0]
                numbers = rec_result[1::]
            else:
                alpha = ''
                numbers = rec_result

            numbers_set = set(numbers)
            if alpha == '5':
                alpha = 's'
            if 's' in numbers_set:
                numbers = numbers.replace('s', '5')
            if 'i' in numbers_set:
                numbers = numbers.replace('i', '1')
            if 'l' in numbers_set:
                numbers = numbers.replace('l', '1')
            if 'o' in numbers_set:
                numbers = numbers.replace('o', '0')
            if 'z' in numbers_set:
                numbers = numbers.replace('z', '2')
            if ' ' in numbers_set:
                numbers = numbers.replace(' ', '')
            if '.' in numbers_set:
                numbers = numbers.replace('.', '')
            res_final = alpha + numbers

        return res_final


class Retrival(BaseModule):

    def __init__(self, config, topk=3, embedding_size=512):
        self.img_size = (224, 224)
        self.model_path = config['retrival_model_path']
        self.register_path = config['retrival_register_path']
        self.feature_library_path = config['retrival_feature_lib_path']
        self.cls_index = config['retrival_cls_index_path']
        self.embedding_size = embedding_size
        self.topk = topk
        self.build_session(self.model_path)

        # 不存在特征库就注册，存在就直接加载这个仓库
        if not os.path.exists(self.feature_library_path):
            self.supports, self.sdict = self.generate(self.register_path)
        else:
            self.supports = faiss.read_index(self.feature_library_path)
            # 读取文件
            # with open("signDict.pkl", "rb") as tf:
            #     self.sdict = pickle.load(tf)
            with open(self.cls_index, 'r') as js:
                self.sdict = json.load(js)
                # print(self.sdict)

    def generate(self, path):
        '''
            Used to register pictures for each traffic sign class
            and generate the search library

            Parameters:
                path: the path of folder for register
            Returns:
                index_l2: index of faiss used for searching
                inds: a dictionary key: index of vector, value: category of traffic sign
        '''
        # 初始化faiss，使用欧氏距离度量向量距离
        index_l2 = faiss.IndexFlatL2(self.embedding_size)
        # 字典{ 向量索引：向量对应图片的种类}
        inds = []
        folder_list = os.listdir(path)
        # folder 的名字是交通标志的种类
        for folder in folder_list:
            folderpath = os.path.join(path, folder)
            filelist = os.listdir(folderpath)
            for file in filelist:
                inds.append(folder)
                src = cv2.imread(os.path.join(folderpath, file))
                input = self.preprocess(src)
                feature = self.model(input.to(
                    self.device)).detach().cpu().numpy()
                # feature注册
                index_l2.add(feature)
        faiss.write_index(index_l2, self.feature_library_path)

        # 保存这个注册的库
        inds_js = json.dumps(inds)
        with open(self.cls_index, 'w') as js:
            js.write(inds_js)
        js.close()
        return index_l2, inds

    def run(self, image):
        '''
            Used to retrieve pictures for each traffic sign class

            Parameters:
                im: the img is read by opencv
                k: search topk nearest vectors in faiss

            Returns:
                traffic sign category of im

        '''
        image = self.preprocess(image)
        # 提取图片特征
        feature_search = self.session.run(
            None, {self.session.get_inputs()[0].name: image[None]})[0]
        maxIterItem, distance = self.postprocess(feature_search)

        return maxIterItem, distance

    def postprocess(self, feature_search):
        D, idx = self.supports.search(feature_search, self.topk)
        res = []
        for i in range(self.topk):
            tt = self.sdict[idx[0].tolist()[i]]
            res.append(tt)
        # 获取出现次数最多的item，若出现次数相同，返回第一个
        maxIterItem = max(res, key=res.count)
        distance = D[0][res.index(maxIterItem)]
        return maxIterItem, distance

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resnet_mean = [0.485, 0.456, 0.406]
        resnet_std = [0.229, 0.224, 0.225]
        image = cv2.resize(image, self.img_size)
        image = image / 255.0
        image -= resnet_mean
        image /= resnet_std
        image = image.astype(np.float32).transpose((2, 0, 1))
        return image


class AssetInfo(object):

    def __init__(self,
                 bbox=[],
                 score=None,
                 label=None,
                 covered_result=False,
                 track_id=None,
                 ocr_out=None,
                 retrival_out=None):
        self.bbox = bbox
        self.score = score
        self.label = label

        self.covered_result = covered_result
        self.track_id = track_id
        self.ocr_out = ocr_out
        self.retrival_out = retrival_out

    def __str__(self) -> str:
        pass


class Asset(BaseModule):

    def __init__(self, config):

        # 检测模型初始化
        detcet_config = config['asset_detect']
        self.img_size = (1280, 1280)
        self.model_path = detcet_config['detect_model_path']
        self.build_session(self.model_path)
        self.confs = {
            0: 0.5,
            1: 0.3,
            2: 0.5,
            3: 0.3,
            4: 0.4,
            5: 0.3,
            6: 0.3,
            7: 0.6
        }
        self.class_names = detcet_config['detect_names']
        self.jtbz_threshold = detcet_config['small_jtbz_filter']
        self.jtbz_cals = detcet_config['jtbz_cals']
        self.tracker = build_tracker(detcet_config['DEEPSORT'], use_cuda = True) # 记录使用过的trackid
        self.used_trackids = [] # 记录使用过的trackid

        # OCR模型初始化
        ocr_config = config['asset_ocr']
        self.textRecognizer = TextRecognizer(ocr_config)
        self.ocr_names = ocr_config['ocr_names']
        self.ocr_results = []

        # 检索模型初始化
        retrival_config = config['asset_retrival']
        self.retrival = Retrival(retrival_config)
        self.retrival_name = retrival_config['retrival_name']

    def run(self, image):
        ori_shape = image.shape
        ori_image = image.copy()
        image = self.preprocess(image)
        det_shape = image.shape
        
        output = self.session.run(
            None, {self.session.get_inputs()[0].name: image[None]})[0]
        
        output = torch.tensor(output).to('cuda')
        out = self.non_max_suppression(output, multi_confs = self.confs)[0]
        out[:, :4] = self.scale_coords(det_shape[1:], out[:, :4],
                                       ori_shape[:2]).round()
        out = self.postprocess(out, ori_image)
        return out

    def preprocess(self, image):
        h0, w0 = image.shape[:2]  # orig hw
        r = self.img_size[0] / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img = cv2.resize(
                image, (int(w0 * r), int(h0 * r)),
                interpolation=cv2.INTER_LINEAR)
        img = self.letterbox(
            img, self.img_size, auto=False, scaleup=False, stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = img.astype('float32')
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, ori_img):
        # 记录一张图片中所有的交通资产信息AssetInfo
        img_info = []
        # 待追踪的信息记录下来
        tracker_preds = np.ones((1, 6))

        for *xyxy, conf, cls in reversed(preds):
            xyxy = np.array([item.detach().cpu() for item in xyxy])
            conf = np.array(conf.detach().cpu())
            cls = cls.detach().cpu()
            label = self.class_names[int(cls)]

            # retrival_switch = True  # retrival 开关

            # 交通标志面积过滤
            if label in self.jtbz_cals and (int(xyxy[3]) - int(xyxy[1])) * (
                    int(xyxy[2]) - int(xyxy[0])) < self.jtbz_threshold:
                continue

            # 实例化asset的输出
            asset_info = AssetInfo(bbox=xyxy, score=conf, label=label)

            print(f'detect output: {label}, conf: {conf}, xyxy:{xyxy}')
            
            # 功能1：crop 公里桩ocr
            if label == 'lichengzhuang':
                h = int(xyxy[3]) - int(xyxy[1])
                w = int(xyxy[2]) - int(xyxy[0])
                if h * w > 400:
                    ww = int(w / 10 * 1.5)  # 缩小公里桩的宽度
                    hh = int(h / 10)  # 缩小公里桩的高度
                    crop_img = ori_img[int(xyxy[1]) + hh:int(xyxy[1] +
                                                             2 / 3 * h),
                                       int(xyxy[0] + ww):int(xyxy[2] -
                                                             ww), ::1]

                    ocr_out = self.textRecognizer.run(image=crop_img)[0]
                    asset_info.ocr_out = ocr_out
                    self.ocr_results.append(ocr_out)
                    # label_result += f'ocr: {ocr_out} ||'
                    print(f'ocr output is {ocr_out} ')
                else:
                    print('the size of glz is smaller than 400')

            # 功能2：如果是遮挡类别，就把 asset_info.label 里面换一下
            if label == 'covered':
                asset_info.label = 'covered'
                asset_info.covered_result = True
                print(f'covered output is {asset_info.covered_result} ')

            # 功能3：三角标志进行检索分类
            if label == self.retrival_name:
                h = int(xyxy[3]) - int(xyxy[1])
                w = int(xyxy[2]) - int(xyxy[0])
                hh, ww = int(h / 20), int(w / 20)
                crop_img = ori_img[int(xyxy[1]) + hh:int(xyxy[3]) - hh,
                                   int(xyxy[0]) + ww:int(xyxy[2]) - ww, ::1]
                retrival_out = self.retrival.run(image=crop_img)[0]
                asset_info.retrival_out = retrival_out
                print(f'retrival output is {retrival_out} ')
            print('=' * 20)
            img_info.append(asset_info)
            tracker_preds = np.append(tracker_preds,
                                      np.expand_dims(np.hstack((xyxy, conf, cls)).astype('float32'), 0),
                                      axis=0)
    
        # 判断图片中是否存在被遮挡的交通标志
        tracker_preds = tracker_preds[1:, :]
        # breakpoint()
        xyxys = tracker_preds[:,0:4]
        print(f"=============={xyxys}==============")
        xywhs = torch.Tensor(self.xyxy2xywh(tracker_preds[:,0:4]))
        confs = torch.Tensor(tracker_preds[:,4])
        clss = tracker_preds[:,5]
        outputs = self.tracker.update(xyxys, xywhs, confs, clss, ori_img)
        
        for value in list(outputs):
            x1, y1, x2, y2, cls_, track_id = value       
            for asset_info in img_info:
                if ([x1,y1,x2,y2]==asset_info.bbox).all():
                    asset_info.track_id = track_id
        """
            对输出进行去重操作
        """
        boxes_ = []
        scores_ = []
        labels_ = []
        is_covered_ = []
        ocr_names_ = []
        retrival_names_ = []
        track_ids_ = []
        
        for asset_info in img_info:
            boxes_.append(asset_info.bbox)
            scores_.append(asset_info.score)
            retrival_names_.append(asset_info.retrival_out) 
            track_ids_.append(asset_info.track_id)
            
            # 使用追踪去重：1）非gonglizhuang的label字段 2）covered字段
            if asset_info.track_id == None:
                labels_.append(asset_info.label)
                # 遮挡字段的去重 1
                is_covered_.append(asset_info.covered_result)
                ocr_names_.append(asset_info.ocr_out)
            else:
                if asset_info.track_id in self.used_trackids:
                    labels_.append(asset_info.label)
                    # 遮挡字段的去重2 如果已经被追踪了，肯定是False
                    is_covered_.append(False)
                    ocr_names_.append('')
                    print('It is tracked and label is None')
                else:
                    self.used_trackids.append(asset_info.track_id)
                    print(self.used_trackids)
                    if asset_info.label in self.jtbz_cals:
                        labels_.append('biaozhibiaopai')
                    else:
                        labels_.append(asset_info.label)
                    # 遮挡字段的去重3：如果是第一次加入追踪的，是什么就加入什么
                    is_covered_.append(asset_info.covered_result)
                    ocr_names_.append(asset_info.ocr_out)

        # 用于去重保存的buffer最多不超过100
        if len(self.used_trackids) > 100:
            self.used_trackids = self.used_trackids[1:]
        
        return np.array(
            boxes_), scores_, labels_, is_covered_, ocr_names_, retrival_names_, track_ids_

    def letterbox(self,
                  im,
                  new_shape=(640, 640),
                  color=(114, 114, 114),
                  auto=True,
                  scaleFill=False,
                  scaleup=True,
                  stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
            1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[
                0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)  # add border
        return im, ratio, (dw, dh)

    def non_max_suppression(self,
                            prediction,
                            conf_thres=0.1,
                            iou_thres=0.45,
                            classes=None,
                            agnostic=False,
                            multi_label=False,
                            labels=(),
                            max_det=300,
                            multi_confs={}):
        """Runs Non-Maximum Suppression (NMS) on inference results

        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS
        # insert by zhinanzhang 20220317
        multi_conf = len(multi_confs) > 0

        output = [torch.zeros(
            (0, 6), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1),
              4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()),
                              1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()),
                              1)  # [conf.view(-1) > conf_thres]
                # insert by zhinanzhang 20220317
                if multi_conf:
                    res = x[0].unsqueeze(0)
                    for c in multi_confs:
                        each_thres = multi_confs[c]
                        each_inds = torch.where(j == c)[0]
                        each_conf, each_box, each_class = conf[each_inds], box[
                            each_inds], j[each_inds]
                        each_x = torch.cat(
                            (each_box, each_conf, each_class.float()),
                            1)[each_conf.view(-1) > each_thres]
                        res = torch.cat((res, each_x), 0)
                    x = res[1:]
                else:
                    x = x[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes,
                                                 device=x.device)).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(
                    descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :
                              4] + c, x[:,
                                        4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n <
                          3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                    1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
        return output

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
    
    def xyxy2xywh(self, x):
        # Convert nx4 boxes from [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right to [x, y, w, h] 
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        if x.shape[0] == 0:
            return y
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # center x
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # center y
        y[:, 2] = x[:, 2] - x[:, 0]  # width of the bbox
        y[:, 3] = x[:, 3] - x[:, 1]  # height of the bbox
        return y

    def box_iou(self, box1, box2):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
                 torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter
                        )  # iou = inter / (area1 + area2 - inter)

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0],
                       img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
                img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords

    def clip_coords(self, boxes, shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[:, 0].clamp_(0, shape[1])  # x1
            boxes[:, 1].clamp_(0, shape[0])  # y1
            boxes[:, 2].clamp_(0, shape[1])  # x2
            boxes[:, 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
