import cv2
import json
import os
import numpy as np
import torch
import yaml
from feature import FeatureExtractor


'''
OpenCV에서는 BGR color format을 사용 
    이유 : BGR ordering was popular among camera manufacturers and other software developers at the time
    Ex) 빨간색은 RGB 값으로 (255, 0, 0)이지만 BGR 값으로는 (0, 0, 255)

# [참고 내용]
# height, width , channel =  cv2.imread("xx/xx/xx").shape
# cv2.rectangle의 ptr1 : 이미지 좌측 상단 (x,y)좌표
# cv2.rectangle의 ptr2 : 이미지 우츨 하단 (x,y)좌표
'''


'''
   parameter format :
    regions -- [xmin, ymin, xmax, ymax]
'''
def show_ImgFeatureInfo(imgPath, sub_region, obj_region, sub_name, obj_name, pred_name):
    feature = FeatureExtractor()
    img = cv2.imread(imgPath)

    sub_color = (0, 0, 255)
    obj_color = (255, 0, 0)
    pred_color = (255, 0, 255)
    attention_color = (0, 255, 0)

    '''
    Subject 정보 사진에 나타내기
    '''
    # subject bbox
    cv2.rectangle(img, (sub_region[0], sub_region[3]), (sub_region[2], sub_region[1]), sub_color, thickness=3)
    # subject label
    cv2.putText(img, "Subject : " + sub_name, (sub_region[0], sub_region[1] - 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=sub_color, thickness=2)

    '''
    Object 정보 사진에 나타내기
    '''
    # object bbox
    cv2.rectangle(img, (obj_region[0], obj_region[3]), (obj_region[2], obj_region[1]), obj_color, thickness=3)
    # object label
    cv2.putText(img, "Object : " + obj_name, (obj_region[0], obj_region[1] - 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=obj_color, thickness=2)

    '''
    Predicate 정보 사진에 나타내기
    '''
    # subject center (x,y)
    sub_center = feature.cal_bbox_center(sub_region[0], sub_region[1], sub_region[2], sub_region[3])
    # object center (x,y)
    obj_center = feature.cal_bbox_center(obj_region[0], obj_region[1], obj_region[2], obj_region[3])
    # predicate arrowed line
    cv2.arrowedLine(img, sub_center, obj_center, color=pred_color, thickness=2)
    # predicate arrowed line direction
    direction = feature.cal_sub2obj_direction(sub_center[0], obj_center[0], sub_center[1], obj_center[1], print_mode=1)
    cv2.putText(img, "("+direction+")", (obj_center[0]+10, obj_center[1]-30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=pred_color, thickness=3)
    # predicate label
    cv2.putText(img, pred_name, (22,22), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=pred_color, thickness=2)

    '''
    attention_region 정보 사진에 나타내기
    '''
    # attention_region = find_attention_window(sub_region, obj_region)
    # cv2.rectangle(img, (attention_region[0], attention_region[3]), (attention_region[2], attention_region[1]), attention_color, thickness=1)

    '''
    모든 정보가 포함된 사진 보여주기
    '''
    cv2.imshow("Image with Infos", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


'''
   parameter format :
    regions -- [xmin, ymin, xmax, ymax]
    final_size -- (height, width)
   return :
     numpy (height, width, 3), numpy (height, width, 3)
'''
def crop_and_resize_Img(img_path, sub_region, obj_region, final_size):
    original_img = cv2.imread(img_path)
    # print("original_img size = " + str(original_img.shape))

    cropped_sub = original_img[sub_region[1]:sub_region[3], sub_region[0]:sub_region[2]]
    cropped_obj = original_img[obj_region[1]:obj_region[3], obj_region[0]:obj_region[2]]

    # print("cropped_sub size = " + str(cropped_sub.shape))
    # print("cropped_obj size = " + str(cropped_obj.shape))

    resized_sub = resize_Img(cropped_sub, final_size)
    resized_obj = resize_Img(cropped_obj, final_size)

    # print("resized_sub size = " + str(resized_sub.shape))
    # print("resized_obj size = " + str(resized_obj.shape))

    # cv2.imshow("cropped_sub", cropped_sub)
    # cv2.imshow("cropped_obj", cropped_obj)
    # cv2.imshow("resized_sub", resized_sub)
    # cv2.imshow("resized_obj", resized_obj)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return resized_sub, resized_obj


'''
    참고 : https://076923.github.io/posts/Python-opencv-8/
'''
def resize_Img(src, resize_shape):
    if src.shape[0] < resize_shape[0] or src.shape[1] < resize_shape[1]:
        return cv2.resize(src, dsize=resize_shape, interpolation=cv2.INTER_CUBIC)
    else:
        return cv2.resize(src, dsize=resize_shape, interpolation=cv2.INTER_AREA)

'''
   parameter format :
    regions -- [xmin, ymin, xmax, ymax]
'''
def find_attention_window(sub_region, obj_region):
    attention_window = []
    for idx in range(0,2):
        if sub_region[idx] < obj_region[idx]:
            attention_window = attention_window + [sub_region[idx]]
        else:
            attention_window = attention_window + [obj_region[idx]]
    for idx in range(2,4):
        if sub_region[idx] > obj_region[idx]:
            attention_window = attention_window + [sub_region[idx]]
        else:
            attention_window = attention_window + [obj_region[idx]]
    return attention_window

'''
   parameter format :
    regions -- [xmin, ymin, xmax, ymax]
   return : 
    torch.Tensor
'''
def make_2channel_img(img_path, sub_region, obj_region):
    img = cv2.imread(img_path)
    height, width, channel = img.shape
    first_subject_channel = torch.zeros(height, width, 1)
    second_object_channel = torch.zeros(height, width, 1)
    sub_region = [sub_region[0]+1, sub_region[1]+1, sub_region[2]-1, sub_region[3]-1]
    obj_region = [obj_region[0]+1, obj_region[1]+1, obj_region[2]-1, obj_region[3]-1]

    for x_idx in range(sub_region[0], sub_region[2] + 1):
        for y_idx in range(sub_region[1], sub_region[3] + 1):
            first_subject_channel[y_idx][x_idx] = 1

    for x_idx in range(obj_region[0], obj_region[2] + 1):
        for y_idx in range(obj_region[1], obj_region[3] + 1):
            second_object_channel[y_idx][x_idx] = 1

    return torch.cat((first_subject_channel, second_object_channel), 2)


def make_attention_square(attention_img):
    if attention_img.shape[0] == attention_img.shape[1]:
        return attention_img
    if attention_img.shape[0] < attention_img.shape[1]:
        append_len = attention_img.shape[1] - attention_img.shape[0]
        if append_len % 2 == 0:
            padding_zero1 = torch.from_numpy(np.zeros((int(append_len/2), attention_img.shape[1], 2)))
            padding_zero2 = torch.from_numpy(np.zeros((int(append_len/2), attention_img.shape[1], 2)))
        else:
            padding_zero1 = torch.from_numpy(np.zeros((int(append_len/2), attention_img.shape[1], 2)))
            padding_zero2 = torch.from_numpy(np.zeros((int(append_len/2)+1, attention_img.shape[1], 2)))
        return torch.cat((padding_zero1, attention_img, padding_zero2), dim=0)
    if attention_img.shape[0] > attention_img.shape[1]:
        append_len = attention_img.shape[0] - attention_img.shape[1]
        if append_len % 2 == 0:
            padding_zero1 = torch.from_numpy(np.zeros((attention_img.shape[0], int(append_len/2), 2)))
            padding_zero2 = torch.from_numpy(np.zeros((attention_img.shape[0], int(append_len/2), 2)))
        else:
            padding_zero1 = torch.from_numpy(np.zeros((attention_img.shape[0], int(append_len/2), 2)))
            padding_zero2 = torch.from_numpy(np.zeros((attention_img.shape[0], int(append_len/2)+1, 2)))
        return torch.cat((padding_zero1, attention_img, padding_zero2), dim=1)


'''
   parameter format :
    img_path -- string
    regions -- [xmin, ymin, xmax, ymax]
    final_size -- (height, width)
   return : 
    numpy (height, width, 2)
'''
def make_interaction_pattern(img_path, sub_region, obj_region, final_size):
    # 2 Channel 이미지 형성
    img2D_tensor = make_2channel_img(img_path, sub_region, obj_region)
    # attention window 구역 찾아내기
    attention_region = find_attention_window(sub_region, obj_region)
    # 찾아낸 attention window 크기로 2 Channel 이미지를 crop 하기
    attention_tensor = img2D_tensor[attention_region[1]:attention_region[3], attention_region[0]:attention_region[2]]
    # attention window 크기로 crop 한 이미지를 정사각형 형태로 만들어주는 zero padding 작업 진행
    square_tensor = make_attention_square(attention_tensor)
    return resize_Img(square_tensor.numpy(), final_size)

def midpoint(start_point, end_point):
    return int((start_point[0] + end_point[0]) / 2), int((start_point[1] + end_point[1]) / 2)


def pred_nameMaker(list):
    msg = "Predicate : "
    for eachName in list:
        msg = msg + "#" + eachName + ", "
    return msg

if __name__ == '__main__':
    cfg_path = 'config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    predicates_categories = json.load(open(os.path.join(cfg['data_root'], 'predicates.json')))
    objects_categories = json.load(open(os.path.join(cfg['data_root'], 'objects.json')))

    # {"predicate": [18], "subject": {"category": 0, "bbox": [192, 650, 266, 697]}, "object": {"category": 0, "bbox": [589, 639, 635, 698]}}
    imgs_path = os.path.join(cfg['data_root'], 'train_images', '000011.jpg')
    sub_region = [192, 650, 266, 697]
    obj_region = [589, 639, 635, 698]
    sub_name = objects_categories[0]
    obj_name = objects_categories[0]
    list = [predicates_categories[18]]
    pred_name = pred_nameMaker(list)

    # {"predicate": [3], "subject": {"category": 49, "bbox": [192, 493, 312, 828]}, "object": {"category": 49, "bbox": [67, 495, 178, 702]}}
    # imgs_path = os.path.join(cfg['data_root'], 'train_images', '000944.jpg')
    # sub_region = [192, 493, 312, 828]
    # obj_region = [67, 495, 178, 702]
    # sub_name = objects_categories[49]
    # obj_name = objects_categories[49]
    # list = [predicates_categories[3]]
    # pred_name = pred_nameMaker(list)

    # {"predicate": [0, 9], "subject": {"category": 71, "bbox": [60, 297, 608, 455]}, "object": {"category": 5, "bbox": [2, 1, 1277, 955]}}
    # imgs_path = os.path.join(cfg['data_root'], 'train_images', '000944.jpg')
    # sub_region = [60, 297, 608, 455]
    # obj_region = [2, 1, 1277, 955]
    # sub_name = objects_categories[71]
    # obj_name = objects_categories[5]
    # list = [predicates_categories[0], predicates_categories[9]]
    # pred_name = pred_nameMaker(list)

    show_ImgFeatureInfo(imgs_path, sub_region, obj_region, sub_name, obj_name, pred_name)
    # crop_and_resize_Img(imgs_path, sub_region, obj_region, (224,224))

    # interaction_pattern = make_interaction_pattern(imgs_path, sub_region, obj_region, (224,224))
    # print("square_img size ==> " + str(interaction_pattern.shape))
    #
    # # for visualization
    # imsi_channel = np.zeros((interaction_pattern.shape[0], interaction_pattern.shape[1], 1))
    # img = np.concatenate((interaction_pattern, imsi_channel), 2)
    # print("square_img visualization size ==> " + str(img.shape))
    # cv2.imshow("FINAL", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()