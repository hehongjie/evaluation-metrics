#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 09:08:21 2020

@author: user
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 10:07:08 2020

@author: 17733
"""


import os
import numpy as np
import cv2 as cv
def evaluation_metrics(groundtruth,premask):
    #二值分割图是一个波段的黑白图，正样本值为1，负样本值为0
    #通过矩阵的逻辑运算分别计算出tp,tn,fp,fn
    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
    true_pos = float(np.logical_and(premask, groundtruth).sum())  # float for division
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(premask, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, groundtruth).sum()
    return true_pos,true_neg,false_pos,false_neg

def evaluate_model(res_path,gt_path,gt_lst,prd_lst):
    true_pos = 0
    true_neg  = 0
    false_pos= 0
    false_neg = 0
    for i in range(len(gt_lst)):
        gt_img=cv.imread(os.path.join(gt_path,gt_lst[i]),2)
        # print(i)
        assert gt_lst[i][:]==prd_lst[i][5:]    # make sure groundtruth images match predicted images
        prd_img=cv.imread(os.path.join(res_path,prd_lst[i]),2)
        met_para=evaluation_metrics(gt_img,prd_img)
        true_pos += met_para[0]
        true_neg  += met_para[1]
        false_pos += met_para[2]
        false_neg += met_para[3]

    prec = true_pos / (true_pos + false_pos + 1e-10)
    rec = true_pos / (true_pos + false_neg + 1e-10)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg + 1e-10)
    F1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-10)
    IoU = true_pos / (true_pos + false_neg + false_pos + 1e-10)
    inv_IoU = true_neg/(true_neg+false_neg + false_pos + 1e-10)
    print("==================================================")
    print("Precision:\t",prec)
    print("==================================================")
    print("Recall:\t",rec)
    print("==================================================")
    print("Accuracy/Pixel accuracy:\t",accuracy)
    print("==================================================")
    print("F1:\t",F1)
    print("==================================================")
    print("IoU:\t",IoU)
    print("==================================================")
    print("mIoU:\t",(IoU+inv_IoU)/2)
    
def mosaic_image(path,img_lst):
    mosaic_img=np.full((8350,8350,3),np.nan)
    j=16
    k=0
    idx=0
    print(path)
    for i in range(len(img_lst)):
        
        img=cv.imread(os.path.join(path,img_lst[i]),-1)
        if k<16 and j>0:
            mosaic_img[j*512-354:(j+1)*512-354,k*512:(k+1)*512,:]=img[:,:,:]
            k=k+1
        elif k==16 and j>0:
            mosaic_img[j*512-354:(j+1)*512-354,k*512:8350,:]=img[:,:158,:]
            j=j-1
            k=0
        elif k<16  and j==0:
            mosaic_img[:158,k*512:(k+1)*512,:]=img[:158,:,:]
            k=k+1
            
        elif k==16  and j==0:
            mosaic_img[:158,k*512:8350,:]=img[:158,:158,:]
            if not os.path.exists(os.path.join(path,'mosaic','mosaic_img')):
                os.makedirs(os.path.join(path,'mosaic'))
                os.makedirs(os.path.join(path,'mosaic','mosaic_img'))
            cv.imwrite(os.path.join(path,'mosaic','mosaic_img',str(idx)+'.jpg'),mosaic_img,[int(cv.IMWRITE_JPEG_QUALITY), 100])
            mosaic_img=np.full((8350,8350,3),np.nan)
            idx=idx+1
            j=16
            k=0
    return os.path.join(path,'mosaic','mosaic_img')
def mosaic_label(path,label_lst):
    mosaic_img=np.full((8350,8350),np.nan)
    j=16
    k=0
    idx=0
    for i in range(len(label_lst)):
        
        label=cv.imread(os.path.join(path,label_lst[i]),2)
        if k<16 and j>0:
            mosaic_img[j*512-354:(j+1)*512-354,k*512:(k+1)*512]=label[:,:]
            k=k+1
        elif k==16 and j>0:
            mosaic_img[j*512-354:(j+1)*512-354,k*512:8350]=label[:,:158]
            j=j-1
            k=0
        elif k<16  and j==0:
            mosaic_img[:158,k*512:(k+1)*512]=label[:158,:]
            k=k+1
            
        elif k==16  and j==0:
            mosaic_img[:158,k*512:8350]=label[:158,:158]
            if not os.path.exists(os.path.join(path,'mosaic','mosaic_label')):
                #os.makedirs(os.path.join(path,'mosaic'))
                os.makedirs(os.path.join(path,'mosaic','mosaic_label'))
            cv.imwrite(os.path.join(path,'mosaic','mosaic_label',str(idx)+'.jpg'),mosaic_img,[int(cv.IMWRITE_JPEG_QUALITY), 100])
            mosaic_img=np.full((8350,8350),np.nan)
            idx=idx+1
            j=16
            k=0 
    return os.path.join(path,'mosaic','mosaic_label')
def mosaic_pred(path,prd_lst):
    mosaic_img=np.full((8350,8350),np.nan)
    j=16
    k=0
    idx=0
    for i in range(len(prd_lst)):
        
        label=cv.imread(os.path.join(path,prd_lst[i]),2)
        if k<16 and j>0:
            mosaic_img[j*512-354:(j+1)*512-354,k*512:(k+1)*512]=label[:,:]
            k=k+1
        elif k==16 and j>0:
            mosaic_img[j*512-354:(j+1)*512-354,k*512:8350]=label[:,:158]
            j=j-1
            k=0
        elif k<16  and j==0:
            mosaic_img[:158,k*512:(k+1)*512]=label[:158,:]
            k=k+1
            
        elif k==16  and j==0:
            mosaic_img[:158,k*512:8350]=label[:158,:158]
            if not os.path.exists(os.path.join(path,'mosaic','mosaic_prediction')):
                #os.makedirs(os.path.join(path,'mosaic'))
                os.makedirs(os.path.join(path,'mosaic','mosaic_prediction'))
            cv.imwrite(os.path.join(path,'mosaic','mosaic_prediction',str(idx)+'.jpg'),mosaic_img,[int(cv.IMWRITE_JPEG_QUALITY), 100])
            mosaic_img=np.full((8350,8350),np.nan)
            idx=idx+1
            j=16
            k=0
    return os.path.join(path,'mosaic','mosaic_prediction')
def visual_inspection(imgpth,labpth,prdpth):
    img_lst=os.listdir(imgpth)
    img_lst.sort(key=lambda x:int(x[:-4]))
    lab_lst=os.listdir(labpth)
    lab_lst.sort(key=lambda x:int(x[:-4]))
    prd_lst=os.listdir(prdpth)
    prd_lst.sort(key=lambda x:int(x[:-4]))
    assert (len(img_lst)==len(lab_lst)) and (len(lab_lst)==len(prd_lst))
    for i in range(len(img_lst)):
        img = cv.imread(os.path.join(imgpth,img_lst[i]),-1)
        lab = cv.imread(os.path.join(labpth,lab_lst[i]),2)
        prd = cv.imread(os.path.join(prdpth,prd_lst[i]),2)
        comb_img=img[:,:,:]
        for r in range((img.shape)[0]):
            for c in range((img.shape)[1]):
                if prd[r,c]==255 and lab[r,c]==255:     #TP
                    comb_img[r,c,0]=207
                    comb_img[r,c,1]=159
                    comb_img[r,c,2]=114
                elif prd[r,c]==255 and lab[r,c]==0:   #FP
                    comb_img[r,c,0]=69
                    comb_img[r,c,1]=64
                    comb_img[r,c,2]=199
                elif prd[r,c]==0 and lab[r,c]==255:   #FN
                    comb_img[r,c,0]=172
                    comb_img[r,c,1]=219
                    comb_img[r,c,2]=185
        if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(imgpth)),'visual')):
            os.makedirs(os.path.join(os.path.dirname(os.path.dirname(imgpth)),'visual'))
        cv.imwrite(os.path.join(os.path.dirname(os.path.dirname(imgpth)),
        'visual',img_lst[i]),comb_img,[int(cv.IMWRITE_JPEG_QUALITY), 100])
def indvisual_inspection(path,img_lst,lab_lst,prd_lst):
    assert (len(img_lst)==len(lab_lst)) and (len(lab_lst)==len(prd_lst))
    for i in range(len(img_lst)):
        img = cv.imread(os.path.join(path,img_lst[i]),-1)
        lab = cv.imread(os.path.join(path,lab_lst[i]),2)
        prd = cv.imread(os.path.join(path,prd_lst[i]),2)
        comb_img=img[:,:,:]
        for r in range((img.shape)[0]):
            for c in range((img.shape)[1]):
                if prd[r,c]==255 and lab[r,c]==255:     #TP
                    comb_img[r,c,0]=207
                    comb_img[r,c,1]=159
                    comb_img[r,c,2]=114
                elif prd[r,c]==255 and lab[r,c]==0:   #FP
                    comb_img[r,c,0]=69
                    comb_img[r,c,1]=64
                    comb_img[r,c,2]=199
                elif prd[r,c]==0 and lab[r,c]==255:   #FN
                    comb_img[r,c,0]=172
                    comb_img[r,c,1]=219
                    comb_img[r,c,2]=185
        if not os.path.exists(os.path.join(path,'small_visual')):
            os.makedirs(os.path.join(path,'small_visual'))
        cv.imwrite(os.path.join(path,'small_visual',img_lst[i]),comb_img,[int(cv.IMWRITE_JPEG_QUALITY), 100])
if __name__=="__main__":
    #res_path = r'/home/user/Desktop/tensorflow_image_segmentation/HRNetV2/HRNetV2_keras_tensorflow_semisupervised-master/WBD/WBD on mas/results'      #FCNresults Deeplabv3+results UNETresults
    res_path = r'/home/user/Desktop/tensorflow_image_segmentation/HRNetV2/HRNetV2_keras_tensorflow_semisupervised-master/composite/WHU/results'
    #gt_path = r'/home/user/Desktop/Waterloo building dataset/test/label'
    #img_path = r'/home/user/Desktop/Waterloo building dataset/test/image'
    gt_path = r'/home/user/Desktop/toronto building dataset/WHU building dataset/3. The cropped aerial image tiles and raster labels/3. The cropped image tiles and raster labels/test/image'
    img_path = r'/home/user/Desktop/toronto building dataset/WHU building dataset/3. The cropped aerial image tiles and raster labels/3. The cropped image tiles and raster labels/test/label'
    #gt_path = r'/home/user/Desktop/masbuilding/test/pad/label'
    #img_path = r'/home/user/Desktop/masbuilding/test/pad/image'
    gt_lst=os.listdir(gt_path)
    prd_lst=os.listdir(res_path)
    img_lst=os.listdir(img_path)
    gt_lst.sort()
    prd_lst.sort()
    # for name in file_lst:
    #     if name[:4]=="mask":
    #         gt_lst.append(name)
    #     elif name[:4]=="pred":
    #         prd_lst.append(name)
    #     elif name[:6]=="origin":
    #         img_lst.append(name)
    # gt_pth=r'C:\Users\17733\Desktop\combination\combination\groundtruth'
    # prd_pth=r'C:\Users\17733\Desktop\combination\combination\v3+'
    # gt_lst=os.listdir(gt_pth)
    # prd_lst=os.listdir(prd_pth)
    gt_lst.sort(key=lambda x:int(x[:-4]))
    prd_lst.sort(key=lambda x:int((x[5:])[:-4]))
    img_lst.sort(key=lambda x:int(x[:-4]))
    assert len(gt_lst)==len(prd_lst)
    #===================================evulation==================================
    evaluate_model(res_path,gt_path,gt_lst,prd_lst)

    # #===================================mosaic=====================================
    # mo_imgpth = mosaic_image(res_path,img_lst)
    # mo_labpth = mosaic_label(res_path,gt_lst)
    # mo_prdpth = mosaic_pred(res_path,prd_lst)
    
    # #==================================visual inspection==============================
    # #RGB:   185 219 172 groundtruth    199 64 69 wrong classified(false positive)  114 159 207 predicted
    # # mo_imgpth = r'C:\Users\17733\Desktop\new reault\new reault\fcn-results\mosaic\mosaic_img'
    # # mo_labpth = r'C:\Users\17733\Desktop\new reault\new reault\fcn-results\mosaic\mosaic_label'
    # # mo_prdpth = r'C:\Users\17733\Desktop\new reault\new reault\fcn-results\mosaic\mosaic_prediction'
    # visual_inspection(mo_imgpth,mo_labpth,mo_prdpth)
    
    # indvisual_inspection(res_path,img_lst,gt_lst,prd_lst)