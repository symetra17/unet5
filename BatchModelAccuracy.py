#!/usr/bin/env python
# coding: utf-8


import os
import io
import cv2
import numpy as np
import pandas as pd
import json
import time

# CalmodelAccuracy is a selfdefined py file
# provide precision, recall, overall accuracy calculation
import CalModelAccuracy as cma



# retrive json data by reading the json file path
def retrive_json_data(json_path):
    with open(json_path) as f:
        json_data = json.load(f)
        
    return json_data



# produce one image by drawing the polygons saved in json
def obtain_tagged_image(json_data):
    shapes_list = json_data['shapes']
    img_width = json_data['imageWidth']
    img_height = json_data['imageHeight']
    num_polygon = len(shapes_list)
    tag_image = np.zeros(shape = [img_height, img_width, 3], dtype = np.uint8)
    
    for item in enumerate(shapes_list):
        pts = item[1]['points']
        list_points = np.array(pts, np.int32)
        cv2.fillPoly(tag_image, [list_points], (255, 255, 255))
        
    return tag_image, img_width, img_height




# calculate non zero area
# the input data is three-bands
def obtain_area(img_data):
    img_area = np.count_nonzero(img_data)
    return img_area


# calculate the accuracy relate parameters for one image
def obtain_accuracy(overlap_area, tagged_area, predict_area, img_width, img_height):
    TP = overlap_area / 3
    FN = tagged_area / 3 - TP
    FP = predict_area / 3 - TP
    imgheight = img_height
    imgwidth = img_width
    tot_pixels = imgheight*imgwidth
    
    #TN = ImageSize*3 - TP - FP - FN
    TN = cma.get_TN(tot_pixels, TP, FP, FN)
    precision_rate = cma.model_precision(TP, FP)
    recall_rate = cma.model_recall(TP, FN)
    accuracy_rate = cma.model_accuracy(TP, TN, tot_pixels)
    cf_matrix = cma.confusion_matrix(TP, TN, FP, FN)
    
    return precision_rate, recall_rate, accuracy_rate, cf_matrix



# create log file by given directory
def create_log(log_path):
    filename = "Model_accuracy_log.txt"
    f = open(os.path.join(log_path, filename), "w+")
    second = time.time()
    f.write("The log is created at -%s." %time.ctime(second)+"\n")
    f.close()



# updated the log file based on full file path and information
def write_log(file, info_txt):
    with io.open(file,"a",encoding="utf-8") as f:
        f.write(info_txt)
        f.write('\n')
        f.write('\n')


def format_matrix(df):
    new_df = df.div(100000)
    new_matrix = new_df.round(1)
    new_matrix['Unit'] = '100K'
    return new_matrix
        
        

# batch calculation, calculate the accuracy of detection results
# should provide the following path:
# the json folder stores json files
# the folder stores all the detection result in bmp format
# the folder to store accuracy result
def batch_cal(json_folder, detect_folder,output_path):
    log_path = output_path
    create_log(log_path)
    info_txt = "The detection results are in - " + detect_folder+'\n' +\
    "The JSON files are in - " + json_folder + '\n'
    
    write_log(os.path.join(output_path,"Model_accuracy_log.txt"), info_txt)
    
    overall_precision = 0.0
    overall_recall = 0.0
    overall_accuracy = 0.0
    num_img = 0
    # empty_data = np.zeros((3,3))
    # col_name = ["True Structure", "True Background", "True Total"]
    # row_name = ["Predict Structure", "Predict Background", "Predict Total"]
    # tot_cf = pd.DataFrame(empty_data, row_name, col_name)
    
    for root, dirs, files in os.walk(detect_folder, topdown=False):
        for name in files:
            if os.path.splitext(os.path.join(root, name))[1].lower() == ".bmp":
                # filename is the bmp file name
                filename = name.split('.')[0]
                imgname = filename.split('_')[0]
                predict_path = os.path.join(root, name)
                json_path = os.path.join(json_folder, imgname) + ".json"
                                              
                
                if os.path.isfile(json_path):
                    print("Accuracy Assesment for %s:" %imgname)
                    predict_img = cv2.imread(predict_path)
                    json_data = retrive_json_data(json_path)
                    tagged_img, img_width, img_height = obtain_tagged_image(json_data)
                    img_overlap = cv2.bitwise_and(tagged_img, predict_img, mask =None)
                    
                    # calculate the area of the tagged, predicted and overlapped
                    overlap_area = obtain_area(img_overlap)
                    tagged_area = obtain_area(tagged_img)
                    predict_area = obtain_area(predict_img)
                    # calculte the precision_rate, recall_rate, accuracy_rate 
                    # and confusion Matrix of the dection result
                    precision_rate, recall_rate, accuracy_rate, cf_matrix = obtain_accuracy(overlap_area,
                                                                                            tagged_area,
                                                                                            predict_area,
                                                                                            img_width,
                                                                                            img_height)
                    # write the confusion matrix into single csv file
                    cf_matrix.to_csv(os.path.join(output_path,filename+'_Confusion_matrix.csv'),header = True)
                    info_txt = "Detection Accuracy for - "+ imgname +':\n' +\
                    "The model's precision is: " + str(round(precision_rate,1)) + '%.\n' +\
                    "The model's recall is: "+  str(round(recall_rate,1)) + '%.\n' +\
                    "The model's accuracy is: " + str(round(accuracy_rate,1))+ '%.\n' +\
                    "Confusion Matrix:" + '\n' +\
                    format_matrix(cf_matrix).to_string() + '\n'
                    # cf_matrix.to_string()
                    
                    write_log(os.path.join(output_path,"Model_accuracy_log.txt"), info_txt)
                    cv2.imwrite('ground_true.png',tagged_img)
                    overall_precision += precision_rate
                    overall_recall += precision_rate
                    overall_accuracy += accuracy_rate
                    num_img += 1
                    # tot_cf = tot_cf + cf_matrix
                    if num_img == 1:
                        tot_cf = cf_matrix
                    else:
                        tot_cf = tot_cf + cf_matrix
                    print("\n")
                    
                else:
                    print("JSON file for "+ file_name +" is not found!" +"\n")
                    info_txt = "JSON file for "+ file_name +" is not found!" +'\n'
                    write_log(os.path.join(output_path,"Model_accuracy_log.txt"), info_txt)
                    
    
    if num_img > 0:
        info_txt = "Overall Detection Accuracy: "+'\n' +\
        "The model's avg precision is: " +\
        str(round(overall_precision/num_img,1)) + '%.\n' +\
        "The model's avg recall is: "+\
        str(round(overall_recall/num_img,1)) + '%.\n' +\
        "The model's avg overall accuracy is: " +\
        str(round(overall_accuracy/num_img,1))+ '%.\n'+\
        "Confusion Matrix:" + '\n' +\
        format_matrix(tot_cf).to_string()
        # tot_cf.to_string()
        
        print("The final confusion matrix:")
        print(format_matrix(tot_cf))
        tot_cf.to_csv(os.path.join(output_path,'Confusion_Matrix.csv'),header = True)
        write_log(os.path.join(output_path,"Model_accuracy_log.txt"), info_txt)


