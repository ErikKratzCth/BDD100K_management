import json
from pathlib import Path, PurePath
import numpy as np
import math
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import time

def load_bdd100k_attribute_spec(img_folder, labels_file, attributes_normal, attributes_outlier, num_train, num_val, num_test, outlier_frac, output_type = 'file_lists'):
    # labels_file: full path of the JSON file with BDD100K labels.
    # train_dir: path of directory containing training images
    # val_dir: path of directory containing validation images
    # test_dir: path of directory containing testing images
    # attributes_normal: n-by-2 list of attributes to be included in the training and validation sets. 1<=n<=3. The first input in each row is the type of attribute, i.e. "weather", and the second is the value of the attribute, i.e. "rainy". Must be defined so that there is no overlap with outlier class.
    # attributes_outlier: n-by-2 list of attributes to be included as outliers in the testing set. 1<=n<=3. Shape is same as attributes_normal. Must be defined so that there is no overlap with normal class.
    # num_train: number of images to be loaded into training set
    # num_val: number of images to be loaded into validation set
    # num_test: number of images to be loaded into testing set
    # outlier_frac: fraction of outliers in testing set
    # output_type: one of 'image_data' and 'file_lists'. 'image_data' returns datasets as np.ndarrays, 'file_lists' returns a list of filenames for each set
    
    print('Loading json data ...', end='')
    start_time = time.time()
    with open(labels_file) as json_data:
        loaded_json_data = json.load(json_data)
    print('\rLoaded json data (%.2fs)' % (time.time()-start_time))
    
    start_time = time.time()
    num_outlier_to_choose = int(math.ceil(num_test * outlier_frac))
    num_normal_test = (num_test - num_outlier_to_choose)
    num_normal_to_choose = num_train + num_val + num_normal_test
    
    print('Parsing json data ...')
    # Find images with right attributes
    normal_filenames, scale_factor = parse_json_data(loaded_json_data, attributes_normal, num_normal_to_choose)
    if scale_factor >= 0:
        num_train = int(math.ceil(scale_factor * num_train))
        num_val = int(math.ceil(scale_factor * num_val))
        num_normal_test = int(math.ceil(scale_factor * num_normal_test))
        num_outlier_to_choose = int(math.ceil(scale_factor * num_outlier_to_choose))
        print('Image sets downsized by factor %.2f' % scale_factor)
    print('Normal file list complete (%.2fs)' % (time.time()-start_time))
    
    start_time = time.time()   
    outlier_filenames, scale_factor = parse_json_data(loaded_json_data, attributes_outlier, num_outlier_to_choose)
    if scale_factor >= 0:
        num_train = int(math.ceil(scale_factor * num_train))
        num_val = int(math.ceil(scale_factor * num_val))
        num_normal_test = int(math.ceil(scale_factor * num_normal_test))
        num_outlier_to_choose = int(math.ceil(scale_factor * num_outlier_to_choose))
        print('Image sets downsized by factor %.2f' % scale_factor)
    print('Outlier file list complete (%.2fs)' % (time.time()-start_time))
    
    
    # Assert there is no overlap between target and outlier data
    overlap_counter = 0;
    for filename in outlier_filenames:
        if filename in normal_filenames:
            overlap_counter += 1
            outlier_filenames.remove(filename)
    if overlap_counter > 0:
        print("\nWARNING: overlap between target and outlier class: removed %d images from Outlier file list\n" % overlap_counter)
        num_outlier_to_choose -= overlap_counter
    
    if output_type == 'file_lists':
        start_time = time.time()
    
        train_list = normal_filenames[:num_train] 
        print("Generated train_list (%.2fs)" % (time.time()-start_time))
    
        start_time = time.time()
        val_list = normal_filenames[num_train:num_train + num_val]
        print("Generated val_list (%.2fs)" % (time.time()-start_time))
    
        start_time = time.time()
        test_list = normal_filenames[num_train + num_val:] + outlier_filenames[:num_outlier_to_choose]
        print("Generated test_list (%.2fs)" % (time.time()-start_time))
        
        return train_list, val_list, test_list
        
    elif output_type == 'image_data':
        # Specify image format
        channels = 3
        image_height = 720 # 180
        image_width = 1280 # 320

        normal_data = np.ndarray(shape=(num_train + num_val + num_normal_test, image_height, image_width, channels), dtype=np.uint8)
        outlier_data = np.ndarray(shape=(num_outlier_to_choose, image_height, image_width, channels), dtype=np.uint8)

        start_time = time.time()
        # Load normal images
        for i, _file in enumerate(normal_filenames):
            img = load_img(str(img_folder / _file))  # this is a PIL image
            img.thumbnail((image_width, image_height))
            x = img_to_array(img)  
            normal_data[i] = x
        print("Normal image data loaded (%.2fs)" % (time.time()-start_time))

        start_time = time.time()  
        # Load outlier images
        for i, _file in enumerate(outlier_filenames):
            img = load_img(str(img_folder / _file))  # this is a PIL image
            img.thumbnail((image_width, image_height))
            x = img_to_array(img)
            outlier_data[i] = x
        print("Outlier image data loaded (%.2fs)" % (time.time()-start_time))

        start_time = time.time()

        train_data = normal_data[:num_train] 
        print("Generated train_data (%.2fs)" % (time.time()-start_time))

        start_time = time.time()
        val_data = normal_data[num_train:num_train + num_val]
        print("Generated val_data (%.2fs)" % (time.time()-start_time))

        start_time = time.time()
        test_data = np.concatenate((normal_data[num_train + num_val:], outlier_data), axis=0)
        print("Generated test_data (%.2fs)" % (time.time()-start_time))

        return train_data, val_data, test_data
    else:
        print("Output_type argument '" + output_type + "' not supported." )


def parse_json_data(json_data, attributes_to_choose, num_to_choose):
    img_names = []
    
    for entry in json_data:
        add_flag = True
        for attribute in attributes_to_choose:
            if isinstance(attribute[1], list): 
                add_flag = False
                for option in attribute[1]: # if several attribute options are specified, one match is enough => True
                    if entry["attributes"][attribute[0]] == option:
                        add_flag = True
                if not add_flag: # if no option matched, do not evaluate other attributes
                    break
            elif entry["attributes"][attribute[0]] != attribute[1]: # if only one option is specified, any other value => False
                add_flag = False
                break
        if add_flag:
            img_names.append(entry["name"])
                
        # check if enough images are found
        if len(img_names) == num_to_choose:
            break
            
    scale_factor = -1
    if len(img_names) < num_to_choose:
        attribute_str = ""
        for attribute in attributes_to_choose:
            attribute_str += attribute[0]+": "
            if isinstance(attribute[1], list): 
                for attribute_value in attribute[1]:
                    attribute_str += attribute_value + "/"
            else:
                attribute_str += attribute[1]
            attribute_str += ', '
        err_str = '\tNot enough files with specified attributes (' + attribute_str + ')\n\t\tRequested: %d\n\t\tFound: %d'
        print( err_str % (num_to_choose, len(img_names)))
        scale_factor = len(img_names)/num_to_choose
        
    return img_names, scale_factor