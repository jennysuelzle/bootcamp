# -*- coding: utf-8 -*-
"""
Written by Sofia Magkiriadou, sofia.magkiriadou@epfl.ch .
"""
import os
import re
from PIL import Image
import numpy
import nptdms
from nptdms import TdmsWriter, RootObject, GroupObject, ChannelObject, TdmsFile
import imageio
import h5py
import pims
import tifffile
import pathlib

###########################
### General information ###
###########################

###########################
##   About this script   ##

# The purpose of this script is to convert among different file formats that you might encounter as an iSCAT user. The formats of interest are: .tiff, .tdms, .mp, and .h5.
# .tiff and .h5 are common formats. .tdms and .mp are proprietary formats. We are interested in these proprietary formats because they are compatible with existing iSCAT analysis software: .tdms works with the Oxford LabView Analysis module and .mp works with Discover MP, from Refeyn. Note also that Refeyn provides a file converter from .tdms to .mp.
# On the other hand, .tiff files are much more standard and currently a possible output of the Adimec version of our homemade iSCAT setup. This setup also outputs .png files, but .tiff is strongly preferred as it is truly lossless. This script also contains a function for .png files.
# The goal of this script is to enable using already existing analysis software with standard image files. It bridges standard image file formats to the two proprietary formats, .tdms and .mp.
# Special thanks to Sam Tusk from the Kukura group at Oxford and Lewis Carney from the software team at Refeyn for their help.

##########################################
## About the .tdms and .mp file formats ##

# These two file formats have similar architecture and they are both similar to .h5 files.
# Briefly: This architecture is hierarchical. If you are familiar with python dictionaries, you can think of a file as dictionary. The hierarchy goes as follows.
# A file can contain multiple groups; each group can either contain other groups, or it can contain channels. The channels contain data.
# You can think of the file itself as a group; it is sometimes called the 'root' group.
# In .tdms files the channels contain flattened data; in order words, the data is a 1-d array. Since we will be typically using images as data, this means that you need to take care when folding and unfolding the data. For example, if you have a movie of ten frames that are each 100 by 200 pixels in x by y, you will need to convert it into a 200,000 by 1 array in order to save it into a channel of a .tdms file. There are many ways to do that: for instance to flatten a single frame you could append the pixel values row by row or column by column. If you then unfold it incorrectly, you might end up frames rotated by 90 degrees - if you misinterpret x for y - or with nonesense-looking frames - if you misinterpret the number of frames for a frame dimension. In this script, I have taken care to fold the movies in the right way such that the Oxford analysis software unfolds reads them properly.
# In contrast, in .h5 or .mp files the data can maintain its original shape. Much simpler!
# The flexibility in file content and organization offered by the architecture of all these file formats makes it easy to store metadata, which explains their appeal.
# For more information see the following links. They were written as part of the API for python libraries that handle these types of files, but they are helpful in general for understanding the basic concepts.
# for h5
# https://docs.h5py.org/en/stable/quick.html
# for .tdms
# https://nptdms.readthedocs.io/en/stable/

###################################################
## Important concepts for this script ##

# The goal is twofold:
# 1. to correctly convert data from .tiff (or .png) format to .tdms or .mp
# 2. in such a way that the analysis software will recognize it correctly; in other words, emulating the core features of the files that the two analysis programs expect to find.

# Keep in mind that in this version the .mp files you generate will have no metadata.
# I have confirmed with Lewis that this does not affect the accuracy of the mass measurement with Discover MP, if you use a calibration file taken with the same conditions as your experiment - as should anyway be the case.
# This might affect the score panel, where the movie quality is judged with quantities such as sharpness and signal.
# You can still compare these quantities between movies taken under the same conditions, just not between movies taken with the Refeyn and with the homemade iSCAT.
# Since the script for running the iSCAT with the Adimec camera outputs metadata, it is possible to include this metadata in the final .mp file in a manner consistent with what Discover MP expects. This will be done in a subsequent version of this script. It remains to then be checked, if this information will suffice for Refeyn's score calculations of signal, sharpness etc, or if it needs further information that is not currently stored.

# Finally, a note on file paths. To avoid confusion between mac and Windows, I am using pathlib to specify the locations of files. This is a python library that converts a string to a directory path with the syntax that is appropriate for the operating system you are running on:
# https://docs.python.org/3/library/pathlib.html#basic-use
# To specify paths in any operating system, you type p = pathlib.Path('main_directory/subdirectory1/subdirectory2/') etc with all slashes being forward slashes, i.e. '/'. Since paths are often expected from the functions in this script, keep this in mind.

############################
### Conversion functions ###
############################

def tiff_to_mp(tiff_location, mp_file, single_stack = True, filename_core = None):
    '''
    Convert a .tiff stack or a sequence of .tiff files to an .mp file. Note that .mp files are secretly .h5 files, and this is how we handle them here.
    
    INPUT
    -----
    tiff_location : a pathlib Path (see last comment under "Important concepts for this script")
        If you have a single .tiff stack, this is the full path to the file, including the filename. In this case you should also make sure to set single_stack = True.
        If you have a sequence of .tiff files, this is the full path name to the folder that contains the .tiff files you want to convert. In this case you should also make sure to set single_stack = False.
        
    mp_file : a pathlib Path (see last comment under "Important concepts for this script")
        The full name for the .mp file you will save (file location and name).
    
    single_stack : boolean, defaults to True
        Set to True if you have one single .tiff file that contains the whole stack of frames. Set to False if you have a collection of .tiff files, one per frame.
        
    filename_core : string
        If you have a collection of .tiff files, here you specify the string that all filenames have in common. For example, if all your frames are named "frame00000.tiff, frame00001.tiff, ..." the filename_core would be "frame".

    OUTPUT
    ------
    A numpy array of all images.
    '''
    
    # Load the .tiff file(s) and convert it (them) into a single numpy array.
    if single_stack:
        numberAboveTotalFrames = 100000
        with tifffile.TiffFile(tiff_location) as _file:
            data =_file.asarray(key=slice(0,numberAboveTotalFrames))

    else:
        names = pathlib.Path(filename_core + '*.tif')
        tiff_location = tiff_location / names
#        tiff_location = os.path.join(tiff_location, names)  # if the variables tiff_location, names are plain strings. If you insert them as pathlib Paths, use the line above. 
        image_sequence = tifffile.TiffSequence(tiff_location)
#        print(tiff_location)
        data = image_sequence.asarray()
#        print(data.shape)
        data = data[0]
#        print(data.shape)

    # Save this array as an .mp file. "frame" is a feature required by Discover MP. It is currently the only feature. More features might be added in later versions. See also comment above on metadata.
    
    with h5py.File(mp_file, "w") as f:
        f.create_dataset("frame", data=data)
    
    return data

'''
WINDOWS COMMAND LINES - WILL REMOVE THIS PARAGRAPH AS IT IS IN THE INSTRUCTIONS NOW
py -c "import iSCAT_file_conversions; import pathlib; iSCAT_file_conversions.tiff_to_mp(pathlib.Path('C://Users/laboleb/Documents/code/iSCAT_file_conversions/test_data/210623/sofiacopy_totiff_all_frames/'), pathlib.Path('C://Users/laboleb/Documents/code/iSCAT_file_conversions/test_data/210623/test02.mp'), single_stack = False, filename_core = 'sofia_copy_totiffstack')"
'''

def convert_images_to_tdms(imagepath, tdms_template_file = pathlib.Path('Z:/SHARED/_Scientific projects/iSCAT_JS_SoM/03_data/data_from_others/Kukura/20200804-Sam/take1/event0.tdms'), effective_frame_rate = 100.5, exposure_time =  100.5, bin_frame_size = 4, z = 1.5, af_radius = 300, image_file_extension = '.tif', target_file = pathlib.Path('Z:/SHARED/_Scientific projects/iSCAT_JS_SoM/03_data/data_from_others/Refeyn_Fabian/event00.tdms')):
    '''
    Convert many image files into one .tdms file.
    This file can then be opened with the LabView analysis software from Oxford. It can also be opened with Discover MP, the analysis software from Refeyn, after being converted from .tdms to .mp with their file converter program.
        
    Note that the LabView analysis software will not recognize a .tdms file if its filename does not contain the string 'event'. Therefore, this string is currently appended to the filename automatically.
    
    The parameters 'effective_frame_rate, exposure_time, bin_frame_size, z, af_radius' are present in the Oxford .tdms files, and I am emulating this format in this script.
    
    A note on the metadata saved in Oxford .tdms files: As I have verified with Sam, in the Oxford files these parameters were not always entered accurately, nor used further downstream, so their values are inconsequential for the analysis.
    The only important one for some functions that they use presently (April 2021) is the effective_frame_rate.
        
    INPUT
    -----
    imagepath : a pathlib Path (see last comment under "Important concepts for this script")
        The path to the folder that contains the image files. The script will take all the image files in that folder.
        
    tdms_template_file : a pathlib Path (see last comment under "Important concepts for this script")
        The full path + filename of the .tdms file that you will use as a template. The format of this file will also be the format of the final .tdms file.
        
    effective_frame_rate : float
        This is the frame rate after time-averaging.
        At Oxford they use two steps of time-averaging. The first one is a regular average, which they apply before saving the images. The second done is a rolling average, which they apply during the analysis. Here we are concerned with the first one. The effective_frame_rate is the frame rate of the saved data, related to the acquisition frame rate and to the number of frames per time-average in the first time-averaging step.
        
    exposure_time : float
        The duration of exposure, i.e. the time for which the shutter was open. Note that it is often registered as 0.0 in the Oxford files; see note above on Oxford metadata.
        
    bin_frame_size : float
        This is the size of the time bins for the first step of time-averaging. In other words, bin_frame_size * effective_frame_rate = acquisition_frame_rate.
        
    z : float
        Likely the z position of the stage, although inconsequential; see note above on Oxford metadata.
        
    af_radius : float
        The radius of the autofocus signal; also inconsequential, see note above on Oxford metadata.
        
    image_file_extension : string
        The extension of the image files you will convert to a .tdms file. Typically this will be '.tif' (avoid '.png' in general).
        
    target_file : a pathlib Path (see last comment under "Important concepts for this script")
        The full filename, including directory path, of the .tdms file you will generate and save.
    
    OUTPUT
    ------
    The most important outcome when you run this function is the saved .tdms file that contains your images in a format that the Oxford Labview analysis program can read.
    For your information, the function currently also outputs three items.
    The first item is a channel_object; this is the object in which the data from the images is stored.
    The second item is a 1-d numpy array of the data itself, flattened to be compatible with the .tdms requirements of the software.
    The third item is a numpy array of the original images, in their original dimensions.
    '''
    
    images = []                                                 # initialize the list of all images
    filenames = [x for x in os.listdir(str(imagepath)) if re.search(image_file_extension, x)]  # collect the list of all filenames
    filenames = sorted(filenames)    ### !!! Sort the filenames. This is very important! Without this the order of the images might not follow their temporal order.
    number_of_frames = len(filenames)
    
    print('There are ' + str(number_of_frames) + ' frames.')    # sanity-check: did you load as many frames as you thought you had?
    
    for i in filenames:                                         # create the list of images; load each image and append it to the list
        image = imageio.imread(str(imagepath) + '/' + i)
        images.append(image)
    images = numpy.array(images, dtype = numpy.float32)         # convert the image list into a numpy array. According to Lewis, the Refeyn team uses 32bit floating numbers.

    xlength = float(image.shape[0])                             # extract the dimensions along x, y
    ylength = float(image.shape[1])
    print('Each image is ' + str(xlength) + ' by ' + str(ylength) + ' pixels.' )  # sanity-check: are the image dimensions what you thought they were?

    data = numpy.reshape(images, (image.shape[0] * image.shape[1] * number_of_frames))  # reshape the numpy array of images. Previously it had dimensions (number of images) by x by y; now it has dimensions (number of images * x * y) by 1.

    tdms_template = nptdms.TdmsFile.open(str(tdms_template_file))    # open the template file without reading it; faster than .read() for large files, which we expect to have

    # next, create a new tdms file object that will include your image data. Following standard Oxford .tdms architecture, this file will include one group called 'img'; this group will include one channel called 'cam1'; and in this channel will reside the image data.
    root_object = RootObject(properties=tdms_template.properties)  # create the main object ("root"), a new object with the same properties as the .tdms file you are using for template.
        #Q: this line is not necessary when I run the program on the iSCAT desktop computer (Windows)
    print('The properties of the template file: ')              # print the properties you found in the template file
    print(tdms_template.properties)

    group_object = GroupObject('img', properties=None)  # create a group object with the same properties as the first group object of the template file. In the Oxford .tdms files there is only one group object, so I am only emulating that one.
    # Q: this line is not necessary when I run the program on the iSCAT desktop computer (Windows)

#    group_object = GroupObject(tdms_template.groups()[0], properties=None)  # create a group object with the same properties as the first group object of the template file. In the Oxford .tdms files there is only one group object, so I am only emulating that one.
    # Q: this line is not necessary when I run the program on the iSCAT desktop computer (Windows)

    # create a channel object. This channel will be called 'cam1' and it will belong to a group called 'img'. It will contain as data the reshaped image array you just created. It will have the same properties as the Oxford template tile, which I state here explicitly for clarity. Remember that, except for the effective_frame_rate, the others are so far inconsequential and are coming along for the ride. Of course you can insert meaningful values that describe your experiment and then you can use them as true metadata, for your information.
    # The names 'img' and 'cam1' are the names used in the Oxford files for the group and channel. You can get these by typing tdms_template.objects.keys().
    channel_object = ChannelObject("img", "cam1", data,  properties={'Effective frame rate' : effective_frame_rate, 'Exposure Time' : exposure_time, 'Bin frame size': bin_frame_size, 'Image size' : xlength, 'Z position (um)' : z, 'Autofocus radius': af_radius, 'Image size 2' : ylength})

    # Finally, write the data to the new tdms file, and save the file.
    with TdmsWriter(str(target_file)) as tdms_writer:
        print(target_file)
        tdms_writer.write_segment([root_object, group_object, channel_object]) # Q: this line does not work when I run the program on the iSCAT desktop computer (Windows). See also two lines above commented out.

    # If you need to write another segment with more data for the same channel:
    #    more_data = numpy.array([6.0, 7.0, 8.0, 9.0, 10.0])
    #    channel_object = ChannelObject("group_1", "channel_1", more_data, properties={})
    #    tdms_writer.write_segment([channel_object])
    
    return channel_object, data, images

def tdms_to_images(tdms_filename, destination, save = True, image_file_extension = '.tiff'):
    '''
    Convert a .tdms file to a series of .tiff frames and save them in a dedicated folder called destination.

    INPUT
    -----
    tdms_filename : a pathlib Path (see last comment under "Important concepts for this script")
        The full name of the .tdms file, including the path to the file.
        
    destination : a pathlib Path (see last comment under "Important concepts for this script")
        The full path to the folder where the final images will be stored.
        
    save : boolean, defaults to True
        If True, the images will be saved in the destination folder. If False, they will not be saved. Either way, you will get a numpy array of the images as the function's output.
    
    image_file_extension : string, defaults to '.tiff'
        The format of the saved image files. .tiff is recommended.

    OUTPUT
    ------
    A numpy array of your images. It is shaped such that, if d is the array, d[0] is the first image, d[1] the second, etc.
    '''
    tdms_filename = str(tdms_filename)
    
    f = nptdms.TdmsFile(tdms_filename)
    
    data_raw = f.channel_data('img', 'cam1')  # Typically, the data in Oxford .tdms files is contained in a channel called 'cam1' that is contained in a group called 'img'. You can extract these names by typing tdms_file.objects.keys(), where tdms_file is the Oxford file.

    attributes = f.object('img', 'cam1').properties  # read the properties of the data - in other words, the metadata
    ylength = int(attributes['Image size'])          # these properties contain, among other information on the experimental conditions, the image dimensions - which we use here in order to shape the data into image-like format
    xlength = int(attributes['Image size 2'])
    number_of_frames = int(len(data_raw) / (ylength * xlength))  # you can finally extract the number of frames by dividing the length of the data_raw array by the total number of pixels per frame
    print('I found ' + str(number_of_frames) + ' frames.')  # sanity-check: does the total number of frames make sense?
    
    data = numpy.reshape(data_raw, (number_of_frames, ylength, xlength))  # shape the data: from the 1-d array that was stored into the .tdms, to a numpy array with shape (number of frames by y by x)
    
    if save:
        if os.path.isdir(destination):                # check if the destination folder exists already
            if len(os.listdir(destination)) != 0:     # if it exists, check if it has content
                answer = input('This directory is already occupied. Do you wish to rewrite its contents? Type y if yes.')                            # if it has content, ask whether you want to rewrite its content
            else:
                answer = 'y'
        else:
            os.mkdir(destination)                     # if the destination folder does not yet exist, make it
            answer = 'y'
        if answer == 'y':
            for i, j in enumerate(data):
                k = Image.fromarray(j)                 # convert each frame to an image
                k_file_tag = 'frame' + str(i).zfill(5) + image_file_extension
                final_destination = destination / k_file_tag
                k.save(final_destination) # save the image with the name 'frame' and its framenumber
#                k.save(destination + 'frame' + str(i).zfill(5) + image_file_extension) # save the image with the name 'frame' and its framenumber
        else:
            print('The contents of ' + str(destination) + ' were not overwritten.')
                   
    return data

#######################
### Other Functions ###
#######################

def read_tdms_properties(filename):
    '''
    Extract the properties of a template .tdms file.
    The purpose of this function is to help decipher the format of the .tdms files generated by the Kukura group, so as to then convert our image files to .tdms according to that format. This is useful for cases where the original data is in other format, such as .png or .tiff; see the function convert_images_to_tdms() below for more context.
    
    INPUT
    -----
    filename : a pathlib Path (see last comment under "Important concepts for this script")
    The full path + name of the template .tdms file whose properties you want to extract. Typically this will be a file from the Kukura group, whose structure you want to mimick later on.
    
    OUTPUT
    ------
    Two items: 1. an ordered dictionary (dictionary that remembers insertion order) of the properties of the file, and 2. a dictionary with information about the file structure.
    '''
    
    properties_dict = {}  # initialize the dictionary where you will insert the file properties
    
    filename = str(filename)
    
    tdms_file = nptdms.TdmsFile.open(filename)  # open the file without reading it; this is faster than nptdms.TdmsFile.read() for large files, and large files are common
    properties = tdms_file.properties # read the properties
    groups = tdms_file.groups()  # read the groups
    properties_dict['groups'] = groups # add the groups to the properties dictionary
    
    channels = []                                     # initialize the list of channels
    for i in groups:
        channels.append(tdms_file.group_channels(i))  # add all channels to the list
        properties_dict[str(i)] = channels            # add the list of channels to the properties dictionary
    
    #   channel_data = f.channel_data()
    #   channel_properties = channel.properties
    
    #properties_dict['channels'] = channels
    #   print('groups: ')
    #  print(groups)
    #print('channels: ')
    #   print(channels)
    
    return properties, properties_dict

# to print all properties, from https://stackoverflow.com/questions/47947991/nptdms-python-module-not-getting-all-channel-properties

def load_images(image_location, filename_core = 'frame', extension = '.tif'):
    '''
    Load the images found in image location.
    
    INPUT
    -----
    image_location : a pathlib Path (see last comment under "Important concepts for this script")
    The full path name to the folder that contains the images you want to load.
    
    OUTPUT
    ------
    A sequence containing the loaded images (see pims documentation for more details).
    '''
    
    names = filename_core + '*' + extension
    image_location = image_location / names
    image_location = str(image_location)
    frames = pims.ImageSequence(image_location)
    
    return frames

def png_to_mp(image_location, mp_location, extension = '.png'):
    '''
    Convert a series of images to an .mp file. Note that .mp files are secretly .h5 files, and this is how we handle them here.
    Note: in principle, this should also work for .tiff files, however this is not currently the case. Use tiff_to_mp for .tiff files.
    
    INPUT
    -----
    image_location : a pathlib Path (see last comment under "Important concepts for this script")
        The full path name to the folder that contains the images you want to load.
    
    mp_location : a pathlib Path (see last comment under "Important concepts for this script")
        The full path for the location and name of the .mp file you will save.
    
    OUTPUT
    ------
    A numpy array of all images.
    '''
    
    images = load_images(image_location, extension = extension)  # load your images
    image_array = numpy.asarray(images)              # make a single numpy array with all images
    
    with h5py.File(mp_location, "w") as f:
        f.create_dataset("frame", data=image_array)  # Then save this array as an .mp file. "frame" is a feature required by Discover MP. It is currently the only feature required. More features will be added in a later versions, to include metadata.
    
    return image_array
