
# Creating a custom Object Detection model - Tutorial

Created for the Phoenix Tensorflow meetup group.  I learned from Sentdex's video series and thus this tutorial follows closely to his.  His videos can be found at https://www.youtube.com/playlist?list=PLQVvvaa0QuDcNK5GeCQnxYnSSaar2tpku.  

-------------------------------------
#### Q:  What is object detection?
A:  Object detection is a computer vision technique that detects instances of objects in an image or video, i.e., it is image classification combined with localization.

------------------------------------

## Setup and Installation
---------------------

Tensorflow requires python 3, if you don't have it, I recommend the Anaconda installation, which can be found at https://www.anaconda.com/download/

Clone the repository from https://github.com/tensorflow/models.  It may download the directory as "models-master", in that case just rename the folder to "models"

Follow the instructions on https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md, make sure that all of the dependancies are installed.  If you have the Anaconda distribution of Python then the only package you will have to install is tensorflow and pillow.

#### Notes:
- Tensorflow supports Nvidia GPUs, if your GPU is an AMD or you don't have one, use the cpu install
- The COCO API installation is optional, I did not use it

### Installing Protobuf

The instructions to install Protobuf are a bit vague, so here are some extra details.  Get the latest version of protoc for your OS from https://github.com/google/protobuf/releases (e.g., protoc-3.5.1-linux-x86_64.zip).  Unzip the file, for example as a new folder protoc3, then navigate to the models/research/ directory and run

``` bash
path/to/protoc3/bin/protoc object_detection/protos/*.proto --python_out=.
```

### Add Libraries to PYTHONPATH

When running locally, the tensorflow/models/research/ and slim directories
should be appended to PYTHONPATH. This can be done by running the following from
tensorflow/models/research/:


``` bash
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

**Note: This command needs to run from every new terminal you start.** If you wish
to avoid running this manually, you can add it as a new line to the end of your
~/.bashrc file.  (I tried adding this to .bashrc profile and for some reason it never worked, but it's managable)

Finally, install the object_detection library by navigating to the directory models/research, and running
``` bash
sudo python setup.py install
```

---------------------

## Collect images

In order identify custom objects, the model will need training data.  You can use Google's image search and download images of the object and save them to a dedicated directory.  For my own project, the images I trained on were taken from videos, and I used the MoviePy (pip install moviepy) library to extract one image every second from the video as follows:

``` python
from moviepy.editor import VideoFileClip
filename = 'MyVideo'
video = 'path/to/file/' + filename + '.mp4'
clip = VideoFileClip(video)
for second in range(int(clip.duration)):
        clip.save_frame('images/' + filename + '_' str(second) + '.png', second)
```

Now, in my "images" folder I have a bunch of screenshots from the video, labeled with the second at which it was taken from the video:

![image.png](images/files.png?raw=true)

The amount of images that you'll need for training will probably range from 100-500, depending on how complex the object is and how powerful/accurate you want your object detection model to be.  In my project, the objects are 2D images that only vary in shading, so I didn't have to worry about things such as size and angle, and I acheived good accuracy with ~150 images of each digit.  If you're not getting the results you want you can always go back and collect more data for training.

## Label images
Once you have all of your images saved, it is time to start labeling them.  There is a very nice program that we will use for this called LabelImg which you can download from https://github.com/tzutalin/labelImg.  Clone the repo and navigate into it, then run the installation code that corresponds to your setup (e.g., Ubuntu, python 3).  Once you've launched the program with
``` bash
python labelImg.py
```
it is time to start annotating your images.  Use "Open Dir" to add the directory that contains your saved images.  Use the shortcut key 'w' to create a bounding box and drag it over your object(s).  Once you've labeled everything in your image, click save or use 'ctrl + s' to save the xml file in the same directory as your image, and click next to go to the next image.

*Example of labeled image*
![image.png](images/lblimg.png?raw=true)

Once you've completed the annotations, copy ~10% of the images along with their corresponding XML files into a directory within your images directory called "test" and copy the rest of the images/XMLs into another directory called "train."

----------------

## Create TFRecords

The next step is to create a TF record file that our model will train from.  To do that, we use the file xml_to_csv.py to first create .csv files which contain the metadata of our training images and labels, then use generate_tfrecord.py to create the records (both of these files make use of the Pandas library, so if you don't have that package then you can get it with pip).  Download both of those files to the same directory as your images folder (so for example if you have your images saved in Desktop/images/ then you'll want to have Desktop/xml_to_csv.py) and create the csv files with 

```bash
python xml_to_csv.py
```

This will create two csv files that will then be used to create the tfrecords.

Next you'll need to modify the code of generate_tfrecord.py, starting on line 31, and enter the names of the object classes that you are identifying.  In my project I identified digits and the percent sign, so my function looks like this

```python
def class_text_to_int(row_label):
    if row_label == 1:
        return 1
    elif row_label == 2:
        return 2
    elif row_label == 3:
        return 3
    elif row_label == 4:
        return 4
    elif row_label == 5:
        return 5
    elif row_label == 6:
        return 6
    elif row_label == 7:
        return 7
    elif row_label == 8:
        return 8
    elif row_label == 9:
        return 9
    elif row_label == 0:
        return 10
    elif row_label == 'percent':
        return 11
    else:
        None
```

Once complete, you are ready to create the records.  Create the training record with
```bash
python generate_tfrecord.py --csv_input=train_labels.csv  --output_path=train.record
```

And create the test record with
```bash
python generate_tfrecord.py --csv_input=test_labels.csv  --output_path=test.record
```


----------------

## Set up the model

There are pretrained models that you can choose from at https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md.  The models have a trade off of speed vs accuracy, so depending on what your goals are, e.g., object detection in real time vs. correctly classifying images, you can choose accordingly.  I went with faster_rcnn_resnet101_coco, which provides a good balance of both speed and accuracy.  Download your model of choice and extract the folder to models/research/object_detection.

### Modify the .config file

In the model folder, open the .config file with a text editor, there are a few changes that need to be made.  First, change the num_classes in the beginning to the number of classes that you are detecting.  In my case, this is 11.  Then at the bottom, there are 5 lines that contain PATH_TO_BE_CONFIGURED that you will modify.  The lines originally look something like this:
```
...
fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"
  from_detection_checkpoint: true
}
train_input_reader {
  label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"
  tf_record_input_reader {
    input_path: "PATH_TO_BE_CONFIGURED/mscoco_train.record"
  }
}
eval_config {
  num_examples: 8000
  max_evals: 10
  use_moving_averages: false
}
eval_input_reader {
  label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"
  shuffle: false
  num_readers: 1
  tf_record_input_reader {
    input_path: "PATH_TO_BE_CONFIGURED/mscoco_val.record"
  }
}
```

Change the fine_tune_checkpoint to point to the model directory that you just created (and are currently editing the file in).  The remaining paths should be changed to "data" and we will create a label map called object-detection.pbtxt in the next step.  Finally, modify the .record files to be simply train.record and test.record, the names of the files we created earlier.  You can move those files into the models/research/object_detection/data/ directory now, too.  Once you modified everything, it should look something like this:
```
...
fine_tune_checkpoint: "faster_rcnn_resnet101_coco_2018_01_28/model.ckpt"
  from_detection_checkpoint: true
}
train_input_reader {
  label_map_path: "data/object-detection.pbtxt"
  tf_record_input_reader {
    input_path: "data/train.record"
  }
}
eval_config {
  num_examples: 8000
  max_evals: 10
  use_moving_averages: false
}
eval_input_reader {
  label_map_path: "data/object-detection.pbtxt"
  shuffle: false
  num_readers: 1
  tf_record_input_reader {
    input_path: "data/test.record"
  }
}
```

### Create the label map
In the data directory, use a text editor to create a new file called object-detection.pbtxt.  You can use the other label maps in that folder as an example to follow.

### Move images folder
Move your images folder that contains your images, xml files, train and test folders into models/research/object_detection directory

-----------------------------

## Train the model

Navigate to the models/research/object_detection directory, this is where we launch the file to train the model, but first create a new directory called *training*.  Now to train the model, I used
```bash
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=faster_rcnn_resnet101_coco_2018_01_28/pipeline.config
```
So the program is taking three arguments, if you are using a different model then your 3rd argument will be different, but it is simply pointing to the .config file that we created earlier, so set yours accordingly.  The training directory that we created is where the program will put the output of its training session.

Now the training begins and you should start seeing your loss value.  Here is a screenshot from my training:
![image.png](images/training.png?raw=true)
It's taking ~30 seconds per step, which is pretty slow but expected since I am not using a GPU.  Once your loss starts averaging around 1 then you stop the training by using ctrl + z in the terminal, but feel free to let it train longer if you want.  

--------------------------

## Export the inference graph
The next step is to use the export_inference_graph.py to create the frozen model for detecting objects.  Run the program from the models/research/object_detection directory, it takes four arguments and my code was as follows:
```bash
python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path faster_rcnn_resnet101_coco_2018_01_28/pipeline.config \
    --trained_checkpoint_prefix training/model.ckpt-396 \
    --output_directory digits_graph
```
Leave the input_type as image_tensor.  The pipeline_config_path once again points to your config file.  The trained_checkpoint_prefix should point to the latest checkpoint that you have.  In my case my loss was reasonably low after ~390 steps, but yours may take longer.  In any case, make sure that you have 3 checkpoint files (meta, index, and data) for whichever number checkpoint you use.  The last argument is the directory the program creates with the inference graph, so use a descriptive name related to your objects.

------------------------

## Detect objects
Now we can use the object_detection_tutorial.ipynb in the object_detection directory to test out the model.  You'll want to make a few changes to the code under the header "Variables."  Mine looks like this:
```python
# What model to download.
MODEL_NAME = 'digits_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'object-detection.pbtxt')

NUM_CLASSES = 10
```
Next, add whatever images you would like to test your detection on to the directory models/research/object_detection/test_images and rename the files to image3.jpg, image4.jpg, ... etc.  In the detection section, modify the range of the loop to correspond to your test images.  If you get a memory error, just run it on a couple of images at a time, i.e.,
```python
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.png'.format(i)) for i in range(5, 7) ]
```

Check out the images folder on this repo to see some of my results.
