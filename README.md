# Object Detection in an Urban Environment

## Data

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/).

[OPTIONAL] - The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. We have already provided the data required to finish this project in the workspace, so you don't need to download it separately.

## Structure

### Data

The data you will use for training, validation and testing is organized as follow:
```
/home/workspace/data/waymo
    - training_and_validation - contains 97 files to train and validate your models
    - train: contain the train data (empty to start)
    - val: contain the val data (empty to start)
    - test - contains 3 files to test your model and create inference videos
```

The `training_and_validation` folder contains file that have been downsampled: we have selected one every 10 frames from 10 fps videos. The `testing` folder contains frames from the 10 fps video without downsampling.

You will split this `training_and_validation` data into `train`, and `val` sets by completing and executing the `create_splits.py` file.

### Experiments
The experiments folder will be organized as follow:
```
experiments/
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training with the unchanged config file
    - experiment0/ - create a new folder for each experiment you run
    - experiment1/ - create a new folder for each experiment you run
    - experiment2/ - create a new folder for each experiment you run
    - label_map.pbtxt
    ...
```

## Prerequisites

### Local Setup

For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile and requirements in the [build directory](./build).

Follow [the README therein](./build/README.md) to create a docker container and install all prerequisites.

### Download and process the data

**Note:** ”If you are using the classroom workspace, we have already completed the steps in the section for you. You can find the downloaded and processed files within the `/home/workspace/data/preprocessed_data/` directory. Check this out then proceed to the **Exploratory Data Analysis** part.

The first goal of this project is to download the data from the Waymo's Google Cloud bucket to your local machine. For this project, we only need a subset of the data provided (for example, we do not need to use the Lidar data). Therefore, we are going to download and trim immediately each file. In `download_process.py`, you can view the `create_tf_example` function, which will perform this processing. This function takes the components of a Waymo Tf record and saves them in the Tf Object Detection api format. An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records). We are already providing the `label_map.pbtxt` file.

You can run the script using the following command:
```
python download_process.py --data_dir {processed_file_location} --size {number of files you want to download}
```

You are downloading 100 files (unless you changed the `size` parameter) so be patient! Once the script is done, you can look inside your `data_dir` folder to see if the files have been downloaded and processed correctly.

### Classroom Workspace

In the classroom workspace, every library and package should already be installed in your environment. You will NOT need to make use of `gcloud` to download the images.

## Instructions

### Exploratory Data Analysis

You should use the data already present in `/home/workspace/data/waymo` directory to explore the dataset! This is the most important task of any machine learning project. To do so, open the `Exploratory Data Analysis` notebook. In this notebook, your first task will be to implement a `display_instances` function to display images and annotations using `matplotlib`. This should be very similar to the function you created during the course. Once you are done, feel free to spend more time exploring the data and report your findings. Report anything relevant about the dataset in the writeup.

Keep in mind that you should refer to this analysis to create the different spits (training, testing and validation).


### Create the training - validation splits
In the class, we talked about cross-validation and the importance of creating meaningful training and validation splits. For this project, you will have to create your own training and validation sets using the files located in `/home/workspace/data/waymo`. The `split` function in the `create_splits.py` file does the following:
* create three subfolders: `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`
* split the tf records files between these three folders by symbolically linking the files from `/home/workspace/data/waymo/` to `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`

Use the following command to run the script once your function is implemented:
```
python create_splits.py --data-dir /home/workspace/data
```

### Edit the config file

Now you are ready for training. As we explain during the course, the Tf Object Detection API relies on **config files**. The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`.

### Training

You will now launch your very first experiment with the Tensorflow object detection API. Move the `pipeline_new.config` to the `/home/workspace/experiments/reference` folder. Now launch the training process:
* a training process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
Once the training is finished, launch the evaluation process:
* an evaluation process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```

**Note**: Both processes will display some Tensorflow warnings, which can be ignored. You may have to kill the evaluation script manually using
`CTRL+C`.

To monitor the training, you can launch a tensorboard instance by running `python -m tensorboard.main --logdir experiments/reference/`. You will report your findings in the writeup.

### Improve the performances

Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup.

Keep in mind that the following are also available:
* experiment with the optimizer: type of optimizer, learning rate, scheduler etc
* experiment with the architecture. The Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) offers many architectures. Keep in mind that the `pipeline.config` file is unique for each architecture and you will have to edit it.

**Important:** If you are working on the workspace, your storage is limited. You may to delete the checkpoints files after each experiment. You should however keep the `tf.events` files located in the `train` and `eval` folder of your experiments. You can also keep the `saved_model` folder to create your videos.


### Creating an animation
#### Export the trained model
Modify the arguments of the following function to adjust it to your models:

```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/
```

This should create a new folder `experiments/reference/exported/saved_model`. You can read more about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model).

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path /data/waymo/testing/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif
```

## Submission Template

### Project overview
In this project the goal is to perform 2D object detection by using a collection of images from the Waymo Open Dataset which contains labels for the classes 'Vehicle', 'Pedestrian', and 'Cyclist'. Moreover, we leverage the tensorflow object detection API for training a custom object detector that is based on a neural network pretrained on the coco dataset. In other words we are finetuning a given neural network, adapting its configuration and parameters to our needs in what could be considered as 'Transfer Learning'. Object detection is a crucial task in the field of Autonomous Driving since perception of the obstacles in the environment lets the car compute an optimal path planning in the next step to avoid such obstacles and thus drive collision-free.

### Set up
I decided to develop this project locally with Docker, so please refer to the section above for [local setup](#local-setup) to install the container.
One diference to note is that inside the container instead of having `/home/workspace`, you should work on a directory named `/app/project/`, so all commands from the [Structure](#structure) and [Instructions](#instructions) sections should be adjusted accordingly if necessary.

I did not download the dataset from the google cloud as suggested in the instructions, but I downloaded the dataset directly from the Udacity workspace where the splits 'train', 'val' and 'test' were already available.

### Troubleshooting
I experienced a major bug with the provided Dockerfile to build the container. For instance the keras version seems to be incompatible with the default installed tensorflow, so the solution I found was to reinstall keras again with the correct version, `pip install keras==2.5.0rc0`

It may be required to install the gpu package of tensorflow as well, `pip install tensorflow-gpu==2.5.0`

Then I saw some error coming from numpy *"InvalidArgumentError: TypeError: 'numpy.float64' object cannot be interpreted as an integer"*, which I was able to solve as indicated in the last post of this [thread](https://github.com/tensorflow/models/issues/2961)

Also, in case you see an error similar to *"tensorflow/stream_executor/cuda/cuda_driver.cc:328] failed call to cuInit: CUDA_ERROR_UNKNOWN: unknown error"*, please consider to turn off Secure Boot in the Bios menu of your computer so that tensorflow can have acces to your GPU device as suggested [here](https://stackoverflow.com/questions/67045622/tensorflow-stream-executor-cuda-cuda-driver-cc328-failed-call-to-cuinit-cuda)






 
### Dataset
#### Dataset analysis
Please find below some samples of the images available in the Waymo Open Dataset. We can observe that the data was recorded both in day and night time conditions, the former case being easier to distinguish objects than the latter with much darker images and shadows. Also we see some data was recorded under harsh weather like rain and wind which unfortunately produce artifacts and diminish the quality of the images. Some of these images have ground-truth bounding boxes for objects at a very far distance, at the resolution of 640x640 even impossible for the human eye to see clearly. Some images even do not have any labels as no traffic participant was there in that moment.
| ![](figures/image1.png)  |  ![](figures/image2.png) |
:-------------------------:|:-------------------------:
| ![](figures/image3.png)  |  ![](figures/image4.png) |
| ![](figures/image5.png)  |  ![](figures/image6.png) |
| ![](figures/image7.png)  |  ![](figures/image8.png) |
| ![](figures/image9.png)  |  ![](figures/image10.png) |
| ![](figures/image11.png)  |  ![](figures/image12.png) |
| ![](figures/image13.png)  |  ![](figures/image14.png) |
| ![](figures/image15.png)  |  ![](figures/image16.png) |

The following plots all display information from a subset of 20k images.
With regards to the distribution of labels, we see that 'Vehicle' is by far the most represented class in the dataset, 'Pedestrian' class is less than a third of 'Vehicle', and 'Cyclist' is the rarest type of label by several orders of magnitude less represented.

![](figures/plot1.png) 

In the next plot we can determine that most images have either between 5 and 20, or more than 20 bounding boxes per image, only 1/4 (of 20k images) have less than 5 boxes annotated.

![](figures/plot2.png) 

As illustrated below, most bounding boxes have a size between 100 and 1000 pixels, i.e. medium size, compared and relative to this amount, almost a half have a size less than 100 (small size) and only very few boxes have an area of more than 1000 pixels (big size).

![](figures/plot3.png) 

The following plot which displays how often multiple instances of a certain class appear on all images, basically confirms again that the likelihood of finding vehicles on an image is a lot bigger than finding pedestrians, and in turn cyclists. Therefore, one conclusion we can get from the figure is that the distribution of ground-truth labels is very unbalanced and this will lead to a poor detection of cyclists.
![](figures/plot4.png)

#### Cross validation
As already mentioned, I used the dataset that was already present in the Udacity workspace that was splitted into 87% training, 10% validation and 3% testing from 99 .tfrecords. The fact that the files have been randomly suffled and the proportion of the splits give us confidence to have diverse and equally representative enough data for training and validation. The validation subset also helps us to spot and eventually avoid an overfitting stage when training the neural network.

### Training
#### Reference experiment
The configuration file of my reference experiment is located in the folder `experiments/ref`. I used the default horizontal flip and random crop image data augmentations, and as a learning rate base I set 0.03 with a warmup lr of 0.01
The network was trained for 3000 steps. The orange curve Loss/total_loss and the blue dots in the different charts of Detection_Boxes_Precision/Recall are important metrics to find out how well the network was trained and its ability to yield good predictions. It seems the validation loss (blue dot) with value 6.3 at step 2k is slightly bigger than the training loss 6.522 also at 2k. This is an early indication of overfitting, and encourages us to try new ideas for better results than the baseline.
![](figures/ref_loss.png)
![](figures/ref_precision.png)
![](figures/ref_recall.png)







#### Improve on the reference
The configuration file of 'experiment1' is located in the folder `experiments/exp1`. In order to increase the variability of the training data and help object detector to genealize to new unseen data, I decided to apply additional data augmentation strategies such as random rgb to gray conversion, and modification of properites like saturation, brightness and contrast. Example of images altered by the techniques of data augmentation can be observed below.

 ![](figures/dataug1.png)
![](figures/dataug2.png)
![](figures/dataug3.png)
![](figures/dataug4.png)
![](figures/dataug5.png)
![](figures/dataug6.png)
![](figures/dataug6.png)
![](figures/dataug7.png)
![](figures/dataug8.png)


This time I changed also the learning rate base and warmup lr to be 2 orders of magnitude lower than before for a more optimal convergence of the loss function, 3e-4 and 1e-4 respectively (vs 3e-2 and 1e-2 as in the reference trial). The network was also trained for 3000 steps.

In the plot of the loss it is very clear that the red curve (exp1) is always lower than the orange curve (ref). The light blue dot loss (exp1) at step 2k is also lower than the dark blue dot (ref). This means that in this new experiment the network was doing much better against the training data, and was generalizing better as well in the validation set as compared to the case of the reference baseline.

![](figures/exp1vsref_loss.png)

In the charts regarding precision and recall we can see that the light blue dot (exp1) is always above the dark blue dot (ref), confirming again the success of the new experiment being able to improve all performance metrics of the baseline.
![](figures/exp1vsref_precision.png)

![](figures/exp1vsref_recall.png)


### Inference demo samples
![](gifs/animation.gif)
![](gifs/animation2.gif)
![](gifs/animation3.gif)








