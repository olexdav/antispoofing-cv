# DataRoot Univeristy
## Tensorflow Project Template for Deep Learning projects
A simple and well designed structure is essential for any Deep Learning project. 
Every Deep learning project has its own unique parts, but after a lot of practice with such kind of projects using Tensorflow - all neccessary parts can be gathered in a general structure. 

We inspired in creation of such a structure from [stanford](https://cs230-stanford.github.io/project-code-examples.html) template and a popular more practical [template](https://www.reddit.com/r/MachineLearning/comments/7ven9f/p_best_practice_for_tensorflow_project_template/). We took best things from both, merged it with our experience and created this structure to be used even outside of DataRoot University. 

This template combines  **simplcity**, **best practice for folder structure** and **good OOP design**.

**So, here's a simple tensorflow template that help you get into your main project faster and just focus on your core (Model, Training, ...etc)**
# Table Of Contents
-  [Intro](#intro)
-  [Try it first](#try-it-first)
-  [In a Nutshell](#in-a-nutshell)
-  [In Details](#in-details)
    -  [Project architecture](#project-architecture)
    -  [Folder structure](#folder-structure)
    -  [ Main Components](#main-components)
        -  [Models](#models)
        -  [Trainer](#trainer)
        -  [Data Loader](#data-loader)
        -  [Logger](#logger)
        -  [Configuration](#configuration)
        -  [Main](#main)
 -  [Speed it up](#speed-it-up)
 -  [Contributing](#contributing)


# Intro
Current version of template consists of two examples for image classification tasks which we will run and evaluate in the [next](#try-it-first) chapter.

You will have 4 activities in total related to this structure:
- First two are in module 3 - training and evaluation
- Others are in module 4 - provisioning and deployment

Final result of such activities will lead you to a few deployed Deep Learning projects within our ecosystem.
You will receive DRP reward upon completing such projects.
Current template can be tweaked in a few minutes and adapted for any image classification task.
In order to do some object detection - you will need to spend a little bit more time.

#### Important
Each two weeks we will add new examples / features and will extend the structure based on votes of DRU community.

# Try it first
In order to get more familiar with the template and how it works let's run a few examples.

First of all you need to install all the libraries. You are free to create virtualenv here.
We are assuming you're using anaconda distribution and 3.5+ version of python.

Clone repository
```bash
git clone https://github.com/dataroot/DRU-DL-Project-Structure.git
cd DRU-DL-Project-Structure
```

##### Important note
All further code snippents with or without `cd`'s are made under assumption you're in project root folder `DRU-DL-Project-Structure`

Install requirements
```bash
pip install -r requirements.txt
```

Now we are ready for the first example. Let's go with cifar100.

[Download](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) and unpack cifar100 dataset. It should be under `data/cifar100/cifar-100-python` relative path.

Run dataset preprocessor
```bash
cd data/cifar100
python prepare_cifar100.py
```

Now you are ready to train your model
```bash
cd mains
python example_cifar_numpy_dataloader.py -c ../configs/cifar100.json
```

You should see something like this in your terminal
<img align="center" hight="600" width="800" src="https://github.com/dataroot/DRU-DL-Project-Structure/blob/master/figures/training_process.png?raw=true">

Open one more terminal window and run tensorboard. Watch metrics in near real-time
```bash
tensorboard --logdir experiments/cifar_example_exp_1/summary
```

You should get something like this [here](http://localhost:6006)
<div align="center">
<img align="center" hight="600" width="800" src="https://github.com/dataroot/DRU-DL-Project-Structure/blob/master/figures/tensorboard_1.png?raw=true">
</div>

Ok, as you can see the accuracy for such a complex task in not that high for just a few epochs. Let's try to play with simpler task.

[Download](https://drive.google.com/file/d/1ufiR6hUKhXoAyiBNsySPkUwlvE_wfEHC/view?usp=sharing) and unpack signs dataset. It should be under `data/signs/SIGNS` relative path.

Run dataset preprocessor
```bash
cd data/signs
python prepare_signs.py -c ../configs/signs.json
```

Let's train a model
```bash
cd mains
python example_signs_tf_dataloader.py -c ../configs/signs.json
```

Open one more terminal window and run tensorboard. Watch metrics in near real-time
```bash
tensorboard --logdir experiments/signs_1/summary
```

Tensorboard has more advanced capabilities. For example you can log miss-classified images there
<div align="center">
<img align="center" hight="600" width="800" src="https://github.com/dataroot/DRU-DL-Project-Structure/blob/master/figures/tensorboard_2.png?raw=true">
</div>

One more interesting feature - DAG visualization
<div align="center">
<img align="center" hight="600" width="800" src="https://github.com/dataroot/DRU-DL-Project-Structure/blob/master/figures/tensorboard_3.png?raw=true">
</div>

# In a Nutshell
In a nutshell here's how to use this template, so **for example** assume you want to implement VGG model so you should do the following:
-  In models folder create a class named VGG that inherit the "base_model" class

```python
    class VGGModel(BaseModel):
        def __init__(self, config):
            super(VGGModel, self).__init__(config)
            #call the build_model and init_saver functions.
            self.build_model() 
            self.init_saver() 
  ```
- Override these two functions "build_model" where you implement the vgg model, and "init_saver" where you define a tensorflow saver, then call them in the initalizer
    
```python
     def build_model(self):
        # here you build the tensorflow graph of any model you want and also define the loss.
        pass
            
     def init_saver(self):
        # here you initalize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
  ```
   
- In trainers folder create a VGG trainer that inherit from "base_train" class
```python
    class VGGTrainer(BaseTrain):
        def __init__(self, sess, model, data, config, logger):
            super(VGGTrainer, self).__init__(sess, model, data, config, logger)
```
- Override these two functions "train_step", "train_epoch" where you write the logic of the training process
```python
    def train_epoch(self):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step
       -add any summaries you want using the summary
        """
        pass

    def train_step(self):
        """
       implement the logic of the train step
       - run the tensorflow session
       - return any metrics you need to summarize
       """
        pass
```
- In main file, you create the session and instances of the following objects "Model", "Logger", "Data_Generator", "Trainer", and config
```python
    sess = tf.Session()
    # create instance of the model you want
    model = VGGModel(config)
    # create your data generator
    data = DataGenerator(config)
    # create tensorboard logger
    logger = Logger(sess, config)
```
- Pass the all these objects to the trainer object, and start your training by calling "trainer.train()" 
```python
    trainer = VGGTrainer(sess, model, data, config, logger)

    # here you train your model
    trainer.train()
```


# In Details

Project architecture 
--------------

<div align="center">
<img align="center" hight="600" width="600" src="https://github.com/dataroot/DRU-DL-Project-Structure/blob/master/figures/diagram.png?raw=true">
</div>


Folder structure
--------------

```
├──  base
│   ├── base_model.py   - this file contains the abstract class of the model.
│   └── base_train.py   - this file contains the abstract class of the trainer.
│
│
├── models              - this folder contains any model of your project.
│   └── example_model.py
│
│
├── trainers            - this folder contains trainers of your project.
│   └── example_trainer.py
│   
│   
├──  mains                - here's the main(s) of your project (you may need more than one main).
│    └── example_main.py  - here's an example of main that is responsible for the whole pipeline.
│ 
│ 
├── configs               - experiment configurationdirectory
│    └── config.json
│ 
│ 
├── data                  - dataset with preprocessors
│    └── signs
│    └── any_other_dataset_you_need
│ 
│ 
├── experiments           - where the tf.checkpoinst and tensorboard summary remains
│    └── experiment_name
│ 
│ 
├──  data_generators      - here's the data_generator that is responsible for all data handling.
│    └── data_generator.py  
│ 
│ 
└── utils
     ├── logger.py
     └── any_other_utils_you_need

```


## Main Components

Models
--------------
- #### **Base model**
    
    Base model is an abstract class that must be Inherited by any model you create, the idea behind this is that there's much shared stuff between all models.
    The base model contains:
    - ***Save*** -This function to save a checkpoint to the desk. 
    - ***Load*** -This function to load a checkpoint from the desk.
    - ***Cur_epoch, Global_step counters*** -These variables to keep track of the current epoch and global step.
    - ***Init_Saver*** An abstract function to initialize the saver used for saving and loading the checkpoint, ***Note***: override this function in the model you want to implement.
    - ***Build_model*** Here's an abstract function to define the model, ***Note***: override this function in the model you want to implement.
- #### **Your model**
    Here's where you implement your model.
    So you should :
    - Create your model class and inherit the base_model class
    - override "build_model" where you write the tensorflow model you want
    - override "init_save" where you create a tensorflow saver to use it to save and load checkpoint
    - call the "build_model" and "init_saver" in the initializer.

Trainers
--------------
- #### **Base trainer**
    Base trainer is an abstract class that just wrap the training process.
    
- #### **Your trainer**
     Here's what you should implement in your trainer.
    1. Create your trainer class and inherit the base_trainer class.
    2. override these two functions "train_step", "train_epoch" where you implement the training process of each step and each epoch.
### Data Loader
This class is responsible for all data handling and processing and provide an easy interface that can be used by the trainer.
### Logger
This class is responsible for the tensorboard summary, in your trainer create a dictionary of all tensorflow variables you want to summarize then pass this dictionary to logger.summarize().


This class also supports reporting to **Comet.ml** which allows you to see all your hyper-params, metrics, graphs, dependencies and more including real-time metric.
Add your API key in the configuration file:

For example: "comet_api_key": "your key here"


### Comet.ml Integration
This template also supports reporting to Comet.ml which allows you to see all your hyper-params, metrics, graphs, dependencies and more including real-time metric. 

Add your API key in the configuration file:

For example:  `"comet_api_key": "your key here"` 

Here's how it looks after you start training:
<div align="center">
<img align="center" width="800" src="https://comet-ml.nyc3.digitaloceanspaces.com/CometDemo.gif">
</div>

You can also link your Github repository to your comet.ml project for full version control. 

 
### Configuration
Use Json as configuration method and then parse it, so write all configs you want then parse it using `utils.config.process_config` and pass this configuration object to all other objects.
### Main
Here's where you combine all previous part.
1. Parse the config file.
2. Create a tensorflow session.
2. Create an instance of "Model", "Data_Generator" and "Logger" and parse the config to all of them.
3. Create an instance of "Trainer" and pass all previous objects to it.
4. Now you can train your model by calling "Trainer.train()"


# Speed it up
In order to achieve better performance for free consider using [google colab](https://colab.research.google.com/notebooks/welcome.ipynb) or [floydhub](https://www.floydhub.com/). Their capabilities described [here](https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d) and [here](https://medium.com/@rupak.thakur/aws-vs-paperspace-vs-floydhub-choosing-your-cloud-gpu-partner-350150606b39).


# Contributing
Any pull request which is accepted by us is rewarded by DRP using our oracle.
In order to receive DRP for such a contribution you will need to link your github account in our system.

