1. Do you need these for deep learning?

   - Lots of math T / F
    - F
   - Lots of data T / F
    - F
   - Lots of expensive computers T / F
    - F
   - A PhD T / F
    - F
   
1. Name five areas where deep learning is now the best in the world.
    - NLP
    - Computer Vision
    - Medicine, finding anomalies in mediccal images such as ultrasounds, x-rays, mri, etc
    - Image generation, generating images given a text prompt
    - Recommendation systems, i.e. providing a shopper with an item that they will likely be interested in.
1. What was the name of the first device that was based on the principle of the artificial neuron?
    - The perceptron
1. Based on the book of the same name, what are the requirements for parallel distributed processing (PDP)?
    1. A set of *processing units*
    1. A *state of activation*
    1. An *output function* for each unit 
    1. A *pattern of connectivity* among units 
    1. A *propagation rule* for propagating patterns of activities through the network of connectivities 
    1. An *activation rule* for combining the inputs impinging on a unit with the current state of that unit to produce an output for the unit
    1. A *learning rule* whereby patterns of connectivity are modified by experience 
    1. An *environment* within which the system must operate
1. What were the two theoretical misunderstandings that held back the field of neural networks?
    - In theory, adding just one extras layer of neurons (total of 2 layers) should be enough to allow neural networks to compute any mathematical function, but in practice neural networks with 2 layers were too big and slow to be useful.
1. What is a GPU?
    - Graphics Processing Unit. The GPU is also known as a 'Graphics Card', it is designed for displaying 3d environment on a computer typically meant for playing games. Games/3d graphics & neural networks rely on a similar set of basic tasks. GPU's are able to run neural networks hundres of times faster than regular CPU's.
1. Open a notebook and execute a cell containing: `1+1`. What happens?
    - The notebook outputs '2'
1. Follow through each cell of the stripped version of the notebook for this chapter. Before executing each cell, guess what will happen.
    - [X]
1. Complete the Jupyter Notebook online appendix.
    - [X]
1. Why is it hard to use a traditional computer program to recognize images in a photo?
    - Because traditional computer programs have to be explicitly told what to do step by step. Whereas neural networks and more similar to brains and able to learn their content or area of focus dynamically.
1. What did Samuel mean by "weight assignment"?
    - Weights are variables while weight assignments are values for the variables. Once a model is finished training and therefore is optimized to best accomplish a given task, the weight assignments are final and do not change.
1. What term do we normally use in deep learning for what Samuel called "weights"?
    - Weights would typically be referred to as model parameters.
1. Draw a picture that summarizes Samuel's view of a machine learning model.
    ```
    Inputs ---------> 
                        Model ---------> Results                
    Weights -------->
    ```
1. Why is it hard to understand why a deep learning model makes a particular prediction?
    - Its hard to understand why a deep learning model makes a particular prediction because it is giving a prediction, and a confidence level, but no explanation of why a prediction was made is provided. If you want to understand why a particular prediction is made for a given model you have dig into the layers of the model to see how the model develops further with each layer. In 2013 a PhD student, Matt Zeiler, and his supervisor, Rob Fergus, published the paper "Visualizing and Understanding Convolutional Networks", which outputs images showing how a neural networks weights look after each layer of training.
1. What is the name of the theorem that shows that a neural network can solve any mathematical problem to any level of accuracy?
    - universal approximation theorem
1. What do you need in order to train a model?
    - To train a model we need input data and labels for the input data
1. How could a feedback loop impact the rollout of a predictive policing model?
    - If the model is predicting crime in an area with a typically higher crime rate, its using past police officers arrests (which could be incorrect or biased) to tell current police officers where they should be monitoring for future crime. This sort of pattern could create a snowball effect where past bias impacts future arrests and the future arrests impart more bias into the model. This is known as a positive feedback loop. The more the model is used, the more biased the data becomes.
1. Do we always have to use 224×224-pixel images with the cat recognition model?
    - No, this is just the number that has been used historically. More pixels will result in a more accurate but slower to train model.
1. What is the difference between classification and regression?
    - Classification attempts to predict a class or category. Regression models attempt to predict one or more numeric qualities like temperature or a location.
1. What is a validation set? What is a test set? Why do we need them?
    - A validation set is a set used to measure the accuracy of the model. A validation set is used to improve the model, if the results of the validation set are not up to our expectations then we can modify the model to get a better result. The test set is another subset of data which we hide from even ourselves. The test set cant be used to improve the model, it can only be used to evaluate the model at the very end of our efforts. Training data is fully exposed, the validation data is less exposed, and test data is totally hidden.
1. What will fastai do if you don't provide a validation set?
    - fastai will use 20% of the data as a validation set by default
1. Can we always use a random sample for a validation set? Why or why not?
    - Its best to use a seed which will make it so the random validation set selected is the same validation set each time the model is run. With this we know if we change our model, retrain it, any differences are due to changes in the model not changes in the validation set.
1. What is overfitting? Provide an example.
    - Overfitting is when a model is trained for too long and starts to memorize the validation set.
1. What is a metric? How does it differ from "loss"?
    - A metric is a measure of model quality. A metric is intended to be understood by a human. Common metrics include error_rate and accuracy (1 - error_rate). Loss is a measure of poerformance that the training system can use to update weights automatically. Metrics are used by humans while loss is used by computers to aid in stochastic gradient descent.
1. How can pretrained models help?
    - A pretrained model is a model that has weights the have already been trained on another dataset.
1. What is the "head" of a model?
    - The head of a model is the last layer. Pre trained models remove the head of the model and replace it with one or more new layers with randomized weights in preparation for training it on your dataset. After we train a pretrained model the head will be a newly added layer specific to our new dataset.
1. What kinds of features do the early layers of a CNN find? How about the later layers?
    - Early layers of CNN's find edge, gradient, and color detection. The later layers will be of more detailed patterns for example faces, wheels, eyes.
1. Are image models only useful for photos?
    - No, image models can also be used for non image specific tasks. For example sound can be converted to a spectrogram (a chart  showing the frequency at differrent time intervals) which can be used to classify different sounds.
1. What is an "architecture"?
    - An architecture is a functional form of a model
1. What is segmentation?
    - Segmentation is taking an image and segmenting it by its individual pixels and classifying each pixel within the image. This is useful for example in finding harmful tumors in images, the model can highlight which pixels are indicative of a tumor.
1. What is `y_range` used for? When do we need it?
    - y_range is how we tell the fastai CollabDataLoaders class/object what range we want our prediction to be in. For example if were predicting a movie rating from 0-5 stars the y_range would be 5. y_range needs to be specified when predicting a continious number.
1. What are "hyperparameters"?
    - Hyperparameters are higher-level choices that govern the meaning of weight parameters for example, network architecture, learning rates, data augmentation strategies.
1. What's the best way to avoid failures when using AI in an organization?
    - Understand what test and validation sets are and why theyre important. For instance, if you're considering bringing in an external vendor or service, make sure that you hold out some test data that the vendor never gets to see. Then you check their model on your test data, using a metric that you choose based on what actually matters to you in practice, and you decide what level of performance is adequate.