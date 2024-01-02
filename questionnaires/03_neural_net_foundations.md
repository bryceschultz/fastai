At this point we have something that is rather magical:

A function that can solve any problem to any level of accuracy (the neural network) given the correct set of parameters
A way to find the best set of parameters for any function (stochastic gradient descent)

1. How is a grayscale image represented on a computer? How about a color image?
    - grayscale images can be represented as each pixel in the image having a number from 0 - 255. 0 being white and 255 being black.
1. How are the files and folders in the `MNIST_SAMPLE` dataset structured? Why?
    - The data is in seperate folders, one for validation, one for training. Inside each training/validation folder there are seperate folders for each digit 0-9.
    - The data being structured like this allows for a common set of images to train from and a common set to test against which ensures that inference models created from this dataset can be adequately compared. Also each digit having its own folder serves as a label for the images within the folder which is necessary for training.
1. Explain how the "pixel similarity" approach to classifying digits works.
    - Pixel similarity involves getting an average pixel value for every pixel of all images within a given folder (say training/3/*) that we want to classify. Then to classify an image we get the average pixel value of the image and see which folders average pixel value its closest to.
1. What is a list comprehension? Create one now that selects odd numbers from a list and doubles them.
    - A list comprehension is a shorter way & faster way to write a list rather than using a loop.
    - List comprehension that selects odd numbers and doubles them:
    - `new_list = [(o*2) for o in a_list if o%2 == 1]`
1. What is a "rank-3 tensor"?
    - A 3 dimensional tensor. for example the below dimensions:
    - [4, 4, 2]
    - Would by [x, y, z]. The tensor wouldnt expand just on the x and y axis but would have a third axis as well that would go back.
1. What is the difference between tensor rank and shape? How do you get the rank from the shape?
    - rank is the number of axes or dimensions in a tensor
        - e.g. the following tensors rank would be 3: [6131, 28, 28]
    - shape is the size of each axis of a tensor.
        - e.g. the following tensors shape would be [6131, 28, 28]: [6131, 28, 28]
1. What are RMSE and L1 norm?
    - L1 norm is the mean of the absolute value of differences (also known as absolute difference)
        - dist_3_abs = (a_3 - mean3).abs().mean()
    - Root Mean Squared Error (also known as L2 norm.) is taking the square of the differences then taking the square root.
        - dist_3_sqr = ((a_3 - mean3)**2).mean().sqrt()
1. How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?
    - By using numpy arrays or pytorch tensors. Numpy arrays and pytorch tensors have a wide variety of operators and methods that can run computations on these compact structures at the same speed as optimized C, because they are written in optimized C.
1. Create a 3×3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.
    - [X]
    ```
    orig_array = array([1,2,3,4,5,6,7,8,9])
    new_array = array([orig_array, orig_array])
    new_array[1,5:9]
    ```
1. What is broadcasting?
    - When pytorch tries to perform an operation between two tensors of different ranks, broadcasting must be used. Pytorch will automatically expand the tensor with the smaller rank to have the same size as the one with the larger rank.
1. Are metrics generally calculated using the training set, or the validation set? Why?
    - Metrics are generally calculated when using the validation set. This is because when running the model against the validation set we are actually testing how effective it is and reporting the metric back to a human. When were interacting with the training set the model will use loss which aids in sgd.
1. What is SGD?
    - SGD or stochastic gradient descent is the process by which we find appropriate weights for our model. SGD uses a loss function, by attempting to make the result of the loss function as low as possible we are able to find the appropriate weights.
1. Why does SGD use mini-batches?
    - Calculating the loss for the full dataset would take a very long time, and calculating the loss for one item of data would result in a very unstable gradient. Instead we use a middle ground which is mini-batches. A larger batch size means that you will get a more accurate and stable estimate of your dataset's gradients from the loss function, but it will take longer, and you will process fewer mini-batches per epoch.  
1. What are the seven steps in SGD for machine learning?
    1. Initialize the weights.
    2. For each image, use these weights to predict the correct image
    3. Based on these predictions, calculate how good the model is (its loss).
    4. Calculate the gradient, which measures for each weight, how changing that weight would change the loss.
    5. Step (that is, change) all the weights based on that calculation.
    6. Go back to the step 2 and repeat the process.
    7. Iterate until you decide to stop the training process (for instance, because the model is good enough or you do not want to wait any longer).
1. How do we initialize the weights in a model?
    - We initialize the weights to random values. Since we already have a routine to improve the weights it doesnt make a significant difference to attempt to initialize them any closer to their end value.
1. What is "loss"?
    - We need some function that will return a number that is small if the performance of the model is good (the standard approach is to treat a small loss as good, and a large loss as bad, although this is just a convention). Loss is the result of the function.
1. Why can't we always use a high learning rate?
    - If we use too high of a learning rate it can result in the loss getting worse.
1. What is a "gradient"?
    - The gradients are the slope of our loss function. gradient is defined as rise/run, that is, the change in the value of the function, divided by the change in the value of the parameter.
1. Do you need to know how to calculate gradients yourself?
    - no, we call the .backwards() method on a tensor, tensorflow will calculate the gradients for us. Then we can access the gradients using the grad attribute of the tensor.
1. Why can't we use accuracy as a loss function?
    - accuracy is intended to be a metric which is calculated from a validation set. A loss function is derived from a training set.
1. Draw the sigmoid function. What is special about its shape?
    ```
   1.0  .                                                                               . 
        .                                                                       ....... . 
        .                                                            ...........        . 
        .                                                        ...                    . 
        .                                                      ..                       . 
        :                                                   ..                          . 
        .                                                 .                             . 
        .                                               ..                              . 
        .                                             ..                                . 
        .                                            .                                  . 
        :                                          ..                                   . 
        .                                         .                                     . 
        .                                       .                                       . 
        .                                      .                                        . 
        .                                    ..                                         . 
        :                                   .                                           . 
        .                                 .                                             . 
        .                               ..                                              . 
        .                              .                                                . 
        .                           ..                                                  . 
        :                         ..                                                    . 
        .                     ...                                                       . 
        .          ...........                                                          . 
        .   .......                                                                     . 
   0.0  .                                                                               . 
    ```
    - sigmoid takes any input value, positive or negative and smooshes it into a value between 0 and 1.
    - It's also a smooth curve that only goes up, which makes it easier for SGD to find meaningful gradients.
1. What is the difference between a loss function and a metric?
    - A loss function is a function that returns a value based on a prediction and a target. Lower values returned by the loss function correspond with better predictions while higher numbers correspond with worse predictions. Loss functions are used on the training set.
    - Metrics are for human consumption and are generated from the validation set.
1. What is the function to calculate new weights using a learning rate?
    - backpropogation or backward()
1. What does the `DataLoader` class do?
    - Takes a python collection and batch size as input and batches the collection into seperate tensors:
    ```
    coll = range(15)
    dl = DataLoader(coll, batch_size=5, shuffle=True)
    list(dl)
    [tensor([ 3, 12,  8, 10,  2]),
    tensor([ 9,  4,  7, 14,  5]),
    tensor([ 1, 13,  0,  6, 11])]
    ```
1. Write pseudocode showing the basic steps taken in each epoch for SGD.
    ```
    for x,y in data:
        pred = model(x)
        loss = loss_func(pred, y)
        loss.backward()
        parameters -= parameters.grad * lr
    ```
1. Create a function that, if passed two arguments `[1,2,3,4]` and `'abcd'`, returns `[(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]`. What is special about that output data structure?
    ```
        def concat_arrays(nums, string):
        array = L()
        for count, ele in enumerate(string):
            array.append((ele, nums[count]))
        return array

        cctd_array = concat_arrays([1,2,3,4], 'abcd')
        cctd_array
    ```
    - The output of this data structure is special because they are mini-batches which are themselves tuples of tensors representing batches of independent and dependent variables.
1. What does `view` do in PyTorch?
    - Changes the shape of a tensor without changing its contents
1. What are the "bias" parameters in a neural network? Why do we need them?
    - The formula for a line is y = wx + b. Bias is the b in this formula. If there was no bias then a prediction (y or dependent variable) would always be equal to 0 when its independent variable is equal to 0, which would be a poor and inaccurate prediction.
1. What does the `@` operator do in Python?
    - @ allows us to do matrix multiplacation in python.
1. What does the `backward` method do?
    - calculates the gradients
1. Why do we have to zero the gradients?
    - loss.backward actually adds the gradients of loss to any gradients that are currently stored. Need to start with the gradients at 0 for each param.
1. What information do we have to pass to `Learner`?
    1. A DataLoaders object
    2. A model
    3. Whether its pretrained or not
    4. A loss function
    5. A metric
1. Show Python or pseudocode for the basic steps of a training loop.
    ```
    def train_model(model, epochs):
    for i in range(epochs):
        train_epoch(model)
        print(validate_epoch(model), end=' ')
    ```
1. What is "ReLU"? Draw a plot of it for values from `-2` to `+2`.
    - Rectified Linear Unit. You get a ReLU by taking the line/model (linear classifier) from a layer and replace every negative number with a zero. e.g.:
    ```
    res = xb@w1 + b1
    res = res.max(tensor(0.0))
    ```
1. What is an "activation function"?
    - An activation function is the process of getting a ReLU from a linear classifier.
1. What's the difference between `F.relu` and `nn.ReLU`?
    - Theyre the same thing just from different packages. F is from fastai nn is from pytorch.
1. The universal approximation theorem shows that any function can be approximated as closely as needed using just one nonlinearity. So why do we normally use more?
    - Because in practice using one nonlinearity is too slow to be practically useful.