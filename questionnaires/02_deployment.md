1. Provide an example of where the bear classification model might work poorly in production, due to structural or style differences in the training data.
    - Working with video data instead of images
    - Handling nighttime images, which may not appear in this dataset
    - Dealing with low-resolution camera images
    - Ensuring results are returned fast enough to be useful in practice
    - Recognizing bears in positions that are rarely seen in photos that people post online (for example from behind, partially covered by bushes, or when a long way away from the camera)
1. Where do text models currently have a major deficiency?
    - A major deficiency of text models is that theyre not good at generating correct responses. Theres no reliable way to combine a given knowledge base with a deep learning model. This is dangerous because nlp models can create content that appears to a layman to be compelling but is factually incorrect.
1. What are possible negative societal implications of text generation models?
    - One possible negative societal implication is that text generation models will be used on social media at a massive scale to sread disinformation, create unrest, and encourage conflict.
1. In situations where a model might make mistakes, and those mistakes could be harmful, what is a good alternative to automating a process?
    - Its recommended to use a deep learning model as a part of a process rather than using it to entirely automate a process. For example a automated system could be used to identify potential stroke victims directly from CT scans, and send a high-priority alert to have those scans looked at quickly. There is only a three hour window to treat strokes so this fast feedback look could save lives. At the same time all scans can continue to be sent to raadiologists the usualy way so there would be no reduction in human input.
1. What kind of tabular data is deep learning particularly good at?
    - Deep learning is particularly good at handling high-cardinality categorical variables (i.e., something that contains a large number of discrete choices, such as zip code or product ID). Reccomendation systems are a good use case for this reason.
1. What's a key downside of directly using a deep learning model for recommendation systems?
    - Deep learning models, as do most other machine learning reccomendation system models only tell you what products a user might like, rather than what recommendations would be helpful for a user. For example if a user is already familiar with the products or if theyre simply different packagings of a product they already own then the reccomendation isnt very helpful.
1. What are the steps of the Drivetrain Approach?
    1. Define the objective you want to achieve
    2. Think about what actions you can take to meet that objective
    3. Think about what data you have or can acquire that vcan help meet your objective
    4. Build a model that you can use to determine the best actions to take to get the best results in terms of your objective
1. How do the steps of the Drivetrain Approach map to a recommendation system?
    1. We want to reccomend products that a user will be interested in buying
    2. We can collect a list of information that will help us predict what theyre interested in purchasing, and if we dont already collect that information we can start collecting it going forward
    3. We have a record of their purchases from us, as well as the pages they have visited, as well as a list of their ratings of products
1. Create an image recognition model using data you curate, and deploy it on the web.
    - []
1. What is `DataLoaders`?
    - DataLoaders: A fastai class that stores multiple DataLoader objects you pass to it, normally a train and a valid, although it's possible to have as many as you like. The first two are made available as properties.
1. What four things do we need to tell fastai to create `DataLoaders`?
    1. blocks, where we specify what types we want for independent and dependent variables
    2. get_items, where we tell the DataLoader object/class which path it will be loading our data from
    3. splitter, where we tell the DataLoader what validation percentage we want to use and what random seed to use
    4. get_y, where we tell the DataLoader what method to call to create the labels in our dataset
1. What does the `splitter` parameter to `DataBlock` do?
    - splitter, where we tell the DataLoader what validation percentage we want to use and what random seed to use
1. How do we ensure a random split always gives the same validation set?
    - By specifying the random seed.
1. What letters are often used to signify the independent and dependent variables?
    - independent is referred to as x, while dependent is referred to as y
1. What's the difference between the crop, pad, and squish resize approaches? When might you choose one over the others?
    - crop crops or cuts the images to all be the same size. Pad pads the images with black to on either the x or y axis whichever is shorter to make it even. Resize stretches the images either vertically or horizontally to all the same size. We typically would use RandomResizedCrop which randomly selects part of an image and crops to just that part.
1. What is data augmentation? Why is it needed?
    - Data augmentation refers to creating random variations of our input data, so that they appear different but do not actually change the meaning of our data. Examples for images include rotation, flipping, perspective warping, brightness changes, and contrast changes.
1. What is the difference between `item_tfms` and `batch_tfms`?
    - batch_tfms tells fastai that we want to use the provided transformation on a batch. Whereas item_tfms is a transformation that is applied to each item individually.
1. What is a confusion matrix?
    - A confusion matrix is a graph that shows how many times our model predicted incorrectly and what the incorrect prediction was compared to the actual classification.
1. What does `export` save?
    - A pkl file that holds our architecture/model and can be used to generate predictions.
1. What is it called when we use a model for getting predictions, instead of training?
    - Inference.
1. What are IPython widgets?
    - GUI Widgets that can be used inside a jupyter notebook. They bring together python and javascript functionality.
1. When might you want to use CPU for deployment? When might GPU be better?
    - GPUs are only useful when youre doing lots of identical work in paralllel. For this reason if youre doing image classification and only classifying one image at a time then a CPU is a better choice. If you were instead waiting for multiple users to submit their images then batching them up and classifying them then that would be more suitable for a GPU.
1. What are the downsides of deploying your app to a server, instead of to a client (or edge) device such as a phone or PC?
    - The app would require internet connection to be able to classify images. If the model/app was deployed on a phone or PC native app then predictions/inference could be run without any internet connection but it is more complex to set up.
1. What are three examples of problems that could occur when rolling out a bear warning system in practice?
    - Same answer as question 1
1. What is "out-of-domain data"?
    - Data that might be missing from our training set but pops up when the model is being used in production.
1. What is "domain shift"?
    - When the type of data that our model sees changes over time. For instance, an insurance company may use a deep learning model as part of its pricing and risk algorithm, but over time the types of customers that the company attracts, and the types of risks they represent, may change so much that the original training data is no longer relevant.
1. What are the three steps in the deployment process?
    - Manual Process
        - Run model in parallel with existing process. 
        - Humans check all predictions
    - Limited Scope Deployment
        - Careful human supervisoin
        - Time or geography limited
    - Gradual Expansion
        - Good reporting systems needed
        - Consider what could go wrong