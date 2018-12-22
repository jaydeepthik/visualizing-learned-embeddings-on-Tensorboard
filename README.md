# visualizing-learned-embeddings-on-Tensorboard
This repository contains code for learning a 128 dimensional embeddings of the IMDB dataset and also visualizing it on Tensorboard  
# using callbacks in keras
Use callbacks in keras to store data about the model during training which then can be visualized on Tensorboard.

1. It requires a directory or a folder to be specified to store various checkpoints and metadata.
  "callbacks =[keras.callbacks.TensorBoard(log_dir='<YOUR_DIR_NAME>', histogram_freq=1, embeddings_freq=1)]"
   this will log the data and the embeddings for each epoch
   
2. From the console fire up tensorboard $ tensorboard --logdir=<YOUR_DIR_NAME>
    in your broser go to http://localhost:6006 to access the logs graphically
