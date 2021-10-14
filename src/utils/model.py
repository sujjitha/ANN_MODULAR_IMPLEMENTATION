import tensorflow as tf
import time
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_model(LOSS_FUNCTION,OPTIMIZER,METRICS,NUM_CLASSES):
   
   LAYERS=[
            tf.keras.layers.Flatten(input_shape=[28,28], name="input_layer"),
            tf.keras.layers.Dense(300, activation="relu", name="hiddenlayer1"),
            tf.keras.layers.Dense(100, activation="relu", name="hiddenlayer2"),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="output_layer")
            ]
    
   model_clf=tf.keras.models.Sequential(LAYERS)
   logging.info(f"Sequential model got created: {model_clf}")
   model_clf.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)
   logging.info("model created")
   return model_clf


def get_unique_filename(file_name):
        unique_file_name=time.strftime(f"%Y%m%d-%H%M%S-{file_name}")
        logging.info(f"unique file created: {unique_file_name}")
        return unique_file_name

def save_model(model, model_name,model_dir):
        unique_filename=get_unique_filename(model_name)
        path_to_model=os.path.join(model_dir,unique_filename)
        logging.info(f"model saved in : {path_to_model} file")
        model.save(path_to_model)


def save_plot(img, plottitle):
    plot_dir = "data_plots"
    os.makedirs(plot_dir, exist_ok=True)  # ONLY CREATE IF MODEL_DIR DOES NOT EXISTS
    plotPath = os.path.join(plot_dir, plottitle)  # model/filename
    plt.imshow(img, cmap="binary")
    plt.axis("off")
    plt.plot()
    plt.figure(figsize=(15,15))
    #plt.savefig(plotPath)
    logging.info(f"saving the plot at {plotPath}")
    sns.heatmap(img, annot=True, cmap="binary")
    #sns.heatmap(img/255, annot=True, cmap="binary")
    print("plot got called")
    logging.info("image got saved")
    plt.savefig(plotPath)
    plt.show()

def save_model_history(df,filepath, plttitle):
        os.makedirs(filepath, exist_ok=True)  # ONLY CREATE IF MODEL_DIR DOES NOT EXISTS
        historyplot_path = os.path.join(filepath, plttitle)  # model/filename
        logging.info(f"model history plot path:  {historyplot_path}")
        #logging.info("model history plot path: ",historyplot_path)
        df.plot(figsize=(10,7))
        plt.grid(True)
        plt.savefig(historyplot_path)
        logging.info("history plot got saved")
        plt.show()

def model_predict(model,x_new, y_test_new):
        y_prob=model.predict(x_new)
        
        y_pred=np.argmax(y_prob, axis=-1)
        logging.info(f"predicted y value is:  {y_pred}")
        for img_array, pred, actual in zip(x_new, y_pred, y_test_new):
                plt.imshow(img_array, cmap="binary")
                plt.title(f"predicted: {pred}, actual: {actual}")
                plt.show()
                print("--*20")
        logging.info("all the predicted values got printed")




