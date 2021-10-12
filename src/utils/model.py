import tensorflow as tf
import time

def create_model(LOSS_FUNCTION,OPTIMIZER,METRICS,NUM_CLASSES):
    LAYERS=[
            tf.keras.layers.Flatten(input_shape=[28,28], name="input_layer"),
            tf.keras.layers.Dense(300, activation="relu", name="hiddenlayer1"),
            tf.keras.layers.Dense(100, activation="relu", name="hiddenlayer2"),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="output_layer")
            ]
    
    model_clf=tf.keras.models.Sequential(LAYERS)
    model_clf.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)
    return model_clf


def get_unique_filename(file_name):
        unique_file_name=time.strftime(f"%Y%m%d-%H%M%S-{file_name}")
        return unique_file_name

def save_model(model, model_name,model_dir):
        unique_filename=get_unique_filename(model_name)
        path_to_model=os.path.join(model_dir,unique_filename)
        model.save(path_to_model)

    

