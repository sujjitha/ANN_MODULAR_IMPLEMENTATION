import os
from src.utils.data_mgmt import get_data
from src.utils.model import create_model, save_model
import logging
from src.utils.common import read_config
import argparse
def training(config_path):
    logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename= os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str, filemode="a")



    config=read_config(config_path)
    validation_datasize = config["params"]["validation_datasize"]
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)
    LOSS_FUNCTION= config["params"]["loss_function"]
    OPTIMIZER=config["params"]["optimizer"]
    METRICS=[config["params"]["metrics"]]
    NUM_CLASSES=config["params"]["num_classes"]
    logging.info(f"loss function, optimizer, metrics and num classes from config.yaml file{LOSS_FUNCTION,OPTIMIZER,METRICS,NUM_CLASSES}")
    model=create_model(LOSS_FUNCTION,OPTIMIZER,METRICS,NUM_CLASSES)
    print("model_clf layer1 name  ",model.layers[1].name)
    weights, biases= model.layers[1].get_weights()
    print("weights shape: ",weights.shape)
    print("biases shape: ", biases.shape)
    logging.info(f"weights and biases: {weights.shape, biases.shape}")
    EPOCHS=config["params"]["epochs"]
    VALIDATION_SET=(X_valid, y_valid)
    logging.info(f"X_valid shape: {X_valid.shape}")
    logging.info(f"y_valid shape: {y_valid.shape}")
    logging.info(f"X_train shape: {X_train.shape}")
    logging.info(f"y_train shape: {y_train.shape}")
    history=model.fit(X_train, y_train,epochs=EPOCHS, validation_data=VALIDATION_SET)

    model_name=config["artifacts"]["model_name"]
    model_dir=config["artifacts"]["model_dir"]
    artifacts_dir=config["artifacts"]["artifacts_dir"]

    model_dir_path=os.path.join(artifacts_dir,model_dir)
    os.makedirs(model_dir_path, exist_ok=True)

    save_model(model, model_name, model_dir_path)
    


if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config.yaml")
    parsed_args=args.parse_args()
    training(config_path=parsed_args.config)

