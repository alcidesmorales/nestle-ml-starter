# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernel_info:
#     name: local-env
#   kernelspec:
#     display_name: Python 3.9.6 64-bit
#     language: python
#     name: python3
# ---

# %%
import argparse
from pathlib import Path
import pickle
import mlflow

import os 
import json


# %% jupyter={"outputs_hidden": false, "source_hidden": false} nteract={"transient": {"deleting": false}}
# Define Arguments for this step

class MyArgs:
    def __init__(self, /, **kwargs):
        self.__dict__.update(kwargs)

args = MyArgs(
                model_name = "taxi-model",
                model_path = "/tmp/train",
                evaluation_output = "/tmp/evaluate", 
                model_info_output_path = "/tmp/model_info_output_path"
                )


# %% jupyter={"outputs_hidden": false, "source_hidden": false} nteract={"transient": {"deleting": false}}
def main(args):
    '''Loads model, registers it if deply flag is True'''
    
    with open((Path(args.evaluation_output) / "deploy_flag"), 'rb') as infile:
        deploy_flag = int(infile.read())

    mlflow.log_metric("deploy flag", int(deploy_flag))
    
    if deploy_flag==1:

        print("Registering ", args.model_name)

        # load model
        model =  mlflow.sklearn.load_model(args.model_path) 

        # log model using mlflow
        mlflow.sklearn.log_model(model, args.model_name)

        # register logged model using mlflow
        run_id = mlflow.active_run().info.run_id
        model_uri = f'runs:/{run_id}/{args.model_name}'
        mlflow_model = mlflow.register_model(model_uri, args.model_name)
        model_version = mlflow_model.version

        # write model info
        print("Writing JSON")
        dict = {"id": f"{args.model_name}:{model_version}"}
        output_path = os.path.join(args.model_info_output_path, "model_info.json")
        with open(output_path, "w") as of:
            json.dump(dict, fp=of)

    else:
        print("Model will not be registered!")


# %% jupyter={"outputs_hidden": false, "source_hidden": false} nteract={"transient": {"deleting": false}}
mlflow.start_run()

lines = [
    f"Model name: {args.model_name}",
    f"Model path: {args.model_path}",
    f"Evaluation output path: {args.evaluation_output}",
]

for line in lines:
    print(line)

main(args)

mlflow.end_run()
