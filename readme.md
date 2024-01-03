# Xgboost Federated Learning code explain

import dependencies
```python
import xgboost as xgb
import flwr as fl
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from flwr.common import Code, Status
from flwr.common.logger import log
from logging import INFO
from sklearn.model_selection import train_test_split
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)
```

import data and preprocessing
```python
# Define features and labels
features = ["Height", "Weight", "BMI", "FSR1", "FSR2", "FSR3", "FSR4", "FSR5", "FSR6", "FSR7", "FSR8", "FSR9"]
target = "Posture"
column_names = ["Posture", "Height", "Weight", "BMI", "FSR1", "FSR2", "FSR3", "FSR4", "FSR5", "FSR6", "FSR7", "FSR8", "FSR9"]

# Read training data
train_data_path = "client_data/record_train1.csv"
train_data = pd.read_csv(train_data_path, names=column_names, encoding='utf-8', index_col=False)
# Read validation data
valid_data_path = "client_data/record_valid1.csv"
valid_data = pd.read_csv(valid_data_path, names=column_names, encoding='utf-8', index_col=False)

# Extract features and labels from training and validation data
X = train_data[features].astype(float)
y = train_data[target]

X_valid = valid_data[features].astype(float)
y_valid = valid_data[target]

# Create LabelEncoder
label_encoder = LabelEncoder()

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Encode target variable as numerical labels
y_train_encoded = label_encoder.fit_transform(y_train)

y_test_encoded = label_encoder.fit_transform(y_test)

y_valid_encoded = label_encoder.fit_transform(y_valid)

# convert data format to DMatrix, which is acceptable by xgboost
dtrain = xgb.DMatrix(X_train, label=y_train_encoded)

dtest = xgb.DMatrix(X_test, label=y_test_encoded)

dvalid = xgb.DMatrix(X_valid, label=y_valid_encoded)
```

model parameters setting
```python
params = {
    "objective": "multi:softmax",  # or "objective": "multi:softprob",
    "num_class": 10,  # Set num_classes if known
    "eta": 0.3,
    "max_depth": 6,
    # Other hyperparameters
}

# how many rounds of local boost
num_rounds = 1
```

building client
```python
class XgbClient(fl.client.Client):
    
    # in the first step, we make initial model(bst) and its congfig = none
    def __init__(self):
        self.bst = None
        self.config = None
    
    # Unlike neural network training, XGBoost trees are not started from a specified random weights. In this case, we do not use get_parameters and set_parameters to initialise model parameters for XGBoost. As a result, let’s return an empty tensor in get_parameters when it is called by the server at the first round.
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )
    
    # local training
    def _local_boost(self):
        # Update trees based on local training data.
        for i in range(num_rounds):
            self.bst.update(dtrain, self.bst.num_boosted_rounds())

        # Extract the last N=num_local_round trees for sever aggregation
        bst = self.bst[
            self.bst.num_boosted_rounds()
            - num_rounds : self.bst.num_boosted_rounds()
        ]
        
        return bst
    
    def fit(self, ins: FitIns) -> FitRes:
        if not self.bst:
            # First round local training
            log(INFO, "Start training at round 1")
            # build the first xgboost model
            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=num_rounds,
                evals=[(dtest, "test"), (dtrain, "train")],
            )
            
            # save model to local storage
            self.config = bst.save_config()
            self.bst = bst
        else:
            # get global_model
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            # Load global model into booster
            self.bst.load_model(global_model)
            self.bst.load_config(self.config)
            
            bst = self._local_boost()
            
        # save model as json format
        local_model = bst.save_raw("json")
        # transfer json formet to byte array
        local_model_bytes = bytes(local_model)

        # return model parameters back to server to do aggregation
        return FitRes(
            parameters=fl.common.Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=len(X_train),
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:

        # Local model evaluation, compute validation loss
        valid_predictions = self.bst.predict(dvalid)
        valid_accuracy = accuracy_score(y_valid_encoded, valid_predictions)
        
        # return evaluation back to server to do aggregation
        return fl.common.EvaluateRes(
            loss=0.0,
            num_examples=len(X_valid),
            metrics={"accuracy": valid_accuracy},
            status=Status(
                code=Code.OK,
                message="OK",
            ),
        )

```
Start the Flower client
```python
# Start the Flower client
fl.client.start_client(server_address="127.0.0.1:8080", client=XgbClient())
```