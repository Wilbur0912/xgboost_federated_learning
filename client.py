import xgboost as xgb
import flwr as fl
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from flwr.common import Code, Status
from flwr.common.logger import log
from logging import INFO
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
X_train = train_data[features].astype(float)
y_train = train_data[target]

X_valid = valid_data[features].astype(float)
y_valid = valid_data[target]

# Create LabelEncoder
label_encoder = LabelEncoder()

# Encode target variable as numerical labels
y_train_encoded = label_encoder.fit_transform(y_train)
y_valid_encoded = label_encoder.transform(y_valid)

params = {
    "objective": "multi:softmax",  # or "objective": "multi:softprob",
    "num_class": 10,  # Set num_classes if known
    "eta": 0.3,
    # Other hyperparameters
}
dtrain = xgb.DMatrix(X_train, label=y_train_encoded)
dvalid = xgb.DMatrix(X_valid, label=y_valid_encoded)

num_rounds = 1

# Integrate the above data into the Flower client
class XgbClient(fl.client.Client):
    def __init__(self):
        self.bst = None
        self.config = None

    def _local_boost(self):
        # Update trees based on local training data
        for i in range(num_rounds):
            self.bst.update(dtrain, self.bst.num_boosted_rounds())

        # Extract the last N=num_local_round trees for server aggregation
        bst = self.bst[
            self.bst.num_boosted_rounds()
            - num_rounds : self.bst.num_boosted_rounds()
        ]

        return bst
    
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )
    
    def fit(self, ins: FitIns) -> FitRes:
        if not self.bst:
            # First round local training
            log(INFO, "Start training at round 1")
            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=num_rounds,
                evals=[(dvalid, "validate"), (dtrain, "train")],
            )
            self.config = bst.save_config()
            self.bst = bst
        else:
            # Perform model training here
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            # Load global model into booster
            self.bst.load_model(global_model)
            self.bst.load_config(self.config)

            bst = self._local_boost()

        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        # Local model evaluation, compute training loss

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

        return fl.common.EvaluateRes(
            loss=0.0,
            num_examples=len(X_valid),
            metrics={"accuracy": valid_accuracy},
            status=Status(
                code=Code.OK,
                message="OK",
            ),
        )

# Start the Flower client
fl.client.start_client(server_address="127.0.0.1:8080", client=XgbClient())
