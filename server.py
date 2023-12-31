import flwr as fl
from flwr.server.strategy import FedXgbBagging
from typing import List, Tuple
from flwr.common import Metrics

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

# Define strategy
strategy = FedXgbBagging(
    fraction_fit=1.0,
    min_fit_clients=2,
    min_available_clients=2,
    min_evaluate_clients=2,
    fraction_evaluate=1.0,
    evaluate_metrics_aggregation_fn=weighted_average
)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=100),
    strategy=strategy,
)

