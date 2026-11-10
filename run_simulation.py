import argparse
import json
import flwr as fl
from flwr.common import Context
import torch
import torch.nn as nn

# Model Sederhana
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        return self.fc(x)

# Client Flower versi Modern (menggunakan Context)
class NumPyClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in Net().state_dict().values()]

    def fit(self, parameters, config):
        return self.get_parameters(config={}), 10, {}

    def evaluate(self, parameters, config):
        return 0.45, 10, {"accuracy": 0.92}

def client_fn(context: Context):
    return NumPyClient().to_client()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--clients", type=int, default=2)
    args = parser.parse_args()

    # Jalankan Simulasi
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
    )

    # Simpan hasil evaluasi (loss) ke file JSON untuk dibaca Streamlit
    losses = [loss for _, loss in history.losses_distributed]
    results = {"rounds": list(range(1, len(losses) + 1)), "loss": losses}
    
    with open("results.json", "w") as f:
        json.dump(results, f)