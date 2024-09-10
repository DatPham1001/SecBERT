import flwr as fl
def start_server():
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3),
                              server_address="0.0.0.0:8088"
   )
# Run this cell in a separate Colab cell
start_server()