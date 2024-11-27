import torch
import torch.nn as nn

class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(
            'cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.begin_step = 0
        self.begin_epoch = 0
        self.ddpm = None  # DDPM model will be added later

    def feed_data(self, data):
        """
        Preprocess and move data to the right device.
        """
        self.data = self.set_device(data)

    def optimize_parameters(self):
        """
        Optimization logic for training.
        For now, this will be specific to your model's training steps.
        """
        pass

    def get_current_visuals(self):
        """
        Fetch current visualizations for the model.
        In the context of graph anomaly detection, this could return
        enhanced graphs, original graphs, or anomaly scores.
        """
        pass

    def get_current_losses(self):
        """
        Return current losses during training.
        For example, the total loss, DDPM loss, and GCLAD loss.
        """
        return self.log_dict

    def print_network(self):
        """
        Print the network description, including total parameters.
        """
        s, n = self.get_network_description(self.netG)
        print(f"Network structure:\n{s}\nTotal parameters: {n}")

    def set_device(self, x):
        """
        Moves model or data to the correct device (GPU or CPU).
        """
        if isinstance(x, dict):
            for key, item in x.items():
                if item is not None:
                    x[key] = item.to(self.device)
        elif isinstance(x, list):
            for item in x:
                if item is not None:
                    item = item.to(self.device)
        else:
            x = x.to(self.device)
        return x

    def get_network_description(self, network):
        """
        Returns the string representation and parameter count of the network.
        """
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def set_ddpm(self, ddpm):
        """
        Integrate the DDPM model into the base model.
        """
        self.ddpm = ddpm

    def forward(self, features, adj):
        """
        Perform a forward pass through both GCLAD and DDPM.
        This method will leverage the DDPM for feature enhancement before passing the data through GCLAD.
        """
        # Apply DDPM to enhance features (e.g., noisy input graph features)
        enhanced_features = self.ddpm(features)

        # Now pass the enhanced features through GCLAD (or any other model components)
        logits_1, logits_2, subgraph_embed, node_embed = self.netG(enhanced_features, adj)

        return logits_1, logits_2, subgraph_embed, node_embed

    def anomaly_detection(self, features, adj):
        """
        Use the model for anomaly detection tasks. For example, calculating anomaly scores.
        We can apply the anomaly detection method after forward pass.
        """
        logits_1, logits_2, subgraph_embed, node_embed = self.forward(features, adj)
        
        # Anomaly detection could involve calculating an anomaly score based on logits
        anomaly_scores = self.calculate_anomaly_score(logits_1, logits_2, subgraph_embed, node_embed)

        return anomaly_scores

    def calculate_anomaly_score(self, logits_1, logits_2, subgraph_embed, node_embed):
        """
        Calculate anomaly scores based on model outputs (logits and embeddings).
        This could involve any logic depending on how anomaly scores are derived.
        """
        # Example: anomaly score could be based on the distance between embeddings or other criteria
        score_1 = torch.sigmoid(logits_1).mean(dim=1)
        score_2 = torch.sigmoid(logits_2).mean(dim=1)
        anomaly_score = score_1 + score_2  # A simple example, modify as needed

        return anomaly_score
