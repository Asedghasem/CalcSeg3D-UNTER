import torch.nn as nn

class IntermediateLayerExtractor(nn.Module):
    def __init__(self, model, target_layer_name):
        super().__init__()
        self.model = model
        self.target_layer_name = target_layer_name
        self.intermediate_output = None

        # Register a forward hook to capture the intermediate layer output
        for name, module in model.named_modules():
            if name == target_layer_name:
                print(name)
                module.register_forward_hook(self.save_intermediate_output)

    def save_intermediate_output(self, module, input, output):
        self.intermediate_output = output

    def forward(self, x):
        final_output = self.model(x)  # Perform a forward pass
        return self.intermediate_output, final_output  # Return intermediate and final output
