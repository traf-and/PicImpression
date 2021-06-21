import torch
from models.Image_encoder import VisualTransformer

encoder = VisualTransformer()
state_dict=torch.load('models/weights.pt')
encoder.load_state_dict(state_dict)
print(encoder.state_dict())