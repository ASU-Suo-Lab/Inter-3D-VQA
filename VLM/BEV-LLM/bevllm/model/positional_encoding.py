import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self,scale_factor, d_model, max_seq_length, feature_space_shape=None):
        super(PositionalEncoding, self).__init__()

        

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe * scale_factor

        self.register_buffer('pe', pe.unsqueeze(0))
        
        self._view_map_cache = {}


    def calculate_view_map(self, spatial_tokens):
        width = int(math.sqrt(spatial_tokens))
        height = int(math.sqrt(spatial_tokens))
        if width * height != spatial_tokens:
            raise ValueError(f"Expected square BEV map, got {spatial_tokens} spatial tokens")



        # Calculate the center of the feature map
        center_x = (height - 1) / 2  # center x-coordinate
        center_y = (width - 1) / 2  # center y-coordinate

        view_map = torch.zeros(width, height).long()

        # Loop through each element in the feature map
        for i in range(width):
            for j in range(height):
                # Calculate the vector from the current element to the center
                x = torch.tensor(j - center_x)
                y = torch.tensor(center_y - i)
                
                # Calculate the angle using atan2
                angle = torch.atan2(y, x)
                
                # Convert angle from radians to degrees if needed
                angle_degrees = torch.rad2deg(angle)
                


                if angle_degrees < 30 and angle_degrees > -30:
                    view_map[i,j] = 0

                elif angle_degrees < 90 and angle_degrees > 30:
                    view_map[i,j] = 1

                elif angle_degrees < 150 and angle_degrees > 90:
                    view_map[i,j] = 2
                
                elif angle_degrees < -150 and angle_degrees > -180 or angle_degrees < 180 and angle_degrees > 150:
                    view_map[i,j] = 3

                elif angle_degrees > -150 and angle_degrees < -90:
                    view_map[i,j] = 4

                elif angle_degrees > -90 and angle_degrees < -30:
                    view_map[i,j] = 5
        
        return view_map

    def get_view_map(self, spatial_tokens):
        if spatial_tokens not in self._view_map_cache:
            self._view_map_cache[spatial_tokens] = self.calculate_view_map(spatial_tokens)
        return self._view_map_cache[spatial_tokens]


    def classify_views(self, bev_map, view):

        channels = bev_map.shape[1]
        width = int(math.sqrt(bev_map.shape[2]))
        height = int(math.sqrt(bev_map.shape[2]))
        view_map = self.get_view_map(bev_map.shape[2])

        pos_vectors = []

        for v in view:
            indicies = view_map.view(-1)
            pos_vector = self.pe[0,indicies].to(bev_map.device)
            if int(v) == 6:
                pos_vector = pos_vector.view(width,height,channels).permute(2,0,1).unsqueeze(0)
            else:
                pos_vector_view = self.pe[0, int(v)].unsqueeze(0).to(bev_map.device)
                placeholder = torch.zeros(1,channels).to(bev_map.device)
                pos_vector = torch.where(pos_vector == pos_vector_view, pos_vector_view, placeholder)
                pos_vector = pos_vector.view(width,height,channels).permute(2,0,1).unsqueeze(0)
            
            pos_vectors.append(pos_vector)

    
        return torch.flatten(torch.cat(pos_vectors), start_dim=2).to(bev_map.device)

    def forward(self, x, view):
        feature_space_shape = x.shape
        x = torch.flatten(x, start_dim=2).permute(0,2,1)
        x = x + self.classify_views(x,view)
        x = x.reshape(feature_space_shape)
        return x
