import torch
import numpy as np
import torch.nn as nn

class RegreessionNetwork(torch.nn.Module):
    def __init__(self):
        """
        1.1 d)
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()
        self.action_classes = torch.tensor([[-1. ,  0. ,  0. ],  # steer, gas and brake
                            [-1. ,  0. ,  0.8],
                            [-1. ,  0.5,  0. ],
                            [ 0. ,  0. ,  0. ],
                            [ 0. ,  0. ,  0.8],
                            [ 0. ,  0.5,  0. ],
                            [ 1. ,  0. ,  0. ],
                            [ 1. ,  0. ,  0.8],
                            [ 1. ,  0.5,  0. ]])


        self.num_classes = 4
        # setting device on GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cnn_blocks_1 = nn.ModuleList([torch.nn.Sequential(  
            torch.nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1),    # (3,96,96) -> (12,48,48)
            torch.nn.BatchNorm2d(12),
            torch.nn.ReLU(),
            # torch.nn.LeakyReLU(negative_slope =0.2),
            torch.nn.MaxPool2d(2, 2),
            # torch.nn.Dropout(0.25),
            ),
            
            torch.nn.Sequential(torch.nn.Conv2d(in_channels=12, out_channels=36, kernel_size=3, stride=1, padding=1),    # (12,48,48) -> (36,24,24)
            torch.nn.BatchNorm2d(36),
            # torch.nn.LeakyReLU(negative_slope =0.2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  
            # torch.nn.Dropout(0.25),
            ),])
        self.cnn_residual_1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=3, out_channels=36, kernel_size=3, stride=2, padding=2),    # (3,96,96) -> (36,24,24)
            torch.nn.BatchNorm2d(36),
            # torch.nn.LeakyReLU(negative_slope =0.2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  
            # torch.nn.Dropout(0.25)
            )
            

        self.cnn_1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=36, out_channels=72, kernel_size=3, stride=1, padding=1),    # (36,24,24) -> (72,12,12)
            nn.BatchNorm2d(72),
            # nn.LeakyReLU(negative_slope =0.2),
            torch.nn.ReLU(),
            nn.MaxPool2d(2, 2),  
            # nn.Dropout(0.25),
            
            nn.ConvTranspose2d(in_channels=72, out_channels=36, kernel_size=4, stride=2, padding=1),    # (72,12,12) -> (36,24,24)
            nn.BatchNorm2d(36),
            # nn.LeakyReLU(negative_slope =0.2),
            torch.nn.ReLU(),
            # nn.Dropout(0.25)
            )

        self.cnn_blocks_2 = nn.ModuleList([torch.nn.Sequential(torch.nn.Conv2d(in_channels=36, out_channels=72, kernel_size=3, stride=1, padding=1),    # (36, 24,24) -> (72,12,12)        
            torch.nn.BatchNorm2d(72),
            # torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  
            # torch.nn.Dropout(0.25),
            ),
            
            torch.nn.Sequential(torch.nn.Conv2d(in_channels=72, out_channels=144, kernel_size=3, stride=1, padding=1),    # (72, 12,12) -> (144,6,6)        
            torch.nn.BatchNorm2d(144),
            # torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  
            # torch.nn.Dropout(0.25),
            ),
            
            torch.nn.Sequential(torch.nn.Conv2d(in_channels=144, out_channels=288, kernel_size=3, stride=1, padding=1),    # (144, 6,6) -> (288,6,6)                   
            torch.nn.BatchNorm2d(288),
            # torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.25)
            )])
            
        self.cnn_residual_2 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=36, out_channels=288, kernel_size=5, stride=2, padding=2),    # (36, 24,24) -> (288,6,6)        
            torch.nn.BatchNorm2d(288),
            # torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  
            # torch.nn.Dropout(0.25),
            )
        
        self.cnn_2 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=288, out_channels=576, kernel_size=3, stride=1, padding=1),    # (144, 6,6) -> (288,6,6)        
            torch.nn.BatchNorm2d(576),
            # torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.25),
            torch.nn.Conv2d(in_channels=576, out_channels=288, kernel_size=3, stride=1, padding=1),    # (144, 6,6) -> (288,6,6)        
            torch.nn.BatchNorm2d(288),
            # torch.nn.LeakyReLU(negative_slope=0.2),  
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.25)  
            )

        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=288*6*6+7, out_features=1000),
            # torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.25),
            torch.nn.Linear(in_features=1000, out_features=100),
            # torch.nn.LeakyReLU(negative_slope =0.2),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.25),
            torch.nn.Linear(in_features=100, out_features=self.num_classes),
            torch.nn.Sigmoid())

        # self.apply(self._init_weights)

        import vision_transformer
        self.vit = vision_transformer.vit_tiny()
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=158+7, out_features=30),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=30, out_features=self.num_classes))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant(m.bias, 0.01)
            # torch.nn.init.zeros_(m.bias)

    def forward(self, observation):
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, C)
        """
        if len(observation.shape)<=3: 
            observation = observation.unsqueeze(dim=0)  # add batch_size to dim (NN take 4 dim input, not 3)
        if observation.shape[-1] != 96:     # i.e., ==3
            observation = torch.permute(observation,(0,3,1,2))   # change the shape to (batch_size, 3, 96, 96)
        if torch.max(observation)>2:    
            observation = observation/255    # restric the pixels value to range of [0,1]

        batch_size = observation.shape[0]

        x = observation
        for blk in self.cnn_blocks_1:
            x = blk(x)
        x = x + self.cnn_residual_1(observation)

        # x = self.cnn_1(x)+x    # residual; helpful in reducing the loss, but not helpful in evaluation (generalization)

        residual = x
        for blk in self.cnn_blocks_2:
            x = blk(x)
        x = x + self.cnn_residual_2(residual)

        # x = self.cnn_2(x)+x    # residual; not so helpful in evaluation 

        x = torch.flatten(x, start_dim=1)
        speed, abs_sensors, steering, gyroscope = self.extract_sensor_values(observation,batch_size)
        x = torch.cat((x,speed, abs_sensors, steering, gyroscope),1)     # concatanate additional information from environment, before feeding it into MLP
        return  self.mlp1(x) # return the probability distribution over the action-classes

    def actions_to_classes(self, actions):
        """
        1.1 c)
        For a given set of actions map every action to its corresponding
        action-class representation. Every action is represented by a 1-dim vector 
        with the entry corresponding to the class number.
        actions:        python list of N torch.Tensors of size 3. It has 9 unique elements, so we split is into 9 classes.
        return          python list of N torch.Tensors of size 1
        """
        # turn action into torch.Tensors of size 4: [steer_right, accelerate, brake, steer_left]  (where steer=-1 means steer_left)

        if type(actions) != list:   # here actions is a tensor of size 3
            c = torch.zeros(4)
            c[:3] = actions.clone()
            if actions[0] == -1.0:
                c[0]=0.0
                c[-1]=1.0
            return c

        classes = []
        for act in actions:
            c = torch.zeros(4)
            c[:3] = act.clone()
            if act[0]==-1:
                c[0]=0
                c[-1]=1
            classes.append(c)

        return classes

    def scores_to_action(self, scores):
        """
        1.1 c)
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
                        C = number of classes
        scores:         python list of torch.Tensors of size C
        return          (float, float, float)
        """
        if len(scores.shape)==2 and scores.shape[0]==1:
            scores = scores[0]
            action = torch.zeros(3)
            if abs(scores[0]-scores[-1])<0.01: # go straight
                action[0]=0
            elif scores[0]>scores[-1]:  # go right
                action[0]=scores[0]-scores[-1]
            elif scores[0]<scores[-1]:  # go left
                action[0]=-(scores[-1]-scores[0])
            action[1:] = scores[1:3]
        # max_action = torch.max(scores)
        # max_action_index = torch.where(scores==max_action)[1][0].item()
        # action = self.action_classes[max_action_index]
        return (float(i) for i in action)

    def extract_sensor_values(self, observation, batch_size):
        """
        observation:    python list of batch_size many torch.Tensors of size
                        (96, 96, 3)
        batch_size:     int
        return          torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 4),
                        torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 1)
        """
        if observation.shape[-1] != 3:  # in this case shape = (B,3,96,96)
            observation = torch.permute(observation,(0,2,3,1))
        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255
        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = steer_crop.sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope
