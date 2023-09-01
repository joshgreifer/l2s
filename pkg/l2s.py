import logging
import math

import numpy as np
import torch
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from pkg.config import Config
from pkg.simple_dataset import SimpleDataset
from pkg.model import LandMarks2ScreenModel

"""
export interface LandmarkFeatures {
    face_oval: Number[][];
    left_eye: number[][];
    right_eye: number[][];
    left_iris: number[][];
    right_iris: number[][];
    nose: number[][];
    eye_blendshapes: number[];
}
"""


class Landmarks2ScreenCoords:

    def __init__(self, logger):

        self.config = Config()
        self.logger = logger
        self.mode = 'eval'
        self.device = torch.device(self.config.device)
        self.target = np.ndarray((2,))

        self.model = LandMarks2ScreenModel(logger=logger, filename=self.config.checkpoint).to(self.device)
        # self.optimizer = torch.optim.SGD(self.model.parameters(),
        #                                  lr=self.config.lr,
        #                                  momentum=self.config.momentum,
        #                                  nesterov=self.config.nesterov)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.config.lr, betas=self.config.betas)

        self.scheduler = StepLR(self.optimizer, step_size=self.config.step_size, gamma=self.config.gamma)

        self.dataset = SimpleDataset(capacity=self.config.dataset_capacity, logger=logger)
        self.dataset.load(self.config.dataset_path)
        self.losses = {"h_loss": 0., "v_loss": 0., "loss": 0.}

        # self.model.eval()


    def save(self):
        self.model.save(self.config.checkpoint)

    def train(self, epochs):

        if len(self.dataset) >= self.config.dataset_min_size:

            self.model.train()

            loader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.config.batch_size,
                                                 shuffle=True)

            logging.getLogger('app').info(f"Training with dataset size: {len(self.dataset)}")

            # If number of epochs not set,  do more and more epochs as the dataset grows
            if epochs <= 0:
                epochs = 1 + len(self.dataset) // 100

            for epoch in range(epochs):
                losses = {"h_loss": 0., "v_loss": 0., "loss": 0.}
                n_batches = 0
                for n_batches, (idx, x, y) in tqdm(enumerate(loader), total=len(self.dataset) // self.config.batch_size):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    pred = self.model(x)

                    try:
                        # The loss is a simple L1 dist, with more weight assigned to vertical loss,
                        # which is harder to learn

                        # dist = torch.mean(torch.nn.PairwiseDistance()(pred, y))
                        # dist_loss += dist.item()

                        dists = torch.mean(abs(pred - y), dim=0)
                        h_dist = dists[0]
                        v_dist = dists[1]
                        loss = torch.pow(v_dist, 3) + torch.pow(h_dist, 3)

                        losses["loss"] += loss.cpu().item()
                        losses["h_loss"] += h_dist.cpu().item()
                        losses["v_loss"] += v_dist.cpu().item()
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    # TODO: Figure out why I'm getting this error sometimes:

                    # File "/pkg/l2s/l2s.py", line 128, in train_with_dataset
                    # v_loss = 5 * abs(pred[:, 1] - y[:, 1])
                    # IndexError: too many indices for tensor of dimension 1

                    except IndexError:
                        continue

                losses["loss"] /= n_batches + 1
                losses["h_loss"] /= n_batches + 1
                losses["v_loss"] /= n_batches + 1
                self.losses = losses
                self.scheduler.step()
                logging.getLogger('app').info(f'Epoch {epoch + 1}: lr {self.scheduler.get_last_lr()}  h_loss: {self.losses["h_loss"]: .4f} v_loss: {self.losses["v_loss"]: .4f}')

                if (epoch % self.config.model_checkpoint_frequency) == 0:
                    self.save()

        return self.losses

    def predict(self, landmarks, label):

        # print("-------", landmarks, "-------")
        # print("-------", label, "-------")

        # Return some of the landmarks back to the client.
        # This is not really necessary, as the client itself found the landmarks in the first place.
        # This is a legacy from the old app, where the client posted video frames
        # and the server found the landmarks.

        face_oval_landmarks = [[landmark[0], landmark[1]] for landmark in landmarks["face_oval"]]
        nose_landmarks = [[landmark[0], landmark[1]] for landmark in landmarks["nose"]]
        landmarks_for_display = face_oval_landmarks + nose_landmarks

        # Pack the landmarks into a tensor.  This tensor is what's saved in the dataset
        # as well as being passed to the model.
        landmarks = self.model.pack_input(landmarks)

        # If we're gathering training data, add the landmarks (x) and label (y)
        # In the training dataset.
        if label is not None:

            self.dataset.add_item(landmarks, torch.Tensor(label).to(torch.float32))

            # Save dataset periodically
            if (self.dataset.idx % self.config.dataset_checkpoint_frequency) == 0:
                self.dataset.save(self.config.dataset_path)
                logging.getLogger('app').info(
                    f"Saved dataset to {self.config.dataset_path}. Dataset size {len(self.dataset)}")

        # Predict the gaze coordinates
        with torch.no_grad():
            self.model.eval()
            pred = self.model(torch.unsqueeze(landmarks, 0).to(self.device))

            pred = torch.squeeze(pred)
            gaze_location = pred.cpu().detach().numpy()

        # print(label, features, gaze_location)
        return {
            'data_index': self.dataset.idx,
            'faces': 1,
            'eyes': 2,
            'gaze': {
                'x': float(gaze_location[0]),
                'y': float(gaze_location[1])
            },
            'landmarks': landmarks_for_display,
            'losses': self.losses
        }


if __name__ == '__main__':
    def main():
        l2s = Landmarks2ScreenCoords(logging.getLogger())
        landmarks = {'face_oval': [[0.6597680449485779, 0.2846783399581909, 0.02011331543326378], [0.6916626691818237, 0.28863203525543213, 0.02311854250729084], [0.7180102467536926, 0.29853254556655884, 0.03404863551259041], [0.7418010830879211, 0.31692731380462646, 0.051516950130462646], [0.7563264966011047, 0.3423432409763336, 0.07167249172925949], [0.7638235092163086, 0.3733297884464264, 0.08934614062309265], [0.7659592032432556, 0.4062206447124481, 0.10748021304607391], [0.7651310563087463, 0.44563058018684387, 0.11888182163238525], [0.7627451419830322, 0.4820328950881958, 0.12049499899148941], [0.7596564292907715, 0.5181971192359924, 0.11438115686178207], [0.7547968029975891, 0.5563021898269653, 0.1028803214430809], [0.746695876121521, 0.5964900255203247, 0.08542519062757492], [0.7375945448875427, 0.6269826889038086, 0.06318315118551254], [0.7284680604934692, 0.6479064226150513, 0.04124224931001663], [0.7154057621955872, 0.66550612449646, 0.023711491376161575], [0.7025173902511597, 0.676508903503418, 0.009818675927817822], [0.6875787973403931, 0.6855582594871521, -0.004265191033482552], [0.6688691973686218, 0.6920061111450195, -0.014192178845405579], [0.6442789435386658, 0.6926613450050354, -0.018169552087783813], [0.6196812391281128, 0.6893594861030579, -0.015547491610050201], [0.6005250215530396, 0.6805289387702942, -0.006625800393521786], [0.5845233201980591, 0.6694071888923645, 0.006629698909819126], [0.5705219507217407, 0.656780481338501, 0.019714344292879105], [0.5561204552650452, 0.6378079056739807, 0.03630077466368675], [0.545821487903595, 0.6160451173782349, 0.057509493082761765], [0.5363157987594604, 0.5852473974227905, 0.0785936638712883], [0.5295211672782898, 0.5439784526824951, 0.09484972059726715], [0.5264604687690735, 0.5051727890968323, 0.10587431490421295], [0.5260522365570068, 0.46897900104522705, 0.11181195825338364], [0.5262702703475952, 0.432611882686615, 0.11003497987985611], [0.5305110812187195, 0.39336079359054565, 0.09889073669910431], [0.5377034544944763, 0.36064761877059937, 0.08169376850128174], [0.5506983995437622, 0.3303617537021637, 0.06510597467422485], [0.5698858499526978, 0.30641886591911316, 0.04631621390581131], [0.5975141525268555, 0.2907431423664093, 0.030544206500053406], [0.6264278888702393, 0.28443223237991333, 0.021308358758687973]], 'left_eye': [[0.7307468056678772, 0.4137488901615143, 0.031687136739492416], [0.7274676561355591, 0.41858965158462524, 0.02787458524107933], [0.72365802526474, 0.42134207487106323, 0.024562940001487732], [0.7180708050727844, 0.4231474995613098, 0.020729610696434975], [0.7097107172012329, 0.4231460988521576, 0.01775573566555977], [0.7004796266555786, 0.4207664430141449, 0.017107632011175156], [0.692231297492981, 0.4175884425640106, 0.018465382978320122], [0.6865452527999878, 0.41544315218925476, 0.020903488621115685], [0.7307468056678772, 0.4137488901615143, 0.031687136739492416], [0.7290292978286743, 0.4094896912574768, 0.027364354580640793], [0.7263791561126709, 0.40511733293533325, 0.02398771233856678], [0.7213027477264404, 0.3999103605747223, 0.02071245387196541], [0.7129966020584106, 0.3966664969921112, 0.01814168319106102], [0.7035887241363525, 0.3973284363746643, 0.017391569912433624], [0.6941576600074768, 0.4022024869918823, 0.01884762942790985], [0.6870318651199341, 0.4086908996105194, 0.02026272378861904]], 'right_eye': [[0.5782067775726318, 0.4064215421676636, 0.026112079620361328], [0.5819177031517029, 0.4100458025932312, 0.022493649274110794], [0.5856572389602661, 0.41222628951072693, 0.019409295171499252], [0.5910873413085938, 0.41407153010368347, 0.016080912202596664], [0.5995978713035583, 0.4145445227622986, 0.013604472391307354], [0.6084585189819336, 0.41297900676727295, 0.013507375493645668], [0.6167783737182617, 0.41077354550361633, 0.015381951816380024], [0.622902512550354, 0.40961790084838867, 0.018034914508461952], [0.5782067775726318, 0.4064215421676636, 0.026112079620361328], [0.5807124376296997, 0.4023098349571228, 0.02192329429090023], [0.5838922262191772, 0.3980889916419983, 0.018847282975912094], [0.58929443359375, 0.39369335770606995, 0.01585214026272297], [0.598107099533081, 0.3909417986869812, 0.013944203034043312], [0.6069024801254272, 0.39209210872650146, 0.01373014785349369], [0.6158038377761841, 0.3971264660358429, 0.01564604416489601], [0.622642457485199, 0.4037325978279114, 0.017415562644600868]], 'left_iris': [[0.7216776609420776, 0.41053029894828796, 0.01913151703774929], [0.7111383676528931, 0.3959817588329315, 0.019146066159009933], [0.6989750862121582, 0.4085950255393982, 0.019117701798677444], [0.7093791961669922, 0.4236399829387665, 0.019134389236569405]], 'right_iris': [[0.6099461317062378, 0.40221813321113586, 0.013537051156163216], [0.5987530946731567, 0.3890780508518219, 0.013541892170906067], [0.5878461599349976, 0.4025461971759796, 0.013536199927330017], [0.5989033579826355, 0.4162907600402832, 0.013537613674998283]], 'nose': [[0.6460637450218201, 0.5596886277198792, -0.04085496813058853], [0.6549197435379028, 0.4824334383010864, -0.05895925685763359], [0.6520858407020569, 0.5083318948745728, -0.03487655892968178], [0.6469296216964722, 0.43671897053718567, -0.038429733365774155]], 'eye_blendshapes': [0.022959042340517044, 0.05858038738369942, 0.19824594259262085, 0.2000114470720291, 0.030696338042616844, 0.10777127742767334, 0.11985012143850327, 0.051411449909210205, 0.0507938414812088, 0.04627503082156181, 0.10346048325300217, 0.28729820251464844, 0.04115169867873192, 0.01789393275976181]}
        label = [0., 0.]
        while len(l2s.dataset) < l2s.config.dataset_min_size:
            l2s.predict(landmarks, label)

        l2s.train(1)

    main()
