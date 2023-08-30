import logging
import math

import numpy as np
import torch
import torch.utils.data

from pkg import Config
from pkg.gaze2screen.simple_dataset import SimpleDataset
from pkg.l2s.model import LandMarks2ScreenModel

"""
export interface LandmarkFeatures {
    face_oval: Number[][];
    left_eye: number[][];
    right_eye: number[][];
    left_iris: number[][];
    right_iris: number[][];
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

        self.model = LandMarks2ScreenModel(logger=logger, filename=self.config.g2s.checkpoint).to(self.device)
        # self.optimizer = torch.optim.SGD(self.model.parameters(),
        #                                  lr=self.config.g2s.lr,
        #                                  momentum=self.config.g2s.momentum,
        #                                  nesterov=self.config.g2s.nesterov)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.config.g2s.lr, betas=self.config.g2s.betas)

        feature_shape = [(36 + 16 + 16 + 4 + 4) * 3 + 14]

        self.dataset = SimpleDataset(capacity=self.config.g2s.dataset_capacity, logger=logger)
        self.dataset.load(self.config.g2s.dataset_path)
        # self.model.eval()
        self.loss = -1.0

    def save(self):
        self.model.save(self.config.g2s.checkpoint)

    def train_with_dataset(self, epochs):

        if len(self.dataset) >= self.config.g2s.dataset_min_size:

            self.model.train()

            loader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.config.g2s.batch_size,
                                                 shuffle=True)

            logging.getLogger('app').info(f"Training with dataset size: {len(self.dataset)}")

            # If number of epochs not set,  do more and more epochs as the dataset grows
            if epochs <= 0:
                epochs = 1 + len(self.dataset) // 100
            total_loss = 0.
            for epoch in range(epochs):
                total_loss = 0.
                for batch, (idx, x, y) in enumerate(loader):

                    pred = self.model(x)
                    try:
                        # The loss is a simple L2 dist, with more weight assigned to vertical loss,
                        # which is harder to learn
                        v_loss = 5 * abs(pred[:, 1] - y[:, 1])
                        h_loss = abs(pred[:, 0] - y[:, 0])
                        loss = torch.sqrt((h_loss * h_loss) + (v_loss * v_loss))
                        loss = torch.linalg.norm(loss)
                        total_loss += loss.item()
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    # Can't figure out why  I'm getting

                    # File "/pkg/l2s/l2s.py", line 128, in train_with_dataset
                    # v_loss = 5 * abs(pred[:, 1] - y[:, 1])
                    # IndexError: too many indices for tensor of dimension 1

                    except IndexError:
                        continue

                total_loss /= len(loader)
                logging.getLogger('app').info(f"Epoch {epoch + 1} loss: {total_loss: .4f}")

            if math.isnan(total_loss):
                total_loss = -1.
            self.loss = total_loss

            return self.loss

    def process_landmarks(self, landmarks, label):

        # print("-------", landmarks, "-------")
        # print("-------", label, "-------")

        landmarks_for_display = [[landmark[0], landmark[1]] for landmark in landmarks["face_oval"]]

        landmarks = self.model.pack_input(landmarks)

        if label is not None:

            self.dataset.add_item(landmarks, torch.Tensor(label).to(torch.float32))

            # Save dataset periodically
            if (self.dataset.idx % 100) == 0:
                self.dataset.save(self.config.g2s.dataset_path)
                logging.getLogger('app').info(
                    f"Saved dataset to {self.config.g2s.dataset_path}. Dataset size {len(self.dataset)}")
            # return {
            #     'data_index': self.dataset.idx,
            #     'faces': 1,
            #     'eyes': 2,
            #     'gaze': {
            #         'x': 0.,
            #         'y': 0.,
            #     },
            #     'landmarks': [],
            #     'loss': self.loss
            # }

        with torch.no_grad():
            self.model.eval()
            pred = self.model(torch.unsqueeze(landmarks, 0))

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
            'loss': self.loss
        }


if __name__ == '__main__':

    l2s = Landmarks2ScreenCoords(logging.getLogger())
    # landmarks =  {'face_oval': [[0.6403485536575317, 0.6403485536575317, 0.03160703554749489], [0.6725339293479919, 0.6725339293479919, 0.0346776507794857], [0.6992780566215515, 0.6992780566215515, 0.04565609246492386], [0.7235185503959656, 0.7235185503959656, 0.06290891021490097], [0.7384771108627319, 0.7384771108627319, 0.08247029781341553], [0.7464713454246521, 0.7464713454246521, 0.09894852340221405], [0.7499455213546753, 0.7499455213546753, 0.11511904001235962], [0.7505308389663696, 0.7505308389663696, 0.12421286106109619], [0.749548614025116, 0.749548614025116, 0.12408576160669327], [0.7481286525726318, 0.7481286525726318, 0.11579795181751251], [0.7442265748977661, 0.7442265748977661, 0.10231031477451324], [0.7367194890975952, 0.7367194890975952, 0.08245524019002914], [0.7281894087791443, 0.7281894087791443, 0.05830126255750656], [0.718886137008667, 0.718886137008667, 0.03434177115559578], [0.705854058265686, 0.705854058265686, 0.015201503410935402], [0.6932834386825562, 0.6932834386825562, -9.067614155355841e-05], [0.679110586643219, 0.679110586643219, -0.015623323619365692], [0.6613640189170837, 0.6613640189170837, -0.026873033493757248], [0.6378408670425415, 0.6378408670425415, -0.031283456832170486], [0.6134511828422546, 0.6134511828422546, -0.02842816337943077], [0.5939010381698608, 0.5939010381698608, -0.018594039604067802], [0.5768530368804932, 0.5768530368804932, -0.004142350051552057], [0.561392068862915, 0.561392068862915, 0.010045778937637806], [0.5441551804542542, 0.5441551804542542, 0.028008824214339256], [0.5304983854293823, 0.5304983854293823, 0.05096305534243584], [0.5175678730010986, 0.5175678730010986, 0.0739249661564827], [0.5079851746559143, 0.5079851746559143, 0.09275997430086136], [0.5037161111831665, 0.5037161111831665, 0.10597322136163712], [0.5030210614204407, 0.5030210614204407, 0.11395861208438873], [0.503226637840271, 0.503226637840271, 0.11427601426839828], [0.5078939199447632, 0.5078939199447632, 0.10517556965351105], [0.5160671472549438, 0.5160671472549438, 0.0895349308848381], [0.5297261476516724, 0.5297261476516724, 0.07424017041921616], [0.549360454082489, 0.549360454082489, 0.056399621069431305], [0.5772887468338013, 0.5772887468338013, 0.04117993637919426], [0.6065606474876404, 0.6065606474876404, 0.03235342353582382]], 'left_eye': [[0.7177810072898865, 0.7177810072898865, 0.03653912991285324], [0.7144370675086975, 0.7144370675086975, 0.032477967441082], [0.7103997468948364, 0.7103997468948364, 0.02904035709798336], [0.7045572996139526, 0.7045572996139526, 0.025089338421821594], [0.6964907646179199, 0.6964907646179199, 0.0220643300563097], [0.6875395774841309, 0.6875395774841309, 0.021477840840816498], [0.6794940233230591, 0.6794940233230591, 0.022980928421020508], [0.6739122867584229, 0.6739122867584229, 0.02562803402543068], [0.7177810072898865, 0.7177810072898865, 0.03653912991285324], [0.715407133102417, 0.715407133102417, 0.032323434948921204], [0.7123090028762817, 0.7123090028762817, 0.02903810702264309], [0.7065203785896301, 0.7065203785896301, 0.02593950368463993], [0.6980273127555847, 0.6980273127555847, 0.02344280295073986], [0.6890565752983093, 0.6890565752983093, 0.022693542763590813], [0.6802819967269897, 0.6802819967269897, 0.023826170712709427], [0.6737405061721802, 0.6737405061721802, 0.025177165865898132]], 'right_eye': [[0.5577681660652161, 0.5577681660652161, 0.02996217831969261], [0.5618841648101807, 0.5618841648101807, 0.026120442897081375], [0.5661395788192749, 0.5661395788192749, 0.02292001061141491], [0.5720781087875366, 0.5720781087875366, 0.019475895911455154], [0.5809409022331238, 0.5809409022331238, 0.016998443752527237], [0.5899168848991394, 0.5899168848991394, 0.01708366535604], [0.5984569191932678, 0.5984569191932678, 0.019226383417844772], [0.6045258641242981, 0.6045258641242981, 0.02210376225411892], [0.5577681660652161, 0.5577681660652161, 0.02996217831969261], [0.5607002973556519, 0.5607002973556519, 0.025916555896401405], [0.5643216371536255, 0.5643216371536255, 0.02294384501874447], [0.5705076456069946, 0.5705076456069946, 0.02019372209906578], [0.5799547433853149, 0.5799547433853149, 0.018434492871165276], [0.588789701461792, 0.588789701461792, 0.018324993550777435], [0.5978462100028992, 0.5978462100028992, 0.019976358860731125], [0.6045617461204529, 0.6045617461204529, 0.021682122722268105]], 'left_iris': [[0.7078792452812195, 0.7078792452812195, -0.0007986108539626002], [0.6963465213775635, 0.6963465213775635, -0.0007977886125445366], [0.6841669082641602, 0.6841669082641602, -0.0008117841789498925], [0.6954725980758667, 0.6954725980758667, -0.0008106166496872902]], 'right_iris': [[0.5924440622329712, 0.5924440622329712, -8.089517359621823e-05], [0.5804612636566162, 0.5804612636566162, -7.28491140762344e-05], [0.5688429474830627, 0.5688429474830627, -7.887563697295263e-05], [0.5810323357582092, 0.5810323357582092, -8.245230128522962e-05]], 'eye_blendshapes': [0.06972628831863403, 0.07498575001955032, 0.40850526094436646, 0.4078928232192993, 0.0378396175801754, 0.083381287753582, 0.10249002277851105, 0.0541406087577343, 0.01844826154410839, 0.014409594237804413, 0.13917632400989532, 0.11299274861812592, 0.015118532814085484, 0.011725246906280518]}
    # label = [0., 0.]
    #
    # #
    # for i in range(1000):
    #      l2s.process_landmarks(landmarks, label)
    while True:
        l2s.train_with_dataset(5)
        l2s.save()