"""
@author: Jun Wang 
@date: 20201019 
@contact: jun21wangustc@gmail.com    
"""

import sys
import yaml

from classifier.MatchingHead import MatchingHead2
sys.path.append('../../')
from classifier.Dense2Layer import Dense2Layer, Dense2LayerSoftmax, Dense2LayerTemp, DenseEmbed
from classifier.ConvDense import ConvDense, Conv2Dense
from classifier.attention import Attention
from classifier.NConvMDense import NConvMDense

class ClassifierFactory:
    """Factory to produce classifier according the classifier_conf.yaml.
    
    Attributes:
        classifier_type(str): which classifier will produce.
        classifier_param(dict):  parsed params and it's value. 
    """
    def __init__(self, classifier_id, classifier_conf_file, experiment, backbone):
        self.classifier_type = classifier_id.split('_')[0]
        with open(classifier_conf_file) as f:
            classifier_conf = yaml.load(f, Loader=yaml.FullLoader)
            self.classifier_param = classifier_conf[experiment][backbone][classifier_id]

    def get_classifier(self):
        if self.classifier_type == "NConvMDense":
            classifer = NConvMDense(self.classifier_param)
        elif self.classifier_type == 'Dense2LayerSoftmax':
            feat_dim = self.classifier_param['feat_dim'] # dimension of the input features, e.g. 512.
            n_l1 = self.classifier_param['n_l1'] # height of the feature map before the final features.
            n_l2 = self.classifier_param['n_l2'] # width of the feature map before the final features.
            classifier = Dense2LayerSoftmax(feat_dim, n_l1, n_l2)
        elif self.classifier_type == 'Dense2LayerTemp':
            feat_dim = self.classifier_param['feat_dim'] # dimension of the input features, e.g. 512.
            n_l1 = self.classifier_param['n_l1'] # height of the feature map before the final features.
            n_l2 = self.classifier_param['n_l2'] # width of the feature map before the final features.
            classifier = Dense2LayerTemp(feat_dim, n_l1, n_l2)
        elif self.classifier_type == 'Attention':
            N = self.classifier_param['N'] # dimension of the input features, e.g. 512.
            in_channels = self.classifier_param['in_channels'] # height of the feature map before the final features.
            in_features = self.classifier_param['in_features'] # height of the feature map before the final features.
            out_features = self.classifier_param['out_features'] # height of the feature map before the final features.
            classifier = Attention(in_channels, in_features, out_features, N)
        elif self.classifier_type == 'Dense2Layer':
            feat_dim = self.classifier_param['feat_dim'] # dimension of the input features, e.g. 512.
            n_l1 = self.classifier_param['n_l1'] # height of the feature map before the final features.
            n_l2 = self.classifier_param['n_l2'] # width of the feature map before the final features.
            classifier = Dense2Layer(feat_dim, n_l1, n_l2)
        elif self.classifier_type == 'DenseEmbed':
            feat_dim = self.classifier_param['feat_dim'] # dimension of the input features, e.g. 512.
            n_l1 = self.classifier_param['n_l1'] # height of the feature map before the final features.
            n_l2 = self.classifier_param['n_l2'] # width of the feature map before the final features.
            classifier = DenseEmbed(feat_dim, n_l1, n_l2)
        elif self.classifier_type == 'MatchingHead2':
            feat_dim = self.classifier_param['feat_dim'] # dimension of the input features, e.g. 512.
            out_h = self.classifier_param['out_h'] # height of the feature map before the final features.
            out_w = self.classifier_param['out_w'] # width of the feature map before the final features.
            n_l1 = self.classifier_param['n_l1'] # height of the feature map before the final features.
            n_l2 = self.classifier_param['n_l2'] # width of the feature map before the final features.
            classifier = MatchingHead2(feat_dim, out_h, out_w, n_l1, n_l2)
        elif self.classifier_type == 'ConvDense':
            k_l1 = self.classifier_param['k_l1']
            in_ch_l1 = self.classifier_param['in_ch_l1']
            out_ch_l1 = self.classifier_param['out_ch_l1']
            s_l1 = self.classifier_param['s_l1']
            in_l2 = self.classifier_param['in_l2']
            out_l2 = self.classifier_param['out_l2']
            classifier = ConvDense(k_l1, in_ch_l1, out_ch_l1, s_l1, in_l2, out_l2)
        elif self.classifier_type == 'Conv2Dense':
            k = [self.classifier_param['k_l1'], self.classifier_param['k_l2']]
            in_ch = [self.classifier_param['in_ch_l1'], self.classifier_param['in_ch_l2']]
            out_ch = [self.classifier_param['out_ch_l1'], self.classifier_param['out_ch_l2']]
            s = [self.classifier_param['s_l1'], self.classifier_param['s_l2']]
            lin_in = self.classifier_param['in_l2']
            lin_out = self.classifier_param['out_l2']
            classifier = Conv2Dense(k, in_ch, out_ch, s, lin_in, lin_out)
        return classifier
