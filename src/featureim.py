import torch
import torch.nn.functional as F
# import torchvision.transforms as transforms
# import torchvision.models as models

# import captum
# from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
# from captum.attr import visualization as viz

# from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
# from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation
import os
import sys
import json
import numpy as np
import copy
# from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from model import create_model
from utils import load_config

from data import create_loader

# load model
config = load_config('config/CA.yml')
model = create_model(config)
path = config.savepoint
model.load_state_dict(torch.load(path))
model.eval()

# get input
dataloader_train = create_loader(config, 'train')
dataloader_valid = create_loader(config, 'valid')
x_train, y_train = next(iter(dataloader_train))
t_train = (x_train[-1], y_train[-1])
x_valid, y_valid = next(iter(dataloader_valid))
# t_valid = (x_valid, y_valid)
xs, y_trues = [x.float() for x in x_valid], [
    y.float() for y in y_valid]
original_output = model.forward(xs, y_trues)['y_preds']
# print(len(original_output['y_preds'][0]))

# list feature
features = [
    '0add',
    'uniadd',
    'newadd',
    'actadd',
    'transcount',
    'Ltranscount',
    'Meantransvalue',
    'high',
    'low',
    'open',
    'volfrom',
    'volto',
    'close',
    'Mom7',
    'Vmeanfrom7',
    'Vmeanto7',
    'retvol7',
    'HLHV7',
    'Mom14',
    'Vmeanfrom14',
    'Vmeanto14',
    'retvol14',
    'HLHV14',
    'Mom28',
    'Vmeanfrom28',
    'Vmeanto28',
    'retvol28',
    'HLHV28',
    'Mom56',
    'Vmeanfrom56',
    'Vmeanto56',
    'retvol56',
    'HLHV56',
    'Mom112',
    'Vmeanfrom112',
    'Vmeanto112',
    'retvol112',
    'HLHV112',
    'Mom350',
    'Vmeanfrom350',
    'Vmeanto350',
    'retvol350',
    'HLHV350',
    'Mom700',
    'Vmeanfrom700',
    'Vmeanto700',
    'retvol700',
    'HLHV700',
    'Mom1400',
    'Vmeanfrom1400',
    'Vmeanto1400',
    'retvol1400',
    'HLHV1400',
    'maxret',
    'history',
    'newadd7',
    'newactadd7',
    'newuniadd7',
    'transcount7',
    'ltranscount7',
    'Meantransvalue7',
    'newadd28',
    'newactadd28',
    'newuniadd28',
    'transcount28',
    'ltranscount28',
    'avgtransvalue28',
    'newadd56',
    'newactadd56',
    'newuniadd56',
    'transcount56',
    'ltranscount56',
    'Meantransvalue56',
    'newadd112',
    'newactadd112',
    'newuniadd112',
    'transcount112',
    'ltranscount112',
    'avgtransvalue112'
]

fimp_metric_dict = dict()
for j in range(0, xs[0].shape[1]):
    new_xs, y_trues = [x.float() for x in x_valid], [
        y.float() for y in y_valid]
    for i in range(0, len(xs)):
        new_xs[i][:, j] = 0
    y_pred_fimp = model.forward(new_xs, y_trues)['y_preds']
    fimp_metric_dict.update({features[j]: np.sum([np.sum(np.power(original_output[i].detach().numpy() - y_pred_fimp[i].detach().numpy(), 2))
                             for i in range(len(original_output))])})
# print(fimp_metric_dict.keys())
all_grade = sum(fimp_metric_dict.values())
result = {key: value / all_grade for key, value in fimp_metric_dict.items()}
sort_result = sorted(result.items(), key=lambda x: x[1], reverse=True)

# plot
labels, ys = zip(*sort_result[0:15])
xs = np.arange(len(labels))

# Create horizontal bars
plt.barh(xs, ys)

# Create names on the x-axis
plt.yticks(xs, labels)

# Show graphic
plt.show()
plt.gca().invert_yaxis()
plt.savefig('featureimp.png')
# plt.barh(xs, ys)

# # Replace default x-ticks with xs, then replace xs with labels
# plt.xticks(xs, labels)
# plt.yticks(ys)

# plt.savefig('netscore.png')

# plt.figure()
# plt.bar(list(fimp_metric_dict.keys()), list(fimp_metric_dict.values()))
# plt.show()
# # fit
# ig = IntegratedGradients(model)
# ig_nt = NoiseTunnel(ig)
# dl = DeepLift(model)
# gs = GradientShap(model)
# fa = FeatureAblation(model)

# ig_attr_test = ig.attribute(t_valid, n_steps=50)
# ig_nt_attr_test = ig_nt.attribute(t_valid)
# dl_attr_test = dl.attribute(t_valid)
# # gs_attr_test = gs.attribute(x_valid[-1], x_train[-1])
# fa_attr_test = fa.attribute(t_valid)


# # prepare attributions for visualization

# x_axis_data = np.arange(x_valid.shape[1])
# # x_axis_data_labels = list(map(lambda idx: feature_names[idx], x_axis_data))

# ig_attr_test_sum = ig_attr_test.detach().numpy().sum(0)
# ig_attr_test_norm_sum = ig_attr_test_sum / \
#     np.linalg.norm(ig_attr_test_sum, ord=1)

# ig_nt_attr_test_sum = ig_nt_attr_test.detach().numpy().sum(0)
# ig_nt_attr_test_norm_sum = ig_nt_attr_test_sum / \
#     np.linalg.norm(ig_nt_attr_test_sum, ord=1)

# dl_attr_test_sum = dl_attr_test.detach().numpy().sum(0)
# dl_attr_test_norm_sum = dl_attr_test_sum / \
#     np.linalg.norm(dl_attr_test_sum, ord=1)

# # gs_attr_test_sum = gs_attr_test.detach().numpy().sum(0)
# # gs_attr_test_norm_sum = gs_attr_test_sum / \
# #     np.linalg.norm(gs_attr_test_sum, ord=1)

# fa_attr_test_sum = fa_attr_test.detach().numpy().sum(0)
# fa_attr_test_norm_sum = fa_attr_test_sum / \
#     np.linalg.norm(fa_attr_test_sum, ord=1)

# lin_weight = model.lin1.weight[0].detach().numpy()
# y_axis_lin_weight = lin_weight / np.linalg.norm(lin_weight, ord=1)

# width = 0.14
# # legends = ['Int Grads', 'Int Grads w/SmoothGrad', 'DeepLift',
# #            'GradientSHAP', 'Feature Ablation', 'Weights']
# legends = ['Int Grads', 'Int Grads w/SmoothGrad', 'DeepLift',
#            'Feature Ablation', 'Weights']

# plt.figure(figsize=(20, 10))

# ax = plt.subplot()
# ax.set_title(
#     'Comparing input feature importances across multiple algorithms and learned weights')
# ax.set_ylabel('Attributions')

# FONT_SIZE = 16
# plt.rc('font', size=FONT_SIZE)            # fontsize of the text sizes
# plt.rc('axes', titlesize=FONT_SIZE)       # fontsize of the axes title
# plt.rc('axes', labelsize=FONT_SIZE)       # fontsize of the x and y labels
# plt.rc('legend', fontsize=FONT_SIZE - 4)  # fontsize of the legend

# ax.bar(x_axis_data, ig_attr_test_norm_sum, width,
#        align='center', alpha=0.8, color='#eb5e7c')
# ax.bar(x_axis_data + width, ig_nt_attr_test_norm_sum,
#        width, align='center', alpha=0.7, color='#A90000')
# ax.bar(x_axis_data + 2 * width, dl_attr_test_norm_sum,
#        width, align='center', alpha=0.6, color='#34b8e0')
# # ax.bar(x_axis_data + 3 * width, gs_attr_test_norm_sum,
# #        width, align='center',  alpha=0.8, color='#4260f5')
# ax.bar(x_axis_data + 4 * width, fa_attr_test_norm_sum,
#        width, align='center', alpha=1.0, color='#49ba81')
# ax.bar(x_axis_data + 5 * width, y_axis_lin_weight,
#        width, align='center', alpha=1.0, color='grey')
# ax.autoscale_view()
# plt.tight_layout()

# ax.set_xticks(x_axis_data + 0.5)
# ax.set_xticklabels(x_axis_data_labels)

# plt.legend(legends, loc=3)
# plt.show()
