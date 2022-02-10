import numpy as np
from scipy.stats import wilcoxon, kruskal, mannwhitneyu

#emodb
emodb_vgg_p1 = 47
emodb_vgg_p2 = 52
emodb_vgg_o1 = 50
emodb_vgg_o2 = 47
emodb_alexnet_p1 = 47
emodb_alexnet_p2 = 67
emodb_alexnet_o1 = 49
emodb_alexnet_o2 = 71
emodb_resnet_p1 = 48
emodb_resnet_p2 = 72
emodb_resnet_o1 = 73
emodb_resnet_o2 = 46
emodb_max_p = 72
emodb_max_o = 73

#ravdess
ravdess_vgg_p1 = 41.06
ravdess_vgg_p2 = 45.3
ravdess_vgg_o1 = 49.85
ravdess_vgg_o2 = 53.79
ravdess_alexnet_p1 = 46.36
ravdess_alexnet_p2 = 51.06
ravdess_alexnet_o1 = 43.94
ravdess_alexnet_o2 = 47.58
ravdess_resnet_p1 = 43.48
ravdess_resnet_p2 = 18.79
ravdess_resnet_o1 = 55.15
ravdess_resnet_o2 = 52.42
ravdess_max_p = 51.06
ravdess_max_o = 55.15

#tess
tess_vgg_p1 = 97.62
tess_vgg_p2 = 99.52
tess_vgg_o1 = 97.6
tess_vgg_o2 = 97.85
tess_alexnet_p1 = 98.01
tess_alexnet_p2 = 98.01
tess_alexnet_o1 = 98.56
tess_alexnet_o2 = 98.81
tess_resnet_p1 = 97.38
tess_resnet_p2 = 57.53
tess_resnet_o1 = 99.76
tess_resnet_o2 = 99.28
tess_max_p =  99.52
tess_max_o = 99.76

#iemocap
iemocap_vgg_p1 = 62.87
iemocap_vgg_o1 = 71.1
iemocap_alexnet_p1 = 63.33
iemocap_alexnet_o1 = 70.31
iemocap_resnet_p1 = 57.2
iemocap_resnet_o1 = 71.2
iemocap_max_p = 63.33
iemocap_max_o = 71.2
#emodb
emodb_mean_nopre = ((emodb_vgg_o1-emodb_vgg_p1) +
                    (emodb_alexnet_o1-emodb_alexnet_p1) +
                    (emodb_resnet_o1-emodb_resnet_p1)) / 3
emodb_mean_pre = ((emodb_vgg_o2-emodb_vgg_p2) +
                    (emodb_alexnet_o2-emodb_alexnet_p2) +
                    (emodb_resnet_o2-emodb_resnet_p2)) / 3
emodb_mean_overall = (emodb_mean_nopre + emodb_mean_pre) / 2
emodb_absolute = emodb_max_o - emodb_max_p

#ravdess
ravdess_mean_nopre = ((ravdess_vgg_o1-ravdess_vgg_p1) +
                    (ravdess_alexnet_o1-ravdess_alexnet_p1) +
                    (ravdess_resnet_o1-ravdess_resnet_p1)) / 3
ravdess_mean_pre = ((ravdess_vgg_o2-ravdess_vgg_p2) +
                    (ravdess_alexnet_o2-ravdess_alexnet_p2) +
                    (ravdess_resnet_o2-ravdess_resnet_p2)) / 3
ravdess_mean_overall = (ravdess_mean_nopre + ravdess_mean_pre) / 2
ravdess_absolute = ravdess_max_o - ravdess_max_p

#tess
tess_mean_nopre = ((tess_vgg_o1-tess_vgg_p1) +
                    (tess_alexnet_o1-tess_alexnet_p1) +
                    (tess_resnet_o1-tess_resnet_p1)) / 3
tess_mean_pre = ((tess_vgg_o2-tess_vgg_p2) +
                    (tess_alexnet_o2-tess_alexnet_p2) +
                    (tess_resnet_o2-tess_resnet_p2)) / 3
tess_mean_overall = (tess_mean_nopre + tess_mean_pre) / 2
tess_absolute = tess_max_o - tess_max_p

#iemocap
iemocap_mean_nopre = ((iemocap_vgg_o1-iemocap_vgg_p1) +
                    (iemocap_alexnet_o1-iemocap_alexnet_p1) +
                    (iemocap_resnet_o1-iemocap_resnet_p1)) / 3
iemocap_absolute = iemocap_max_o - iemocap_max_p

#significanze  test
emodb_p1 = [emodb_vgg_p1, emodb_alexnet_p1, emodb_resnet_p1]
emodb_p2 = [emodb_vgg_p2, emodb_alexnet_p2, emodb_resnet_p2]
emodb_o1 = [emodb_vgg_o1, emodb_alexnet_o1, emodb_resnet_o1]
emodb_o2 = [emodb_vgg_o2, emodb_alexnet_o2, emodb_resnet_o2]

ravdess_p1 = [ravdess_vgg_p1, ravdess_alexnet_p1, ravdess_resnet_p1]
ravdess_p2 = [ravdess_vgg_p2, ravdess_alexnet_p2, ravdess_resnet_p2]
ravdess_o1 = [ravdess_vgg_o1, ravdess_alexnet_o1, ravdess_resnet_o1]
ravdess_o2 = [ravdess_vgg_o2, ravdess_alexnet_o2, ravdess_resnet_o2]

tess_p1 = [tess_vgg_p1, tess_alexnet_p1, tess_resnet_p1]
tess_p2 = [tess_vgg_p2, tess_alexnet_p2, tess_resnet_p2]
tess_o1 = [tess_vgg_o1, tess_alexnet_o1, tess_resnet_o1]
tess_o2 = [tess_vgg_o2, tess_alexnet_o2, tess_resnet_o2]

_, w_emodb_nopre = wilcoxon(emodb_p1, emodb_p2, alternative="two-sided")
_, w_emodb_pre = wilcoxon(emodb_p2, emodb_o2, alternative="two-sided")
_, w_emodb_overall = wilcoxon(emodb_p1+emodb_p2, emodb_o1+emodb_o2, alternative="two-sided")

_, w_ravdess_nopre = wilcoxon(ravdess_p1, ravdess_o1, alternative="two-sided")
_, w_ravdess_pre = wilcoxon(ravdess_p2, ravdess_o2, alternative="two-sided")
_, w_ravdess_overall = wilcoxon(ravdess_p1+ravdess_p2, ravdess_o1+ravdess_o2, alternative="two-sided")

_, w_tess_nopre = wilcoxon(tess_p1, tess_o1, alternative="two-sided")
_, w_tess_pre = wilcoxon(tess_p2, tess_o2, alternative="two-sided")
_, w_tess_overall = wilcoxon(tess_p1+tess_p2, tess_o1+tess_o2, alternative="two-sided")

all_p = emodb_p1 + ravdess_p1 + tess_p1 + emodb_p2 + ravdess_p2 + tess_p2
all_o = emodb_o1 + ravdess_o1 + tess_o1 + emodb_o2 + ravdess_o2 + tess_o2

all_p1 = emodb_p1 + ravdess_p1 + tess_p1
all_p2 = emodb_p2 + ravdess_p2 + tess_p2
all_o1 = emodb_o1 + ravdess_o1 + tess_o1
all_o2 = emodb_o2 + ravdess_o2 + tess_o2

print(len(all_p))
#all_p = [1,2,3,4,5,6,7,8,9]
#all_o = [32,45,43,67,87,4,5,6,3]

w = wilcoxon(all_p, all_o, alternative="two-sided")
w1 = wilcoxon(all_p1, all_o1,  alternative="less")
w2 = wilcoxon(all_p2, all_o2,  alternative="less")

print ("IMPROVEMENT PER-DATASET (percentage points):")
print ("EmoDB | No pretrain: ",
        np.round(emodb_mean_nopre, decimals=2),
        "| Pretrain: ",
        np.round(emodb_mean_pre, decimals=2),
        "| Mean overall: ",
        np.round(emodb_mean_overall, decimals=2),
        "| Absolute overall: ",
        np.round(emodb_absolute, decimals=2))
print ("Ravdess | No pretrain: ",
        np.round(ravdess_mean_nopre, decimals=2),
        "| Pretrain: ",
        np.round(ravdess_mean_pre, decimals=2),
        "| Mean overall: ",
        np.round(ravdess_mean_overall, decimals=2),
        "| Absolute overall: ",
        np.round(ravdess_absolute, decimals=2))
print ("Tess | No pretrain: ",
        np.round(tess_mean_nopre, decimals=2),
        "| Pretrain: ",
        np.round(tess_mean_pre, decimals=2),
        "| Mean overall: ",
        np.round(tess_mean_overall, decimals=2),
        "| Absolute overall: ",
        np.round(tess_absolute, decimals=2))
print ("Iemocap | Mean overall: ",
        np.round(iemocap_mean_nopre, decimals=2),
        "| Absolute overall: ",
        np.round(iemocap_absolute, decimals=2))


print ("Wilcoxon:", w)
print ("Wilcoxon:", w1)
print ("Wilcoxon:", w2)
