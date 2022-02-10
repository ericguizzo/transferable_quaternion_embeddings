import numpy as np
import matplotlib.pyplot as plt


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

#nopre
emodb_alexnet_nopre = 62.5
emodb_resnet_nopre = 53.75
emodb_vgg_nopre = 57.5
ravdess_alexnet_nopre = 31.515
ravdess_resnet_nopre = 42.727
ravdess_vgg_nopre = 37.273
tess_alexnet_nopre = 97.143
tess_resnet_nopre = 96.429
tess_vgg_nopre = 90.94

#nobackprop
emodb_alexnet_noback = 45
emodb_resnet_noback = 53
emodb_vgg_noback = 47

ravdess_alexnet_noback = 47.576
ravdess_resnet_noback = 51.818
ravdess_vgg_noback = 51.667

tess_alexnet_noback = 98.571
tess_resnet_noback = 99.524
tess_vgg_noback = 97.619

diff_emodb_nopre = ((emodb_alexnet_o1 - emodb_alexnet_nopre) + \
             (emodb_resnet_o1 - emodb_resnet_nopre) + \
             (emodb_vgg_o1 - emodb_vgg_nopre) / 3)
diff_ravdess_nopre = ((ravdess_alexnet_o1 - ravdess_alexnet_nopre) + \
             (ravdess_resnet_o1 - ravdess_resnet_nopre) + \
             (ravdess_vgg_o1 - ravdess_vgg_nopre) / 3)
diff_tess_nopre = ((tess_alexnet_o1 - tess_alexnet_nopre) + \
             (tess_resnet_o1 - tess_resnet_nopre) + \
             (tess_vgg_o1 - tess_vgg_nopre) / 3)

diff_emodb_noback = ((emodb_alexnet_o1 - emodb_alexnet_noback) + \
             (emodb_resnet_o1 - emodb_resnet_noback) + \
             (emodb_vgg_o1 - emodb_vgg_noback) / 3)
diff_ravdess_noback = ((ravdess_alexnet_o1 - ravdess_alexnet_noback) + \
             (ravdess_resnet_o1 - ravdess_resnet_noback) + \
             (ravdess_vgg_o1 - ravdess_vgg_noback) / 3)
diff_tess_noback = ((tess_alexnet_o1 - tess_alexnet_noback) + \
             (tess_resnet_o1 - tess_resnet_noback) + \
             (tess_vgg_o1 - tess_vgg_noback) / 3)

diff_nopre = np.round([-diff_emodb_nopre, -diff_ravdess_nopre, -diff_tess_nopre], decimals=2)
diff_noback = np.round([-diff_emodb_noback, -diff_ravdess_noback, -diff_tess_noback], decimals=2)

y_labels = ["EmoDb", "Ravdess", "Tess"]
x = np.arange(len(y_labels))
width = 0.35  # the width of the bars

fig, ax = plt.subplots()

rects1 = ax.barh(x - width/2, diff_nopre, width, label='No pretraining')
rects2 = ax.barh(x + width/2, diff_noback, width, label='No backpropagation')

ax.set_xlabel('Performance drop (percentage points)')
ax.set_title('Ablation Study: removing R2Hemo pretraining/backpropagation')
ax.set_yticks(x)
ax.set_yticklabels(y_labels)

ax.set_xlim(-34,5)

plt.axvline(x=0, color='black', label='Baseline: Full R2Hemo')

ax.legend()


for i, v in enumerate(diff_nopre):
        ax.text(v-3.4, i - width/2-0.03, str(v))

for i, v in enumerate(diff_noback):
        if v <= 0:
            ax.text(v-3.2, i + width/2-0.03, str(v))
        else:
            ax.text(v, i + width/2-0.03, str(v))

ax.grid(color='gray', linestyle='dashed', alpha=0.3)

#plt.show()

fig_name = "../ablation_study_train.png"
plt.savefig(fig_name, format = 'png', dpi=300)

print ("DIFF nopre ", diff_nopre)
print ("DIFF noback ", diff_noback)
