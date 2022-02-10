import numpy as np
import matplotlib.pyplot as plt


r = {
100:{'alexnet':{
         'baseline': [46.36, 51.06],
         'our': [43.94, 47.58]},
    'resnet':{
        'baseline': [43.48, 18.79],
        'our': [55.15, 52.42]},
    'vgg':{
        'baseline': [41.06, 45.3],
        'our': [49.85, 53.79]},
    },
75:{'alexnet':{
         'baseline': [44.848, 51.667],
         'our': [48.182, 51.061]},
    'resnet':{
        'baseline': [38.182, 25],
        'our': [49.394, 49.545]},
    'vgg':{
        'baseline': [51.97, 48.182],
        'our': [45.909, 50]},
    },
50:{'alexnet':{
         'baseline': [39.091, 47.121],
         'our': [42.273, 43.485]},
    'resnet':{
        'baseline': [31.818, 20.909],
        'our': [46.818, 43.939]},
    'vgg':{
        'baseline': [42.727, 42.424],
        'our': [41.364, 50.606]},
    },
25:{'alexnet':{
         'baseline': [41.364, 40],
         'our': [37.879, 43.333]},
    'resnet':{
        'baseline': [35.303, 20],
        'our': [43.03, 40.909]},
    'vgg':{
        'baseline': [37.879, 39.697],
        'our': [38.788, 42.727]},
    },
15:{'alexnet':{
         'baseline': [37.576, 35.152],
         'our': [36.061, 34.394]},
    'resnet':{
        'baseline': [32.879, 20.303],
        'our': [40.152, 33.636]},
    'vgg':{
        'baseline': [35.909, 38.333],
        'our': [33.788, 36.061]},
    },
10:{'alexnet':{
         'baseline': [28.636, 29.848],
         'our': [29.697, 33.788]},
    'resnet':{
        'baseline': [27.727, 19.848],
        'our': [36.818, 32.273]},
    'vgg':{
        'baseline': [30.152, 35.909],
        'our': [30.455, 31.364]},
    },
5:{'alexnet':{
         'baseline': [29.091, 25.152],
         'our': [23.485, 24.545]},
    'resnet':{
        'baseline': [24.545, 23.939],
        'our': [29.242, 28.939]},
    'vgg':{
        'baseline': [21.061, 23.636],
        'our': [27.727, 29.545]},
    },
2:{'alexnet':{
         'baseline': [25, 27.727],
         'our': [25.455, 25.606]},
    'resnet':{
        'baseline': [26.667, 18.939],
        'our': [27.727, 22.727]},
    'vgg':{
        'baseline': [24.545, 23.636],
        'our': [27.121, 26.364]},
    },

1:{'alexnet':{
         'baseline': [21.061, 21.515],
         'our': [20, 19.394]},
    'resnet':{
        'baseline': [18.939, 17.424],
        'our': [23.333, 18.485]},
    'vgg':{
        'baseline': [22.121, 18.788],
        'our': [18.939, 25.152]},
    }

}

nets = ['alexnet', 'resnet', 'vgg']
sets = ['baseline', 'our']
steps = [75, 50, 25, 15, 10, 5, 2, 1]
steps = [100, 75, 50, 25, 10, 5, 1]
steps = [1, 5, 10, 25, 50, 75, 100]
steps_labels = [str(i)+'%' for i in steps]

b1 = np.zeros(len(steps))
b2 = np.zeros(len(steps))
o1 = np.zeros(len(steps))
o2 = np.zeros(len(steps))

for k in range(len(steps)):
    step = steps[k]
    for net in nets:
        b1_temp = r[step][net]['baseline'][0]
        b2_temp = r[step][net]['baseline'][1]
        o1_temp = r[step][net]['our'][0]
        o2_temp = r[step][net]['our'][1]
        b1[k] += b1_temp
        b2[k] += b2_temp
        o1[k] += o1_temp
        o2[k] += o2_temp

b1 /= len(nets)
b2 /= len(nets)
o1 /= len(nets)
o2 /= len(nets)
print ("*******")

i_1_1 = o1[0] - b1[0]
i_1_2 = o2[0] - b2[0]
i_100_1 = o1[-1] - b1[-1]
i_100_2 = o2[-1] - b2[-1]
print (i_1_1, i_1_2, i_100_1, i_100_2)





plt.figure(1)
plt.suptitle("Ablation study: reducing training data")
plt.plot(b1, label="Real, no pre", color='red')
plt.plot(b2, label="Real pre", color='orange')
plt.plot(o1, label="Quat+R2Hemo no pre", color='blue')
plt.plot(o2, label="Quat+R2Hemo pre", color="green")

plt.legend()
plt.xticks(range(len(steps)), steps_labels)
plt.ylabel('Mean test accuracy')
plt.xlabel('Training data amount')
plt.grid(color='gray', linestyle='dashed', alpha=0.3)

fig_name = "../ablation_study_reduced.png"
plt.savefig(fig_name, format = 'png', dpi=300)
