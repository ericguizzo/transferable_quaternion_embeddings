import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
'''
r_vgg = [51.9, 56.7, 23.81, 23.81, 12.14]
p_vgg = [49.52, 44.52, 23.81, 23.81, 12.14]
pp_vgg = [41.9, 52.62, 45.95, 18.81, 17.14]

r_alex = [54.29, 45.24, 43.57, 23.81, 23.81]
p_alex = [60.0, 48.57, 52.62, 23.81, 0.08]
pp_alex = [52.62, 51.67, 36.19, 13.81, 7.38]

r_res = [60.24, 58.57, 56.9, 23.81, 24.54]
p_res = [50.95, 52.62, 45.95, 13.81, 11.43]
pp_res = [59.28, 52.62, 56.66, 38.57, 14.04]


r_vgg = [51.9, 56.7, 23.81, 23.81, 23.81, 23.81, 12.14]
p_vgg = [49.52, 44.52, 23.81, 23.81, 23.81, 23.81, 12.14]
pp_vgg = [41.9, 52.62, 45.95, 18.81, 33.57, 18.81, 17.14]

r_alex = [54.29, 45.24, 43.57, 23.81, 23.81, 23.81, 23.81]
p_alex = [60.0, 48.57, 52.62, 23.81, 23.81, 23.81, 0.08]
pp_alex = [52.62, 51.67, 36.19, 33.57, 13.81, 11.43, 7.38]

r_res = [60.24, 58.57, 56.9, 39.52, 23.81, 26.19, 24.54]
p_res = [50.95, 52.62, 45.95, 35.23, 13.81, 13.81, 11.43]
pp_res = [59.28, 52.62, 56.66, 45.23, 38.57, 33.57, 14.04]

r_vgg = [51.9, 56.7, 23.81, 23.81, 23.81, 12.14]
p_vgg = [49.52, 44.52, 23.81, 23.81, 23.81, 12.14]
pp_vgg = [41.9, 52.62, 45.95, 18.81, 18.81, 17.14]

r_alex = [54.29, 45.24, 43.57, 23.81, 23.81, 23.81]
p_alex = [60.0, 48.57, 52.62, 23.81, 23.81, 0.08]
pp_alex = [52.62, 51.67, 36.19, 33.57, 11.43, 7.38]

r_res = [60.24, 58.57, 56.9, 39.52, 26.19, 24.54]
p_res = [50.95, 52.62, 45.95, 35.23, 13.81, 11.43]
pp_res = [59.28, 52.62, 56.66, 45.23, 33.57, 14.04]
'''

r_vgg = [60, 51.9, 56.7, 23.81, 23.81, 23.81, 12.14]
p_vgg = [51.19, 49.52, 44.52, 23.81, 23.81, 23.81, 12.14]
pp_vgg = [59.29, 41.9, 52.62, 45.95, 18.81, 18.81, 17.14]

r_alex = [62.62, 54.29, 45.24, 43.57, 23.81, 23.81, 23.81]
p_alex = [61.19, 60.0, 48.57, 52.62, 23.81, 23.81, 0.08]
pp_alex = [57.62, 52.62, 51.67, 36.19, 33.57, 11.43, 7.38]

r_res = [61.9, 60.24, 58.57, 56.9, 39.52, 26.19, 24.54]
p_res = [53.58, 50.95, 52.62, 45.95, 35.23, 13.81, 11.43]
pp_res = [60, 59.28, 52.62, 56.66, 45.23, 33.57, 14.04]

r = [r_vgg, r_alex, r_res]
p = [p_vgg, p_alex, p_res]
pp = [pp_vgg, pp_alex, pp_res]


r_means = np.zeros(len(r_vgg))
p_means = np.zeros(len(r_vgg))
pp_means = np.zeros(len(r_vgg))

positions = range(len(r_vgg))
labels = ["100%", "75%", "50%", "25%", "10%", "5%", "1%"]

for s in r:
    for i in range(len(r_vgg)):
        r_means[i] += s[i] / 3
for s in p:
    for i in range(len(r_vgg)):
        p_means[i] += s[i] / 3
for s in pp:
    for i in range(len(r_vgg)):
        pp_means[i] += s[i] / 3

print (r_means)
print (p_means)
print (pp_means)

'''
to_pop = [2]
for i in [r_means, p_means, pp_means]:
    for p in to_pop:

        i.pop(p)
'''

impr_p = np.subtract(r_means, p_means)
impr_pp = np.subtract(r_means, pp_means)
plt.figure(1)
plt.suptitle("Ablation study: reduced data")
#plt.plot(r_means)
plt.plot(impr_p)
plt.plot(impr_pp)
plt.axhline(y=0, color='black')
#plt.plot(p_means)
#plt.plot(pp_means)
plt.legend(['Impr. over baseline', 'Impr. over pretrained baseline'])
plt.xticks(positions, labels)
plt.ylabel('Mean improvement (percentage points)')
plt.xlabel('Data amount')
plt.show()
