import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

b_vgg = 51.19
b_alex = 61.19
b_res = 53.58

bp_vgg = 59.29
bp_alex = 57.62
bp_res = 60

r_vgg = 60
r_alex = 62.62
r_res = 61.9


vgg_1 = 61.6
vgg_2 = 51.9
vgg_3 = 61.6
vgg_4 = 60
vgg_5 = 53.8

res_1 = 59.52
res_2 = 53.57
res_3 = 58.85
res_4 = 61.19
res_5 = 63.57

alex_1 = 58.57
alex_2 = 44.52
alex_3 = 58.57
alex_4 = 61.9
alex_5 = 60.2

r2he = ((r_vgg-b_vgg) + (r_alex-b_alex) + (r_res-b_res)) / 3
recon_only = (((vgg_1-b_vgg) + (alex_1-b_alex) + (res_1-b_res)) / 3) - r2he
no_recon_only_classification = (((vgg_2-b_vgg) + (alex_2-b_alex) + (res_2-b_res)) / 3) - r2he
recon_only_discrete_class = (((vgg_3-b_vgg) + (alex_3-b_alex) + (res_3-b_res)) / 3) - r2he
recon_only_vad = (((vgg_4-b_vgg) + (alex_4-b_alex) + (res_4-b_res)) / 3) - r2he
real_model = (((vgg_5-b_vgg) + (alex_5-b_alex) + (res_5-b_res)) / 3) - r2he

p_r2he = ((r_vgg-bp_vgg) + (r_alex-bp_alex) + (r_res-bp_res)) / 3
p_recon_only = (((vgg_1-bp_vgg) + (alex_1-bp_alex) + (res_1-bp_res)) / 3) - r2he
p_no_recon_only_classification = (((vgg_2-bp_vgg) + (alex_2-bp_alex) + (res_2-bp_res)) / 3) - r2he
p_recon_only_discrete_class = (((vgg_3-bp_vgg) + (alex_3-bp_alex) + (res_3-bp_res)) / 3) - r2he
p_recon_only_vad = (((vgg_4-bp_vgg) + (alex_4-bp_alex) + (res_4-bp_res)) / 3) - r2he
p_real_model = (((vgg_5-bp_vgg) + (alex_5-bp_alex) + (res_5-bp_res)) / 3) - r2he


print ("OUR APPROACH: ", r2he)
print ("reconstruction only: ", recon_only)
print ("no recon, only classification (dicr. + vad): ", no_recon_only_classification)
print ("recon + only discrete class.: ", recon_only_discrete_class)
print ("recon + only vad: ", recon_only_vad)
print ("real_model: ",real_model)

y = [real_model, recon_only, no_recon_only_classification, recon_only_discrete_class, recon_only_vad]
#y = [recon_only_vad, recon_only_discrete_class, no_recon_only_classification, recon_only, real_model, r2he]
#y1 = [p_recon_only_vad, p_recon_only_discrete_class, p_no_recon_only_classification, p_recon_only, p_real_model, p_r2he]
y1 = [p_real_model, p_recon_only, p_no_recon_only_classification, p_recon_only_discrete_class, p_recon_only_vad]

y = [np.round(x, decimals=1) for x in y]
y1 = [np.round(x, decimals=1) for x in y1]

x = pd.Series(y)

#plt.style.use('ggplot')

#y_labels = ["OUR APPROACH", "only reconstruction", "only emotion", "no vad", "no discrete", "real r2Hemo"]
#y_labels = ["no discrete" , "no vad", "only \n emotion", "only \n reconstruction", "real \n r2Hemo", "quat \n r2Hemo"]
y_labels = ["real \n r2Hemo", "only \n recon","only \n emo","no \n vad", "no \n discrete"]

labels = y_labels
men_means = y
women_means = y1

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.barh(x, y, width, label='R2Hemo variants')
#rects2 = ax.bar(x + width/2, women_means, width, label='Impr. over pretrained baseline')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Performance drop (percentage points)')
ax.set_title('Ablation Study: removing R2Hemo components')
ax.set_yticks(x)
ax.set_yticklabels(labels)

ax.set_xlim(-13,0.5)

plt.axvline(x=0, color='black', label='Baseline: Full R2Hemo')
ax.legend()

for i, v in enumerate(y):
    ax.text(v-1.1, i-0.055, str(v))

ax.grid(color='gray', linestyle='dashed', alpha=0.3)


#plt.show()




fig_name = "../ablation_study.png"
plt.savefig(fig_name, format = 'png', dpi=300)
