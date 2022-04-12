from __future__ import print_function
import numpy as np
import sys, os
import xlsxwriter
import argparse
'''
Automatically compile an xls spreadsheet from training results contained in a
npy dict
'''

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str)
parser.add_argument('--output_name', type=str)
parser.add_argument('--profile', type=str, default='profile_autoencoder')
parser.add_argument('--column_width', type=int, default=18)
parser.add_argument('--highlight', type=str, default='colorscale')  #colorscale or minmax


args = parser.parse_args()

profile_autoencoder = ['train_loss_total', 'val_loss_total', 'test_loss_total', '/',
                       'train_loss_recon', 'val_loss_recon', 'test_loss_recon', '/',
                       'train_loss_emo', 'val_loss_emo', 'test_loss_emo', '/',
                       'train_loss_acc', 'val_loss_acc', 'test_loss_acc', '/',
                       'train_loss_at', 'val_loss_at', 'test_loss_at', '/',
                       ]
profile_autoencoder_vad = ['train_loss_total', 'val_loss_total', 'test_loss_total', '/',
                       'train_loss_recon', 'val_loss_recon', 'test_loss_recon', '/',
                       'train_loss_emo', 'val_loss_emo', 'test_loss_emo', '/',
                       'train_loss_acc', 'val_loss_acc', 'test_loss_acc', '/',
                       'train_loss_at', 'val_loss_at', 'test_loss_at', '/',
                       'train_loss_vad', 'val_loss_vad', 'test_loss_vad', '/',
                       'train_loss_valence', 'val_loss_valence', 'test_loss_valence', '/',
                       'train_loss_arousal', 'val_loss_arousal', 'test_loss_arousal', '/',
                       'train_loss_dominance', 'val_loss_dominance', 'test_loss_dominance', '/',
                       'train_loss_acc_valence', 'val_loss_acc_valence', 'test_loss_acc_valence', '/',
                       'train_loss_acc_arousal', 'val_loss_acc_arousal', 'test_loss_acc_arousal', '/',
                       'train_loss_acc_dominance', 'val_loss_acc_dominance', 'test_loss_acc_dominance', '/',
                       ]

profile_emotion_recognition = ['train_loss', 'val_loss', 'test_loss', '/',
                       'train_acc', 'val_acc', 'test_acc', '/',
                       ]


profile = eval(args.profile)

workbook = xlsxwriter.Workbook(args.output_name)
worksheet = workbook.add_worksheet()

#define formats
header_format = workbook.add_format({'align': 'center', 'bold':True,'border': 4, 'bg_color':'green'})
header_format_white = workbook.add_format({'align': 'center', 'bold':True,'border': 4, 'bg_color':'white'})
header_format_yellow = workbook.add_format({'align': 'center', 'bold':True,'border': 4, 'bg_color':'yellow'})
header_format_red = workbook.add_format({'align': 'center', 'bold':True,'border': 4, 'bg_color':'red'})

values_format = workbook.add_format({'align': 'center','border': 1})
blank_format = workbook.add_format({'align': 'center','border': 1})
bestvalue_format = workbook.add_format({'align': 'center', 'bold':True, 'border':1, 'bg_color':'green'})
bestvalue_format_red = workbook.add_format({'align': 'center', 'bold':True, 'border':1, 'bg_color':'red'})

worksheet.write(0, 0, 'ID', header_format)
worksheet.write(0, 1, 'comment_1', header_format_red)
worksheet.write(0, 2, 'comment_2', header_format_red)
offset_h = 3
offset_v = 0
#write column names
for i in range(len(profile)):
    if profile[i] == '/':
        worksheet.write(0, i+offset_h, profile[i],header_format_white)
    else:
        worksheet.write(0, i+offset_h, profile[i],header_format_yellow)

worksheet.set_column(1,len(profile)+offset_h-1,args.column_width)

#iterate experiments
contents = os.listdir(args.input_folder)
contents = list(filter(lambda x: '.npy' in x, contents))
num_exps = len(contents)
out_name = os.path.join(args.input_folder, args.output_name)
for i in contents:
    temp_path = os.path.join(args.input_folder, i)
    dict = np.load(temp_path, allow_pickle=True).item()
    #print (dict[0].keys())
    exp_ID = i.split('_')[-1].split('.')[0][3:]
    curr_row = int(exp_ID)+offset_v
    worksheet.write(curr_row, 0, exp_ID,values_format)
    worksheet.write(curr_row, 1, dict[0]['parameters']['comment_1'],values_format)
    worksheet.write(curr_row, 2, dict[0]['parameters']['comment_2'],values_format)
    #write results:
    for i in range(len(profile)):
        #print (profile[i

        if profile[i] == '/':
            worksheet.write(curr_row, i+offset_h, '|',values_format)
        else:
            value = dict['summary'][profile[i]]
            if type(value)==list:value=value[0]
            value = np.round(value, decimals=5)
            worksheet.write(curr_row, i+offset_h, value,values_format)
#highlight best results
for i in range(len(profile)):
    curr_column = offset_h + i
    if profile[i] != '/':
        if args.highlight == 'minmax':
            worksheet.conditional_format(offset_v, curr_column, offset_v+num_exps, curr_column,
                                     {'type': 'bottom','value': '1','format': bestvalue_format})
            worksheet.conditional_format(offset_v, curr_column, offset_v+num_exps, curr_column,
                                    {'type': 'top','value': '1','format': bestvalue_format_red})
        elif args.highlight == 'colorscale':
            worksheet.conditional_format(offset_v, curr_column, offset_v+num_exps, curr_column,
                                    {'type': '3_color_scale'})

workbook.close()
