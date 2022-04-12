import argparse
import os
import gdown

'''
Download our RH-emo pre-trained weights
Command line arguments define which task to download and where to put the checkpoint file.
'''



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default="models")
    args = parser.parse_args()

    #file_id = '1vCX0KHW44Q9plKTdkgyKZRcyjfgVA7jX'
    file_id = '1vCX0KHW44Q9plKTdkgyKZRcyjfgVA7jX'
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    output_path = os.path.join(args.output_path, "pretrained_RHemo")

    gdown.download(id=file_id, output=output_path, quiet=False)

    #print ('Pre-trained weights successfully downloaded')
