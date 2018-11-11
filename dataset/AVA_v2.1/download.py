import os

train_eval_url = 'https://s3.amazonaws.com/ava-dataset/trainval/'
test_url = 'https://s3.amazonaws.com/ava-dataset/test/'
train_file = '/home/yikang.liao/workspace/cvpr19/dataset/AVA_v2.1/ava_file_names_trainval_v2.1.txt'
test_file = '/home/yikang.liao/workspace/cvpr19/dataset/AVA_v2.1/ava_file_names_test_v2.1.txt'


def downlaod(url, infile, outdir):
    with open(infile) as fp:
        filenames = [line.strip() for line in fp.readlines()]

    for filename in filenames:
        # will automatically skip existing files
        cmd = "wget -c %s -P %s" % (url + filename, outdir)
        os.system(cmd)

if __name__ == '__main__':

    downlaod(train_eval_url, train_file, '/home/yikang.liao/share/dataset/AVA/train-validation/')
    downlaod(test_url, test_file, '/home/yikang.liao/share/dataset/AVA/test/')


