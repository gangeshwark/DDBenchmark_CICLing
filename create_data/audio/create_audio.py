from os import system
import io, os
import pandas as pd


def extract_features(input_file, output_file):
    """
    /projects/TensorflowProjects/DeceptionDetection/extract_scripts/audio/opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract
    """
    cmd = '/projects/TensorflowProjects/DeceptionDetection/extract_scripts/audio/opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract -C /projects/TensorflowProjects/DeceptionDetection/extract_scripts/audio/opensmile-2.3.0/config/IS13_ComParE.conf -I %s -O %s' % (
        input_file, output_file)
    print(cmd)
    system(cmd)


def get_features(input):
    with io.open(input) as f:
        l = f.readlines()[-1].split(',')
        print(len(l))
        l = l[1:-1]
        print(len(l))
        assert len(l) == 6373
        return l


if __name__ == '__main__':
    data = pd.DataFrame(columns=['file', 'features'])
    td = 'Deceptive/'
    x = 0
    for i in range(61):
        fol = 'trial_lie_%03d/' % (i + 1)
        folder = '/projects/TensorflowProjects/contextual-sentiment-analysis/RealLifeDeceptionDetection.Split.2017/'
        out_fol = '/home/gangeshwark/PycharmProjects/DDBenchmark_CICLing/create_data/audio/all_features/'
        folder += td
        folder += fol
        # folder += 'Audio_Noiseless_Splits/'
        print(folder)
        for s, sub, file in os.walk(folder):
            for f in sorted(file)[:]:
                if os.path.isfile(folder + f) and f.endswith('noiseless.wav'):
                    i = folder + f
                    o = out_fol + f[:-4] + '.csv'
                    extract_features(i, o)
                    feats = get_features(o)
                    d = pd.DataFrame({'file': fol[:-1], 'features': [feats]})
                    data = data.append(d)
                    x += 1
    data.to_csv('features_%s.csv' % td[:-1], index=False, header=True, columns=['file', 'features'])
    data.to_pickle('features_%s.pkl' % td[:-1])
