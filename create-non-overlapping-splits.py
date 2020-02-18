from utils import ensure_empty_directory_exists
import argparse
import fnmatch
import random
import os


test_synthnames = set([
    'ENSTDkCl',
    'ENSTDkAm',
])

train_synthnames = set([
    'StbgTGd2',
    'SptkBGCl',
    'SptkBGAm',
    'AkPnStgb',
    'AkPnCGdD',
    'AkPnBsdf',
    'AkPnBcht'
])


def desugar(c):
    prefix = 'MAPS_MUS-'
    last = c[::-1].find('_')
    pid = c[len(prefix):(-last - 1)]
    return prefix, last, pid


def collect_all_piece_ids(base_dir, synthnames):
    pids = set()
    for synthname in synthnames:
        for base, dirs, files in os.walk(os.path.join(base_dir, synthname)):
            candidates = fnmatch.filter(files, '*MUS*')
            if len(candidates) > 0:
                for c in candidates:
                    _, _, pid = desugar(c)
                    pids.add(pid)

    return pids


def collect_all_filenames(base_dir, synthnames, include):
    filenames = set()
    for synthname in synthnames:
        for base, dirs, files in os.walk(os.path.join(base_dir, synthname)):
            candidates = fnmatch.filter(files, '*MUS*')
            if len(candidates) > 0:
                for c in candidates:
                    _, _, pid = desugar(c)
                    if pid in include:
                        path, ext = os.path.splitext(c)
                        filenames.add(os.path.join(base, path))
    return list(filenames)


def write_pairs(filename, lines):
    pairs = []
    for line in lines:
        pairs.append('{}.wav,{}.mid'.format(line, line))
    with open(filename, 'w') as f:
        f.writelines('\n'.join(pairs) + '\n')


def main():
    random.seed(155853)

    parser = argparse.ArgumentParser(description='create non-overlapping splits')
    parser.add_argument('maps_base_directory', help='path must be relative to the working directory')
    args = parser.parse_args()
    print('#' * 30)
    print('generating the splits as defined in:')
    print('Curtis Hawthorne, Erich Elsen, Jialin Song, Adam Roberts,')
    print('Ian Simon, Colin Raffel, Jesse Engel, Sageev Oore, Douglas Eck,')
    print('"Onsets and Frames: Dual-Objective Piano Transcription",')
    print('Proceedings of the 19th International Society for Music Information Retrieval')
    print('Conference ISMIR 2018, September 23-27, 2018.')
    print('#' * 30)
    print('these splits were also used in:')
    print('Rainer Kelz, Sebastian Böck and Gerhard Widmer,')
    print('"Deep Polyphonic ADSR Piano Note Transcription"')
    print('ICASSP 2019 - 2019 IEEE International Conference on Acoustics,')
    print('Speech and Signal Processing (ICASSP), Brighton, United Kingdom, 2019, pp. 246-250.')
    print('#' * 30)
    print('IMPORTANT:')
    print('if you train the old 2016 models on these splits, you will notice')
    print('a performance drop by about ~3 percentage points for the AllConv2016 model.')
    print('for much better results on MAPS, train models from the ICASSP19 paper!')
    print('https://github.com/rainerkelz/ICASSP19')
    print('#' * 30)

    train_pids = collect_all_piece_ids(args.maps_base_directory, train_synthnames)
    test_pids = collect_all_piece_ids(args.maps_base_directory, test_synthnames)

    print('len(train_pids)', len(train_pids))
    print('len(test_pids)', len(test_pids))

    train_filenames = sorted(collect_all_filenames(
        args.maps_base_directory,
        train_synthnames,
        train_pids - test_pids
    ))
    test_filenames = sorted(collect_all_filenames(
        args.maps_base_directory,
        test_synthnames,
        test_pids
    ))

    # we're validating on a subset of the trainset!
    # this is going to tell us **how close we are to learning the trainset by heart**...
    # ... and be a **bad estimate of generalization error** ...
    valid_filenames = random.sample(train_filenames, 10)

    print('len(train_filenames)', len(train_filenames))
    print('len(valid_filenames)', len(valid_filenames))
    print('len(test_filenames)', len(test_filenames))

    dirname = 'non-overlapping'
    ensure_empty_directory_exists(dirname)

    write_pairs(os.path.join(dirname, 'train'), train_filenames)
    write_pairs(os.path.join(dirname, 'valid'), valid_filenames)
    write_pairs(os.path.join(dirname, 'test'), test_filenames)


if __name__ == '__main__':
    main()
