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


def collect_all_filenames(base_dir, synthnames):
    filenames = set()
    for synthname in synthnames:
        for base, dirs, files in os.walk(os.path.join(base_dir, synthname)):
            candidates = fnmatch.filter(files, '*MUS*')
            if len(candidates) > 0:
                for c in candidates:
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
    print('generating the splits called "Configuration II", as defined in:')
    print('Siddharth Sigtia, Emmanouil Benetos, Simon Dixon,')
    print('"An End-to-End Neural Network for Polyphonic Piano Music Transcription",')
    print('IEEE/ACM Trans. Audio Speech & Language Processing, vol. 24, no. 5, pp. 927-939, 2016.')
    print('#' * 30)
    print('these are the splits necessary to reproduce the numbers for "Configuration II" in the paper:')
    print('Rainer Kelz, Matthias Dorfer, Filip Korzeniowski,')
    print('Sebastian Böck, Andreas Arzt, Gerhard Widmer')
    print('"On the Potential of Simple Framewise Approaches to Piano Transcription",')
    print('Proceedings of the 17th International Society for Music Information Retrieval Conference,')
    print('ISMIR 2016, New York City, United States, August 7-11, 2016')
    print('#' * 30)

    train_filenames = sorted(collect_all_filenames(
        args.maps_base_directory,
        train_synthnames
    ))
    test_filenames = sorted(collect_all_filenames(
        args.maps_base_directory,
        test_synthnames
    ))

    valid_filenames = random.sample(train_filenames, 30)
    train_filenames = list(set(train_filenames) - set(valid_filenames))

    print('len(train_filenames)', len(train_filenames))
    print('len(valid_filenames)', len(valid_filenames))
    print('len(test_filenames)', len(test_filenames))

    dirname = 'configuration-II'
    ensure_empty_directory_exists(dirname)

    write_pairs(os.path.join(dirname, 'train'), train_filenames)
    write_pairs(os.path.join(dirname, 'valid'), valid_filenames)
    write_pairs(os.path.join(dirname, 'test'), test_filenames)


if __name__ == '__main__':
    main()
