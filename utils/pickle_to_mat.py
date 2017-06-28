import sys

import pickle

def main():
    filename = sys.argv[1]
    fd = open(filename, 'rb')
    pkl = pickle.load(fd)
    fd.close()

    dic = {}
    dic['gram'] = pkl['']
    pass


if __name__ == "__main__":
    main()