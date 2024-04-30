import os

import iscan

##### set parameters #####
filename = os.getcwd() + r"\iscan_test.csv"


def main(filename):
    metadata, frames = iscan.parse(filename)


##### run #####
main(filename)
