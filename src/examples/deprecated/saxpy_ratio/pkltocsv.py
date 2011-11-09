
import sys
import pickle
import csv

def main(iname):
  ifile = open(iname, 'r')
  data = pickle.load(ifile)
  writer = csv.writer(open(iname.rstrip('pkl')+'csv', 'wb'))
  for name, vals in data.iteritems():
    lbls = ['ratio','gbs','gflops'] 
    for lbl, valarr in zip(lbls, vals):
      writer.writerow([name,lbl]+valarr)

if __name__ == "__main__":
  main(sys.argv[1])
