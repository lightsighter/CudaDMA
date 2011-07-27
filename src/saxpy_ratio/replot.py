
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def main():
  import pickle
  output = open('data.pkl', 'r')
  data = pickle.load(output)
  print data

  lbl_sz = 12
  styles = ['-','--',':']
  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  ax2 = ax1.twinx()
  for name, vals, style in zip(data.keys(),data.values(),styles):
    xs, ys_gbs, ys_gflops = vals
    ax1.plot(xs,ys_gbs,'b'+style, label=name)
    ax2.plot(xs, ys_gflops, 'r'+style, label=name)

  ax1.set_xlabel('B/FLOP', size=lbl_sz)
  ax1.set_ylabel('GB/s', size=lbl_sz, color='b')
  for t1 in ax1.get_yticklabels():
    t1.set_color('b')
  ax2.set_ylabel('GFLOP/s', size=lbl_sz, color='r')
  for t1 in ax2.get_yticklabels():
    t1.set_color('r')
  ax1.set_xscale('log', basex=2)
  ax2.set_xscale('log', basex=2)
  ax1.invert_xaxis()
  ax2.invert_xaxis()
  ax1.set_ylim(0)
  ax2.set_ylim(0)
  leg = plt.legend(loc=7)
  for t in leg.get_texts():
    t.set_fontsize('small')
  for l in leg.get_lines():
    l.set_color('k')
  plt.show()

if __name__ == "__main__":
    main()
