#!/usr/bin/env python

'''Script to encode the list of 8-tuples emitted by the imfit program
and format them for use in the shader. The order emitted by imfit is:

  u: center x-coordinate of Gabor basis function
  v: center y-coordinate
  r: rotation angle of Gabor basis function
  p: phase of basis function
  l: wavelength (inverse frequency) of Gabor function
  t: width perpendicular to sinusoidal component
  s: width along sinusoidal component
  h: amplitude

The eight numbers are rescaled and quantized into the range [0, 511]
and encoded into four floating-point numbers (uv, rp, lt, sh).

Indexes of Gabor function parameters

GABOR_PARAM_U  = 0  # [-1, 1]
GABOR_PARAM_V  = 1  # [-1, 1]
GABOR_PARAM_R  = 2  # [0, 2*pi]
GABOR_PARAM_P0 = 3  # [0, 2*pi]
GABOR_PARAM_P1 = 4  # [0, 2*pi]
GABOR_PARAM_P2 = 5  # [0, 2*pi]
GABOR_PARAM_L  = 6  # [2.5*px, 4]
GABOR_PARAM_T  = 7  # [px, 4]
GABOR_PARAM_S  = 8  # [px, 2]
GABOR_PARAM_H0 = 9  # [0, 2]
GABOR_PARAM_H1 = 10 # [0, 2]
GABOR_PARAM_H2 = 11 # [0, 2]

GABOR_NUM_PARAMS = 12

GABOR_RANGE = np.array([
    [ -1, 1 ],
    [ -1, 1 ],
    [ -np.pi, np.pi ],
    [ -np.pi, np.pi ],
    [ -np.pi, np.pi ],
    [ -np.pi, np.pi ],
    [ 0, 4 ],
    [ 0, 4 ],
    [ 0, 2 ],
    [ 0, 2 ],
    [ 0, 2 ],
    [ 0, 2 ] ])

'''

import sys
import numpy as np
import json

def wrap_twopi(f):
    while f < 0: f += 2*np.pi
    while f > 2*np.pi: f -= 2*np.pi
    return f

if len(sys.argv) != 3:
    print ('usage:', sys.argv[0], 'params.txt', 'outparams.txt')
    sys.exit(0)

infile = open(sys.argv[1], 'r')
outfilename = sys.argv[2]
two_pi = 2*np.pi
lower_bound = np.array([-1.0, -1.0, 0.0,    0.0,   0.0,    0.0,    0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
upper_bound = np.array([ 1.0,  1.0, two_pi, two_pi,two_pi, two_pi, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0])
ranges = upper_bound-lower_bound

var_names = 'uvrpltsh'
tol = 1e-4
outfile = np.array([]);

for line in infile:

    line = line.rstrip()
    if not line:
        break
    linemap = map(float,line.split(','))
    nums = np.array(list(linemap))
    nums[2] = wrap_twopi(nums[2])
    nums[3] = wrap_twopi(nums[3])
    nums[4] = wrap_twopi(nums[4])
    nums[5] = wrap_twopi(nums[5])    
    nums = np.clip(nums, lower_bound, upper_bound)
    assert(np.all(nums >= lower_bound) and np.all(nums <= upper_bound))
    u = np.round(511*(nums - lower_bound)/ranges).astype(int)

    gu = np.array([u[0],u[1],u[2],u[3],u[6],u[7],u[8],u[9]])
    cu = np.array([u[4],u[5],u[10],u[11]])

    outputs = np.array([])
    
    for j in range(4):
        na = gu[2*j + 0]
        nb = gu[2*j + 1]
        n = 512*na + nb
        #assert(n % 512 == nb)
        #assert(n / 512 == na)
        outputs = np.append(outputs,n)
    outfile = np.concatenate((outfile,outputs,cu))

with open(outfilename, "w") as txt_file:
    json.dump(outfile.tolist(),txt_file,separators=', ')