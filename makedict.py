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

'''

import sys
import numpy as np

def wrap_twopi(f):
    while f < 0: f += 2*np.pi
    while f > 2*np.pi: f -= 2*np.pi
    return f

if len(sys.argv) != 2:
    print ('usage:', sys.argv[0], 'params.txt')
    sys.exit(0)

infile = open(sys.argv[1], 'r')
two_pi = 2*np.pi
lower_bound = np.array([-1.0, -1.0, 0.0,    0.0,    0.0, 0.0, 0.0, 0.0])
upper_bound = np.array([ 1.0,  1.0, two_pi, two_pi, 4.0, 4.0, 2.0, 2.0])
ranges = upper_bound-lower_bound

var_names = 'uvrpltsh'
tol = 1e-4

print ('"gabor-list": [')
for line in infile:

    line = line.rstrip()
    if not line:
        break
    linemap = map(float,line.split(','))
    nums = np.array(list(linemap))
    nums[2] = wrap_twopi(nums[2])
    nums[3] = wrap_twopi(nums[3])    
    nums = np.clip(nums, lower_bound, upper_bound)
    assert(np.all(nums >= lower_bound) and np.all(nums <= upper_bound))
    u = np.round(511*(nums - lower_bound)/ranges).astype(int)

    outputs = []
    
    for j in range(4):
        na = u[2*j + 0]
        nb = u[2*j + 1]
        n = 512*na + nb
        #assert(n % 512 == nb)
        #assert(n / 512 == na)
        outputs.append(n)

    uvrp = 'vec4({},{},{},{})'.format(*nums[:4])
    ltsh = 'vec4({},{},{},{})'.format(*nums[4:])
    # print ('    k += gabor(p, vec4({:}.,{:}.,{:}.,{:}.));'.format(*outputs))
    # simpler space separated print output for Max
    print ('{}., {}., {}., {}.,'.format(*outputs))

print (']')
