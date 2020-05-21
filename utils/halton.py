"""Halton low discrepancy sequence.

This snippet implements the Halton sequence following the generalization of
a sequence of *Van der Corput* in n-dimensions.

---------------------------

MIT License

Copyright (c) 2017 Pamphile Tupui ROY

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import tensorflow as tf


def primes_from_2_to(n):

	sieve = np.ones(n // 3 + (n % 6 == 2), dtype=np.bool)
	for i in range(1, int(n ** 0.5) // 3 + 1):
		if sieve[i]:
			k = 3 * i + 1 | 1
			sieve[k * k // 3::2 * k] = False
			sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
	return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


def van_der_corput(n_sample, base=2):

	sequence = []
	for i in range(n_sample):
		n_th_number, denom = 0., 1.
		while i > 0:
			i, remainder = divmod(i, base)
			denom *= base
			n_th_number += remainder / denom
		sequence.append(n_th_number)

	return sequence


def halton(dim, n_sample):

	big_number = 10
	while 'Not enought primes':
		base = primes_from_2_to(big_number)[:dim]
		if len(base) == dim:
			break
		big_number += 1000

	sample = [van_der_corput(n_sample + 1, dim) for dim in base]
	sample = np.stack(sample, axis=-1)[1:]

	return sample


def halton_batch(batch_size, dim, n_sample, extent=1.5, z_offset=0.):

	if z_offset == 0:
		return tf.cast(tf.constant([halton(dim, n_sample) for _ in range(batch_size)]), tf.float32) * (extent / 0.5)
	else:
		arr = np.array([halton(dim, n_sample) for _ in range(batch_size)], dtype=np.float32) * (extent / 0.5)
		arr[:, :, 2] = z_offset
		arr[:, :, :2] = arr[:, :, :2] - np.mean(arr[:, :, :2], axis=1).reshape(arr.shape[0], -1, 2)
		return tf.constant(arr)
