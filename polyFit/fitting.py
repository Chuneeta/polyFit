from astropy.io import fits
import numpy as np

class Fit(object):
	def __init__(self, calfits):
		"""
		Object that takes in .fits containing the calibration solutions file readable by astropy
		and initializes them as global varaibles
		- calfits : .fits file containing the calibration solutions
		"""
		self.calfits = calfits
		
	def polynomial_fit(self):
		pass