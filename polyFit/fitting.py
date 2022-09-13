from astropy.io import fits
import statsmodels.api as sm
from numpy.polynomial import Polynomial
import numpy as np

# Number of coarse bands
Nband = 24
polstrs = ['XX', 'XY', 'YX', 'YY']

class Fit(object):
	def __init__(self, calfits):
		"""
		Object that takes in .fits containing the calibration solutions
		file readable by astropy
		and initializes them as global varaibles
		- calfits : .fits file containing the calibration solutions
		"""
		self.calfits = calfits
		with fits.open(self.calfits) as hdus:
			data = hdus['SOLUTIONS'].data
			self.gain_array = data[:, :, :, ::2] + data[:, :, :, 1::2] * 1j
			_sh = self.gain_array.shape
			self.convergence = hdus['RESULTS'].data
			self.Ntimes = _sh[0]
			self.Nants = _sh[1]
			self.Nfreq = _sh[2]
			self.Npols = _sh[3]
			self.Nband = Nband
			assert self.Nfreq % Nband == 0
			self.Nchan = self.Nfreq // self.Nband
			
	def unflagged_channels(self, edge_width, clip_width):
		# getting the unflagged channels
		channels = np.arange(self.Nchan)
		channels = np.delete(channels, self.Nchan // 2) # centre width
		channels = channels[edge_width + clip_width: -edge_width + clip_width]
		all_channels = []
		for bd in range(self.Nband):
			for ch in channels:
				all_channels.append(ch + (bd * self.Nchan))
		return all_channels

	def _construct_xdata_fit(self, data_x, order):
		# construct xdata for OLS fit
		# statsmodels like to have 2D arrays, even if one axis is 
		# one dimensional
		xdata = data_x[:, np.newaxis]
		xdata_fit = np.empty((len(data_x), order))
		for ord in range(1, order + 1):
			xdata_fit[:, ord - 1] = xdata.flatten() ** ord
		# Add a constant for fitting to complete the polynomial model
		xdata_fit = sm.add_constant(xdata_fit)
		return xdata_fit

	def _polynomial_fit(self, data_x, data_y, order):
		# setup the polynomial coordinates, with enough space to 
		# hold the required orders
		xdata = self._construct_xdata_fit(data_x, order)
		ydata = data_y[:, np.newaxis]
		# Fit the model with Ordinary Least squares as we don't have any errors
		model = sm.OLS(ydata, xdata).fit()
		return model

	def _polynomial_numpy(self, data_x, data_y, order, weights):
		data_real = np.real(data_y)
		data_imag = np.imag(data_y)
		fit_real = Polynomial.fit(data_x, data_real, deg=order, w=weights)
		fit_imag = Polynomial.fit(data_x, data_imag, deg=order, w=weights)
		return fit_real, fit_imag

	def fit_data(self, order, weights=None, edge_width=2, clip_width=0):
		_sh = self.gain_array.shape
		self.poly_array = np.zeros_like(self.gain_array, dtype=complex)
		self.bic_array = np.zeros((_sh[0], _sh[1], _sh[3]))
		unflagged_chs = self.unflagged_channels(edge_width, clip_width)
		cols = []
		times , antennas = [], []
		polarizations, complx = [], []
		coeffs = []
		for t in range(self.Ntimes):
			# filter any channels where result is NaN
			good_chs = np.array([ch for ch in unflagged_chs if not np.isnan(self.convergence[t, ch])])
			# Remove flagged tiles which are nan at first unflagged frequency and pol
			good_tiles = np.argwhere(~np.isnan(self.gain_array[t, :, good_chs[0], 0]))
			for ant in good_tiles:
				for pol in range(self.Npols):
					model_real = self._polynomial_fit(good_chs, np.real(self.gain_array[t, ant, good_chs, pol]), order=order)
					model_imag = self._polynomial_fit(good_chs, np.imag(self.gain_array[t, ant, good_chs, pol]), order=order)
					xpredicted = self._construct_xdata_fit(np.arange(self.Nfreq), order)
					self.poly_array[t, ant, : , pol] = model_real.predict(xpredicted) + 1j * model_imag.predict(xpredicted)
					self.bic_array[t, ant, pol] = max([model_real.bic, model_imag.bic])
					# writing to fits colums
					for i, cp in enumerate(['REAL', 'IMAG']):
						times.append(t)
						antennas.append(ant[0])
						polarizations.append(polstrs[pol])
						complx.append(cp)
						coeffs.append(model_real.params.tolist()) if i % 2 == 0 else coeffs.append(model_imag.params.tolist())

		# asssigning nans to bad antennas
		self.bic_array[np.where(self.bic_array == 0.)] = np.nan
		coeffs = np.array(coeffs)
		_sh = coeffs.shape
		cols.append(fits.Column(name = 'Timestamp', format='I', array=np.array(times)))
		cols.append(fits.Column(name = 'Antenna', format='I', array=np.array(antennas)))
		cols.append(fits.Column(name = 'Pol', format='2A', array=np.array(polarizations)))
		cols.append(fits.Column(name = 'Complex', format='4A', array=np.array(complx)))
		for cf in range(_sh[1]):
			cols.append(fits.Column(name = 'Coeff{}'.format(cf), format='E', array=coeffs[:, cf]))
		self.poly_params = fits.BinTableHDU.from_columns(cols, name='FIT_COEFFS')

	def bic(self):
		return np.nanmin(self.bic_array)

	def reduced_chisq(self, order):
		try:
			chisq_real = np.nansum((np.real(self.gain_array) - np.real(self.poly_array)) ** 2 / np.nanstd(np.real(gain_array)))
			chisq_imag = np.nansum((np.imag(self.gain_array) - np.real(self.poly_array)) ** 2 / np.nanstd(np.imag(gain_array))) 
			divisor = len(self.gain_array.flatten()) - (order + 1)
			return np.nanmax([chisq_real / divisor , chisq_imag / divisor])
		except AttributeError:
			print ('Fit Object has no fitting attribute. Fitting of the data is required.')
			return None

	def mean_square_error(self):
		try:
			mse_real = 1 / self.Nfreq * np.nansum((np.real(self.gain_array) - np.real(self.poly_array)) ** 2)
			mse_imag = 1 / self.Nfreq * np.nansum((np.imag(self.gain_array) - np.imag(self.poly_array)) ** 2)
			return max([mse_real, mse_imag])
		except AttributeError:
			print ('Fit Object has no fitting attribute. Fitting of the data is required.')
			return None

	def write_to(self, outfile=None, overwrite=True, **kwargs):
		if outfile is None:
			outfile = self.calfits.replace('.fits', '_poly.fits')
		try:
			with fits.open(self.calfits) as hdus:
				hdus['SOLUTIONS'].data[:, :, :, ::2] = np.real(self.poly_array)
				hdus['SOLUTIONS'].data[:, :, :, 1::2] = np.imag(self.poly_array)
				hdus.append(self.poly_params)
				for key, value in kwargs.items():
					hdus['FIT_COEFFS'].header.insert('TTYPE1', (key, value))
				hdus.writeto(outfile, overwrite=overwrite)
		except AttributeError:
			print ('Fit Object has no fitting attribute. Fitting of the data is required. No file is being written.')