import time
import xarray as xr
import numpy as np
import pandas as pd
from glob import glob
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors

from pytmatrix import refractive, tmatrix_aux
import snowScatt
from scattering_functions import waves, vector_liquid_Z, scat_cloud, scat_rain
from scattering_functions import z_tm_solid_sphere, z_tm_BF95_spheroid

from microphysical_function import Dice, Dcloud, Drain, gammaPSD


plt.rcParams.update({'font.size':10})

cached_scattering = True # turn to False if you want to recalculate the scattering
data = xr.open_dataset('data/P3/fwdP3.nc')

if not cached_scattering:
	print("doing scattering")
	frequency_list = ['X', 'Ka', 'W']
	frequency = xr.IndexVariable(dims='frequency', data=frequency_list, attrs={'unit':'frequency band label use pytmatrix for wavelength'})
	coords = {'time':data.time,
			  'height':data.height,
			  'frequency':frequency}
	xrZssrga = xr.DataArray(dims=['time', 'height', 'frequency'], coords=coords, attrs={'long name':'Zice computed using ssrga for snow'})
	xrZtm = xr.DataArray(dims=['time', 'height', 'frequency'], coords=coords, attrs={'long name':'Zice computed using TM for snow'})
	xrZliu09 = xr.DataArray(dims=['time', 'height', 'frequency'], coords=coords, attrs={'long name':'Zice computed using Liu09 for snow'})
	xrZliquid = xr.DataArray(dims=['time', 'height', 'frequency'], coords=coords, attrs={'long name':'Z of the liquid part'})
	#################################################################################
	# SCATTERING PART
	#################################################################################
	for freq_str in frequency.values:
		wl = waves[freq_str]
		n_water = refractive.m_w_0C[wl]
		rayleigh_coeff = wl**4/(tmatrix_aux.K_w_sqr[wl]*np.pi**5)
		wl *= 1.0e-3 # back to meters

		a_graupel = np.linspace(21, 472, 1000)
		a_partrimed = np.linspace(0.0121, 0.024, 1000)

		z_tm_graupel = pd.read_csv('data/tables/z_tm_graupel_'+freq_str+'.csv', index_col=0, dtype=np.float64, engine='c')
		z_tm_graupel.columns = z_tm_graupel.columns.astype(np.float)
		z_tm_partrimed = pd.read_csv('data/tables/z_tm_partrimed_'+freq_str+'.csv', index_col=0, dtype=np.float64, engine='c')
		z_tm_partrimed.columns = z_tm_partrimed.columns.astype(np.float)
		
		def iceP3tm_backscattering(dcrit, dcrits, dcritr):
			functions = [lambda x, ag, apr: np.nan,
						 lambda x, ag, apr: z_tm_solid_sphere.loc[x, freq_str],
						 lambda x, ag, apr: z_tm_BF95_spheroid.loc[x, freq_str],
						 lambda x, ag, apr: z_tm_graupel.iloc[z_tm_graupel.index.get_loc(ag, 'nearest')].loc[x],
						 lambda x, ag, apr: z_tm_partrimed.iloc[z_tm_partrimed.index.get_loc(apr, 'nearest')].loc[x]
						]
			def backscattering(D, ag, apr):
				conditions = [D<=0.0,
							  (0.0<D)*(D<=dcrit),
							  (dcrit<D)*(D<=dcrits),
							  (dcrits<D)*(D<=dcritr),
							  dcritr<D
							 ]
				return np.piecewise(D, conditions, functions, ag, apr)
			return backscattering

		def calc_ice_Z_tm(N0, mu, lam, dcrit, dcrits, dcritr, cs1, ds1, cs, ds, cgp, dg, csr, dsr, D):
			N = gammaPSD(N0, mu, lam)
			backscattering = iceP3tm_backscattering(dcrit, dcrits, dcritr)
			return (N(D)*backscattering(D, cgp, csr)*np.gradient(D)).sum()
		vector_ice_Z_tm = np.vectorize(calc_ice_Z_tm, excluded=['D'], otypes=[np.float])

		#a_graupel = np.linspace(21, 472, 1000)
		#a_partrimed = np.linspace(0.0121, 0.024, 1000)
		z_ssrga_partrimed = pd.DataFrame(index=a_partrimed, columns=Dice)
		start = time.time()
		z_ssrga_BF95 = pd.DataFrame(index=Dice, columns=['BF95'],
									data=snowScatt.backscatter(Dice, wavelength=wl,
															   properties='Leinonen15tabB01', ref_index=refractive.mi(wl*1e3, 0.900),#temperature=270.0,
															   massScattering=np.minimum(0.0121*Dice**1.9, np.pi*900.0*0.6*Dice**3/6) # limit mass as in TM
															  )*1.0e6 # convert to mm2
								   )
		for j in a_partrimed:
			z_ssrga_partrimed.loc[j] = snowScatt.backscatter(Dice, wavelength=wl,
															 properties='Leinonen15tabB01', ref_index=refractive.mi(wl*1e3, 0.900),#temperature=270.0,
															 massScattering=np.minimum(j*Dice**1.9, np.pi*900.0*0.6*Dice**3/6) # limit mass as in TM
															)*1.0e6 # pass from m2 to mm2
		print(time.time() - start )
		def iceP3ssrga_backscattering(dcrit, dcrits, dcritr):
			functions = [lambda x, ag, apr: np.nan,
						 lambda x, ag, apr: z_tm_solid_sphere.loc[x, freq_str],
						 lambda x, ag, apr: z_ssrga_BF95.loc[x, 'BF95'],
						 lambda x, ag, apr: z_tm_graupel.iloc[z_tm_graupel.index.get_loc(ag, 'nearest')].loc[x],
						 lambda x, ag, apr: z_ssrga_partrimed.iloc[z_ssrga_partrimed.index.get_loc(apr, 'nearest')].loc[x],
						]
			def backscattering(D, ag, apr):
				conditions = [D<=0.0,
							  (0.0<D)*(D<=dcrit),
							  (dcrit<D)*(D<=dcrits),
							  (dcrits<D)*(D<=dcritr),
							  dcritr<D
							 ]
				return np.piecewise(D, conditions, functions, ag, apr)
			return backscattering

		def calc_ice_Z_ssrga(N0, mu, lam, dcrit, dcrits, dcritr, cs1, ds1, cs, ds, cgp, dg, csr, dsr, D):
			N = gammaPSD(N0, mu, lam)
			backscattering = iceP3ssrga_backscattering(dcrit, dcrits, dcritr)
			return (N(D)*backscattering(D, cgp, csr)*np.gradient(D)).sum()
		vector_ice_Z_ssrga = np.vectorize(calc_ice_Z_ssrga, excluded=['D'], otypes=[np.float])

		#def piecewise_linear(x, x0, y0, k1, k2):
		#	return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
		RHfile = 'data/tables/scatdb.csv'
		dataRH = pd.read_csv(RHfile)
		dataRH = dataRH[dataRH.flaketype == 9] # sector snowflake
		#dataRH = dataRH[dataRH.flaketype == 20] # aggregate spherical
		#dataRH = dataRH[dataRH.flaketype == 21] # aggregate oblate
		#dataRH = dataRH[dataRH.flaketype == 22] # aggregate prolate
		dataRH = dataRH[abs(dataRH.temperaturek-270) < 4.9]
		dataRH = dataRH[abs(dataRH.frequencyghz-1.0e-9*snowScatt._compute._c/wl) < 0.5]
		print('RH frequency ', dataRH.frequencyghz.drop_duplicates().values)
		dataRH['max_dimension'] = dataRH.max_dimension_mm*1.0e-3
		dataRH.set_index('max_dimension', inplace=True)
		dataRH.sort_index(inplace=True)
		dataRH['aeff'] = dataRH.aeffum*1.0e-6
		dataRH['mass'] = 4.0*np.pi*900.0*dataRH.aeff**3/3
		#pm , em = curve_fit(piecewise_linear, np.log10(dataRH.index.values.astype(np.float)),
		#									  np.log10(dataRH.mass.values.astype(np.float)))
		#pc , ec = curve_fit(piecewise_linear, np.log10(dataRH.index.values.astype(np.float)),
		#									  np.log10(dataRH.cbk.values.astype(np.float)))
		#bm, am = np.polyfit(x=np.log10(dataRH.index.values.astype(np.float)[:]),
		#					y=np.log10(dataRH.mass.values.astype(np.float)[:]),
		#					deg=1)
		#am = 10.0**am
		#bc, ac = np.polyfit(x=np.log10(dataRH.index.values.astype(np.float)[:]),
		#					y=np.log10(dataRH.cbk.values.astype(np.float)[:]),
		#					deg=1)
		#ac = 10.0**ac
		#f_cbk = interp1d(dataRH.index.values.astype(np.float),
		#				 dataRH.cbk.values.astype(np.float),
		#				 fill_value='extrapolate')
		#f_mkg = interp1d(dataRH.index.values.astype(np.float),
		#				 dataRH.mass.values.astype(np.float),
		#				 fill_value='extrapolate')
		l_cbk = interp1d(np.log10(dataRH.index.values.astype(np.float)),
						 np.log10(dataRH.cbk.values.astype(np.float)),
						 fill_value='extrapolate')
		l_mkg = interp1d(np.log10(dataRH.index.values.astype(np.float)),
						 np.log10(dataRH.mass.values.astype(np.float)),
						 fill_value='extrapolate')
		#def dda_scattering(d):
		#	return am*d**bm, ac*d**bc
		#def dda_scattering_lin(d):
		#	return f_mkg(d), f_cbk(d)
		def dda_scattering_log(d):
			return 10**l_mkg(np.log10(d)), 10**l_cbk(np.log10(d))
		#def dda_scattering_piece(d):
		#	return 10**piecewise_linear(np.log10(d), *pm), 10**piecewise_linear(np.log10(d), *pc)
		#mass_fit, bck_fit = dda_scattering(dataRH.index)
		#mass_piece, bck_piece = dda_scattering_piece(dataRH.index)
		#mass_lin, bck_lin = dda_scattering_lin(dataRH.index)
		#mass_log, bck_log = dda_scattering_log(dataRH.index)
		def z_dda(d, m):
			mdda, bck = dda_scattering_log(d)
			return bck*1.0e6*(m/mdda)**2# convert to mm2 #*(m/mdda)**2

		def iceP3dda_backscattering(dcrit, dcrits, dcritr):
			functions = [lambda x, ag, apr: np.nan,
						 lambda x, ag, apr: z_tm_solid_sphere.loc[x, freq_str],
						 lambda x, ag, apr: z_dda(x, np.minimum(0.0121*x**1.9, np.pi*900.0*0.6*x**3/6)),
						 lambda x, ag, apr: z_tm_graupel.iloc[z_tm_graupel.index.get_loc(ag, 'nearest')].loc[x],
						 lambda x, ag, apr: z_dda(x, np.minimum(apr*x**1.9, np.pi*900.0*0.6*x**3/6)),
						]
			def backscattering(D, ag, apr):
				conditions = [D<=0.0,
							  (0.0<D)*(D<=dcrit),
							  (dcrit<D)*(D<=dcrits),
							  (dcrits<D)*(D<=dcritr),
							  dcritr<D
							 ]
				return np.piecewise(D, conditions, functions, ag, apr)
			return backscattering

		def calc_ice_Z_dda(N0, mu, lam, dcrit, dcrits, dcritr, cs1, ds1, cs, ds, cgp, dg, csr, dsr, D):
			N = gammaPSD(N0, mu, lam)
			backscattering = iceP3dda_backscattering(dcrit, dcrits, dcritr)
			return (N(D)*backscattering(D, cgp, csr)*np.gradient(D)).sum()
		vector_ice_Z_dda = np.vectorize(calc_ice_Z_dda, excluded=['D'], otypes=[np.float])

		start = time.time()
		Zcloud = vector_liquid_Z(data.N0c.values, data.mu_c.values, data.lamc.values, D=Dcloud, z=scat_cloud[freq_str].values)
		Zrain = vector_liquid_Z(data.N0r.values, data.mu_r.values, data.lamr.values, D=Drain, z=scat_rain[freq_str].values)
		print('Calculating liquid Z', time.time()-start, 'seconds') # less than 2 minutes for the liquid part, 
		start = time.time()
		ZiceTM = vector_ice_Z_tm(data.N0i.values, data.mui.values, data.lami.values,
								 data.dcrit.values, data.dcrits.values, data.dcritr.values,
								 data.cs1.values, data.ds1.values, data.cs.values, data.ds.values,
								 data.cgp.values, data.dg.values, data.csr.values, data.dsr.values, D=Dice)
		print('Calculating ice Z tm ', time.time()-start, 'seconds') # less than 2 minutes for the liquid part, 
		start = time.time()
		ZiceSSRGA = vector_ice_Z_ssrga(data.N0i.values, data.mui.values, data.lami.values,
									   data.dcrit.values, data.dcrits.values, data.dcritr.values,
									   data.cs1.values, data.ds1.values, data.cs.values, data.ds.values,
									   data.cgp.values, data.dg.values, data.csr.values, data.dsr.values, D=Dice)
		print('Calculating ice Z ssrga ', time.time()-start, 'seconds') # less than 2 minutes for the liquid part, 
		start = time.time()
		ZiceDDA = vector_ice_Z_dda(data.N0i.values, data.mui.values, data.lami.values,
								   data.dcrit.values, data.dcrits.values, data.dcritr.values,
								   data.cs1.values, data.ds1.values, data.cs.values, data.ds.values,
								   data.cgp.values, data.dg.values, data.csr.values, data.dsr.values, D=Dice)
		print('Calculating ice Z dda ', time.time()-start, 'seconds') # less than 2 minutes for the liquid part,
		Zliquid = 10.0*np.log10(rayleigh_coeff*(Zrain+Zcloud))
		Zssrga = 10.0*np.log10(rayleigh_coeff*ZiceSSRGA)
		Ztm = 10.0*np.log10(rayleigh_coeff*ZiceTM)
		Zdda = 10.0*np.log10(rayleigh_coeff*ZiceDDA)
		xrZliquid.loc[dict(frequency=freq_str)] = Zliquid
		xrZssrga.loc[dict(frequency=freq_str)] = Zssrga
		xrZtm.loc[dict(frequency=freq_str)] = Ztm
		xrZliu09.loc[dict(frequency=freq_str)] = Zdda

else:
	xrZssrga = data.Zssrga
	xrZtm = data.Ztm
	xrZliu09 = data.Zliu09
	xrZliquid = data.Zliquid 
	
ssrgaXKa = (xrZssrga.loc[dict(frequency='X')] - xrZssrga.loc[dict(frequency='Ka')]).values.flatten()
tmXKa = (xrZtm.loc[dict(frequency='X')] - xrZtm.loc[dict(frequency='Ka')]).values.flatten()
liu09XKa = (xrZliu09.loc[dict(frequency='X')] - xrZliu09.loc[dict(frequency='Ka')]).values.flatten()
ssrgaKaW = (xrZssrga.loc[dict(frequency='Ka')] - xrZssrga.loc[dict(frequency='W')]).values.flatten()
tmKaW = (xrZtm.loc[dict(frequency='Ka')] - xrZtm.loc[dict(frequency='W')]).values.flatten()
liu09KaW = (xrZliu09.loc[dict(frequency='Ka')] - xrZliu09.loc[dict(frequency='W')]).values.flatten()

# Plot Fig 8
freq = 'W'
grid = True
fig, axs = plt.subplots(3, 2, figsize=(9, 9), sharex=True, sharey=True, constrained_layout=True)
mesh = axs[0, 0].pcolormesh(data.time, data.height*1.0e-3, data.Zssrga.loc[dict(frequency=freq)].T, vmin=-30, vmax=25, cmap='jet', rasterized=True)
axs[0, 0].set_title('SSRGA')
mesh = axs[1, 0].pcolormesh(data.time, data.height*1.0e-3, data.Ztm.loc[dict(frequency=freq)].T, vmin=-30, vmax=25, cmap='jet', rasterized=True)
axs[1, 0].set_title('T matrix')
mesh = axs[2, 0].pcolormesh(data.time, data.height*1.0e-3, data.Zliu09.loc[dict(frequency=freq)].T, vmin=-30, vmax=25, cmap='jet', rasterized=True)
axs[2, 0].set_title('Liu sector snowflake')
fig.colorbar(mesh, ax=axs[:, 0], label='Reflectivity    [dBZ]', location='left', aspect=60, use_gridspec=grid)


mesh = axs[1, 1].pcolormesh(data.time, data.height*1.0e-3, data.Ztm.loc[dict(frequency=freq)].T-data.Zssrga.loc[dict(frequency=freq)].T, 
                            vmin=-5, vmax=5, cmap='Spectral', rasterized=True)
axs[1, 1].set_title('Tmatrix - SSRGA')
mesh = axs[2, 1].pcolormesh(data.time, data.height*1.0e-3, data.Zliu09.loc[dict(frequency=freq)].T-data.Zssrga.loc[dict(frequency=freq)].T, 
                            vmin=-5, vmax=5, cmap='Spectral', rasterized=True)
axs[2, 1].set_title('sector snowflake - SSRGA')
fig.colorbar(mesh, ax=axs[1:, 1], label='Reflectivity difference   [dB]', aspect=40, use_gridspec=grid)

mesh = axs[0, 1].pcolormesh(data.time, data.height*1.0e-3, 1.0e0*(data.qi.T/data.qni.T), vmax=1.0e-7, rasterized=True)
axs[0, 1].set_title('mean mass  q/N')
fig.colorbar(mesh, ax=[axs[0, 1]], label='[kg]', aspect=20, use_gridspec=grid)

[a.xaxis.set_major_formatter(mdates.DateFormatter('%H')) for a in axs.flatten()]
axs[0, 0].set_ylabel('Height   [km]')
axs[1, 0].set_ylabel('Height   [km]')
axs[2, 0].set_ylabel('Height   [km]')
axs[2, 0].set_xlabel('hour of the day (UTC)')
axs[2, 1].set_xlabel('hour of the day (UTC)')

for ax, l in zip(axs.flatten(), ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']):
    ax.text(0, 1.02, l,  fontweight='black', transform=ax.transAxes)

fig.savefig('comparison_Wband.png')
fig.savefig('comparison_Wband.pdf')
	
# Plot Fig 9
radar = xr.open_dataset('data/P3/tripex_joy_tricr00_l2_any_v00_20151124000000.nc')
radar = radar.where(radar.quality_flag_offset_w<=4096, drop=True)
radar = radar.where(radar.quality_flag_offset_x<=4096, drop=True)
radar = radar.where(radar.ta<-5)
radar = radar.where(radar.ta>-20)
XKa = (radar.dbz_x-radar.dbz_ka).values.flatten()
KaW = (radar.dbz_ka-radar.dbz_w).values.flatten()
fig, axs = plt.subplots(1, 1, figsize=(6, 4.5))

mask = np.isfinite(XKa)*np.isfinite(KaW)
hst, xe, ye = np.histogram2d(KaW[mask], XKa[mask], bins=100)
norm = mcolors.LogNorm(vmin=np.nanmin(hst[np.nonzero(hst)]),
                       vmax=np.nanmax(hst))
mesh = axs.pcolormesh(xe, ye, hst.T, norm=norm, rasterized=True)
cbar = plt.colorbar(mesh, ax=axs, label='Number of measurements')
axs.scatter(ssrgaKaW+0.5, ssrgaXKa, label='SSRGA', c='C3', rasterized=True)#, c=meanMass)
axs.scatter(tmKaW+0.5, tmXKa, label='T-matrix', c='C1', rasterized=True)#, c=meanMass)
axs.scatter(liu09KaW+0.5, liu09XKa, label='Liu sector snowflake', c='C2', rasterized=True)#, c=meanMass)
axs.legend(loc=4)
axs.grid()
axs.set_xlabel('DWR   Ka-W   [dB]')
axs.set_ylabel('DWR   X-Ka   [dB]')
fig.savefig('triple_frequency.png')
fig.savefig('triple_frequency.pdf')

