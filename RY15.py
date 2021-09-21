import numpy as np
from scipy import constants
from cdo import *
#from netCDF4 import Dataset as ncfile
import xarray as xr

cdo = Cdo()
'''
python class for decomposing temperature change to radiative terms accoring to räisänen 2017

*  1) SW-absorbtion, decomsode to clear-sky and cloudy components (SW-component)
*  2) LW-transmissiviti changes (LW-component)
*  3) Horisontal energy transport (convergense) (CONV)
*  4) Pintavuon muutos (SURF)
*  5) Residuaali

SW term is futer decomposed using method by taylor et.al (2006)

code used CMIP standard for field names

Original python code Kalle Nordling, 2019
based on Jouni Räisänen Grads code and Joonas Merikanto CDO mantras

Räisänen, J. (2017). An energy balance perspective on regional CO 2-induced temperature changes in CMIP5 models. Climate Dynamics, 48(9-10), 3441-3454.

LW component is demomposed using radiative kernels, CAMS5
'''




class RY15:
	def __init__(self,dataset1,dataset2,kernel2):
		self.kernel=kernel2
		print('USE KERNEL...:'+self.kernel)
		self.dataset1 ={}
		self.dataset2 ={}
		self.dataset1 =dataset1['coupled']
		self.dataset2 =dataset2['coupled']
		#self.dataset1 =dataset1['fsst']
		#self.dataset2 =dataset2['fsst']
		self.lat = dataset1['lat']
		self.lon = dataset1['lon']
		self.plev = dataset1['plev']
		print(self.plev,'plev')
		#self.dataset1_fsst = dataset1['fsst']
		#self.dataset2_fsst = dataset2['fsst']		
		self.results = {} #datastructure to hold results

		if kernel2 == "hadgem":
			self.surf_kernel={'path':'/scratch/project_2001927/nordlin1/PDRMIP/dT_RY15/code/kernels/hadgem-kernels/',
				    'all_sky':['tsurf_lw','HadGEM2_lw_TOA_L38_tsurf_lw.nc'],
				    'clear_sky':['tsurf_lw_cs','HadGEM2_lw_TOA_L38_tsurf_lw_cs.nc']}

			self.trop_kernel={'path':'/scratch/project_2001927/nordlin1/PDRMIP/dT_RY15/code/kernels/hadgem-kernels/',
				    'all_sky':['ta_lw','HadGEM2_lw_TOA_L17_ta_lw.nc'],
				    'clear_sky':['ta_lw_cs','HadGEM2_lw_TOA_L17_ta_lw_cs.nc']}

			self.q_kernel={'path':'/scratch/project_2001927/nordlin1/PDRMIP/dT_RY15/code/kernels/hadgem-kernels/',
				    'all_sky':['q_lw','HadGEM2_lw_TOA_L17_q_lw.nc'],
				    'clear_sky':['q_lw_cs','HadGEM2_lw_TOA_L17_q_lw_cs.nc']}

		if kernel2 == "echam":
			self.trop_kernel={'path':'/scratch/project_2001927/nordlin1/PDRMIP/dT_RY15/code/kernels/',
				    'all_sky':['Ta_trad0','K_Talw.nc'],
				    'clear_sky':['Ta_traf0','K_Talw_cl.nc']}

			self.surf_kernel={'path':'/scratch/project_2001927/nordlin1/PDRMIP/dT_RY15/code/kernels/',
				    'all_sky':['Ts_trad0','K_Tslw.nc'],
				    'clear_sky':['Ts_traf0','K_Tslw_cl.nc']}

			self.q_kernel={'path':'/scratch/project_2001927/nordlin1/PDRMIP/dT_RY15/code/kernels/',
				    'all_sky':['Q_trad0','K_Qlw.nc'],
				    'clear_sky':['Q_traf0','K_Qlw_cl.nc']}

		if kernel2 == "gfdl":
			self.trop_kernel={'path':'/scratch/project_2001927/nordlin1/PDRMIP/dT_RY15/code/kernels/',
				    'all_sky':['lw_t','TOA_GFDL_Kerns_gauss.nc'],
				    'clear_sky':['lwclr_t','TOA_GFDL_Kerns_gauss.nc']}

			self.surf_kernel={'path':'/scratch/project_2001927/nordlin1/PDRMIP/dT_RY15/code/kernels/',
				    'all_sky':['lw_ts','TOA_GFDL_Kerns_gauss.nc'],
				    'clear_sky':['lwclr_ts','TOA_GFDL_Kerns_gauss.nc']}

			self.q_kernel={'path':'/scratch/project_2001927/nordlin1/PDRMIP/dT_RY15/code/kernels/',
				    'all_sky':['lw_q','TOA_GFDL_Kerns_gauss.nc'],
				    'clear_sky':['lwclr_q','TOA_GFDL_Kerns_gauss.nc']}
		print(self.trop_kernel)
		self.proseduce()

	'''
	Main function where calculations are done
	'''
	def proseduce(self):
		
		self.results['DeltaT'] = self.calcDelta('tas')
		#print(self.results['DeltaT'].shape)
		self.addEpsilonFields()
		self.calcFactorD()
		
		#self.addOverCastFields()
		

		self.calcLWterm()
		self.decompLW()
		
		#self.calcLWterm()
		self.calcSWterms()
		self.calcSURFterm()
		self.calcCONVterms()
		self.doTaylor2()
		#self.results['cloud'] = self.results['dtAcld']+self.results['LW_cre']
		#self.results['Dt_tot'] = self.results['LW'] + self.results['SW'] + self.results['SURF'] + self.results['CONV']
		#self.results['Dt_tot2'] = self.results['LW2'] + self.results['SW'] + self.results['SURF'] + self.results['CONV']
		#self.results['SW_check'] = self.results['dtswin']+self.results['dtAclr']+self.results['dtAcld']+self.results['dtAalfa']+self.results['dtAnl']
		
		#self.calcLWterm()
		#self.decompLW2()
		
		#self.calcERF()
		#self.calcSDCs()
		
	def calcERF(self):
		self.dataset1['NET_SW_TOA'] = -1*(self.dataset1_fsst['rsdt'])+(self.dataset1_fsst['rsut'])
		self.dataset2['NET_SW_TOA'] = -1*(self.dataset2_fsst['rsdt'])+(self.dataset2_fsst['rsut'])
		self.dataset1['rlut_fsst'] = self.dataset1_fsst['rlut']
		self.dataset2['rlut_fsst'] = self.dataset2_fsst['rlut']


		#self.dataset1['NET_SW_TOA_CS'] = self.dataset1_fsst['rsdt']-self.dataset1_fsst['rsutcs']
		#self.dataset2['NET_SW_TOA_CS'] = self.dataset2_fsst['rsdt']-self.dataset2_fsst['rsutcs']
		#self.dataset1['rlutcs_fsst'] = self.dataset1_fsst['rlutcs']
		#self.dataset2['rlutcs_fsst'] = self.dataset2_fsst['rlutcs']

		self.results['ERF'] = (self.calcDelta('NET_SW_TOA')+ self.calcDelta('rlut_fsst'))*-1.0
		#self.results['ERFCS'] = self.calcDelta('NET_SW_TOA_CS')+ self.calcDelta('rlutcs_fsst')
		#self.results['ERF_cloud'] = self.results['ERF']-self.results['ERFCS']

	def calcSDC(self,field,deltaT):
		field_prime = field-np.nanmean(field,axis=0)
		deltaT_prime =deltaT- np.nanmean(deltaT,axis=0)
		tmp = field_prime*deltaT_prime
		tmp2=np.sqrt(deltaT_prime**2)
		return np.nanmean(tmp,axis=0)/np.nanmean(tmp2,axis=0)

	def calcSDCs(self):
		tmp = self.results.copy()
		for key,value in tmp.items():
			self.results[key+"_sdc"] = self.calcSDC(value,tmp['DeltaT'])
 
	'''
	Function used to calc delta of some fields. If order of calculation is needed to chage it can be done easyly
	'''
	def calcDelta(self,field):
		#try:
			return self.dataset1[field]-self.dataset2[field]
		#except:

		#	print('DELTA calculation faild, cannot find field '+field+' from datasets')
			#print(self.dataset1[field])
			#print(self.dataset2[field])

	'''
	function to calculated mean of two fields. Seperate function if method needs to be changes
	'''
	def calcMeanField(self,field):
		#try:
			return (self.dataset1[field]+self.dataset2[field])*0.5
		#except:
		#	print('MEAN calculation faild, cannot find field '+field+' from datasets')
		#	#print(self.dataset1[field])
		#	print(self.dataset2[field])


	'''
	extra helper function to calculate surf term, calculate net surf flux
	'''
	def surfHelp(self,dataset):
		tmp1 = dataset['hfls']+dataset['hfss']
		tmp2 = dataset['rsus']-dataset['rsds']
		tmp3 = dataset['rlus']-dataset['rlds']
		dataset['NETSURF'] = dataset['rsds']-dataset['rsus']+dataset['rlds']-dataset['rlus']-dataset['hfls']-dataset['hfss']
		#dataset['NETSURF'] = tmp1+tmp2+tmp3
	'''
	extra helpper function for taylor decompostion,epsilon is added to avoid dividing by 0 cases
	'''
	def taylorHelp(self,dataset):
		epsilon = 1.2345e-7
		dataset['Q_hat'] = dataset['rsds']/(dataset['rsdt']+epsilon)
		dataset['Q_hatcs'] = dataset['rsdscs']/(dataset['rsdt']+epsilon)
		dataset['Q_hatoc'] = dataset['rsdsoc']/(dataset['rsdt']+epsilon)

	'''
	calculate planetary  albedo
	'''
	def calculatePlanetaryAlbedo(self,dataset):
		epsilon = 1.2345e-7
		dataset['A'] = 1-((dataset['rsdt']-dataset['rsut'])/(dataset['rsdt']+epsilon))
		dataset['A_cs'] = 1-((dataset['rsdt']-dataset['rsutcs'])/(dataset['rsdt']+epsilon))
		dataset['A_oc'] = 1-((dataset['rsdt']-dataset['rsutoc'])/(dataset['rsdt']+epsilon))

	'''
	calculate mu and gamma parametrs, from taylor 2007
	mu = A+Q(1+a), eq. 9,
	gamma = (mu-Q)/(mu-aQ), eq. 10
	'''
	def calcMuGamma(self,dataset):
		epsilon = 1.2345e-7
		dataset['mu'] = (1-dataset['alpha'])*dataset['Q_hat']+dataset['A']
		dataset['mu_cs'] = (1-dataset['alphacs'])*dataset['Q_hatcs']+dataset['A_cs']
		dataset['mu_oc'] = (1-dataset['alphaoc'])*dataset['Q_hatoc']+dataset['A_oc']
		dataset['mu_cl'] = dataset['mu_oc']/(dataset['mu_cs']++epsilon)

		dataset['gamma'] = (dataset['mu']-dataset['Q_hat'])/((dataset['mu']-dataset['alpha']*dataset['Q_hat'])+epsilon)
		dataset['gamma_cs'] = (dataset['mu_cs']-dataset['Q_hatcs'])/((dataset['mu_cs']-dataset['alphacs']*dataset['Q_hatcs'])+epsilon)
		dataset['gamma_oc'] = (dataset['mu_oc']-dataset['Q_hatoc'])/((dataset['mu_oc']-dataset['alphaoc']*dataset['Q_hatoc'])+epsilon)
		dataset['gamma_cl'] = 1-((1-dataset['gamma_oc'])/(1-dataset['gamma_cs']+epsilon))

		dataset['A_ocC'] = (dataset['A']-((1-(dataset['clt']/100.0))*dataset['A_cs']))/((dataset['clt']/100.0)+1.2345e-7)

	'''
	This function treats partly cloudy regions according to Taylor 2007
	R = (1-c)*R_clr+c*R_oc eq.3 in Taylors papers, solve R_oc
	R_oc = (R-(1-c)*R_clr)/(c+epsilon)
	small epsilon is added to avoid divison by 0
	'''
	def taylor_overcast(self,R,R_clr,cloud):
		epsilon = 1.2345e-7
		return (R-(cloud-1)*R_clr)/(cloud+epsilon)

	'''
	This function adds calculated overcast fields to dataset structures
	'''
	def calc_overcasted_fileds(self,dataset):
		dataset['rsutoc']=self.taylor_overcast(dataset['rsut'],dataset['rsutcs'],dataset['clt']/100.0)
		dataset['rsdsoc']=self.taylor_overcast(dataset['rsds'],dataset['rsdscs'],dataset['clt']/100.0)
		dataset['rsusoc']=self.taylor_overcast(dataset['rsus'],dataset['rsuscs'],dataset['clt']/100.0)
	
	'''
	Add overcasted fields to datasets
	'''
	def addOverCastFields(self):
		self.calc_overcasted_fileds(self.dataset1)
		self.calc_overcasted_fileds(self.dataset2)

	'''
	calc planetary emissivity
	LW=outgoing LW radiation (rlut),sigma is stefan-bolzman constant
	LW = epsilon*sigma*T^4, eq.2 in raisainen paper 2017 , from here solve epsilon
	epsilon = LW/(sigma*T^4)
	add result to dataset
	calculation is done for both all-sky and clear-sky terms
	'''
	def calcEpsilon(self,dataset):
		dataset['eps'] = dataset['rlut']/(dataset['tas']**4*constants.sigma)	
		dataset['epscs'] = dataset['rlutcs']/(dataset['tas']**4*constants.sigma)


	'''
	function adds epsilon fields to datasets
	'''
	def addEpsilonFields(self):
		self.calcEpsilon(self.dataset1)
		self.calcEpsilon(self.dataset2)


	'''
	Function calculates factor D in raisanen paper
	D=4*sigma[epsilon]*[T]^3
	epsilon and T are means of two datasets
	'''
	def calcFactorD(self):
		mean_T = self.calcMeanField('tas')
		mean_epsilon = self.calcMeanField('eps')
		self.results['D'] = 4.0*constants.sigma*mean_epsilon*mean_T**3

	'''
	calculate surface albedo
	alpha = Q_s_up/Q_s_down
	'''
	def calcSurfaceAlbedo(self,dataset):
		epsilon = 1.2345e-7
		dataset['alpha'] = dataset['rsus']/(dataset['rsds']+epsilon)
		dataset['alphacs'] = dataset['rsuscs']/(dataset['rsdscs']+epsilon)
		dataset['alphaoc'] = dataset['rsusoc']/(dataset['rsdsoc']+epsilon)
		
	
	'''
	calc A's eq. 7 in taylor paper
	'''
	def calcA(self,mu,gamma,alpha):
		epsilon = 1.2345e-7
		return mu*gamma+((mu*alpha*(1-gamma)**2)/(1-alpha*gamma+epsilon))


	def taylorHepl(self,dataset):
		eps = 1.2345e-6
		dataset['rsutoc'] = (dataset['rsut']+(dataset['c']-1.0)*dataset['rsutcs'])/(dataset['c']+eps)
		dataset['rsdsoc'] = (dataset['rsds']+(dataset['c']-1.0)*dataset['rsdscs'])/(dataset['c']+eps)
		dataset['rsusoc'] = (dataset['rsus']+(dataset['c']-1.0)*dataset['rsuscs'])/(dataset['c']+eps)

		dataset['A'] = dataset['rsut']/(dataset['s']+eps)
		dataset['Aclr'] = dataset['rsutcs']/(dataset['s']+eps)
		dataset['Aoc'] = dataset['rsutoc']/(dataset['s']+eps)

		dataset['alfa']=dataset['rsus']/(dataset['rsds']+eps)
		dataset['alfaclr']=dataset['rsuscs']/(dataset['rsdscs']+eps)
		dataset['alfaoc']=dataset['rsusoc']/(dataset['rsdsoc']+eps)

		dataset['Qrat'] = dataset['rsds']/(dataset['s']+eps)
		dataset['Qratclr'] = dataset['rsdscs']/(dataset['s']+eps)
		dataset['Qratoc'] = dataset['rsdsoc']/(dataset['s']+eps)

		dataset['my']=dataset['A']+dataset['Qrat']*(1-dataset['alfa'])
		dataset['myclr']=dataset['Aclr'] +dataset['Qratclr']*(1-dataset['alfaclr'])
		dataset['myoc']=dataset['Aoc']+dataset['Qratoc']*(1-dataset['alfaoc'])

		dataset['gamma']=(dataset['my']-dataset['Qrat'])/(dataset['my']-dataset['alfa']*dataset['Qrat']+eps)
		dataset['gammaclr']=(dataset['myclr']-dataset['Qratclr'])/(dataset['myclr']-dataset['alfaclr']*dataset['Qratclr']+eps)
		dataset['gammaoc']=(dataset['myoc']-dataset['Qratoc'])/(dataset['myoc']-dataset['alfaoc']*dataset['Qratoc']+eps)


		dataset['gammacld']=1-(1-dataset['gammaoc'])/(1-dataset['gammaclr'])
		dataset['mycld']=(dataset['myoc']+eps)/(dataset['myclr']+eps)

	def Afrompar(self,xc,xalfaclr,xalfaoc,xmyclr,xmycld,xgammaclr,xgammacld):
		xgammaoc=1-(1-xgammaclr)*(1-xgammacld)
		xmyoc=xmyclr*xmycld
		
		xAclr=xmyclr*xgammaclr+xmyclr*xalfaclr*(1-xgammaclr)**2/(1-xalfaclr*xgammaclr)
		xAoc=xmyoc*xgammaoc+xmyoc*xalfaoc*(1-xgammaoc)**2/(1-xalfaoc*xgammaoc)
		
		return (1-xc)*xAclr+xc*xAoc

	'''
	do taylor decomposition
	'''
	def doTaylor2(self):
		self.dataset1['s'] = self.dataset1['rsdt']
		self.dataset2['s'] = self.dataset2['rsdt']
		self.dataset1['c'] = self.dataset1['clt']/100.0
		self.dataset2['c'] = self.dataset2['clt']/100.0
		self.taylorHepl(self.dataset1)
		self.taylorHepl(self.dataset2)


		a=self.calcMeanField('A')
		s=self.calcMeanField('s')
		c=self.calcMeanField('c')
		alfaclr=self.calcMeanField('alfaclr')
		alfaoc=self.calcMeanField('alfaoc')
		myclr=self.calcMeanField('myclr')
		mycld=self.calcMeanField('mycld')
		gammaclr=self.calcMeanField('gammaclr')
		gammacld=self.calcMeanField('gammacld')


		c1=self.dataset1['c']
		c2=self.dataset2['c']
		myclr1=self.dataset1['myclr']
		myclr2=self.dataset2['myclr']
		alfaclr1=self.dataset1['alfaclr']
		alfaoc1=self.dataset1['alfaoc']
		mycld1=self.dataset1['mycld']
		gammaclr1=self.dataset1['gammaclr']
		gammacld1=self.dataset1['gammacld']

		alfaclr2=self.dataset2['alfaclr']
		alfaoc2=self.dataset2['alfaoc']
		mycld2=self.dataset2['mycld']
		gammaclr2=self.dataset2['gammaclr']
		gammacld2=self.dataset2['gammacld']



		self.dataset1['apar'] = self.Afrompar(c1,alfaclr1,alfaoc1,myclr1,mycld1,gammaclr1,gammacld1) 
		self.dataset2['apar'] = self.Afrompar(c2,alfaclr2,alfaoc2,myclr2,mycld2,gammaclr2,gammacld2) 

		self.dataset1['aalfaclr'] = self.Afrompar(c,alfaclr1,alfaoc,myclr,mycld,gammaclr,gammacld) 
		self.dataset2['aalfaclr'] = self.Afrompar(c,alfaclr2,alfaoc,myclr,mycld,gammaclr,gammacld) 

		self.dataset1['aalfaoc'] = self.Afrompar(c,alfaclr,alfaoc1,myclr,mycld,gammaclr,gammacld) 
		self.dataset2['aalfaoc'] = self.Afrompar(c,alfaclr,alfaoc2,myclr,mycld,gammaclr,gammacld) 

		self.dataset1['amycld'] = self.Afrompar(c,alfaclr,alfaoc,myclr,mycld1,gammaclr,gammacld) 
		self.dataset2['amycld'] = self.Afrompar(c,alfaclr,alfaoc,myclr,mycld2,gammaclr,gammacld) 

		self.dataset1['agammacld'] = self.Afrompar(c,alfaclr,alfaoc,myclr,mycld,gammaclr,gammacld1) 
		self.dataset2['agammacld'] = self.Afrompar(c,alfaclr,alfaoc,myclr,mycld,gammaclr,gammacld2) 

		self.dataset1['ac'] = self.Afrompar(c1,alfaclr,alfaoc,myclr,mycld,gammaclr,gammacld)
		self.dataset2['ac'] = self.Afrompar(c2,alfaclr,alfaoc,myclr,mycld,gammaclr,gammacld) 
		
		self.dataset1['amyclr'] = self.Afrompar(c,alfaclr,alfaoc,myclr1,mycld,gammaclr,gammacld) 
		self.dataset2['amyclr'] = self.Afrompar(c,alfaclr,alfaoc,myclr2,mycld,gammaclr,gammacld)

		self.dataset1['agammaclr'] = self.Afrompar(c,alfaclr,alfaoc,myclr,mycld,gammaclr1,gammacld) 
		self.dataset2['agammaclr'] = self.Afrompar(c,alfaclr,alfaoc,myclr,mycld,gammaclr2,gammacld)
		
		self.dataset1['acld'] = self.Afrompar(c1,alfaclr,alfaoc,myclr,mycld1,gammaclr,gammacld1) 
		self.dataset2['acld'] = self.Afrompar(c2,alfaclr,alfaoc,myclr,mycld2,gammaclr,gammacld2) 
		
		self.dataset1['aalfa'] = self.Afrompar(c,alfaclr1,alfaoc1,myclr,mycld,gammaclr,gammacld) 
		self.dataset2['aalfa'] = self.Afrompar(c,alfaclr2,alfaoc2,myclr,mycld,gammaclr,gammacld) 
		
		self.dataset1['aclr'] = self.Afrompar(c,alfaclr,alfaoc,myclr1,mycld,gammaclr1,gammacld) 
		self.dataset2['aclr'] = self.Afrompar(c,alfaclr,alfaoc,myclr2,mycld,gammaclr2,gammacld) 
		
		ddaalfa= self.calcDelta('aalfa') #(aalfa2-aalfa1)S
		ddaclr= self.calcDelta('aclr') #(aclr2-aclr1)
		ddacld= self.calcDelta('acld')#(acld2-acld1)

		daalfa=  self.calcDelta('aalfaclr')+self.calcDelta('aalfaoc') #(aalfaclr2-aalfaclr1)+(aalfaoc2-aalfaoc1)
		daalfaclr= self.calcDelta('aalfaclr')#(aalfaclr2-aalfaclr1)
		daalfaoc= self.calcDelta('aalfaoc')#(aalfaoc2-aalfaoc1)
		dacld= self.calcDelta('amycld')+self.calcDelta('agammacld')+self.calcDelta('ac')#(amycld2-amycld1)+(agammacld2-agammacld1)+(ac2-ac1)
		dacldmy=  self.calcDelta('amycld')#(amycld2-amycld1)
		dacldgamma=self.calcDelta('agammacld') #(agammacld2-agammacld1)
		dacldc= self.calcDelta('ac')#(ac2-ac1)
		daclr= self.calcDelta('amyclr')+self.calcDelta('agammaclr')#(amyclr2-amyclr1)+(agammaclr2-agammaclr1)
		daclrmy= self.calcDelta('amyclr')#(amyclr2-amyclr1)
		daclrgamma= self.calcDelta('agammaclr')#(agammaclr2-agammaclr1)
		da=self.calcDelta('A')
		danl=da-(daalfa+dacld+daclr)
		ddanl=da-(ddaalfa+ddacld+ddaclr)
		
		
		fs=self.calcDelta('s')*(1-a)
		fA=-s*da
		fAclr=-s*daclr
		fAcld=-s*dacld
		fAalfa=-s*daalfa
		fAnl=-s*danl
		sensitivity=1.0/self.results['D']
		self.results['dtswin']=sensitivity*fs
		self.results['dtAclr']=sensitivity*fAclr
		self.results['dtAcld']=sensitivity*fAcld
		self.results['dtAalfa']=sensitivity*fAalfa
		self.results['dtAnl']=sensitivity*fAnl
		self.results['amycld']=-s*self.calcDelta('amycld')*sensitivity
		self.results['agammacld']=-s*self.calcDelta('agammacld')*sensitivity
		self.results['ac']=-s*self.calcDelta('ac')*sensitivity

		self.results['dtswin_flux']=fs
		self.results['dtAclr_flux']=fAclr
		self.results['dtAcld_flux']=fAcld
		self.results['dtAalfa_flux']=fAalfa
		self.results['dtAnl_flux']=fAnl
		self.results['amycld_flux']=-s*self.calcDelta('amycld')
		self.results['agammacld_flux']=-s*self.calcDelta('agammacld')
		self.results['ac_flux']=-s*self.calcDelta('ac')


		#		dataset['gammaoc']=(dataset['myoc']-dataset['Qratoc'])/(dataset['myoc']-dataset['alfaoc']*dataset['Qratoc']+eps)
		self.results['myoc']=self.calcDelta('myoc')
		self.results['Qratoc']=self.calcDelta('Qratoc')
		self.results['alfaoc'] = self.calcDelta('alfaoc')
		self.results['alfaoc_Qratoc'] = self.calcDelta('alfaoc')*self.calcDelta('Qratoc')
	'''
	do taylor decomposition
	'''
	def doTaylor(self):
		self.calcSurfaceAlbedo(self.dataset1)
		self.calcSurfaceAlbedo(self.dataset2)

		#calc mean alphas
		brac_alpha = self.calcMeanField('alpha')
		brac_alphacs = self.calcMeanField('alphacs')
		brac_alphaoc = self.calcMeanField('alphaoc')

		#calc Q_hat terms
		self.taylorHelp(self.dataset1)
		self.taylorHelp(self.dataset2)

		#calc planetary albedo (A term, in taylor paper)
		self.calculatePlanetaryAlbedo(self.dataset1)
		self.calculatePlanetaryAlbedo(self.dataset2)

		brac_a = self.calcMeanField('A')
		brac_acs = self.calcMeanField('A_cs')
		brac_aoc = self.calcMeanField('A_oc')

		#calc mu and gamma
		self.calcMuGamma(self.dataset1)
		self.calcMuGamma(self.dataset2)

		brac_mu = self.calcMeanField('mu')
		brac_mucs = self.calcMeanField('mu_cs')
		brac_muoc = self.calcMeanField('mu_oc')
		brac_mucl = self.calcMeanField('mu_cl')

		brac_gamma = self.calcMeanField('gamma')
		brac_gammacs = self.calcMeanField('gamma_cs')
		brac_gammaoc = self.calcMeanField('gamma_oc')
		brac_gammacl = self.calcMeanField('gamma_cl')

		brac_clt = self.calcMeanField('clt')/100.0
		
		#calc deltaA's
		#delta alpha
		gamma_oc = 1.0-(1.0-brac_gammacl)*(1.0-brac_gammacs)
		mu_oc = brac_mucs*brac_mucl

		self.dataset1['A_clr1'] = self.calcA(brac_mucs,brac_gammacs,self.dataset1['alphacs'])
		self.dataset1['A_oc1'] = self.calcA(brac_muoc,brac_gammaoc,self.dataset1['alphaoc'])
		self.dataset2['A_clr1'] = self.calcA(brac_mucs,brac_gammacs,self.dataset2['alphacs'])
		self.dataset2['A_oc1'] = self.calcA(brac_muoc,brac_gammaoc,self.dataset2['alphaoc'])

		delta_A_alpha_clr = (1-brac_clt)*self.calcDelta('A_clr1')
		delta_A_alpha_oc = brac_clt*self.calcDelta('A_oc1')

		#delta mu
		self.dataset1['A_clr1'] = self.calcA(self.dataset1['mu_cs'],brac_gammacs,brac_alpha)
		self.dataset1['A_oc1'] = self.calcA(brac_mucs*self.dataset1['mu_oc'],brac_gammaoc,brac_alphaoc)
		self.dataset2['A_clr1'] = self.calcA(self.dataset2['mu_cs'],brac_gammacs,brac_alpha)
		self.dataset2['A_oc1'] = self.calcA(brac_mucs*self.dataset2['mu_oc'],brac_gammaoc,brac_alphaoc)

		delta_A_mu_clr = (1-brac_clt)*self.calcDelta('A_clr1')
		delta_A_mu_oc = brac_clt*self.calcDelta('A_oc1')

		#delta gamma
		gamma_oc = 1-(1-self.dataset1['gamma_cl'])*(1-brac_gammacs)
		self.dataset1['A_clr1'] = self.calcA(brac_mucs,self.dataset1['gamma_cs'],brac_alphacs)
		self.dataset1['A_oc1'] = self.calcA(brac_mucs,gamma_oc,brac_alphaoc)
		gamma_oc = 1-(1-self.dataset2['gamma_cl'])*(1-brac_gammacs)
		self.dataset2['A_clr1'] = self.calcA(brac_mucs,self.dataset2['gamma_cs'],brac_alphacs)
		self.dataset2['A_oc1'] = self.calcA(brac_mucs,gamma_oc,brac_alphaoc)

		delta_A_gamma_clr = (1-brac_clt)*self.calcDelta('A_clr1')
		delta_A_gamma_oc = brac_clt*self.calcDelta('A_oc1')
		
		#delta C
		self.dataset1['A_clr1'] = self.calcA(brac_mucs,brac_gammacs,brac_alphacs)
		self.dataset1['A_oc1'] = self.calcA(brac_muoc,brac_gammaoc,brac_alphaoc)

		A_clr = self.calcDelta('A_clr1')
		A_oc = brac_clt*self.calcDelta('A_oc1')

		self.dataset1['A_1'] = (1-(self.dataset1['clt']/100.0))*A_clr+(self.dataset1['clt']/100.0)*A_oc
		self.dataset2['A_1'] = (1-(self.dataset2['clt']/100.0))*A_clr+(self.dataset2['clt']/100.0)*A_oc
		delta_A_c = self.calcDelta('A_1')

		#add to results
		self.results['Delta_SW_alpha'] = ((delta_A_alpha_clr+delta_A_alpha_oc)*-1.0*self.calcMeanField('rsdt'))/self.results['D']
		self.results['Delta_SW_clr'] = ((delta_A_mu_clr+delta_A_gamma_clr)*-1.0*self.calcMeanField('rsdt'))/self.results['D']
		self.results['Delta_SW_cld'] = ((delta_A_mu_oc+delta_A_gamma_oc+delta_A_c)*-1.0*self.calcMeanField('rsdt'))/self.results['D']
		self.results['SW_check'] = self.results['Delta_SW_alpha']+self.results['Delta_SW_clr']+self.results['Delta_SW_cld']

		self.results['Delta_SW_alpha_flux'] = ((delta_A_alpha_clr+delta_A_alpha_oc)*-1.0*self.calcMeanField('rsdt'))
		self.results['Delta_SW_clr_flux'] = ((delta_A_mu_clr+delta_A_gamma_clr)*-1.0*self.calcMeanField('rsdt'))
		self.results['Delta_SW_cld_flux'] = ((delta_A_mu_oc+delta_A_gamma_oc+delta_A_c)*-1.0*self.calcMeanField('rsdt'))
		self.results['SW_check_flux'] = self.results['Delta_SW_alpha_flux']+self.results['Delta_SW_clr_flux']+self.results['Delta_SW_cld_flux']
		
	'''
	function calculate LW change factor all sky and cloudy components
	calculation is done according to raisanen paper eq.4 factor 1
	LW = - sigma*Delta_Epsilon*T^4
	Temperature here is mean temperature
	LW = LW_clear+LW_cloudy part -> LW_cre = LW-LW_Clearsky
	'''
	def calcLWterm(self):
		deltaEPS = self.calcDelta('eps')
		deltaEPSCS = self.calcDelta('epscs')
		mean_T_power4 = (self.dataset1['tas']**4+self.dataset2['tas']**4)*0.5
		
		self.results['LW'] = (-1.0*constants.sigma*deltaEPS*mean_T_power4)/self.results['D']
		self.results['LW_clearsky'] = (-1.0*constants.sigma*deltaEPSCS*mean_T_power4)/self.results['D']
		self.results['LW_cre'] = self.results['LW']-self.results['LW_clearsky']

		self.results['LW_flux'] = (-1.0*constants.sigma*deltaEPS*mean_T_power4)
		self.results['LW_clearsky_flux'] = (-1.0*constants.sigma*deltaEPSCS*mean_T_power4)
		self.results['LW_cre_flux'] = self.results['LW_flux']-self.results['LW_clearsky_flux']

		self.results['LW2_flux'] = self.calcDelta('rlut')
		self.results['LW2_clearsky_flux'] = self.calcDelta('rlutcs')
		self.results['LW2_cre_flux'] = self.results['LW2_flux']-self.results['LW2_clearsky_flux']
		self.results['LW2'] = self.results['LW2_flux']/self.results['D']
		self.results['LW2_clearsky'] = self.results['LW2_clearsky_flux']/self.results['D']
		self.results['LW2_cre'] = self.results['LW2']-self.results['LW2_clearsky']

		self.results['LW3_flux'] = -1.0*self.calcDelta('rlut')
		self.results['LW3_clearsky_flux'] = -1.0*self.calcDelta('rlutcs')
		self.results['LW3_cre_flux'] = self.results['LW3_flux']-self.results['LW3_clearsky_flux']
		self.results['LW3'] = self.results['LW3_flux']/self.results['D']
		self.results['LW3_clearsky'] = self.results['LW3_clearsky_flux']/self.results['D']
		self.results['LW3_cre'] = self.results['LW3']-self.results['LW3_clearsky']

	'''
	calc surface temperature response
	'''
	def calcSurfResponse(self,type_="all"):
		var = self.surf_kernel['all_sky'][0]
		file_=self.surf_kernel['path']+self.surf_kernel['all_sky'][1]
		if type_=="clear":
			var = self.surf_kernel['clear_sky'][0]
			file_=self.surf_kernel['path']+self.surf_kernel['clear_sky'][1]

		self.ts_kernel = file_#cdo.remapcon("t63grid",input='-selvar,'+var+' kernels/cam5-kernels/kernels/ts.kernel.nc')
		ts_kernel = xr.open_dataset(self.ts_kernel,decode_times=False)[var][:,:,:]
		del self.ts_kernel
		ts_kernel = np.tile(ts_kernel,(int(self.results['DeltaT'].shape[0]/ts_kernel.shape[0]),1,1))
		print(self.results['DeltaT'].shape[0],ts_kernel.shape)
		dLW_ts=ts_kernel*self.results['DeltaT'];
		del ts_kernel
		return dLW_ts

	def populatePgrid(self,p,lon,lat,time):
		tmp = np.tile(p,(lon,1)).T
		tmp = self.permute(tmp,(1,0,2))
		tmp = np.tile(tmp,(1,lat,1))
		tmp = self.permute(tmp,(0,1,2,3))
		tmp = np.tile(tmp,(time,1,1,1)) #tmp lon,lat,time,lev
		return tmp

	def permute(self,A,ind):
		return np.transpose( np.expand_dims(A, axis=0), ind )
	'''	
	calculate location of 
	'''
	def calcTropopause(self):
		x=np.cos(np.deg2rad(self.lat));
		p_tropopause_zonalmean=300-200*x;
		tmp = np.tile(p_tropopause_zonalmean,(len(self.lon),1)).T
		tmp = self.permute(tmp,(0,1,2))
		tmp = np.tile(tmp,(len(self.plev),1,1))
		tmp = self.permute(tmp,(0,1,2,3))
		tmp = np.tile(tmp,(self.results['DeltaT'].shape[0],1,1,1)) #tmp lon,lat,time,lev
		#self.results['tropopause'] = tmp
		return tmp

	def checKernel(self,kernel,plevels):
		print('check kernel direction')
		unit=True
		mul=True
		if not ((self.plev==plevels).all()) and ((np.sort(self.plev/100)==np.sort(plevels)).all() or (np.sort(self.plev*100)==np.sort(plevels)).all()):
			print('unit diff')
			unit=False
			if (np.sort(self.plev/100)==np.sort(plevels)).all():
				mul=False
			if (np.sort(self.plev*100)==np.sort(plevels)).all():
				mul=True
		if ((self.plev == plevels[::-1]).all()) or ((self.plev*100 == plevels[::-1]).all()):	
			print('kernel diff',kernel.shape)
			tmp = np.flip(kernel,axis=1)
			return tmp,unit,mul
		return [],unit,mul

	def calcTropospheStratospherResponse(self,type_="all"):
		var = self.trop_kernel['all_sky'][0]
		file_=self.trop_kernel['path']+self.trop_kernel['all_sky'][1]
		if type_=="clear":
			var = self.trop_kernel['clear_sky'][0]
			file_=self.trop_kernel['path']+self.trop_kernel['clear_sky'][1]

		print('read tropo kernel')
		kplev=xr.open_dataset(file_,decode_times=False)['plev'][:].to_masked_array()
		ta_kernel = self.readQkernel(file_,var)
		test,unit,mul = self.checKernel(ta_kernel,kplev)
		dx = np.copy(self.dx)
		p=np.copy(self.p)
		if not unit:
			if mul:
				print('multiply')
				dx *= 100
				#p *= 100
			else:
				print('div')
				dx /= 100
				#p /= 100				
		if len(test) > 1:
			ta_kernel = test
		ta_kernel[np.abs(ta_kernel) < 0.0001] = np.nan
		#self.results['ta_kernel'] = ta_kernel
		#ta_kernel = ta_kernel#(100/(self.dx))
		print('self.p kplev')
		print(self.plev,kplev)
		mask = (p>=self.t_pause)
		delta_ta=self.calcDelta('ta')
		#self.results['delta_ta'] = delta_ta
		dta = delta_ta*mask

		print('iuntegratge')
		#self.results['dta_tropo'] = dta
		mul_factor=1.0
		if self.kernel=="hadgem":
			mul_factor=100.0
		if self.kernel=="gfdl":
			mul_factor=1.0
		if self.kernel=='echam':
			mul_factr=1.0/1.0#100.0

		if self.kernel == "echam":
			dLW_ta=np.squeeze(np.nansum((ta_kernel*mask*dta),axis=1))
		else:
			dLW_ta=np.squeeze(np.nansum(((ta_kernel*mask)*dta*(dx))/mul_factor,axis=1))

		mask = (p<self.t_pause)
		dta = delta_ta*mask
		#self.results['dta_strato'] = dta
		#self.results['dx'] = dx
		print('integrate')
		if self.kernel=="echam":
			dLw_ta_stra=np.squeeze(np.nansum(((ta_kernel*mask)*dta)/mul_factor,axis=1))
		else:
			dLw_ta_stra=np.squeeze(np.nansum(((ta_kernel*mask)*dta*(dx))/mul_factor,axis=1))

		del ta_kernel,mask,dta				
		return dLW_ta,dLw_ta_stra

	def calcSatVapPressure(self,t,p):
   		## Formulae from Buck (1981):
		es = (1.0007+(3.46e-6*p))*6.1121*np.exp(17.502*(t-273.15)/(240.97+(t-273.15)))
		wsl = .622*es/(p-es); # saturation mixing ratio wrt liquid water (g/kg)
  
		es = (1.0003+(4.18e-6*p))*6.1115*np.exp(22.452*(t-273.15)/(272.55+(t-273.15)))
		wsi = .622*es/(p-es); # saturation mixing ratio wrt ice (g/kg)
  
		ws = wsl;
		print(t.shape,"t shape")
		print(ws.shape,"ws shape")
		print(wsi.shape,'wsi shape')
		ws[t<273.15]=wsi[t<273.15];
 
		return ws/(1+ws);
	
	def tileKernel(self,ta_kernel):
		ta_kernel = np.tile(ta_kernel,(int(self.results['DeltaT'].shape[0]/ta_kernel.shape[0]),1,1,1))
		return ta_kernel
	
	def readQkernel(self,kernel,var):
		tmp = xr.open_dataset(kernel,decode_times=False)
		ret = self.tileKernel(tmp[var][:,:,:,:])#*(100/self.dx)
		tmp.close()
		return ret

	def useQkernel(self,file_,var):
		print('Use Q kernel',file_,var)
		q_kernel=self.readQkernel(file_,var)
		q_kernel2=q_kernel/self.dqdt;
		del q_kernel
		print('kernel')
		print(q_kernel2.shape)
		print('dq')
		print(self.dq.shape)
		d_q=np.squeeze(np.nansum(q_kernel2*self.dq*self.dx,axis=1));
		del q_kernel2
		return d_q

	def useQkernelLog(self,file_,var):
		print('Use Q kernel',file_,var)
		q_kernel=self.readQkernel(file_,var)
		q_kernel[np.abs(q_kernel) < 0.0001] = np.nan 
		kplev=xr.open_dataset(file_,decode_times=False)['plev'][:].to_masked_array()
		test,unit,mul = self.checKernel(q_kernel,kplev)
		print("kernel level", kplev)
		dx = np.copy(self.dx)
		p = np.copy(self.p)
		if not unit:
			if mul:
				dx *= 100
				#p *= 100
			else:
				dx /= 100
				#p /= 100	
		if len(test) > 1:
			print('swap kernel')
			q_kernel = test

		if self.kernel=="hadgem":
			mul_factor=100.0
		if self.kernel=="gfdl":
			mul_factor=1.0
		if self.kernel=='echam':
			mul_factor=1/1

		self.dataset1['qs'] = self.calcSatVapPressure(self.dataset1['ta'],p)*1000.0
		self.dataset2['qs'] = self.calcSatVapPressure(self.dataset2['ta'],p)*1000.0
		print('calc q')
		self.dataset1['q'] = self.dataset1['qs']*(self.dataset1['hur']/100.0)
		self.dataset2['q'] = self.dataset2['qs']*(self.dataset2['hur']/100.0)
		self.dataset2['lnq'] = np.log(self.dataset2['q'])
		self.dataset1['lnq'] = np.log(self.dataset1['q'])
		self.results['q1_exp'] = self.dataset1['q']
		self.results['q2_base'] = self.dataset2['q']

		if self.kernel == "echam":
			q_kernel2=q_kernel
			dqlog=(self.calcDelta('lnq'))
			d_q=np.squeeze(np.nansum((q_kernel2*dqlog),axis=1));
		else:
			tmp=np.log10(self.dataset2['qs'])/(np.log(self.calcSatVapPressure(self.dataset2['ta']+1,self.p))+1.1234e-6)
			dqlog = (np.log(self.dataset1['q'])-np.log(self.dataset2['q']))/(tmp+1)
			dlogqdt = dqlog/(self.calcDelta('ta')+1.1234e-6)
			#dqlog=np.log(self.calcDelta('q'))
			q_kernel2=q_kernel/dlogqdt#*(100/(self.dx))#/self.dlogqdt
			d_q=np.squeeze(np.nansum((q_kernel2*dqlog*dx)/mul_factor,axis=1));
		#self.results['q_kernel'] = q_kernel
		self.results['q_kernel2'] = q_kernel2
		self.results['dqlog'] = dqlog
		#self.results['dx'] = dx
		#self.results['p'] = self.p
		#mask = (self.p>=self.t_pause)
		#self.dqlog = self.dqlog*mask
		del q_kernel
		#print('kernel')
		#print(q_kernel2.shape)
		#print('dq')
		#print(self.dqlog.shape)
		#print('q kernel dx',dx[0,:,0,0])

		del q_kernel2
		return d_q
		

	def waterVapor(self):
		print('calc sat vap press')


		print('read kernels and tile them')


		file_=self.q_kernel['path']+self.q_kernel['all_sky'][1]
		file_cs=self.q_kernel['path']+self.q_kernel['clear_sky'][1]

		self.results['dLW_q_flux'] = self.useQkernelLog(file_,self.q_kernel['all_sky'][0])
		self.results['dLW_q_cs_flux'] = self.useQkernelLog(file_cs,self.q_kernel['clear_sky'][0])


		self.results['dLW_q'] = self.results['dLW_q_flux']/self.results['D']
		self.results['dLW_q_cs'] = self.results['dLW_q_cs_flux']/self.results['D']




	'''
	decompose LW term using radiative kernels
	'''
	def decompLW(self):
		print('decomnp LW')
		#interpolote kernels to commont gaussian grid


		self.results['surf_response_flux'] = self.calcSurfResponse("all")
		self.results['surf_response'] = self.results['surf_response_flux']/self.results['D']


		self.results['surf_response_cs_flux'] = self.calcSurfResponse("clear")
		self.results['surf_response_cs'] = self.results['surf_response_cs_flux']/self.results['D']
	
		self.t_pause=self.calcTropopause()
		#self.results['t_pause'] = self.t_pause

		self.p = self.populatePgrid(self.plev,len(self.lon),len(self.lat),self.results['DeltaT'].shape[0])/100
		#self.results['plevels'] = self.p


		self.dx = self.plev[:-1]-self.plev[1:]
		#self.dx = self.plev[1:]-self.plev[:-1]
		self.dx = np.append(self.dx,self.plev[-1])
		#self.dx = np.append(self.dx,self.plev[0])

		self.dx = self.populatePgrid(self.dx,len(self.lon),len(self.lat),self.results['DeltaT'].shape[0])
		self.dx =  np.tile(self.dx,(int(self.results['DeltaT'].shape[0]/self.dx.shape[0]),1,1,1))
		self.dx /= 100
		
		self.results['tropo_response_flux'],self.results['strato_response_flux'] = self.calcTropospheStratospherResponse("all")
		self.results['tropo_response'] = self.results['tropo_response_flux']/self.results['D']
		self.results['strato_response'] = self.results['strato_response_flux']/self.results['D']


		self.results['tropo_response_cs_flux'],self.results['strato_response_cs_flux'] = self.calcTropospheStratospherResponse("clear")
		self.results['tropo_response_cs'] = self.results['tropo_response_cs_flux']/self.results['D']
		self.results['strato_response_cs'] = self.results['strato_response_cs_flux']/self.results['D']
		

		self.waterVapor()

		#decomp LW cloud

		self.results['LW_cloud_kernel_term_flux'] = (self.results['surf_response_flux']-self.results['surf_response_cs_flux'])+(self.results['tropo_response_flux']-self.results['tropo_response_cs_flux'])+(self.results['dLW_q_flux']-self.results['dLW_q_cs_flux'])+(self.results['strato_response_flux']-self.results['strato_response_cs_flux'])
		self.results['LW_cloud_q_flux']=(self.results['dLW_q_flux']-self.results['dLW_q_cs_flux'])
		self.results['LW_cloud_ta_flux']=(self.results['tropo_response_flux']-self.results['tropo_response_cs_flux'])+(self.results['strato_response_flux']-self.results['strato_response_cs_flux'])
		self.results['LW_cloud_tropo_flux']=(self.results['tropo_response_flux']-self.results['tropo_response_cs_flux'])
		self.results['LW_cloud_strato_flux']=(self.results['strato_response_flux']-self.results['strato_response_cs_flux'])
		self.results['LW_cloud_tas_flux']=(self.results['surf_response_flux']-self.results['surf_response_cs_flux'])

		self.results['LW_cloud_kernel_term'] = self.results['LW_cloud_kernel_term_flux']/self.results['D']
		self.results['LW_cloud_q']=self.results['LW_cloud_q_flux']/self.results['D']
		self.results['LW_cloud_ta']=self.results['LW_cloud_ta_flux']/self.results['D']
		self.results['LW_cloud_tropo']=self.results['LW_cloud_tropo_flux']/self.results['D']
		self.results['LW_cloud_strato']=self.results['LW_cloud_strato_flux']/self.results['D']
		self.results['LW_cloud_tas']=self.results['LW_cloud_tas_flux']/self.results['D']
	
		

		
		
		self.results['LW_cre_kernel_flux'] = self.results['LW_cre_flux']-self.results['LW_cloud_kernel_term_flux']
		self.results['LW_cre_kernel'] = self.results['LW_cre_kernel_flux']/self.results['D']
		self.results['LW_clearsky_correct_flux'] = self.results['LW_clearsky_flux']+self.results['LW_cloud_kernel_term_flux']
		self.results['LW_clearsky_correct'] = self.results['LW_clearsky_correct_flux']/self.results['D']

		#decomp LW cloud
		self.results['LW2_cre_kernel_flux'] = self.results['LW2_cre_flux']-self.results['LW_cloud_kernel_term_flux']
		self.results['LW2_cre_kernel'] = self.results['LW2_cre_kernel_flux']/self.results['D']
		self.results['LW2_clearsky_correct_flux'] = self.results['LW2_clearsky_flux']+self.results['LW_cloud_kernel_term_flux']
		self.results['LW2_clearsky_correct'] = self.results['LW2_clearsky_correct_flux']/self.results['D']

		self.results['LW3_cre_kernel_flux'] = self.results['LW3_cre_flux']-self.results['LW_cloud_kernel_term_flux']
		self.results['LW3_cre_kernel'] = self.results['LW3_cre_kernel_flux']/self.results['D']
		self.results['LW3_clearsky_correct_flux'] = self.results['LW3_clearsky_flux']+self.results['LW_cloud_kernel_term_flux']
		self.results['LW3_clearsky_correct'] = self.results['LW3_clearsky_correct_flux']/self.results['D']
	'''

	Calculate total SW term without decomposition
	'''
	def calcSWterms(self):
		self.dataset1['SW'] = self.dataset1['rsdt']-self.dataset1['rsut']
		self.dataset2['SW'] = self.dataset2['rsdt']-self.dataset2['rsut']
		self.dataset1['SW_cs'] = self.dataset1['rsdt']-self.dataset1['rsutcs']
		self.dataset2['SW_cs'] = self.dataset2['rsdt']-self.dataset2['rsutcs']
		
		self.results['SW'] = self.calcDelta('SW')/self.results['D']
		self.results['SW_cs'] = self.calcDelta('SW_cs')/self.results['D']
		self.results['SW_cloud'] = self.results['SW']-self.results['SW_cs']
		#self.results['SW_clearsky'] = self.calcDelta('rsutcs')/self.results['D']
		#self.results['SW_cre'] = self.results['SW_total']-self.results['SW_clearsky']
		
		
	'''
	calculate change in surface fluxes
	'''
	def calcSURFterm(self):
		self.surfHelp(self.dataset1)
		self.surfHelp(self.dataset2)
		self.results['SURF'] = -self.calcDelta('NETSURF')/self.results['D']
		self.results['SURF_flux'] = -self.calcDelta('NETSURF')
		self.results['NETSURF1'] = self.dataset1['NETSURF']
		self.results['NETSURF2'] = self.dataset2['NETSURF']


	'''
	calculate convergence term (horizontal energy transport)
	Calculated as residual from other terms
	'''
	def calcCONVterm(self,dataset):	
		dataset['CONV'] = dataset['rlut']+dataset['NETSURF']-dataset['SW']

	def calcCONVterms(self):
		self.calcCONVterm(self.dataset1)
		self.calcCONVterm(self.dataset2)
		self.results['CONV'] = self.calcDelta('CONV')/self.results['D']
		self.results['CONV_flux'] = self.calcDelta('CONV')

