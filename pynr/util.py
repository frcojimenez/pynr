import numpy as np
import scipy
from scipy.fft import fft, ifft
    # Functions to convert the nr data to physical units && Fourier transform tools
c=2.99792458*10**8;G=6.67259*10**(-11);MS=1.9885*10**30;parsec=3.0857*10**(16);Msol=1.9885*(10**30) ;
ConversionFac=7.4247138240457957979*10**(-28 );
Msec=Msol*ConversionFac/c;
Mm=Msol*ConversionFac;
MmMsec=Msec * Mm;
def time_to_t_NR(tau,M):
    """ It converts Physical times to NR units in [M].
    """  
    return 1/((M*MS*G)/c**3)*tau

def time_to_t_Phys(tau,M):
    """ It converts NR times to physical units in [s].
    """  
    return ((M*MS*G)/c**3)*tau

def f_to_Phys(f,M):
    """ It converts NR frequencies to physical units in [Hz].
    """  
    return (c**3/(M*MS*G))*f

def f_to_NR(f,M):
    """ It converts Physical frequencies to NR units in [1/M].
    """  
    return 1/((c**3/(M*MS*G)))*f

# pycbc spherical harmonics
def spher_harms(l, m, inclination,coa_phase=0):
    """Return spherical harmonic polarizations
    """

    # FIXME: we are using spin -2 weighted spherical harmonics for now,
    # when possible switch to spheroidal harmonics.
    Y_lm = lal.SpinWeightedSphericalHarmonic(inclination, coa_phase, -2, l, m).real

    return Y_lm

def amp_to_phys(amp,mass,distance,l=2,m=2,i=0,apply_spherical_harmonic=True):
    """Return amplitude in physical units. Mass is given in solar masses and distance in Mpc.
    """
    
    if apply_spherical_harmonic:
        spher_harm=spher_harms(l,m,i)
    else:
        spher_harm=1
        
    res=amp*spher_harm*G*mass*MS/(c**2*(distance*10**6*parsec))
    return res

def amp_to_nr(amp,mass,distance,l=2,m=2,i=0,apply_spherical_harmonic=True):
    """Return amplitude in physical units. Mass is given in solar masses and distance in Mpc.
    """
    if apply_spherical_harmonic:
        spher_harm=spher_harms(l,m,i)
    else:
        spher_harm=1
        
    res=amp*1/(spher_harm*G*mass*MS/(c**2*(distance*10**6*parsec)))
    return res

def amp_fd_to_NR(amp,mass,distance,l=2,m=2,i=0,apply_spherical_harmonic=True):
    """Return amplitude in physical units. Mass is given in solar masses and distance in Mpc.
    """
    if apply_spherical_harmonic:
        spher_harm=spher_harms(l,m,i)
    else:
        spher_harm=1
        
    res=amp/spher_harm*1/((mass**2*MmMsec)/(distance*parsec*10**6))
    return res
    

def amp_fd_to_phys(amp,mass,distance,l=2,m=2,i=0,apply_spherical_harmonic=True):
    """Return amplitude in physical units. Mass is given in solar masses and distance in Mpc.
    """
    
    if apply_spherical_harmonic:
        spher_harm=spher_harms(l,m,i)
    else:
        spher_harm=1
        
    res=amp*(1/(spher_harm*1/((mass**2*MmMsec)/(distance*parsec*10**6))))
    return res


def ZeroPadTimeSeries(ts,nleft,nright):
    times=ts[:,0]
    yarray=ts[:,1]
    starttime=times[0]
    endtime=times[-1]

    dt=times[1]-times[0]
    times_left=[starttime-nleft*dt+(i-1)*dt for i in range(nleft)]
    times_right=[endtime+i*dt for i in range(nright)]
    pad=np.pad(yarray, (nleft, nright), 'constant')
    times=np.concatenate((times_left,times,times_right))
    
    padded_ts =np.column_stack((times,pad))
    
    return padded_ts

def Window_Tanh(data,flo,sigmalo,fhi,sigmahi,times=[]):   
    
    if isinstance(data[0], np.ndarray):
        xaxis=data[:,0]
        yaxis=data[:,1] 
        wind=[(1/4.)*(1. + np.tanh(4*(xaxis[f] - flo)/sigmalo))*(1. - np.tanh(4*(xaxis[f] - fhi)/sigmahi))*yaxis[f] for f in range(len(xaxis))]

        res= np.column_stack((xaxis,wind))   
    else:
        xaxis = times
        yaxis = data
        wind=[(1/4.)*(1. + np.tanh(4*(xaxis[f] - flo)/sigmalo))*(1. - np.tanh(4*(xaxis[f] - fhi)/sigmahi))*yaxis[f] for f in range(len(xaxis))]
        res = wind            
    return res

#def Window_Tanh(data,flo,sigmalo,fhi,sigmahi):   
#    xaxis=data[:,0]
#    yaxis=data[:,1]
    
#    res=[(1/4.)*(1. + np.tanh(4*(f - flo)/sigmalo))*(1. - np.tanh(4*(f - fhi)/sigmahi)) for f in xaxis]
#    return np.column_stack((xaxis,res*yaxis))   

def FFT_FreqBins(times):
    Len = len(times)
    DeltaT = times[-1]- times[0]
    dt = DeltaT/(Len-1)
    dnu = 1/(Len*dt)
    maxfreq = 1/(2*dt)
    add = dnu/4

    p = np.arange(0.0,maxfreq+add,dnu)
    m = np.arange(p[-1]-(2*maxfreq)+dnu,-dnu/2+add,dnu)
    res=np.concatenate((p,m))
    
    return res

def FFT_ZeroPadded_Windowed(data,nleft=10,nright=10,x_low=-100,sigma_low=50,x_high=100,sigma_high=50):
    zero_padded = ZeroPadTimeSeries(data,nleft,nright)
    windowed = Window_Tanh(zero_padded,x_low,sigma_low,x_high,sigma_high)
    
    xaxis = windowed[:,0]
    yaxis = windowed[:,1]
    dt = windowed[1,0]-windowed[0,0]
    ft = dt*fft(yaxis)
    xf = FFT_FreqBins(xaxis.real).real
    
    ft_pos = ft[:np.argmax(xf <0)]
    xf_pos = xf[:np.argmax(xf <0)]
    
    return np.column_stack((xf_pos,ft_pos))
    
def select_cases(cases,regexp):
        """Select cases find within the list of 'cases' that match the regular expresion 'regexp'."""
        res = list(filter(regexp.match, cases)) 
        return res
    
def check_type(expr,valtype):
        """Check if expr is type."""
        return isinstance(expr,valtype)
    
def spher_harms(l, m, inclination,coa_phase=0):
    """Return spherical harmonic polarizations
    """

    # FIXME: we are using spin -2 weighted spherical harmonics for now,
    # when possible switch to spheroidal harmonics.
    Y_lm = lal.SpinWeightedSphericalHarmonic(inclination, coa_phase, -2, l, m)

    return Y_lm

def duplicates_pos(test_list):
    a, seen, result = test_list, set(), []
    
    for idx, item in enumerate(a):
        if item not in seen:
            seen.add(item)          # First time seeing the element
        else:
            result.append(idx)    
    return result
