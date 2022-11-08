# Copyright (C) 2021 Xisco Jimenez Forteza
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
# Module to load and postprocess SXS NR data
import numpy as np
import glob
import os
import re
import json
import sympy
import lal
import h5py
from scipy.interpolate import interp1d
import sxs
import romspline
import pynr.util as ut
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True


def nr_waveform(download_Q=True,root_folder=None,pycbc_format=True,modes=[[2,2],[2,-2]],
                distance=100, inclination=0,coa_phase=0,modes_combined=True,tapering=True,RD=False,
                zero_align=True,extrapolation_order=2,resolution_level=-1, **args):
    '''
    Function to load the waveforms from the NR catalogues. 
    
        Parameters
    ----------
    code: {'SXS','RIT','MAYA'}. Select the catalogue you want the data from.
    tag: eg. {'SXS:BBH:0305','RIT-BBH-0001','GT0001'}. Tag of the target waveform.
    download_Q: logical, optional, Default = True. If True downloads the data from the catalogue url.
    root_folder: string, optional, Default = None. If download_Q = False, it must be provided.
    pycbc_format: logical, optional, Default=True. If True it provides hp, hc in the Timeseries format.
    
    mass: Total mass in solar masses. 
    modes: list of modes, optional, Default= [[2,2],[2,-2]].
    distance: Distance, optional, Default = 100 Mpc.
    inclination: Inclination, optional, Default = 0.
    coa_phase: Coalescense phase, optional, Default = 0.
    
    modes_combined: list, optional, Default = True. If False it splits the list of individual modes.
    delta_t: time sampling. 
    tapering: logical, optional, Default = True. If True it applies a Tanh tappering function at the initial and ending points of the data.
    RD: logical, optional, Default= False. If True it provides the ringdown part.
    zero_align: logical, optional, Default= True. If True it aligns the waveform such that the peak of the strain is at t=0.   
    
    -- Applied for the SXS data.
    extrapolation_order: int, {2,3,4}, Default=2. SXS extrapolation order.
    resolution_level: int, {1,2,...}, Default = -1. It takes by default the best resolution available.

    
    Returns
    -------
    hplus: The plus polarization of the waveform.
    hcross: The cross polarization of the waveform.
    
    '''
    
    if args['code'] == 'SXS':
        hp,hc = sxs_waveform(download_Q=download_Q,root_folder=root_folder,pycbc_format=pycbc_format,modes=modes,
                distance=distance, inclination=inclination,coa_phase=coa_phase,modes_combined=modes_combined,tapering=tapering,RD=RD,
                zero_align=zero_align,extrapolation_order=extrapolation_order,resolution_level=resolution_level, **args) 
    elif args['code'] == 'RIT':
        hp,hc = rit_waveform(download_Q=download_Q,root_folder=root_folder,pycbc_format=pycbc_format,modes=modes,
                distance=distance, inclination=inclination,coa_phase=coa_phase,modes_combined=modes_combined,tapering=tapering,RD=RD,
                zero_align=zero_align,**args) 
    elif  args['code'] == 'MAYA':
        hp,hc = maya_waveform(download_Q=download_Q,root_folder=root_folder,pycbc_format=pycbc_format,modes=modes,
                distance=distance, inclination=inclination,coa_phase=coa_phase,modes_combined=modes_combined,tapering=tapering,RD=RD,
                zero_align=zero_align,**args) 
    else:
        print('Catalogue not recognised')
        return
    
    return hp,hc
    
def sxs_waveform(**args):
    
    download_Q = args['download_Q']
    sxs_tag = args['tag']
    extrapolation_order = args['extrapolation_order']
    modes  = args['modes']
    mass  = args['mass']
    distance  = args['distance']
    inclination = args['inclination']
    coa_phase = args['coa_phase']
    modes_combined = args['modes_combined']
    delta_t = args['delta_t']
    tapering = args['tapering']
    RD = args['RD']
    zero_align = args['zero_align']    
    if download_Q:
        home=os.path.expanduser('~')
        resolution_level = args['resolution_level']
        sxs.load(sxs_tag+"/Lev/rhOverM",progress=False)
        h5file=sorted(glob.glob(home+"/.cache/sxs/"+sxs_tag+"*/**/rhOverM_Asymptotic_GeometricUnits_CoM.h5",recursive = True))[resolution_level]
    
    else:
        sxs_root_folder = args['root_folder']   
        sxs_resolutions = sorted(glob.glob(sxs_root_folder+"/"+sxs_tag.replace(":","_")+"/*"))
        res_length = len(sxs_resolutions)
        resolution_level = args['resolution_level']
        sxs_case = sxs_resolutions[resolution_level]
        h5file=glob.glob(sxs_case+"/"+"rhOverM_Asymptotic_GeometricUnits_CoM.h5", recursive = True)[0]
    
    sampling=ut.time_to_t_NR(delta_t,mass)
    if modes_combined:
        print("Extracting and combining modes");
        hwave = Generate_SXS_Waveform(h5file,modes,extrapolation_order=extrapolation_order,zero_align=zero_align,resample=True,sampling_rate=sampling,modes_combined=modes_combined,inclination=inclination,coa_phase=coa_phase)
    else:
        hwave = Generate_SXS_Waveform(h5file,modes,extrapolation_order=extrapolation_order,zero_align=zero_align,resample=True,sampling_rate=sampling,modes_combined=modes_combined,inclination=inclination,coa_phase=coa_phase)[0]
    
    if RD:
        boolean = hwave[:,0]>= 0
        hwave = hwave[boolean]
        tlow = 0
    else:
        tlow = 1000
    
    if tapering: 
        hwave = ut.Window_Tanh(hwave,hwave[0,0]+tlow,100,150,100)
        hwave = ut.ZeroPadTimeSeries(hwave,10000,1000)    
    
    times = ut.time_to_t_Phys(hwave[:,0].real,mass)
    wf = ut.amp_to_phys(hwave[:,1],mass,distance,apply_spherical_harmonic=False)
    tmrg = times[np.argmax(np.abs(wf))]
    

    dt = ut.time_to_t_Phys(sampling,mass)
    start_time = times[0].real
    
    if args['pycbc_format']==True :
        import pycbc
        from pycbc.types import TimeSeries
        wf = TimeSeries(wf, delta_t=dt,epoch=start_time)    
        hp, hc = wf.real(), wf.imag()
    else:
        hp, hc =  np.stack((times,wf.real)).T, np.stack((times,wf.imag)).T

    return hp, hc
    

def rit_waveform(**args):
    
    download_Q = args['download_Q']
    rit_tag = args['tag']
    modes  = args['modes']
    mass  = args['mass']
    distance  = args['distance']
    inclination = args['inclination']
    coa_phase = args['coa_phase']
    modes_combined = args['modes_combined']
    delta_t = args['delta_t']
    tapering = args['tapering']
    RD = args['RD']
    zero_align = args['zero_align']     
    
    rit_tag=rit_tag.replace(':','-')
        
    if download_Q:
        from urllib.request import urlopen
        from bs4 import BeautifulSoup
        import ssl
        import re
        import urllib

        context = ssl._create_unverified_context()
        URL = "https://ccrgpages.rit.edu/~RITCatalog/"
        res = urlopen(URL,context=context)
        soup = BeautifulSoup(res, "html.parser")
        htmldata = soup.find_all(href=True,recursive=True)
        metadata=np.array([str(s['href']) for s in htmldata if "Metadata" in str(s)]);
        nrdata=np.array([str(s['href']) for s in htmldata if "ExtrapStrain_RIT" in str(s)]);
        rit_tag=rit_tag.replace(':','-')
        r = re.compile(".*"+rit_tag+".*")
        mycase=select_cases(nrdata,r)
        export_user=os.path.expanduser('~/.cache/')
        ssl._create_default_https_context = ssl._create_unverified_context

        response=urllib.request.urlretrieve(URL+mycase[0],export_user+rit_tag+'.h5')   
        h5file=export_user+rit_tag+'.h5'

    else:
        rit_root_folder = args['root_folder']   
        rit_resolutions = sorted(glob.glob(rit_root_folder+"/*"+rit_tag.replace(":","-")+"*"))
        rit_case = rit_resolutions[-1]
        h5file=rit_case
    
    sampling=ut.time_to_t_NR(delta_t,mass)
    if modes_combined:
        print("Extracting and combining modes");
        hwave = Generate_Waveform(h5file,modes,zero_align=zero_align,sampling_rate=sampling,modes_combined=modes_combined,inclination=inclination,coa_phase=coa_phase)
    else:
        hwave = Generate_Waveform(h5file,modes,zero_align=zero_align,sampling_rate=sampling,modes_combined=modes_combined,inclination=inclination,coa_phase=coa_phase)[0]
    
    if RD:
        boolean = hwave[:,0]>= 0
        hwave = hwave[boolean]
        tlow = 0
    else:
        tlow = 1000
    
    if tapering: 
        hwave = ut.Window_Tanh(hwave,hwave[0,0]+tlow,100,150,100)
        hwave = ut.ZeroPadTimeSeries(hwave,10000,1000)    
    
    times = ut.time_to_t_Phys(hwave[:,0].real,mass)
    wf = ut.amp_to_phys(hwave[:,1],mass,distance,apply_spherical_harmonic=False)
    tmrg = times[np.argmax(np.abs(wf))]
    dt = ut.time_to_t_Phys(sampling,mass)
    start_time = times[0].real
    
    if args['pycbc_format']==True :
        import pycbc
        from pycbc.types import TimeSeries
        wf = TimeSeries(wf, delta_t=dt,epoch=start_time)    
        hp, hc = wf.real(), wf.imag()
    else:
        hp, hc =  np.stack((times,wf.real)).T, np.stack((times,wf.imag)).T

    return hp, hc


def maya_waveform(**args):
    
    download_Q = args['download_Q']
    maya_tag = args['tag']
    modes  = args['modes']
    mass  = args['mass']
    distance  = args['distance']
    inclination = args['inclination']
    coa_phase = args['coa_phase']
    modes_combined = args['modes_combined']
    delta_t = args['delta_t']
    tapering = args['tapering']
    RD = args['RD']
    zero_align = args['zero_align']     
    
    maya_tag=maya_tag.replace(':','')
    maya_tag=maya_tag.replace('-','')

    if download_Q:
        from urllib.request import urlopen
        from bs4 import BeautifulSoup
        import ssl
        import re
        import urllib

        context = ssl._create_unverified_context()
        URL = "https://github.com/cevans216/gt-waveform-catalog/raw/master/h5files/"
        export_user=os.path.expanduser('~/.cache/')
        ssl._create_default_https_context = ssl._create_unverified_context
        response=urllib.request.urlretrieve(URL+maya_tag+'.h5',export_user+maya_tag+'.h5')   
        h5file=export_user+maya_tag+'.h5'

    else:
        maya_root_folder = args['root_folder']   
        maya_resolutions = sorted(glob.glob(maya_resolutions+"/*"+maya_tag.replace(":","-")+"*"))
        maya_case = maya_resolutions[-1]
        h5file=maya_case
    
    sampling=ut.time_to_t_NR(delta_t,mass)
    if modes_combined:
        print("Extracting and combining modes");
        hwave = Generate_Waveform(h5file,modes,zero_align=zero_align,sampling_rate=sampling,modes_combined=modes_combined,inclination=inclination,coa_phase=coa_phase)
    else:
        hwave = Generate_Waveform(h5file,modes,zero_align=zero_align,sampling_rate=sampling,modes_combined=modes_combined,inclination=inclination,coa_phase=coa_phase)[0]
    
    if RD:
        boolean = hwave[:,0]>= 0
        hwave = hwave[boolean]
        tlow = 0
    else:
        tlow = 1000
    
    if tapering: 
        hwave = ut.Window_Tanh(hwave,hwave[0,0]+tlow,100,150,100)
        hwave = ut.ZeroPadTimeSeries(hwave,10000,1000)    
    
    times = ut.time_to_t_Phys(hwave[:,0].real,mass)
    wf = ut.amp_to_phys(hwave[:,1],mass,distance,apply_spherical_harmonic=False)
    tmrg = times[np.argmax(np.abs(wf))]
    dt = ut.time_to_t_Phys(sampling,mass)
    start_time = times[0].real
    
    if args['pycbc_format']==True :
        import pycbc
        from pycbc.types import TimeSeries
        wf = TimeSeries(wf, delta_t=dt,epoch=start_time)    
        hp, hc = wf.real(), wf.imag()
    else:
        hp, hc =  np.stack((times,wf.real)).T, np.stack((times,wf.imag)).T

    return hp, hc

def sxs_waveform_metadata(**args):
    '''
    Function to find and load the SXS NR strain waveforms. 
    The following inputs must be provided:
    'sxs_root_folder': root folder where SXS data is stored.
    'tag': SXS waveform tag. If the SXS data is stored in your local machine, it must correspond to the folder name of each simulation. In case you want to directly download the data, the tag must be in the following format: SXS_BBH_####.
    'resolution': Resolution level wanted for this waveform. -1 stands for the highest resolution and 1 for the lowest.
    'extrapolation_order': extrapolation order desired. 
    
    '''

    sxs_tag = args['tag']

    if args['download_Q']:
        home=os.path.expanduser('~')
        resolution_level = args['resolution_level']
        sxs.load(sxs_tag+"/Lev/metadata.json",progress=False)
        json_metafile=sorted(glob.glob(home+"/.cache/sxs/"+sxs_tag+"*/**/metadata.json",recursive = True))[resolution_level]
          
    else:
        sxs_root_folder = args['sxs_root_folder']   
        sxs_resolutions = sorted(glob.glob(sxs_root_folder+"/"+sxs_tag.replace(":","_")+"/*"))
        res_length = len(sxs_resolutions)
        resolution_level = args['resolution_level']
        sxs_case = sxs_resolutions[resolution_level]
        json_metafile=glob.glob(sxs_case+"/"+"metadata.json", recursive = True)[0]
    
    
    with open(json_metafile) as file:
        metadata = json.load(file)
    
    return metadata

def find_hd5files(tags,sxs_root_folder,catalogue='SXS',export=False,filepath=""):
        """Find the .json files."""
        if catalogue == 'SXS':
            tags0 = tags[0].replace(":","_")
            rootpath = glob.glob(sxs_root_folder+"/**/"+tags0,recursive = True)[0]
            path = os.path.dirname(rootpath)

            hd5files = [sorted(glob.glob(path+"/"+x.replace(":","_")+"/**/rhOverM_Asymptotic_GeometricUnits_CoM.h5", recursive = True),reverse=True)[0] for x in tags]
        elif catalogue == 'RIT':
            hd5files=find_jsonfiles()
            hd5files = [(x.split("-id")[0].replace("RIT_BBH_","ExtrapStrain_RIT-BBH-")+'.h5').replace('Metadata','Data') for x in hd5files]
            
        if export:
            textfile = open(filepath, "w")
            for element in hd5files:
                   textfile.write(element + "\n")
            textfile.close()
            
        return hd5files
    
def find_jsonfiles(tags,sxs_root_folder,catalogue='SXS',export=False,filepath=""):
        """Find the .json files."""
        if catalogue == 'SXS':
            tags0 = tags[0].replace(":","_")
            rootpath = glob.glob(sxs_root_folder+"/**/"+tags0,recursive = True)[0]
            path = os.path.dirname(rootpath)

            hd5files = [sorted(glob.glob(path+"/"+x.replace(":","_")+"/**/metadata.json", recursive = True),reverse=True)[0] for x in tags]
        elif catalogue == 'RIT':
            hd5files=find_jsonfiles()
            hd5files = [(x.split("-id")[0].replace("RIT_BBH_","Metadata")+'.json').replace('Metadata','Data') for x in hd5files]
            
        if export:
            textfile = open(filepath, "w")
            for element in hd5files:
                   textfile.write(element + "\n")
            textfile.close()
            
        return hd5files

def Generate_SXS_Waveform(h5file,modes,extrapolation_order=3,zero_align=True,resample=True,sampling_rate=0.5,modes_combined=True,inclination=0,coa_phase=0,RD=False,toffset=10):          
    
        #print('Check the sxs convention. Now is h= h+ - i hx')
        gw_nr = h5py.File(h5file, 'r')["Extrapolated_N"+str(extrapolation_order)+".dir"]
        if modes_combined:
            h=0
            for x in modes:
                gw_nr_all = gw_nr["Y_l"+str(x[0])+"_m"+str(x[1])+".dat"]
                times = gw_nr_all[:,0]
                hp=gw_nr_all[:,1]
                hc=gw_nr_all[:,2]
                h+=(hp -1j*hc)*spher_harms(x[0],x[1],inclination,coa_phase=coa_phase)  

            if resample:
                dt= sampling_rate
                h_int=interp1d(times, h, kind='cubic')
                times= np.arange(times[0], times[-1], dt).real
                h=h_int(times)

            if zero_align:
                tmrg = times[np.argmax(np.abs(h))]
                times = times - tmrg 

            h_final = np.stack((times.real,h)).T
        else:
            h=[None]*len(modes)
            h_final=[None]*len(modes)
            a=0
            for x in modes:
                gw_nr_all = gw_nr["Y_l"+str(x[0])+"_m"+str(x[1])+".dat"]
                times = gw_nr_all[:,0].real
                hp=gw_nr_all[:,1]
                hc=gw_nr_all[:,2]
                h[a]=(hp -1j*hc) 
                if resample:
                    dt= sampling_rate
                    h_int=interp1d(times, h[a], kind='cubic')
                    times= np.arange(times[0], times[-1], dt).real
                    h[a]=h_int(times)            

                if zero_align:
                    tmrg = times[np.argmax(np.abs(h[a]))]
                    times = times - tmrg 
                times=times.real
                h_final[a] = np.stack((times,h[a])).T
                a=a+1
            
        if RD:
            tmax22=np.abs(h_final[0][:,1]).argmax()
            tcut=times[tmax22]+toffset
            boolean = times>= tcut
            times=times[boolean]-times[tmax22]
            h_final=[np.stack((times,h_final[i][:,1][boolean])).T for i in range(len(h_final))]
            
        return h_final
    
def Generate_Waveform(h5file,modes,zero_align=True,sampling_rate=0.1,modes_combined=True,inclination=0,coa_phase=0,RD=False,toffset=10):
            
        if modes_combined:
            h=0
            for x in modes:
                amp=romspline.readSpline(h5file,group='amp_l'+str(x[0])+'_m'+str(x[1]))
                phase=romspline.readSpline(h5file,group='phase_l'+str(x[0])+'_m'+str(x[1]))

                time_amp=amp.X
                amp = amp.Y
                time_ph=phase.X
                phase=phase.Y
                amp_int = romspline.ReducedOrderSpline(time_amp, amp,verbose=False)
                ph_int = romspline.ReducedOrderSpline(time_ph, phase,verbose=False)
                tmin=max(time_ph[0],time_amp[0])
                tmax=min(time_ph[-1],time_amp[-1])
                times=np.linspace(tmin,tmax,int((tmax-tmin)/sampling_rate))
                wave=amp_int(times)*np.exp(-1j*ph_int(times)) 
                h+=wave*spher_harms(x[0],x[1],inclination,coa_phase=coa_phase)  

            if zero_align:
                tmrg = times[np.argmax(np.abs(h))]
                times = times - tmrg 

            h_final = np.stack((times.real,h)).T
        else:
            h=[None]*len(modes)
            h_final=[None]*len(modes)
            a=0
            for x in modes:
                amp=romspline.readSpline(h5file,group='amp_l'+str(x[0])+'_m'+str(x[1]))
                phase=romspline.readSpline(h5file,group='phase_l'+str(x[0])+'_m'+str(x[1]))
                time_amp=amp.X
                amp = amp.Y
                time_ph=phase.X
                phase=phase.Y
                amp_int = romspline.ReducedOrderSpline(time_amp, amp,verbose=False)
                ph_int = romspline.ReducedOrderSpline(time_ph, phase,verbose=False)
                tmin=max(time_ph[0],time_amp[0])
                tmax=min(time_ph[-1],time_amp[-1])
                times=np.linspace(tmin,tmax,int((tmax-tmin)/sampling_rate))
                h[a]=amp_int(times)*np.exp(-1j*ph_int(times)) 

                if zero_align:
                    tmrg = times[np.argmax(np.abs(h[a]))]
                    times = times - tmrg 
                times=times.real
                h_final[a] = np.stack((times,h[a])).T
                a=a+1
            
        if RD:
            tmax22=np.abs(h_final[0][:,1]).argmax()
            tcut=times[tmax22]+toffset
            boolean = times>= tcut
            times=times[boolean]-times[tmax22]
            h_final=[np.stack((times,h_final[i][:,1][boolean])).T for i in range(len(h_final))]

        return h_final
    
def select_cases(cases,regexp):
        """Select cases find within the list of 'cases' that match the regular expresion 'regexp'."""
        res = list(filter(regexp.match, cases)) 
        return res
    
# this function will be used to select the data in terms of the physical parameters

    
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

import numpy as np

def sxs_catalogue_select(conditions,verbose=False):
    import requests_cache
    '''Function used to select a subdomain of the sxs catalogue taking the json data stored at https://arxiv.org/src/1904.04831/anc/sxs_catalog.json. 
        The conditions are a list composed of a [keyword, range]. The keywords must be consistent with the ones found at the SXS json files :
        [{BBH,NSNS,BHNS},
        ['reference_mass_ratio',[2,4]],
        ['remnant_dimensionless_spin',[0,1]],
        ['reference_chi_eff',[0,0.5]],
        [reference_chi1_perp,[-0.001,0.001]],
        [reference_chi2_perp,[-0.001,0.001]]] 
    '''
    
    session = requests_cache.CachedSession('cached_sxs')

    request = session.get("https://arxiv.org/src/1904.04831/anc/sxs_catalog.json", headers={'accept': 'application/citeproc+json'})
    sxs_catalog_json = request.json()
    sxs_keys=sxs_catalog_json.keys()

    sublist={}
    for i in sxs_keys:
        if i.split(":")[-2] == "BBH":
            sublist[i]=sxs_catalog_json[i]
        elif i.split(":")[-2] == "NSNS":
            sublist[i]=sxs_catalog_json[i]
        elif i.split(":")[-2] == "BHNS":
            sublist[i]=sxs_catalog_json[i]

    sublist_keys=sublist.keys()

    for j in conditions[1:]:
        subsubkeys=[]
        for k in sublist_keys:
            if len(j)==2:
                if j[0]=='remnant_dimensionless_spin':
                    cond_val=np.linalg.norm(sublist[k][j[0]])
                else:
                    cond_val=sublist[k][j[0]]
                try:
                    if np.logical_and(j[1][0]<=cond_val, j[1][1]>=cond_val):
                        subsubkeys.append(k)
                except:
                    None
        sublist_keys=subsubkeys
        
    if verbose:
        import pandas as pd
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

        condition_tags=[i[0] for i in conditions[1:]]
        vals =[[sxs_catalog_json[j][k] for k in condition_tags] for j in sublist_keys]
        [vals[i].insert(0,sublist_keys[i]) for i in range(len(sublist_keys))]
        pd.set_option('display.max_columns', None)
        df = pd.DataFrame(vals, columns = ['Tag']+condition_tags)
        print(df)
       
    return sublist_keys