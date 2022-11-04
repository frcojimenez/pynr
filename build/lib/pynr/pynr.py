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
import jsons
import sympy
import lal
import h5py
from scipy.interpolate import interp1d
import sxs
import romspline
import pynr.util as ut

def nr_waveform(download_Q=True,root_folder=None,pycbc_format=True,modes=[[2,2],[2,-2]],
                distance=100, inclination=0,coa_phase=0,modes_combined=True,tapering=True,RD=False,
                zero_align=True,extrapolation_order=2,resolution_level=-1, **args):
    '''
    Function to load the waveforms from the NR catalogues. 
    
        Parameters
    ----------
    code: {'SXS','RIT'}. Select the catalogue you want the data from.
    tag: eg. {'SXS:BBH:0305','RIT-BBH-0001'}. Tag of the target waveform.
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
        print
        hp,hc = rit_waveform(download_Q=download_Q,root_folder=root_folder,pycbc_format=pycbc_format,modes=modes,
                distance=distance, inclination=inclination,coa_phase=coa_phase,modes_combined=modes_combined,tapering=tapering,RD=RD,
                zero_align=zero_align,**args) 
    elif  args['code'] == 'MAYA':
        hp,hc = maya_waveform_1(download_Q=True,root_folder=None,pycbc_format=True,modes=[[2,2],[2,-2]],
                distance=100, inclination=0,coa_phase=0,modes_combined=True,tapering=True,RD=False,
                zero_align=True,extrapolation_order=2,resolution_level=-1,**args) 
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
    sxs_init=SXS_catalogue('',[])
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
        hwave = Window_Tanh(hwave,hwave[0,0]+tlow,1000,150,100)
        hwave = ZeroPadTimeSeries(hwave,10000,1000)    
    
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
        hwave = Generate_RIT_Waveform(h5file,modes,zero_align=zero_align,sampling_rate=sampling,modes_combined=modes_combined,inclination=inclination,coa_phase=coa_phase)
    else:
        hwave = Generate_RIT_Waveform(h5file,modes,zero_align=zero_align,sampling_rate=sampling,modes_combined=modes_combined,inclination=inclination,coa_phase=coa_phase)[0]
    
    if RD:
        boolean = hwave[:,0]>= 0
        hwave = hwave[boolean]
        tlow = 0
    else:
        tlow = 1000
    
    if tapering: 
        hwave = Window_Tanh(hwave,hwave[0,0]+tlow,1000,150,100)
        hwave = ZeroPadTimeSeries(hwave,10000,1000)    
    
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
        metadata = jsons.load(file)
    
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
    
def Generate_RIT_Waveform(h5file,modes,zero_align=True,sampling_rate=0.1,modes_combined=True,inclination=0,coa_phase=0,RD=False,toffset=10):
            
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
def SXS_Catalogue_Select_cases(sxs_root_folder,select_patterns,tolerance=0.001,sortcolumn=2,best_resolution=True,verbose=False):
    '''Function used to select a subdomain of the sxs catalogue stored at sxs_root_folder. 
       Some example select_patterns keywords are [['BHBH',None],['Non-Precessing',None],['Massratio',[>=1,<=10],
       ['Spin-eff',['>=-0.5','<=0.5']],['Remnant-Spin', ['>=0.1','<=0.5']],['Remnant-Mass', ['>=0.88','<=0.97']],['Eccentricity',['>=0.01','<=0.5']]] 
    '''
    # find metadata files. If best_resolution = True, it takes the best resolution. Otherwise, it will consider all the resolutions available.
    if best_resolution==-1:
        json_metafiles = sorted(glob.glob(sxs_root_folder+"/**/metadata.json", recursive = True))
        tag_name=[i.split(os.sep)[5] for i in json_metafiles]
        res = [max(idx for idx, val in enumerate(tag_name) if val == x) for x in tag_name]
        pos_best_resolution=list(dict.fromkeys(res))
        json_metafiles = list(np.array(json_metafiles)[pos_best_resolution])
        tags_aux = [i.split(os.sep)[5].replace("_",":") for i in json_metafiles]
    else:
        json_metafiles = sorted(glob.glob(sxs_root_folder+"/**/metadata.json", recursive = True))
        tag_name=[i.split(os.sep)[5] for i in json_metafiles]
        json_metafiles_aux=[None]*len(json_metafiles)
        for i in range(len(json_metafiles)):
            path = os.path.normpath(json_metafiles[i])
            tesp=path.split(os.sep)[0:6]
            tesp[0]='/'
            json_metafiles_aux[i]=os.path.join(*tesp)
        res = []
        [res.append(x) for x in json_metafiles_aux if x not in res];
        res=list(filter(None, res))
        json_metafiles=[None]*len(res)
        for i in range(len(res)):
            json_metafiles[i]=sorted(glob.glob(res[i]+"/**/metadata.json", recursive = True))
            if len(json_metafiles[i])!=1:
                json_metafiles[i]=json_metafiles[i][-2]
            else:
                json_metafiles[i]=json_metafiles[i][0]
        
        tags_aux = [i.split(os.sep)[5].replace("_",":") for i in json_metafiles]
        
    if select_patterns[0,0]=='BHBH':
        r = re.compile(".*BBH.*")
        tags = select_cases(tags_aux,r)
        position = [tags_aux.index(x) for x in tags]
        json_metafiles=[json_metafiles[x] for x in position]
    else:
        r = re.compile(".*"+select_patterns[0,0]+".*")
        tags = select_cases(tags_aux,r)
        position = [tags_aux.index(x) for x in tags]
        json_metafiles=[json_metafiles[x] for x in position]

    # load all metadata
    metadata = {}
    for i in range(len(json_metafiles)):
        with open(json_metafiles[i]) as file:
            metadata[tags[i]] = jsons.load(file)
    
    print('Found ', len(tags), 'metadata.json files')
    
    object_type=np.asarray([None]*len(tags))
    mass_1=np.asarray([None]*len(tags))
    mass_2=np.asarray([None]*len(tags))
    mass_ratio=np.asarray([None]*len(tags))
    spin1=np.asarray([None]*len(tags))
    spin2=np.asarray([None]*len(tags))
    chi_eff=np.asarray([None]*len(tags))
    chi_p=np.asarray([None]*len(tags))
    eccentricity=np.asarray([None]*len(tags))
    remnant_mass=np.asarray([None]*len(tags))
    remnant_spin=np.asarray([None]*len(tags))
    remnant_speed=np.asarray([None]*len(tags))
    angle_1=np.asarray([None]*len(tags))
    angle_2=np.asarray([None]*len(tags))
    tags_aux=np.asarray([None]*len(tags))
    simulation_name=np.asarray([None]*len(tags))
    

    for i in range(len(tags)):
        object_type[i] = metadata[tags[i]]['object_types']
        if  object_type[i]=='BHBH':
            tags_aux[i]=tags[i]
            mass_1[i] = metadata[tags[i]]['initial_mass1']
            mass_2[i] = metadata[tags[i]]['initial_mass2']
            mass_ratio[i] = metadata[tags[i]]['reference_mass_ratio']
            spin1[i] = np.around(metadata[tags[i]]['reference_dimensionless_spin1'],4)
            spin2[i] = np.around(metadata[tags[i]]['reference_dimensionless_spin2'],4)
            chi_eff[i] = mass_1[i]*spin1[i][2] + mass_2[i]*spin2[i][2]
        
            A1=(2+3*mass_ratio[i]/(2.))
            A2=(2+3/(2.*mass_ratio[i]))        
            chi_p1 = metadata[tags[i]]['reference_chi1_perp']*mass_1[i]**2
            chi_p2 = metadata[tags[i]]['reference_chi2_perp']*mass_2[i]**2
            chi_p[i] = max(A1*chi_p1,A2*chi_p2)/(A1*mass_1[i]**2)           

            eccentricity[i] = metadata[tags[i]]['reference_eccentricity']
            remnant_mass[i] = metadata[tags[i]]['remnant_mass']
            #remnant_spin[i] = np.linalg.norm(metadata[tags[i]]['remnant_dimensionless_spin'])
            remnant_spin[i] = metadata[tags[i]]['remnant_dimensionless_spin']
            simulation_name[i] = metadata[tags[i]]["simulation_name"]
            #remnant_speed[i] = metadata[tags[i]]['remnant_velocity']
            #initial_distance = metadata[tags[i]]['initial_separation']
        elif object_type[i]=='BHNS':
            tags_aux[i]=tags[i]
        elif object_type[i]=='NSNS':
            tags_aux[i]=tags[i]
    if any(t == 'BHNS' or t == 'NSNS' for t in select_patterns[:,0]):
        print('Metadata is corrupted for almost all the BHNS and NSNS cases. It has not been possible to classify them.')
        return(tags_aux)
    bh_parameters = np.stack((tags_aux,object_type,mass_ratio,chi_eff,spin1,spin2,chi_p,eccentricity,remnant_mass,remnant_spin,simulation_name)).T    
 
    select_conditions = select_patterns[:,0]
    for i in range(len(select_conditions)):
        
        if select_conditions[i]=='BHBH':  
            boolean = bh_parameters[:,1] ==  select_conditions[i]
            bh_parameters = bh_parameters[boolean]
    
        if select_conditions[i]=='Tag':  
            #boolean = bh_parameters[:,0] ==  select_patterns[i,1] np.intersect1d(a,b)
            intersect, ind_a, ind_b = np.intersect1d(bh_parameters[:,0],select_patterns[i,1], return_indices=True)
            bh_parameters = bh_parameters[ind_a]
            
        elif select_conditions[i]=='Non-Precessing':
            #boolean = bh_parameters[:,6] <=  tolerance
            #bh_parameters = bh_parameters[boolean]
                    
            prec_str=bh_parameters[:,6].astype('str')
            boolean = np.array([sympy.sympify(x.replace("<","").replace("NaN","0")) for x in prec_str])<= tolerance
            boolean = boolean.astype(bool)
            bh_parameters = bh_parameters[boolean]

        elif select_conditions[i]=='Index':
            eval_conditions = np.array(select_patterns[i,1])
            index_str=bh_parameters[:,0].astype('str')
            index_str=np.array([str(int(i.split(":")[-1])) for i in index_str])
            boolean = np.array([sympy.sympify(x+eval_conditions[0]) for x in index_str])
            boolean = boolean.astype(bool)
            bh_parameters = bh_parameters[boolean]
            
            index_str=bh_parameters[:,0].astype('str')
            index_str=np.array([str(int(i.split(":")[-1])) for i in index_str])
            boolean = np.array([sympy.sympify(x+eval_conditions[1]) for x in index_str])          
            boolean = boolean.astype(bool)    
            bh_parameters = bh_parameters[boolean]

        
        elif select_conditions[i]=='Precessing':
            boolean = bh_parameters[:,6] > tolerance
            bh_parameters = bh_parameters[boolean]
        
        elif select_conditions[i]=='Non-Spinning':
            norm1=[np.linalg.norm(x) for x in bh_parameters[:,4]]
            norm2=[np.linalg.norm(x) for x in bh_parameters[:,5]]
            boolean1 = np.array(norm1) <=  tolerance
            boolean2 = np.array(norm2) <=  tolerance
            boolean  = list(boolean1) or list(boolean2)
            bh_parameters = bh_parameters[boolean]
        
        elif select_conditions[i]=='Massratio':
            
            eval_conditions = np.array(select_patterns[i,1])
            mass_ratio_str=bh_parameters[:,2].astype('str')
            boolean = np.asarray([sympy.sympify(x+eval_conditions[0]) for x in mass_ratio_str])
            boolean = boolean.astype(bool)
            bh_parameters = bh_parameters[boolean]
            
            mass_ratio_str=bh_parameters[:,2].astype('str')
            boolean = np.asarray([sympy.sympify(x+eval_conditions[1]) for x in mass_ratio_str])          
            boolean = boolean.astype(bool)    
            bh_parameters = bh_parameters[boolean]
            
        elif select_conditions[i]=='Spin-1':
            
            eval_conditions = np.array(select_patterns[i,1])
            spin_eff_str=[np.linalg.norm(np.array(x).astype('str')) for x in bh_parameters[:,4]]
            boolean = np.asarray([sympy.sympify(str(x)+eval_conditions[0]) for x in spin_eff_str])
            boolean = boolean.astype(bool)
            bh_parameters = bh_parameters[boolean]
            
            spin_eff_str=[np.linalg.norm(np.array(x).astype('str')) for x in bh_parameters[:,4]]
            boolean = np.asarray([sympy.sympify(str(x)+eval_conditions[1]) for x in spin_eff_str])          
            boolean = boolean.astype(bool)    
            bh_parameters = bh_parameters[boolean]
            
        elif select_conditions[i]=='Spin-2':
            
            eval_conditions = np.array(select_patterns[i,1])
            spin_eff_str=[np.linalg.norm(np.array(x).astype('str')) for x in bh_parameters[:,5]]
            boolean = np.asarray([sympy.sympify(str(x)+eval_conditions[0]) for x in spin_eff_str])
            boolean = boolean.astype(bool)
            bh_parameters = bh_parameters[boolean]
            
            spin_eff_str=[np.linalg.norm(np.array(x).astype('str')) for x in bh_parameters[:,5]]
            boolean = np.asarray([sympy.sympify(str(x)+eval_conditions[1]) for x in spin_eff_str])          
            boolean = boolean.astype(bool)    
            bh_parameters = bh_parameters[boolean]
            
        
        elif select_conditions[i]=='Spin-eff':
            
            eval_conditions = np.array(select_patterns[i,1])
            spin_eff_str=bh_parameters[:,3].astype('str')
            boolean = np.asarray([sympy.sympify(x+eval_conditions[0]) for x in spin_eff_str])
            boolean = boolean.astype(bool)
            bh_parameters = bh_parameters[boolean]
            
            spin_eff_str=bh_parameters[:,3].astype('str')
            boolean = np.asarray([sympy.sympify(x+eval_conditions[1]) for x in spin_eff_str])          
            boolean = boolean.astype(bool)    
            bh_parameters = bh_parameters[boolean]
            
        elif select_conditions[i]=='Spin-p':
            
            eval_conditions = np.array(select_patterns[i,1])
            spin_p_str=bh_parameters[:,6].astype('str')
            boolean = np.asarray([sympy.sympify(x+eval_conditions[0]) for x in spin_p_str])
            boolean = boolean.astype(bool)
            bh_parameters = bh_parameters[boolean]
            
            spin_eff_str=bh_parameters[:,6].astype('str')
            boolean = np.asarray([sympy.sympify(x+eval_conditions[1]) for x in spin_p_str])          
            boolean = boolean.astype(bool)    
            bh_parameters = bh_parameters[boolean]
        
        elif select_conditions[i]=='Eccentricity':
            eval_conditions = np.array(select_patterns[i,1])
            ecc_eff_str=bh_parameters[:,7].astype('str')
            boolean = np.asarray([sympy.sympify(x.replace("<","").replace("NaN","0")+eval_conditions[0]) for x in ecc_eff_str])
            boolean = boolean.astype(bool)
            bh_parameters = bh_parameters[boolean]
            
            ecc_eff_str=bh_parameters[:,7].astype('str')
            boolean = np.asarray([sympy.sympify(x.replace("<","").replace("NaN","0")+eval_conditions[1]) for x in ecc_eff_str])          
            boolean = boolean.astype(bool)    
            bh_parameters = bh_parameters[boolean]
            
        elif select_conditions[i]=='Remnant-Spin':
            
            eval_conditions = np.array(select_patterns[i,1])
            spin_rm_str=np.linalg.norm(bh_parameters[:,9].astype('str'))
            boolean = np.asarray([sympy.sympify(x+eval_conditions[0]) for x in spin_rm_str])
            boolean = boolean.astype(bool)
            bh_parameters = bh_parameters[boolean]
            
            spin_rm_str=np.linalg.norm(bh_parameters[:,9].astype('str'))
            boolean = np.asarray([sympy.sympify(x+eval_conditions[1]) for x in spin_rm_str])          
            boolean = boolean.astype(bool)    
            bh_parameters = bh_parameters[boolean]
            
        elif select_conditions[i]=='Remnant-Mass':

            eval_conditions = np.array(select_patterns[i,1])
            mass_rm_str=bh_parameters[:,8].astype('str')
            boolean = np.asarray([sympy.sympify(x+eval_conditions[0]) for x in mass_rm_str])
            boolean = boolean.astype(bool)
            bh_parameters = bh_parameters[boolean]
            
            mass_rm_str=bh_parameters[:,8].astype('str')
            boolean = np.asarray([sympy.sympify(x+eval_conditions[1]) for x in mass_rm_str])          
            boolean = boolean.astype(bool)    
            bh_parameters = bh_parameters[boolean]
        
        else:
            print('Wrong conditional keyword: ', select_conditions[i])
        
        if sortcolumn==0 or sortcolumn==0:
            bh_parameters=np.array(sorted(bh_parameters, key= lambda x: x[sortcolumn]))
        else:
            bh_parameters=np.array(sorted(bh_parameters, key= lambda x: np.abs(x[sortcolumn])))
        
    if verbose:
        import pandas as pd
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

        test=np.array(bh_parameters)
        pd.set_option('display.max_columns', None)

        #df = pd.DataFrame(test[:,[0,2,3,6,10]], columns = ['Tag','q',r'$\chi_{eff}$',r'$\chi_{p}$','name'])
        df = pd.DataFrame(test[:,[0,2,3,6,4,5,10]], columns = ['Tag','q',r'$\chi_{eff}$',r'$\chi_{p}$',r'$\chi_{1}$',r'$\chi_{2}$','name'])
    return bh_parameters
    
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


class SXS_catalogue:
    def __init__(self,sxs_root_folder,conditions,tolerance=0.001,sort=True, sortcolumn=2, catalogue="SXS"):
        """This code incorporates all the functions to load and process SXS data."""

        self.sxs_root_folder = sxs_root_folder
        self.conditions = conditions
        self.tolerance = tolerance
        self.sort = sort
        self.sortcolumn = sortcolumn
        self.catalogue = catalogue
       
      
    def sxs_tags(self,json_metafiles):
        """Find the .json files."""
        if self.catalogue=="SXS":
            r= re.compile(".*SXS_.*")
            tags = [os.path.split(os.path.split(i)[0])[-1].replace("_",":") for i in json_metafiles]        
            tags = [select_cases(i.split(os.sep),r) for i in json_metafiles]
            tags = [i[0] for i in tags]
        elif self.catalogue=="RIT":
            r= re.compile(".*RIT.*")
            tags = [os.path.split(i)[-1] for i in json_metafiles]        
        return tags
    
    def nr_metadata_keys(self):
        if self.catalogue=="SXS":
            keys = ['object_types','initial_mass1','initial_mass2','reference_mass_ratio','reference_dimensionless_spin1',
                    'reference_dimensionless_spin2','reference_chi1_perp','reference_chi2_perp','reference_eccentricity',
                   'remnant_mass','remnant_dimensionless_spin']         
        elif  self.catalogue=="RIT":
            keys = ['catalog-tag','system-type','initial-mass1','initial-mass2','relaxed-mass-ratio-1-over-2','initial-bh-chi1z','initial-bh-chi2z','initial-bh-chi1x','initial-bh-chi1y','initial-bh-chi2x','initial-bh-chi2y','eccentricity','final-mass','final-chi']
            
        return keys
    
    def metadata(self,json_metafile,parameters_list=[]):
        """Output the metadata for a given json_metafile. To output a reduced set of metadata keywords you need to provide the parameter_list = ['reference_mass1',...] argument."""
        metadata = {}
        with open(json_metafile) as file:
            metadata = jsons.load(file)
        
        if len(parameters_list)>0:
            metadata_res=[None]*len(parameters_list)
            for i in range(len(parameters_list)):
                try:
                    metadata_res[i]=metadata[parameters_list[i]]
                except:
                    pass
                
        #elif self.catalogue == 'RIT':
        #    with open(json_metafile) as infile:
        #        data = infile.read()
        #        linesall = data.split('\n')
        #        lint=[]

        #       for line in linesall:
        #            if not line.startswith('*') and not line.startswith('#'):
        #                lint.append(line)
        #        str_list = list(filter(None, lint))[1:]
        #        metadata=self.txt_to_json(str_list)
                
        #    if len(parameters_list)>0:
        #        metadata=[metadata[x] for x in parameters_list]
        
        return metadata_res
    
    def delete_duplicates(self,json_metafiles,tags_aux):
        duplist=duplicates_pos(tags_aux)
        poslist = [i for i in range(len(tags_aux))]
        poslist = list(set(poslist) - set(duplist))
        
        json_metafiles = [json_metafiles[i] for i in poslist]
        json_metafiles=sorted(json_metafiles)
        tags_aux = [tags_aux[i] for i in poslist]
        tags_aux = sorted(tags_aux)
        
        return json_metafiles, tags_aux
    
    def txt_to_json(self,split_Line):
    # Assumes that the first ':' in a line
    # is always the key:value separator

        line_dict = {}
        for part in split_Line:
            key, value = part.split("=", maxsplit=1)
            key = key.rstrip()
            value = value.rstrip()
            line_dict[key] = value.rstrip()

        return line_dict
    
    def load_metadata(self,json_metafiles,keys,tags):
        
        # load all metadata
        if self.catalogue == 'SXS':
            metadata = {}
            for i in range(len(json_metafiles)):
                with open(json_metafiles[i]) as file:
                    metadata[tags[i]] = jsons.load(file)
        elif self.catalogue == 'RIT':
            metadata = {}
            for i in range(len(json_metafiles)):
                with open(json_metafiles[i]) as infile:
                    data = infile.read()
                linesall = data.split('\n')
                lint=[]

                for line in linesall:
                    if not line.startswith('*') and not line.startswith('#'):
                        lint.append(line)
                str_list = list(filter(None, lint))[1:]
                metadata[tags[i]]=self.txt_to_json(str_list)
        
        object_type=np.asarray([None]*len(tags))
        mass_1=np.asarray([None]*len(tags))
        mass_2=np.asarray([None]*len(tags))
        mass_ratio=np.asarray([None]*len(tags))
        spin1=np.asarray([None]*len(tags))
        spin2=np.asarray([None]*len(tags))
        chi_eff=np.asarray([None]*len(tags))
        chi_p=np.asarray([None]*len(tags))
        eccentricity=np.asarray([None]*len(tags))
        remnant_mass=np.asarray([None]*len(tags))
        remnant_spin=np.asarray([None]*len(tags))
        remnant_speed=np.asarray([None]*len(tags))
        angle_1=np.asarray([None]*len(tags))
        angle_2=np.asarray([None]*len(tags))
        tags_aux=np.asarray([None]*len(tags))

        #initial_distance=np.asarray([None]*len(tags))
        #orbits_number =np.asarray([None]*len(tags))
               
        
        for i in range(len(tags)):
            object_type[i] = metadata[tags[i]][keys[0]]
            if self.catalogue == 'RIT':
                object_type[i] =  object_type[i].split(':')[1]

            if  object_type[i]=='BHBH' or object_type[i]=='BBH':
                tags_aux[i]=tags[i]
                mass_1[i] = float(metadata[tags[i]][keys[1]])
                mass_2[i] = float(metadata[tags[i]][keys[2]])
                if self.catalogue == 'SXS': 
                    mass_ratio[i] = max(metadata[tags[i]][keys[3]],1)
                    spin1[i] = float(metadata[tags[i]][keys[4]][-1])
                    spin2[i] = float(metadata[tags[i]][keys[5]][-1])       
                elif self.catalogue == 'RIT':
                    mass_ratio[i] = max(1/float(metadata[tags[i]][keys[3]]),1)
                    spin1[i] = float(metadata[tags[i]][keys[4]])
                    spin2[i] = float(metadata[tags[i]][keys[5]]) 
                              
                chi_eff[i] = mass_1[i]*spin1[i] + mass_2[i]*spin2[i]

                A1=(2+3*mass_ratio[i]/(2.))
                A2=(2+3/(2.*mass_ratio[i]))             
                if self.catalogue == 'SXS':   
                    chi_p1 = metadata[tags[i]][keys[6]]
                    chi_p2 = metadata[tags[i]][keys[7]]
                elif self.catalogue == 'RIT':        
                    if metadata[tags[i]]['system-type']==' Precessing':
                        chi1x = float(metadata[tags[i]][keys[6][0]])
                        chi1y = float(metadata[tags[i]][keys[6][1]])
                        chi2x = float(metadata[tags[i]][keys[7][0]])
                        chi2y = float(metadata[tags[i]][keys[7][1]])
                    else:
                        chi1x,chi1y,chi2x,chi2y =[0,0,0,0]
                    
                    chi_p1 = np.sqrt(chi1x**2  + chi1y**2)
                    chi_p2 = np.sqrt(chi2x**2  + chi2y**2)               

                chi_p[i] = max(A1*np.abs(chi_p1)*mass_1[i]**2,A2*np.abs(chi_p2)*mass_2[i]**2)/(A1*mass_1[i]**2)
                eccentricity[i] = metadata[tags[i]][keys[8]]
                if check_type(eccentricity[i],str):
                    eccentricity[i] = eccentricity[i].replace("<","").replace("NaN","0").strip()
                    if len(eccentricity[i]) == 0:
                        eccentricity[i] = 0
                    eccentricity[i] = np.float(eccentricity[i])

                remnant_mass[i] = float(metadata[tags[i]][keys[9]])
                remnant_spin[i] = float(np.linalg.norm(metadata[tags[i]][keys[10]]))
                #remnant_speed[i] = metadata[tags[i]]['remnant_velocity']
                #initial_distance = metadata[tags[i]]['initial_separation']
            elif object_type[i]=='BHNS':
                tags_aux[i]=tags[i]
            elif object_type[i]=='NSNS':
                tags_aux[i]=tags[i]
                
        
        
        res = np.stack((tags_aux,object_type,mass_ratio,chi_eff,chi_p,eccentricity,remnant_mass,remnant_spin)).T 
        
        return res

    
    def cases_loop(self,select_conditions,bh_parameters):
        
        for i in range(len(select_conditions)):
            if select_conditions[i]=='BHBH' or select_conditions[i]=='BBH':
                boolean = bh_parameters[:,1] ==  select_conditions[i]
                bh_parameters = bh_parameters[boolean]
            
            elif select_conditions[i]=='Tag':
                boolean = np.logical_or(bh_parameters[:,0]==self.conditions[i,1], bh_parameters[:,0] == (self.conditions[i,1].replace(":","_")))
                bh_parameters = bh_parameters[boolean]
        
            elif select_conditions[i]=='Non-Precessing':
                boolean = bh_parameters[:,4] <=  self.tolerance
                bh_parameters = bh_parameters[boolean]
            
            elif select_conditions[i]=='Precessing':
                boolean = bh_parameters[:,4] >  self.tolerance
                bh_parameters = bh_parameters[boolean]
        
            elif select_conditions[i]=='Massratio':
            
                eval_conditions = np.array(self.conditions[i,1])
                mass_ratio_str=bh_parameters[:,2].astype('str')
                boolean = np.asarray([sympy.sympify(x+eval_conditions[0]) for x in mass_ratio_str])
                boolean = boolean.astype(bool)
                bh_parameters = bh_parameters[boolean]
            
                mass_ratio_str=bh_parameters[:,2].astype('str')
                boolean = np.asarray([sympy.sympify(x+eval_conditions[1]) for x in mass_ratio_str])          
                boolean = boolean.astype(bool)    
                bh_parameters = bh_parameters[boolean]
        
            elif select_conditions[i]=='Spin-eff':
            
                eval_conditions = np.array(self.conditions[i,1])
                spin_eff_str=bh_parameters[:,3].astype('str')
                boolean = np.asarray([sympy.sympify(x+eval_conditions[0]) for x in spin_eff_str])
                boolean = boolean.astype(bool)
                bh_parameters = bh_parameters[boolean]
            
                spin_eff_str=bh_parameters[:,3].astype('str')
                boolean = np.asarray([sympy.sympify(x+eval_conditions[1]) for x in spin_eff_str])          
                boolean = boolean.astype(bool)    
                bh_parameters = bh_parameters[boolean]
            
            elif select_conditions[i]=='Spin-p':
            
                eval_conditions = np.array(self.conditions[i,1])
                spin_p_str=bh_parameters[:,4].astype('str')
                boolean = np.asarray([sympy.sympify(x+eval_conditions[0]) for x in spin_p_str])
                boolean = boolean.astype(bool)
                bh_parameters = bh_parameters[boolean]
            
                spin_eff_str=bh_parameters[:,4].astype('str')
                boolean = np.asarray([sympy.sympify(x+eval_conditions[1]) for x in spin_p_str])          
                boolean = boolean.astype(bool)    
                bh_parameters = bh_parameters[boolean]
        
            elif select_conditions[i]=='Eccentricity':
                eval_conditions = np.array(self.conditions[i,1])
                ecc_eff_str=bh_parameters[:,5].astype('str')
                boolean = np.asarray([sympy.sympify(x.replace("<","").replace("NaN","0")+eval_conditions[0]) for x in ecc_eff_str])
                boolean = boolean.astype(bool)
                bh_parameters = bh_parameters[boolean]
            
                ecc_eff_str=bh_parameters[:,5].astype('str')
                boolean = np.asarray([sympy.sympify(x.replace("<","").replace("NaN","0")+eval_conditions[1]) for x in ecc_eff_str])          
                boolean = boolean.astype(bool)    
                bh_parameters = bh_parameters[boolean]
                
            elif select_conditions[i]=='Remnant-Spin':
            
                eval_conditions = np.array(self.conditions[i,1])
                spin_rm_str=bh_parameters[:,7].astype('str')
                boolean = np.asarray([sympy.sympify(x+eval_conditions[0]) for x in spin_rm_str])
                boolean = boolean.astype(bool)
                bh_parameters = bh_parameters[boolean]
            
                spin_rm_str=bh_parameters[:,7].astype('str')
                boolean = np.asarray([sympy.sympify(x+eval_conditions[1]) for x in spin_rm_str])          
                boolean = boolean.astype(bool)    
                bh_parameters = bh_parameters[boolean]
            
            elif select_conditions[i]=='Remnant-Mass':

                eval_conditions = np.array(self.conditions[i,1])
                mass_rm_str=bh_parameters[:,6].astype('str')
                boolean = np.asarray([sympy.sympify(x+eval_conditions[0]) for x in mass_rm_str])
                boolean = boolean.astype(bool)
                bh_parameters = bh_parameters[boolean]
            
                mass_rm_str=bh_parameters[:,6].astype('str')
                boolean = np.asarray([sympy.sympify(x+eval_conditions[1]) for x in mass_rm_str])          
                boolean = boolean.astype(bool)    
                bh_parameters = bh_parameters[boolean]
        
            else:
                print('Wrong conditional keyword: ', select_conditions[i])
                
        return bh_parameters
    
    def cases(self):
        '''Function used to select a subdomain of the sxs catalogue stored at sxs_root_folder. 
           Some example of conditions are [['BHBH',None],['Non-Precessing',None],['Massratio',[>=1,<=10],
            ['Spin-eff',['>=-0.5','<=0.5']],['Remnant-Spin', ['>=0.1','<=0.5']],['Remnant-Mass', ['>=0.88','<=0.97']],['Eccentricity',['>=0.01','<=0.5']]] 
        '''
        if self.catalogue=="SXS":
            json_metafiles = sorted(glob.glob(self.sxs_root_folder+"/**/metadata.json", recursive = True),reverse=True)
        elif self.catalogue=="RIT":
            json_metafiles = sorted(glob.glob(self.sxs_root_folder+"/Metadata/*_Metadata.txt"),reverse=True)        
                
        tags_aux = self.sxs_tags(json_metafiles)
        json_metafiles, tags_aux = self.delete_duplicates(json_metafiles,tags_aux)
        
        
        if self.conditions[0,0]=='BHBH':
            r = re.compile(".*BBH.*")
            tags = select_cases(tags_aux,r)
            position = [tags_aux.index(x) for x in tags]
            json_metafiles=[json_metafiles[x] for x in position]
        else:
            r = re.compile(".*"+self.conditions[0,0]+".*")
            tags = select_cases(tags_aux,r)
            position = [tags_aux.index(x) for x in tags]
            json_metafiles=[json_metafiles[x] for x in position]
                
    
        if any(t == 'BHNS' or t == 'NSNS' for t in self.conditions[:,0]):
            print('Metadata is corrupted for almost all the BHNS and NSNS cases. It has not been possible to classify them.')
            return(tags_aux)
        
        keys = self.nr_metadata_keys()
        
        bh_parameters = self.load_metadata(json_metafiles,keys,tags)   
        select_conditions = self.conditions[:,0]       
        bh_parameters = self.cases_loop(select_conditions,bh_parameters)
        
        if self.sort:
            bh_parameters=np.array(sorted(bh_parameters, key= lambda x: np.abs(x[self.sortcolumn])))


        return bh_parameters
        
    def find_jsonfiles(self,export=False,filepath=""):
        """Find the .json files."""
        tags = self.cases()[:,0]
        if self.catalogue == 'SXS':
            tags0 = tags[0].replace(":","_")
            rootpath = glob.glob(self.sxs_root_folder+"/**/"+tags0,recursive = True)[0]
            path = os.path.dirname(rootpath)
            json_metafiles = [sorted(glob.glob(path+"/"+x.replace(":","_")+"/**/metadata.json",recursive=True),reverse=True)[0] for x in tags0]
        elif self.catalogue == 'RIT':
            tags0 = tags[0].replace(":","_")
            rootpath = glob.glob(self.sxs_root_folder+"/Metadata/**/"+tags0,recursive = True)[0]
            path = os.path.dirname(rootpath)
            json_metafiles = [sorted(glob.glob(path+"/"+x,recursive=True),reverse=True)[0] for x in tags]
        
        if export:
            textfile = open(filepath, "w")
            for element in json_metafiles:
                   textfile.write(element + "\n")
            textfile.close()
        return json_metafiles
    
    def find_hd5files(self,export=False,filepath=""):
        """Find the .json files."""
        tags = self.cases()[:,0]
        if self.catalogue == 'SXS':
            tags0 = tags[0].replace(":","_")
            rootpath = glob.glob(self.sxs_root_folder+"/**/"+tags0,recursive = True)[0]
            path = os.path.dirname(rootpath)

            hd5files = [sorted(glob.glob(path+"/"+x.replace(":","_")+"/**/rhOverM_Asymptotic_GeometricUnits_CoM.h5", recursive = True),reverse=True)[0] for x in tags]
        elif self.catalogue == 'RIT':
            hd5files=self.find_jsonfiles()
            hd5files = [(x.split("-id")[0].replace("RIT_BBH_","ExtrapStrain_RIT-BBH-")+'.h5').replace('Metadata','Data') for x in hd5files]
            
        if export:
            textfile = open(filepath, "w")
            for element in hd5files:
                   textfile.write(element + "\n")
            textfile.close()
            
        return hd5files
    


    def Generate_SXS_Waveform(self,h5file,modes,extrapolation_order=3,zero_align=True,resample=True,sampling_rate=0.5,modes_combined=True,inclination=0,coa_phase=0):
        gw_nr = h5py.File(h5file, 'r')["Extrapolated_N"+str(extrapolation_order)+".dir"]
        if modes_combined:
            h=0
            for x in modes:
                gw_nr_all = gw_nr["Y_l"+str(x[0])+"_m"+str(x[1])+".dat"]
                times = gw_nr_all[:,0]
                hp=gw_nr_all[:,1]
                hc=gw_nr_all[:,2]
                sph=spher_harms(x[0],x[1],inclination,coa_phase=coa_phase)
                h+=(hp -1j*hc)*sph

            if resample:
                dt= sampling_rate
                h_int=interp1d(times, h, kind='cubic')
                times= np.arange(times[0], times[-1], dt)
                h=h_int(times)

            if zero_align:
                tmrg = times[np.argmax(np.abs(h))]
                times = times - tmrg 

            h_final = np.stack((times,h)).T
        else:
            h=[None]*len(modes)
            h_final=[None]*len(modes)
            a=0
            for x in modes:
                gw_nr_all = gw_nr["Y_l"+str(x[0])+"_m"+str(x[1])+".dat"]
                times = gw_nr_all[:,0]
                hp=gw_nr_all[:,1]
                hc=gw_nr_all[:,2]
                h[a]=(hp -1j*hc) 
                if resample:
                    dt= sampling_rate
                    h_int=interp1d(times, h[a], kind='cubic')
                    times= np.arange(times[0], times[-1], dt)
                    h[a]=h_int(times)            

                if zero_align:
                    tmrg = times[np.argmax(np.abs(h[a]))]
                    times = times - tmrg 

                h_final[a] = np.stack((times,h[a])).T
                a=a+1

        return h_final




