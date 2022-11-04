# pynr
Provides simple to call functions to get the waveforms from the RIT, SXS and MAYA catalogues. The code is able to read the data directly from the catalogue websites and from your local computed. Options to load the modes individually are provided.

## Installation

You can also install from source by cloning the repository at https://github.com/frcojimenez/pynr. Required packages are listed in the `requirements.txt` file, which you can install by running `pip install -r requirements.txt` from within the source code directory, followed by `python setup.py install`. Install  `pip install -r optional-requirements.txt` if you want your code to run in the same format as the pycbc code. 

## Examples

 1. Check the notebook Examples/Examples.ipynb notebook:

```
>>> import pykerr
>> hp_sxs,hx_sxs= nr.nr_waveform(code='SXS',pycbc_format=True,
    download_Q = True,
    tag = 'SXS:BBH:0305',
    extrapolation_order = 2,
    resolution_level = -1,
    modes  = [[2,2],[2,-2]],
    mass  = 100,
    distance  = 100,
    inclination = 0,
    coa_phase = 0,
    modes_combined = True,
    delta_t = 1/1024,
    tapering = False,
    RD = True,
    zero_align = True )
```

## References

1. SXS, https://data.black-holes.org/waveforms/index.html.
2. RIT. https://ccrg.rit.edu/content/data/rit-waveform-catalog
3. Pycbc. https://github.com/gwastro/pycbc
