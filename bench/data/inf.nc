CDF      
      TIME             title         \Hourly Argos-tracked drifters location and velocity estimates, with 95% confidence intervals   description       aThis is version 1.01 of the dataset. See Elipot et al. 2016, JGR-Oceans, doi:10.1002/2016JC011716      note     �For all variables of dimension TIME, interruptions in the estimation along a single trajectory are indicated by "Inf" value
s; Individual trajectories are separated by "NaN" values; Thus, one can use the COL2CELL function of the JLAB Matlab toolbox (http://www.jmlilly.net
) to convert each data matrix into a cell array with one component for each individual trajectory, without the need to load and loop for individual ID
s, e.g. lat = col2cell(lat);   creator       (Shane Elipot, University of Miami, RSMAS   	timestamp         16-Jan-2017 11:05:27         TIME                standard_name         time   units         hours since 1979-01-01 00:00:00         �@�'�    @�'�    @�'�    @�'�    @�'�    @�(     @�(    @�(     @�(0    @�(@    @�(P    @�(`    @�(p    �      @�)�    @�*     @�*    @�*     @�*0    @�*@    