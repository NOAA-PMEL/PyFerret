	BLOCK DATA XDSET_PARMS_DATA

* parameters differentiating characteristics of individual data sets

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/5/88
* revision 0.1 - 7/2/88 - generalized Levitus_clim --> dp_clim_air_sst
*			- added Am_factor
* revision 0.2 - 4/19/89- pre-initialize only the LEV air temp set.  Other
*			  3-letter codes to be pulled from the 205 data base
*			  when SET DATA_SET is given

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'tmap_format/tmap_dims.parm'
	include	'ferret_cmn/xdset_parms.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'FERRET_CMN:XDSET_PARMS.CMN'
#endif

* description of dp_ variables used by COX and PHILANDER output sets for air
* temperature sets:
*	dp_const_air_sst	- (LOGICAL) is the airt-sst diff. a constant ?
*		if yes (constant)
*	dp_delta_air_sst	- (REAL) contains the airt-sst difference
*		if no (air temp supplied by another data set)
*	dp_clim_air_sst		- (INTEGER) points a COX or PHILANDER output
*				  set to its corresponding air temp set
*	dp_airt_name		- (STRING) has 3-letter code for air temp set
*	dp_airt_dset		- (INTEER) has data set # for the air temp dset 

	DATA	dp_clim_air_sst / maxdsets*unspecified_int4 /

	DATA	dp_airt_name( 1 ) / 'LEV' /,
     .		dp_airt_name( 2 ) / unspecified_name4 /,
     .		dp_airt_name( 3 ) / unspecified_name4 /,
     .		dp_airt_name( 4 ) / unspecified_name4 /,
     .		dp_airt_name( 5 ) / unspecified_name4 /

	DATA	dp_airt_dset / max_airt_clim * unspecified_int4 /

	END
