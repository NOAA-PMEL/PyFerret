	BLOCK DATA XDSET_PARMS_DATA

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
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
	include	'ferret.parm'
	include	'tmap_dims.parm'
	include	'xdset_parms.cmn'
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
