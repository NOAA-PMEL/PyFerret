	INTEGER FUNCTION KNOWN_GRID ( data_set, category, variable )

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
* find the defining grid for the given variable
* if unknown - return unknown flag

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:   7/6/89 - based on IGRID
* V230:  7/21/92 - check for unspecified data set on calc_var

#ifdef unix
	include	'tmap_format/tmap_dims.parm'
	include	'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xcalc_vars.cmn'
#else
	INCLUDE	'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XCALC_VARS.CMN'
#endif

* calling argument declarations:
	INTEGER	data_set, category, variable

* internal variable declarations:
	INTEGER	dset

* different procedure for determining grid from each category
	IF     ( category .EQ. cat_file_var ) THEN

	   KNOWN_GRID = ds_grid_number( variable )


	ELSEIF ( category .EQ. cat_calc_var ) THEN

           IF ( data_set .EQ. unspecified_int4 ) THEN
              KNOWN_GRID = unspecified_int4
           ELSE
	      KNOWN_GRID = cvar_grid( variable, data_set )
           ENDIF


	ELSEIF ( category .EQ. cat_user_var ) THEN

* ... allow "abstract" expressions that don't depend on data set
	   IF ( .NOT.uvar_need_dset(variable)
     .	   .OR. data_set .EQ. unspecified_int4 ) THEN
	      dset = pdset_irrelevant
	   ELSE
	      dset = data_set
	   ENDIF
	   KNOWN_GRID = uvar_grid( variable, dset )



	ELSEIF ( category .EQ. cat_temp_var ) THEN

	   KNOWN_GRID = mr_grid( variable )



	ELSE

	   STOP 'KNOWN_GR'

	ENDIF

	RETURN
	END
