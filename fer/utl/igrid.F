	INTEGER FUNCTION IGRID ( data_set, category, variable, status )

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
* if unknown - go get it

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/23/86
* revision 0.6 - 11/22/88- new TM IO library var_gridber --> ds_grid_number
* V200:   7/6/89 - most guts removed to KNOWN_GRID
*	 8/29/89 - trap pseudo-variable grid requests
*	 3/26/90 - improved error messages
* V230:  7/21/92 - improved error message for unknown GFDL diagnostic var

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xalgebra.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XALGEBRA.CMN'
#endif

* calling argument declarations:
	INTEGER	data_set, category, variable, status

* internal variable declarations:
	INTEGER	KNOWN_GRID, TM_LENSTR1, dset

* only user variables have a status issue
	status = ferr_ok

* check for pseudo-variable
	IF ( category .EQ. cat_pseudo_var ) CALL ERRMSG
     .		( ferr_invalid_command, status,	'pseudo-variable '
     .		//alg_pvar(variable)(:TM_LENSTR1(alg_pvar(variable)))
     .		//' cannot be used this way',
     .		*5000 )

	IGRID = KNOWN_GRID( data_set, category, variable )

	IF ( IGRID .NE. unspecified_int4 ) RETURN

* grid unknown
	IF ( category .EQ. cat_calc_var ) THEN
	   CALL ERRMSG( ferr_invalid_command, status,
     .		'unknown data set for GFDL diagnostic variable',
     .          *5000 )
	ELSEIF ( category .NE. cat_user_var ) THEN
	   CALL ERRMSG( ferr_internal, status, 'IGRID', *5000 )
	ENDIF

* ... allow "abstract" expressions that don't depend on data set
	IF ( .NOT.uvar_need_dset(variable)
     .	   .OR. data_set .EQ. unspecified_int4 ) THEN
	   dset = pdset_irrelevant
	ELSE
	   dset = data_set
	ENDIF

	CALL GET_UVAR_GRID( variable, dset, status )
	IGRID = uvar_grid( variable, dset )

	RETURN

* error exit
 5000	RETURN
	END
