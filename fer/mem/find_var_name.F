	SUBROUTINE FIND_VAR_NAME ( dset_num, test_name, category, variable )

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

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/4/86
* revision 0.1 - 1/20/87 - changed nvari_prenamed to num_fils_vars for PSI
* revision 0.2 - 3/6/87  - introduced user var search and munknown_var_name
* revision 0.3 - 3/23/87 - added new TMAP library: time axes, file var. names
* revision 0.4 - 3/10/88 - added cat_named_var and unspecified data set
* V200:  6/20/89 - eliminated "named" variables
*	 1/25/90 - padded cvar_name_code with 4 blanks avoid QADXT=QADX match
* V402:  7/95 - Added support for  LET/D=dset var = ...
* V495: 7/7/98 - *sh* allow case-sensitive variable names in quotes from dsets
*       user variable names are allowed in quotations just for
*       consistency. Pseudo-variable names in quotes are not allowed -- so they
*       don't get confused

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xcalc_vars.cmn'
	include	'ferret_cmn/xalgebra.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCALC_VARS.CMN'
	INCLUDE	'FERRET_CMN:XALGEBRA.CMN'
#endif

* calling argument declarations:
	INTEGER		dset_num, category, variable
	CHARACTER*(*)	test_name

* internal variable declarations:
	INTEGER		ivar, uvdset
	LOGICAL		MATCH_NAME, MATCH_QUOTED_NAME, single_dset

* initialize
	single_dset = dset_num .NE. unspecified_int4

* start by checking pseudo-variables names
	DO 10 ivar = 1, num_pseudo_vars
	   IF ( MATCH_NAME( test_name, alg_pvar(ivar) ) ) THEN
	      category = cat_pseudo_var
	      variable = ivar
	      RETURN
	   ENDIF
 10	CONTINUE

* check data user-defined variables
* ... first check for an explicitly named data set
	IF ( single_dset ) THEN
	   DO 40 ivar = 1, max_uvar
	      IF ( uvar_num_items( ivar ) .EQ. uvar_deleted ) GOTO 40
	      IF ( uvar_dset( ivar ) .NE. dset_num ) GOTO 40
	      IF ( MATCH_QUOTED_NAME(test_name,uvar_name_code(ivar)) )
	1	   THEN
	         category = cat_user_var
	         variable = ivar
	         RETURN
	      ENDIF
 40	   CONTINUE
	ELSE
* ... check for an implicitly named data set - the "current" data set
	   uvdset = cx_data_set(cx_last)
	   IF ( uvdset .NE. unspecified_int4 ) THEN
	      DO 60 ivar = 1, max_uvar
	         IF ( uvar_num_items( ivar ) .EQ. uvar_deleted ) GOTO 60
	         IF ( uvar_dset( ivar ) .NE. uvdset ) GOTO 60
	         IF ( MATCH_QUOTED_NAME( test_name,uvar_name_code(ivar)) )
	1	      THEN
	            category = cat_user_var
	            variable = ivar
	            RETURN
	         ENDIF
 60	      CONTINUE
	   ENDIF
	ENDIF

* ... check the global user variable definitions defined without /D
	DO 100 ivar = 1, max_uvar
	   IF ( uvar_num_items( ivar ) .EQ. uvar_deleted ) GOTO 100
	   IF ( uvar_dset( ivar ) .NE. unspecified_int4 ) GOTO 100
	   IF ( MATCH_QUOTED_NAME( test_name,uvar_name_code(ivar)) )
	1	THEN
	      category = cat_user_var
	      variable = ivar
	      RETURN
	   ENDIF
 100	CONTINUE

* check file variables from this data set
	DO 200 ivar = 1, maxvars
	   IF ( single_dset
     .	  .AND. ds_var_setnum( ivar ) .NE. dset_num ) GOTO 200
	   IF ( MATCH_QUOTED_NAME( test_name, ds_var_code( ivar ) ) ) THEN
	      category = cat_file_var
	      variable = ivar
	      RETURN
	   ENDIF
 200	CONTINUE

* ... check the global user variable definitions defined with /D
	DO 250 ivar = 1, max_uvar
	   IF ( uvar_num_items( ivar ) .EQ. uvar_deleted ) GOTO 250
	   IF ( uvar_dset( ivar ) .NE. pdset_irrelevant ) GOTO 250
	   IF ( MATCH_QUOTED_NAME( test_name,uvar_name_code(ivar)) )
	1	THEN
	      category = cat_user_var
	      variable = ivar
	      RETURN
	   ENDIF
 250	CONTINUE

* check table of calculatable variables (padded to 8 characters)
	DO 300 ivar = 1, num_calc_vars
	   IF (MATCH_QUOTED_NAME(test_name,cvar_name_code(ivar)//'    '))
	1	THEN
	      category = cat_calc_var
	      variable = ivar
	      RETURN
	   ENDIF
 300	CONTINUE

* not found
	variable = munknown_var_name
	RETURN

	END
