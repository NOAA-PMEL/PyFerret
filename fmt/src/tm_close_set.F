	SUBROUTINE TM_CLOSE_SET ( ds_num, status )
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
*
* Reclaim a data set by re-initializing all associated common variables.
* Initialize all the variables with the set up values, except for:
*	ds_var_setnum		= set_not_open
*	ds_alert_on_open	= .FALSE.
*	ds_alert_on_output	= .FALSE.
*	ds_regsteps		= .TRUE.
*	sf_setnum		= set_not_open
*	sf_valid		= .FALSE.
*	sf_lunit		= file_not_open	
*	sf_regvars		= .TRUE.
*	sf_regtimes		= .TRUE.
*
* programmer - mark verschell
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 09/11/86 - based on TM_ONLY1_STEP
* revision 1.00 - 03/19/87 - based on TM_CLOSE_SET by S. HANKIN
* revision 1.10 - 05/07/87 - added clean up of grid, and integration_tstep
* revision 1.11 - 07/07/87 - removed ds_revision
* revision 1.12 - 10/30/87 - removed ds_make_dscrip_rev
* revision 1.13 - 07/09/88 - aux_parm becomes parm_name and parm_text, and
*			     aux_modnum, aux_parm(s) now initialized in do loop
* revision 1.14 - 10/31/88 - added ds_var_modline
* revision 1.15 - 11/21/88 - changed re-initialization to specific char_initXX
* revision 1.16 - 02/08/89 - removed ds_var_modline
* revision 1.20 - 02/15/89 - Renamed TM_CLOSE_SET - added var_file variables -
*			     changed GT_CLOSE_STEP to TM_CLOSE_STEP, removed
*			     ds_open
* revision 1.21 - 04/24/91 - increased size of sf_name, ds_name and ds_des_name
* netCDF support *sh* 6/92
* EPIC netCDF *sh* 3/93
* v500 - *kob* 3/99 -  initialize ds_var_code, ds_var_units, ds_var_title
*                      and ds_var_titl_mod w/ char_inits of proper new value
* v500 - *kob* 4/99 - increase size of sf_name and ds_des_name for lengthly
*	              DODS filenames
* V510 *sh* 4/00 - deallocate grids and axes used by variables in this data set
* v540 *acm* 10/01 increase length of ds_name
* v540 *acm* 10/01  remove VMS include statements

* arguments:
*	ds_num	- data set number (see COMMON /XSTEP_FILES/)

	include 'tmap_errors.parm'	! error code parameter defs
#include "tmap_dset.parm"
	include 'tmap_dims.parm'
	include 'xstep_files.cmn_text'
	external xstep_files_data
	include 'xdset_info.cmn_text'
	external xdset_info_data

* internal variable declarations
        LOGICAL   its_cdf
	INTEGER   ds_num,status,f_num,ivar,iparm,iaux

* initialize: do we have an open set?
	IF (ds_name(ds_num) .EQ. char_init256) GOTO 9000

* save the data format
        its_cdf = ds_type(ds_num) .EQ. 'CDF'
     .     .OR.   ds_type(ds_num) .EQ. 'ECDF' 

* First clear variables associated only with data set number
	ds_type(ds_num)			=	char_init04
	ds_format(ds_num)		=	char_init04
	ds_source_class(ds_num)		=	char_init16
	ds_source(ds_num)		=	char_init16
	ds_subsource(ds_num)		=	char_init16
	ds_prog_rev(ds_num)		=	char_init04
********************************************************************************
	ds_expnum(ds_num)		=	char_init04
	ds_modnum(ds_num)		=	char_init04
	ds_title(ds_num)		=	char_init80
	ds_mod_title(ds_num)		=	char_init80
	ds_forcing_set(ds_num)		=	char_init16
	ds_t0time(ds_num)		=	char_init20
	ds_time_unit(ds_num)		=	real4_init
	ds_ntegrate_tstep(ds_num)	=	real4_init
	ds_ntuple(ds_num)		=	int4_init
	ds_time_run(ds_num)		=	char_init20
	DO 100 iaux = 1,num_aux
	  ds_aux_modnum(iaux,ds_num)	=	char_init04
  100	CONTINUE
	DO 110 iparm = 1,num_parm
	  ds_parm_name(iparm,ds_num)	=	char_init20
	  ds_parm_text(iparm,ds_num)	=	char_init80
  110	CONTINUE
********************************************************************************
	ds_alert_on_open(ds_num)	= 	.FALSE.
	ds_alert_on_output(ds_num)	= 	.FALSE.
	ds_message(ds_num)		=	char_init80
********************************************************************************
	ds_des_name(ds_num)		=	char_init256
	ds_name(ds_num)			=	char_init256
	ds_1step(ds_num)		=	real8_init
	ds_lastep(ds_num)		=	real8_init
	ds_delstep(ds_num)		=	real8_init
	ds_regsteps(ds_num)		=	.TRUE.
	ds_basic_axes(1,ds_num)		=	int4_init
	ds_basic_axes(2,ds_num)		=	int4_init
	ds_basic_axes(3,ds_num)		=	int4_init
	ds_basic_axes(4,ds_num)		=	int4_init
	ds_basic_axes(5,ds_num)		=	int4_init
	ds_basic_axes(6,ds_num)		=	int4_init
	ds_time_axis(ds_num)		=	int4_init

* Next clear variables associated with variable in that data set.
	DO 120 ivar = 1,maxvars
	  IF (ds_var_setnum(ivar) .NE. ds_num) GOTO 120
* ... deallocate grids and axes used by the dat aset
	  CALL TM_DEALLO_DYN_GRID( ds_grid_number(ivar))
	  ds_var_code(ivar)		=	char_init64
	  ds_var_title(ivar)		=	char_init128
	  ds_var_titl_mod(ivar)		=	char_init128
	  ds_var_units(ivar)		=	char_init64
	  ds_grid_number(ivar)		=	int4_init
	  ds_missing_flag(ivar)		=	real4_init
	  ds_bad_flag(ivar)		=	real4_init
	  ds_ordering(1,ivar)		=	int4_init
	  ds_ordering(2,ivar)		=	int4_init
	  ds_ordering(3,ivar)		=	int4_init
	  ds_ordering(4,ivar)		=	int4_init
	  ds_grid_start(1,ivar)		=	int4_init
	  ds_grid_start(2,ivar)		=	int4_init
	  ds_grid_start(3,ivar)		=	int4_init
	  ds_grid_start(4,ivar)		=	int4_init
	  ds_grid_end(1,ivar)		=	int4_init
	  ds_grid_end(2,ivar)		=	int4_init
	  ds_grid_end(3,ivar)		=	int4_init
	  ds_grid_end(4,ivar)		=	int4_init
	  ds_precision(ivar)		=	char_init01
	  ds_aux_set_num(ivar)		=	int4_init
********************************************************************************
	  ds_ndataperrec(ivar)		=	int4_init
	  ds_nrecpervar(ivar)		=	int4_init
	  ds_nrecb4var(ivar)		=	int4_init
	  ds_var_setnum(ivar) 		=	set_not_open
  120	CONTINUE

* Now loop through stepfiles and initialize associated entries, closing files
* if necessary

	DO 130 f_num = 1, maxstepfiles
	  IF (sf_setnum(f_num) .NE. ds_num) GOTO 130

* close step file or CDF file ?
	  IF ( sf_lunit (f_num) .NE. file_not_open) THEN	! open ?
            IF ( its_cdf ) THEN
               CALL CD_CLOSE_SET( sf_lunit(f_num), status )
            ELSE
	       CALL TM_CLOSE_STEP (f_num, status)
            ENDIF
	    IF (status .NE. merr_ok) GOTO 9900
	  ENDIF

* delete step file from tables
	  sf_name(f_num)		=	char_init256    ! *kob* 4/99
	  sf_setnum(f_num)		=	set_not_open
	  sf_aux_set_num(f_num)		=	int4_init
	  sf_nrecperstep(f_num)		=	int4_init
	  sf_skip(f_num)		=	int4_init
	  sf_1step(f_num)		=	real8_init
	  sf_lastep(f_num)		=	real8_init
	  sf_delta(f_num)		=	real8_init
	  sf_reclen(f_num)		=	int4_init
	  sf_valid(f_num)		=	.FALSE.
	  sf_lunit(f_num)		=	file_not_open	
	  sf_index(f_num)		=	char_init23
	  sf_regvars(f_num)		=	.TRUE.
	  sf_regtimes(f_num)		=	.TRUE.
  130	CONTINUE

        DO 140 f_num = 1, maxvarfiles
	  vf_firstvar(f_num)		=	int4_init
	  vf_lastvar(f_num)		=	int4_init
  140   CONTINUE

 9000	status = merr_ok

 9900	RETURN
	END
