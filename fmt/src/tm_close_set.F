	SUBROUTINE TM_CLOSE_SET ( ds_num, status )
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

* arguments:
*	ds_num	- data set number (see COMMON /XSTEP_FILES/)

#ifdef unix
	include 'tmap_format/tmap_errors.parm'	! error code parameter defs
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xstep_files.cmn_text'
	external xstep_files_data
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'	! error code PARAMETER defs
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XSTEP_FILES.CMN'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
#endif

* internal variable declarations
        LOGICAL   its_cdf
	INTEGER   ds_num,status,f_num,ivar,iparm,iaux

* initialize: do we have an open set?
	IF (ds_name(ds_num) .EQ. char_init40) GOTO 9000

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
	ds_des_name(ds_num)		=	char_init128
	ds_name(ds_num)			=	char_init40
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
	  ds_var_code(ivar)		=	char_init08
	  ds_var_title(ivar)		=	char_init40
	  ds_var_titl_mod(ivar)		=	char_init40
	  ds_var_units(ivar)		=	char_init24
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
	  sf_name(f_num)		=	char_init128    ! *sh* 4/91
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
	  vf_firstvar(f_num)		=	int4_init
	  vf_lastvar(f_num)		=	int4_init
  130	CONTINUE
 9000	status = merr_ok

 9900	RETURN
	END
