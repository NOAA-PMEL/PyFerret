    	SUBROUTINE FERRET_QUERY_F( query, flags, nflags, ctext, ntext,
     .				   num_strings, num_vals, nums, nnums,
     .				   arg1, arg2, arg3, arg4, arg5, qstatus )

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
* respond to the GUI query API call

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* for FERRET V400: 8/94
*   3/15/95 - fixed error in "strides" for query GCOORDS
* V402: "temporary ferret_query.inc" changed to ferret_cmn/ferret_query.parm
* V420: bug fix: query would crash if grid was unknown
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
*	    4/97 *kob* - added ifdef for MANDATORY_FORMAT_WIDTHS on internal
*			 read because linux/nag f90 needed a format width.
*           12/00*acm* - cleanup: TM_DATE_OK declared but never called
* V533      8/01 *acm* - calendar axis additions: SECS_FROM_BC needs cal_id
*      *acm* 10/01  remove VMS includes

        include 'tmap_dims.parm'
#       include "tmap_dset.parm"
        include 'xdset_info.cmn_text'
        external xdset_info_data
        include 'xtm_grid.cmn_text'
        external xgt_grid_data
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'gfdl_vms.parm'
	include 'ferret_query.parm'
	include 'xprog_state.cmn'
	include 'xcommand.cmn'
	include 'xtext_info.cmn'
	include 'xcontext.cmn'
	include 'xalgebra.cmn'
	include 'xgui.cmn'
	include 'xvariables.cmn' ! JC_ADDITION


* calling argument declarations
	INTEGER query, nflags, flags(nflags), num_strings, num_vals, ntext,
     .		nnums, qstatus
* ... c-type Hollerith string buffers
#ifdef sun
        BYTE      ctext(ntext), arg1(*), arg2(*), arg3(*), arg4(*), arg5(*)
#else
        INTEGER*1 ctext(ntext), arg1(*), arg2(*), arg3(*), arg4(*), arg5(*)
#endif
	REAL*8	nums(nnums)

* local variable declarations:
	LOGICAL GEOG_LABEL
	INTEGER	TM_LENSTR1, FIND_DSET_NUMBER, IGRID, TM_GET_LINENUM,
     .		TM_GET_GRIDNUM, TM_GET_CALENDAR_ID,
     .		i, i2, n, dim, dset, cat, var, axis, grid,
     .		slen, insert_pt, stride, status, cal_id
	REAL*8	TWORLD, TM_WORLD, SECS_FROM_BC, lo, hi, delta
	CHARACTER AX_TITLE*32, TM_FMT*16, buff1*32,
     .		  farg1*80, farg2*80, farg3*80, farg4*80, farg5*80

* local parameter declarations:
	CHARACTER*4 notext
	PARAMETER ( notext = '****')

* legal query ?
	IF ( query .GT. max_queries ) THEN
	   qstatus = QSTAT_UNKNOWN_QUERY
	   RETURN
	ENDIF

* initialize
	qstatus = QSTAT_OK
	insert_pt = 1
	num_strings = 0
	num_vals = 0
	DO 10 i = 1, nflags
 10	flags(i) = 0
	ctext(1) = 0

* convert arguments to FORTRAN character arrays
	CALL TM_CTOF_STRNG(arg1, farg1, 80) 
	CALL TM_CTOF_STRNG(arg2, farg2, 80) 
	CALL TM_CTOF_STRNG(arg3, farg3, 80) 
	CALL TM_CTOF_STRNG(arg4, farg4, 80) 
	CALL TM_CTOF_STRNG(arg5, farg5, 80) 

* respond to the appropriate query
	GOTO ( 100, 200, 300, 400, 500, 600, 700, 800, 900,1000,
     .	      1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,
     .	      2100,2200,2300,2400,2500,2600,2700,2800,2900,3000 ) query
	
* STATUS
* error status of last command
 100	CONTINUE
	qstatus = QSTAT_UNKNOWN_QUERY
	RETURN

* MESSAGE
* messages from last command
 200	CONTINUE
	qstatus = QSTAT_UNKNOWN_QUERY
	RETURN

* DSET
* list of open data sets
 300	DO 310 i = 1,  maxdsets
	   IF ( ds_name(i) .NE. char_init40 ) THEN
	      slen = TM_LENSTR1(ds_name(i))
	      CALL CSTRING_INSERT( ds_name(i),
     .				slen, ctext, ntext, insert_pt)
	      IF ( insert_pt .GT. ntext ) THEN
	         qstatus = QSTAT_BUFF_OVFLOW
	         RETURN
	      ENDIF
	      num_strings = num_strings + 1
	   ENDIF
 310	CONTINUE
	RETURN

* VARIABLE dset
* return the variables in a data set
* ... identify the data set
 400	dset = FIND_DSET_NUMBER(farg1)
	IF ( dset .EQ. unspecified_int4 ) THEN
	   qstatus = QSTAT_INVALID_ARG
	   RETURN
	ENDIF
	DO 410 var = 1, maxvars
	   IF ( ds_var_setnum(var) .NE. dset ) GOTO 410
	   IF ( ds_var_code(var) .EQ. '-' ) GOTO 410
	   slen = TM_LENSTR1(ds_var_code(var))
	   CALL CSTRING_INSERT( ds_var_code(var),
     .				slen, ctext, ntext, insert_pt)
	   IF ( insert_pt .GT. ntext ) THEN
	      qstatus = QSTAT_BUFF_OVFLOW
	      RETURN
	   ENDIF
	   num_strings = num_strings + 1
 410	CONTINUE
	RETURN

* GRID dset variable
* 4-axis description of grid for given variable
* ... identify the data set, then variable, then its grid
 500	dset = FIND_DSET_NUMBER(farg1)
	IF ( dset .EQ. unspecified_int4 ) THEN
	   qstatus = QSTAT_INVALID_ARG
	   RETURN
	ENDIF
	CALL FIND_VAR_NAME( dset, farg2, cat, var )
	IF ( dset .EQ. unspecified_int4 ) THEN
	   qstatus = QSTAT_INVALID_ARG
	   RETURN
	ENDIF
	grid = IGRID ( dset, cat, var, status )
	IF ( grid.EQ.unspecified_int4 .OR. status.NE.ferr_ok) THEN
	   qstatus = -1 * status
	   RETURN
	ENDIF
* ... first return the grid name
	slen = TM_LENSTR1(grid_name(grid))
	CALL CSTRING_INSERT( grid_name(grid),
     .			     slen, ctext, ntext, insert_pt)
* ... then the 4 axis names
	DO 510 dim = 1, 4
	   axis = grid_line( dim, grid )
	   IF ( axis .EQ. munknown ) THEN
	      CALL CSTRING_INSERT( 'UNKNOWN',
     .				   7, ctext, ntext, insert_pt)
	   ELSEIF( axis .EQ. mnormal ) THEN
	      CALL CSTRING_INSERT( 'NORMAL',
     .				   6, ctext, ntext, insert_pt)
	   ELSE
	      slen = TM_LENSTR1(line_name(axis))
	      CALL CSTRING_INSERT( line_name(axis),
     .				   slen, ctext, ntext, insert_pt)
	   ENDIF
	   IF ( insert_pt .GT. ntext ) THEN
	      qstatus = QSTAT_BUFF_OVFLOW
	      RETURN
	   ENDIF
 510    CONTINUE
* ... then the 4 axis orientation codes
	DO 520 dim = 1, 4
	   axis = grid_line( dim, grid )
	   IF ( axis .EQ. munknown .OR. axis .EQ. mnormal ) THEN
	      CALL CSTRING_INSERT( '**',
     .				   2, ctext, ntext, insert_pt)
	   ELSE
	      CALL CSTRING_INSERT( line_direction(axis),
     .				   2, ctext, ntext, insert_pt)
	   ENDIF
	   IF ( insert_pt .GT. ntext ) THEN
	      qstatus = QSTAT_BUFF_OVFLOW
	      RETURN
	   ENDIF
 520    CONTINUE
	num_strings = 9	! 1 + 4 + 4
	RETURN

* AXIS  (args?)
 600	qstatus = QSTAT_UNKNOWN_QUERY
	RETURN

* COORDS axis
* coordinates of (irregular) axis
 700	CONTINUE

* TRANS  *************
* list of transforms
 800	CONTINUE
	qstatus = QSTAT_UNKNOWN_QUERY
	RETURN

* TRNARG (trans) ****will not be used!!*************
* argument description for given transform
 900	CONTINUE
* LVARS  *************
* list of LET-defined variables
 1000	CONTINUE

* DVARS (dset) *************
* list of LET-defined variables defined for data set
* ... identify the data set
*
* first attempt at an upgrade by JC (3/1/96)
*
 1100	dset = FIND_DSET_NUMBER(farg1)
	IF ( dset .EQ. unspecified_int4 ) THEN
	   qstatus = QSTAT_INVALID_ARG
	   RETURN
	ENDIF
	DO 1110 var = 1, max_uvar
           IF (uvar_num_items(var) .EQ. uvar_deleted  ) GOTO 1110
           IF (uvar_name_code(var)(1:3) .EQ. 'EX#'    ) GOTO 1110
           IF ( dset .NE. uvar_dset(var)         ) GOTO 1110
           slen = TM_LENSTR1(uvar_name_code(var))
	   CALL CSTRING_INSERT( uvar_name_code(var),
     .				slen, ctext, ntext, insert_pt)

	   IF ( insert_pt .GT. ntext ) THEN
	      qstatus = QSTAT_BUFF_OVFLOW
	      RETURN
	   ENDIF
	   num_strings = num_strings + 1
 1110	CONTINUE
	RETURN

* VBACKGROUND  (dset variable) *************
* background info about this variable
 1200	CONTINUE
* DBACKGROUND (dset) *************
* background info about this data set
 1300	CONTINUE
* WINDOWS  *************
* list of open windows
 1400	CONTINUE
* WCURRENT  *************
* name of current window
 1500	CONTINUE
* VPORTS  *************
* list of defined viewports
 1600	CONTINUE
* VCURRENT  *************
* name of current viewport
 1700	CONTINUE
* VIEW  (vport) *************
* definition of indicated viewport
 1800	CONTINUE
	qstatus = QSTAT_UNKNOWN_QUERY
	RETURN

* DCURRENT
* name of current data set
 1900	DO 1910 dset = 1, maxdsets
           IF ( ds_name(dset) .EQ. char_init40 ) GOTO 1910
	   IF ( dset .EQ. cx_data_set(cx_last) ) THEN
	      slen = TM_LENSTR1(ds_name(dset))
	      CALL CSTRING_INSERT( ds_name(dset),
     .				   slen, ctext, ntext, insert_pt)
	      num_strings = 1
	      RETURN
	   ENDIF
 1910	CONTINUE
* ... there is no current data set
	CALL CSTRING_INSERT( '*', 1, ctext, ntext, insert_pt)
	num_strings = 1
	RETURN

* SPAN  dset variable
* grid name and axis limits for a variable
 2000	dset = FIND_DSET_NUMBER(farg1)
	IF ( dset .EQ. unspecified_int4 ) THEN
	   qstatus = QSTAT_INVALID_ARG
	   RETURN
	ENDIF
	CALL FIND_VAR_NAME( dset, farg2, cat, var )
	IF ( dset .EQ. unspecified_int4 ) THEN
	   qstatus = QSTAT_INVALID_ARG
	   RETURN
	ENDIF
	grid = IGRID ( dset, cat, var, status )
	IF ( status .NE. ferr_ok ) THEN
	   qstatus = -1 * status
	   RETURN
	ENDIF
* ... first return the grid name
	slen = TM_LENSTR1(grid_name(grid))
	CALL CSTRING_INSERT( grid_name(grid),
     .			     slen, ctext, ntext, insert_pt)
* ... then the 4 axis spans
* ... loss,hiss,loww,hiww (all presented as floating point)
	DO 2010 dim = 1, 4
	   axis = grid_line( dim, grid )
	   i = (dim-1) * 4
	   IF ( axis .EQ. munknown
     .	   .OR. axis .EQ. mnormal  ) THEN
	      nums(i+1) = DBLE( unspecified_int4 ) 
	      nums(i+2) = DBLE( unspecified_int4 ) 
	      nums(i+3) = DBLE(bad_val4)
	      nums(i+4) = DBLE(bad_val4)
	   ELSE
! 4/4/96 - return grid limits rather than span limits (which are meaningful
!               only for file variables)
              nums(i+1) = 1.0D0
              nums(i+2) = DBLE(line_dim(axis))
              nums(i+3) = TM_WORLD( 1,
     .                              grid, dim, box_middle)
              nums(i+4) = TM_WORLD( line_dim(axis),
     .                              grid, dim, box_middle)
	   ENDIF
 2010    CONTINUE
	num_strings = 1
	num_vals = 16
	RETURN

* GAXIS grid orientation(1-4)
 2100	grid = TM_GET_GRIDNUM( farg1 )
	IF ( grid .EQ. unspecified_int4 ) THEN
	   qstatus = QSTAT_INVALID_ARG
	   RETURN
	ENDIF
* *kob* 4/97
#ifdef MANDATORY_FORMAT_WIDTHS
	READ( farg2,* ) dim
#else
	READ( farg2,'(I)' ) dim
#endif
	axis = grid_line(dim, grid)
* ... first return the axis title (units are appended already ...)
	IF ( axis .NE. mnormal ) THEN
	   buff1 = AX_TITLE(dim, grid, slen)
	ELSE
	   buff1 = notext
	   slen = 4
	ENDIF
	CALL CSTRING_INSERT( buff1,
     .			     slen, ctext, ntext, insert_pt)
* ... then the axis units
	IF ( axis .NE. mnormal ) THEN
	   buff1 = line_units(axis)
	   slen = TM_LENSTR1(buff1)
	ELSE
	   buff1 = notext
	   slen = 4
	ENDIF
	CALL CSTRING_INSERT( buff1,
     .			     slen, ctext, ntext, insert_pt)
* ... now return some flags:
	IF ( line_modulo(axis) ) THEN		! modulo ?
	   flags(1) = 1
	ELSE
	   flags(1) = 0
	ENDIF
	IF ( GEOG_LABEL( dim, grid ) ) THEN	! fancy labeling ?
	   flags(2) = 1
	ELSE
	   flags(2) = 0
	ENDIF
	IF ( line_regular(axis) ) THEN		! regular spacing ?
	   flags(3) = 1
	ELSE
	   flags(3) = 0
	ENDIF
	flags(4) = line_unit_code(axis)		! unit code
* ... some low:high:delta information
	nums(1) = DBLE( line_dim(axis) )
	nums(2) = line_start(axis)
	nums(3) = line_delta(axis)
	num_strings = 2
	num_vals = 3
* ... and finially special stuff for calendar axis formatting
	IF ( dim.EQ.t_dim .AND. flags(2).EQ.1 ) THEN

*  See if we can find out the calendar, if grid/axis is known
	IF (grid .EQ. unspecified_int4  .AND.  
     .	    axis .EQ. unspecified_int4)  THEN 
	   cal_id = 1
	ELSE
	   IF (axis .EQ. unspecified_int4) axis = grid_line(t_dim, grid)
	   cal_id = TM_GET_CALENDAR_ID ( line_cal_name(axis) )
	ENDIF

	   nums(4) = SECS_FROM_BC( line_t0(axis), cal_id, status )
	   nums(5) = line_tunit(axis)
	ENDIF
	RETURN

* GCOORDS grid orientation(1-4)   offset   stride
* ... the variable num_valstoj will tell the number of values returned
 2200	grid = TM_GET_GRIDNUM( farg1 )
	IF ( grid .EQ. unspecified_int4 ) THEN
	   qstatus = QSTAT_INVALID_ARG
	   RETURN
	ENDIF
* *kob* 4/97 include axis...line in below ifdef just for simplicity
#ifdef MANDATORY_FORMAT_WIDTHS
	READ( farg2,* ) dim
	axis = grid_line(dim, grid)
	READ( farg3,* ) i2       ! offset
	READ( farg4,* ) stride
#else
	READ( farg2,'(I)' ) dim
	axis = grid_line(dim, grid)
	READ( farg3,'(I)' ) i2       ! offset
	READ( farg4,'(I)' ) stride
#endif
	i2 = i2 + 1
	num_strings = 0
	DO 2210 i = 1, nnums
	   IF ( i2 .GT. line_dim(axis) ) RETURN
	   nums( i ) = TM_WORLD( i2, grid, dim, box_middle)
	   num_vals = i
	   i2 = i2 + stride
 2210	CONTINUE
	RETURN

* SSPOINT grid orientation delta_index index
 2300	CONTINUE

* WWPOINT grid orientation delta_index coordinate
 2400	CONTINUE

* UVAR_DEFINITION dset variable
* text of the algebraic expression defining the variable
 2500	dset = FIND_DSET_NUMBER(farg1)
	IF ( dset .EQ. unspecified_int4 ) THEN
	   qstatus = QSTAT_INVALID_ARG
	   RETURN
	ENDIF
	CALL FIND_VAR_NAME( dset, farg2, cat, var )
	IF ( dset .EQ. unspecified_int4 ) THEN
	   qstatus = QSTAT_INVALID_ARG
	   RETURN
	ENDIF

* ... return the definition
	slen = TM_LENSTR1(uvar_text(var))
	CALL CSTRING_INSERT( uvar_text(var),
     .			     slen, ctext, ntext, insert_pt)
* ... the title
	slen = TM_LENSTR1(uvar_title(var))
	CALL CSTRING_INSERT( uvar_title(var),
     .			     slen, ctext, ntext, insert_pt)
* ... and the units
	slen = TM_LENSTR1(uvar_units(var))
	CALL CSTRING_INSERT( uvar_units(var),
     .			     slen, ctext, ntext, insert_pt)
	num_strings = 3
	RETURN


* ****  *************
 2600	CONTINUE
* ****  *************
 2700	CONTINUE
* ****  *************
 2800	CONTINUE
* ****  *************
 2900	CONTINUE
* ****  *************
 3000	CONTINUE
	qstatus = QSTAT_UNKNOWN_QUERY
	RETURN

	END
