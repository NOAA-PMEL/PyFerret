	SUBROUTINE READ_TM ( memory, cx, mr, status )

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
* read a grid of data from a TMAP data set

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 11/28/88 - based on IS_READ_GT v0.3 and READ_GT_XYZ v1.1
* V200:  5/19/89 - reordered lo/hi_ss args
*	 12/8/89 - changed call arguments in TM_READ for new compiler
* V230:  7/30/92 - must CALL ERRMSG to stop command files, etc. following error
*        11/6/92 - increased variable length to 24 characters
* V312: 5/94 - array "memory" as a calling argument
* v500: 2/99 - *kob* add call to new routine REPLACEABLE_BAD_FLAGS to allow
*              and check for NaN as missing values
#ifdef unix
	include	'tmap_format/tmap_errors.parm'
	include	'tmap_format/tmap_dims.parm'
	include	'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include	'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE	'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE	'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations:
	INTEGER	cx, mr, status
	REAL	memory( mem_blk_size, max_mem_blks )

* internal variable declarations:
	INTEGER 	idim, variable, category, data_set, grid,
     .			sf_num, tmap_status, lo_ss(4), hi_ss(4),
     .			lo_s1, lo_s2, lo_s3, lo_s4, 
     .			hi_s1, hi_s2, hi_s3, hi_s4 
	CHARACTER*24	VAR_CODE, variable_code

* *kob* 2/99 - add declaration for new c function
	INTEGER REPLACEABLE_BAD_FLAGS

	EQUIVALENCE	(lo_ss(1),lo_s1), (lo_ss(2),lo_s2),
     .			(lo_ss(3),lo_s3), (lo_ss(4),lo_s4),
     .			(hi_ss(1),hi_s1), (hi_ss(2),hi_s2),
     .			(hi_ss(3),hi_s3), (hi_ss(4),hi_s4)

* --- end of introductory code ---

* initialize
	data_set = cx_data_set( cx )
	category = cx_category( cx )
	variable = cx_variable( cx )
	grid = cx_grid( cx )
	variable_code = VAR_CODE( category, variable )

* reserve and catalog space for the result
	CALL CREATE_MEM_VAR( cx, mr, status )
	IF ( status .NE. ferr_ok ) GOTO 5900

* diagnostic output
	IF (mode_diagnostic) CALL DIAGNOSTIC_OUT('reading',mr,point_to_mr)

* set up array of subscripts
	DO 100 idim = 1, 4
	   IF ( grid_line( idim, grid ) .EQ. mnormal ) THEN
	      lo_ss( idim ) = 1
	      hi_ss( idim ) = 1
	   ELSE
	      lo_ss( idim ) = cx_lo_ss( cx, idim )
	      hi_ss( idim ) = cx_hi_ss( cx, idim )
	   ENDIF
 100	CONTINUE

* read the data
	CALL TM_READ  ( data_set,
     .			variable_code,
     .			lo_s1, lo_s2, lo_s3, lo_s4,
     .			hi_s1, hi_s2, hi_s3, hi_s4,
     .			lo_ss,
     .			hi_ss,
     .			memory( 1, mr_blk1( mr ) ),
     .			sf_num,
     .			tmap_status		)
	IF ( tmap_status .NE. merr_ok ) THEN
	   CALL DELETE_VARIABLE( mr )
	   GOTO 5010
	ENDIF

* replace multiple bad data flags in some variables
* IF ( ds_bad_flag( variable ) .NE. ds_missing_flag( variable ) )
* *kob* 2/99 - call new routine to check for usage of NaN.
	IF (REPLACEABLE_BAD_FLAGS(ds_bad_flag(variable), 
     .                            ds_missing_flag(variable)) .EQ. 1)
     .	   CALL REPLACE_BAD_DATA
     .		(memory(1,mr_blk1(mr)),mr,cx,ds_missing_flag(variable))

* successful completion
	RETURN

* error exit
 5010	CALL ERRMSG( ferr_TMAP_error, status, ' ', *5900 )   ! 7/92
 5900	RETURN
	END	
