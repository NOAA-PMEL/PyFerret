	SUBROUTINE DIAG_OP( doing, act_class, cx, idim )

*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
* write a diagnostic line describing the operation in progress

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V200:  6/6/89 - extracted from IS_STRIP
* V230: 8/22/92 - document regridding transformation
*       11/4/92 - expanded user variable name length
* v320:   12/94 - added modulo reading as a stack operation
*      12/29/94 - use SPLIT_LIST to redirect tty output for GUI
*	2/21/95 - added isact_reverse 
* V420   9/8/95 - support for implicit grid allocation/deallocation
*	  11/95 - support for context reconciliation (isact_class_reconcile)
*	   1/96 - bug fix: implicit grid kludge use of cx for gridnum crashes
*		- use "strip" to determine if reconcilliation axis is relevant
* V500 *kob* 3/99- up VAR_CODE and var decls from 8 to 64 chars
* V510 *sh* 8/99 - added isact_class_sample
* V541 *sh*  2/02 - added isact_class_subspan_fill
* V552 *acm* 4/03- up VAR_CODE decl from 64 to 128 chars
* v552 *acm* 4/03  up VAR_TRANS to 150 characters (variables are up to 128, plus
*                  space for the transformation specifier)
* V64  10/09 *acm* longer string length for grid_name: limit length written to diag output
* V680  5/12 *acm* Add dataset ID to diagnostic output
* V680  6/12 *acm* 6D Ferret, changes for DEFINE DATA/AGGREGATE
* V6.8  acm  6/12 Implement mode_6d_lab, set by default. If cancelled, the 
*                 output will not list info for E,F dimensions.
* V690  1/14 *sh* report 3-argument var-getting operation
*		  allow for "false 1 point" regridding
* V7    acm  6/16 Ticket 2437: Deprecating mode_6d_lab. 
*		  Adjust output to 6-D or 4-D grids
* V701  acm  7/16 Avoid writing nulls when changing from 6-D to 4-D output
* V702 *sh* 11/16 report gathering on compressed axis
* V702  sh   1/17 Output total_mem along with gather diagnostics
*                 Increase field sizes for long axes
* v767 6/22 *acm* See issue 112: Use the INTEGER*8 routine CGRID_SIZE8

* calling argument declarations:
        IMPLICIT NONE
	INTEGER	act_class, idim, cx
	CHARACTER*(*) doing

* internal variable declarations:
	LOGICAL	FOUR_D_GRID
	INTEGER	TM_LENSTR1, slen, clen, ii, i, listdims
	INTEGER*8 CGRID_SIZE8, available_mem, chunk_size
	CHARACTER*3   ALG_TRANS_CODE
	CHARACTER*128 VAR_CODE,  var
	CHARACTER*150 VAR_TRANS, out_buff

	include	'tmap_dims.parm'
	include	'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include	'interp_stack.parm'
	include	'xvariables.cmn'
	include	'xcontext.cmn'
	include	'xprog_state.cmn'
	include	'xtext_info.cmn'
        include 'xalgebra.cmn'
	include 'xrisc.cmn'		! 12/94 SPLIT_LIST buffer

	listdims = nferdims
	IF ( .NOT. mode_6d_lab) listdims = 4

* initialize
! 1/96 - kludge of using cx for grid # causes this extra IF -- awful!
	IF( act_class .NE. isact_class_impl_grid ) THEN
	   var = VAR_CODE( cx_category(cx), cx_variable(cx) )
           slen = TM_LENSTR1( var )
	ENDIF

* TRANSFORMATION
* output: "VAR_NAME[x=lo:hi@TRN:n,D=#]"
	risc_buff = ' '
	IF ( act_class .EQ. isact_class_trans ) THEN
	   out_buff = VAR_TRANS(idim,cx,slen)
	   WRITE( risc_buff, 3010 ) doing, out_buff(:slen)
 3010      FORMAT (1X,A5,' --> ',A)
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun, risc_buff, 0 )

* REGRIDDING (4/14 - incl false 1pt regridding)
	ELSEIF( act_class .EQ. isact_class_regrid ) THEN
	   IF (cx_aux_stat(idim,cx) .EQ. paux_stat_false_1pt_regrid ) THEN
	      WRITE ( risc_buff, 3019 ) doing,
     .		       var(1:slen),
     .		       ww_dim_name(idim),
     .                 alg_regrid_code(cx_regrid_trans(idim,cx))
 3019	      FORMAT (1X,A5,' false regrid of ',A,' --> ',A1,'@',A)
	   ELSE
	      WRITE ( risc_buff, 3020 ) doing, ww_dim_name(idim),
     .		       var(1:slen), grid_name(cx_grid(cx))(:16),
     .                 alg_regrid_code(cx_regrid_trans(idim,cx))
 3020	      FORMAT (1X,A5,' regrid on ',A1,': ',A,' --> ',A,' @',A)
	   ENDIF
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun, risc_buff, 0 )

* SPLITTING/GATHERING
	ELSEIF( act_class .EQ. isact_class_gather ) THEN
	   chunk_size = CGRID_SIZE8(cx)
	   available_mem = max_mem_allowed-(permanent_mem+essential_mem)
	   WRITE ( risc_buff, 3030 ) doing, var(1:slen),
     .                  ww_dim_name( idim ),
     .			cx_lo_ss(cx,idim), cx_hi_ss(cx,idim),
     .			cx_data_set(cx), chunk_size, available_mem
 3030	   FORMAT
     . (1X,A5,' gathering ',A,' on ',A1,' axis:',2I9,' dset:',I4, 
     .  I14,'=request', I14, '=availableMem')
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun, risc_buff, 0 )

* COMPRESSED AXIS SPLITTING/GATHERING
	ELSEIF( act_class .EQ. isact_class_cmp_gather_dummy ) THEN
	   chunk_size = CGRID_SIZE8(cx)
	   available_mem = max_mem_allowed-(permanent_mem+essential_mem)
	   WRITE ( risc_buff, 3031 ) doing, var(1:slen),
     .                  ww_dim_name( idim ),
     .			cx_lo_ss(cx,idim), cx_hi_ss(cx,idim),
     .			cx_data_set(cx), chunk_size, available_mem
 3031	   FORMAT
     . (1X,A5,' gather/compress ',A,' on ',A1,' axis:',2I9,' dset:',I4, 
     .  I14,'=request', I14, '=availableMem')
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun, risc_buff, 0 )

* SAMPLING (SAMPLE* functions)
	ELSEIF( act_class .EQ. isact_class_sample ) THEN
	   WRITE ( risc_buff, 3035 ) doing, var(1:slen),
     .                  ss_dim_name( idim ),
     .			cx_lo_ss(cx,idim), cx_hi_ss(cx,idim),
     .			cx_data_set(cx)
 3035	   FORMAT
     . (1X,A5,' sampling ',A,' on ',A1,' axis:',2I9,' dset:',I4)
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun, risc_buff, 0 )

* ASSEMBLING MODULO COMPONENTS
	ELSEIF( act_class .EQ. isact_class_modulo ) THEN
	   WRITE ( risc_buff, 3040 ) doing, var(1:slen),
     .                  ww_dim_name( idim ),
     .			cx_lo_ss(cx,idim), cx_hi_ss(cx,idim),
     .			cx_data_set(cx)
 3040	   FORMAT 
     . (1X,A5,' moduloing ',A,' on ',A1,' axis:',2I9,' dset:',I4)
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun, risc_buff, 0 )

* reversed axis reads
	ELSEIF( act_class .EQ. isact_class_reverse ) THEN
	   WRITE ( risc_buff, 3050 ) doing, var(1:slen),
     .                  ww_dim_name( idim ),
     .			cx_lo_ss(cx,idim), cx_hi_ss(cx,idim),
     .			cx_data_set(cx)
 3050	   FORMAT
     . (1X,A5,' reversing ',A,' on ',A1,' axis:',2I9,' dset:',I4)
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun, risc_buff, 0 )

* implicit grid allocation/deallocation
	ELSEIF( act_class .EQ. isact_class_impl_grid ) THEN
* ... special case: "cx" is actually the grid number !!!

	   IF ( FOUR_D_GRID(cx) ) listdims = 4
	   DO 500 ii = 1, listdims
	      i = grid_line(ii,cx)
	      IF ( i .GT. 0 ) THEN
	         WRITE (out_buff((ii-1)*10+1:ii*10),'(A)') line_name(i)(:10)
	      ELSEIF ( i .EQ. 0 ) THEN
	         WRITE (out_buff((ii-1)*10+1:ii*10),'(A)') 'NORMAL    '
	      ELSE
	         out_buff((ii-1)*10+1:ii*10) = '(invalid) '
	      ENDIF
 500	   CONTINUE

	   IF (listdims .EQ. 6)
     .	     WRITE ( risc_buff, 3060 ) doing, grid_name(cx)(:16), 
     .	           out_buff(:60)
	   IF (listdims .EQ. 4)
     .	     WRITE ( risc_buff, 3060 ) doing, grid_name(cx)(:16), 
     .	           out_buff(:40)

 3060	   FORMAT (1X,A,' dynamic grid ',2A)
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun, risc_buff, 0 )

	ELSEIF( act_class .EQ. isact_class_reconcile ) THEN
	   IF ( doing .EQ. 'strip' ) THEN
* ... idim has no meaningful value when stripping reconciliation
	      out_buff = ' '
	      clen = 1
	   ELSE
	      out_buff = ' on '//ww_dim_name(idim)//' axis'
	      clen = 10
	   ENDIF
	   WRITE ( risc_buff, 3070 ) doing, out_buff(:clen),
     .				     var(1:slen)
 3070	   FORMAT (1X,A5,' limits reconciliation',A,': ',A)
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun, risc_buff, 0 )

* subspan modulo axis void filling
	ELSEIF( act_class .EQ. isact_class_subspan_fill ) THEN
	   WRITE ( risc_buff, 3080 ) doing, var(1:slen),
     .                  ww_dim_name( idim ),
     .			cx_lo_ss(cx,idim), cx_hi_ss(cx,idim),
     .			cx_data_set(cx)
 3080	   FORMAT 
     . (1X,A5,' modulo-void-filling ',A,' on ',A1,' axis:',2I9,' dset:',I4)
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun, risc_buff, 0 )


* SPLITTING/GATHERING
	ELSEIF( act_class .EQ. isact_class_aggregate_gather ) THEN
	   WRITE ( risc_buff, 3090 ) doing, var(1:slen),
     .                  ww_dim_name( idim ),
     .			cx_lo_ss(cx,idim), cx_hi_ss(cx,idim),
     .			cx_data_set(cx)
 3090	   FORMAT
     . (1X,A5,' aggregate gathering ',A,' on ',A1,' axis:',2I9,' dset:',I4)
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun, risc_buff, 0 )

* getting arguments needed for curvilinear plotting
	ELSEIF( act_class .EQ. isact_class_aux_plot_vars ) THEN
	   WRITE ( risc_buff, 3100 ) doing, var(1:slen),
     .                  ww_dim_name( idim ),
     .			cx_lo_ss(cx,idim), cx_hi_ss(cx,idim),
     .			cx_data_set(cx)
 3100	   FORMAT
     . (1X,A5,' getting aux vars ',A,' on ',A1,' axis:',2I9,' dset:',I4)
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun, risc_buff, 0 )

	ELSE
	   CALL WARN( 'CORRUPTION!  unkn_act_cls' )

	ENDIF

	RETURN
	END


