	SUBROUTINE DIAG_OP( doing, act_class, cx, idim )

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

* calling argument declarations:
	INTEGER	act_class, idim, cx
	CHARACTER*(*) doing

* internal variable declarations:
	INTEGER	  TM_LENSTR1, slen, clen, ii, i
	CHARACTER VAR_CODE*24, VAR_TRANS*64, var*24, out_buff*64

#ifdef unix
	include	'tmap_format/tmap_dims.parm'
	include	'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/interp_stack.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xprog_state.cmn'
	include	'ferret_cmn/xtext_info.cmn'
        include 'ferret_cmn/xalgebra.cmn'
	include 'ferret_cmn/xrisc.cmn'		! 12/94 SPLIT_LIST buffer

#else
	INCLUDE	'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE	'FERRET_CMN:XTEXT_INFO.CMN'
        INCLUDE 'FERRET_CMN:XALGEBRA.CMN'
	INCLUDE 'FERRET_CMN:XRISC.CMN'		! 12/94 SPLIT_LIST BUFFER
#endif

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

* REGRIDDING
	ELSEIF( act_class .EQ. isact_class_regrid ) THEN
	   WRITE ( risc_buff, 3020 ) doing, ww_dim_name(idim),
     .		       var(1:slen), grid_name(cx_grid(cx)),
     .                 alg_regrid_code(cx_regrid_trans(idim,cx))
 3020	   FORMAT (1X,A5,' regrid on ',A1,': ',A,' --> ',A,' @',A)
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun, risc_buff, 0 )

* SPLITTING/GATHERING
	ELSEIF( act_class .EQ. isact_class_gather ) THEN
	   WRITE ( risc_buff, 3030 ) doing, var(1:slen),
     .                  ww_dim_name( idim ),
     .			cx_lo_ss(cx,idim), cx_hi_ss(cx,idim)
 3030	   FORMAT (1X,A5,' gathering ',A,' on ',A1,' axis:',2I6)
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun, risc_buff, 0 )

* ASSEMBLING MODULO COMPONENTS
	ELSEIF( act_class .EQ. isact_class_modulo ) THEN
	   WRITE ( risc_buff, 3040 ) doing, var(1:slen),
     .                  ww_dim_name( idim ),
     .			cx_lo_ss(cx,idim), cx_hi_ss(cx,idim)
 3040	   FORMAT (1X,A5,' moduloing ',A,' on ',A1,' axis:',2I6)
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun, risc_buff, 0 )

* reversed axis reads
	ELSEIF( act_class .EQ. isact_class_reverse ) THEN
	   WRITE ( risc_buff, 3050 ) doing, var(1:slen),
     .                  ww_dim_name( idim ),
     .			cx_lo_ss(cx,idim), cx_hi_ss(cx,idim)
 3050	   FORMAT (1X,A5,' reversing ',A,' on ',A1,' axis:',2I6)
	   CALL SPLIT_LIST(pttmode_ops, ttout_lun, risc_buff, 0 )

* implicit grid allocation/deallocation
	ELSEIF( act_class .EQ. isact_class_impl_grid ) THEN
* ... special case: "cx" is actually the grid number !!!
	   DO 500 ii = 1, 4
	      i = grid_line(ii,cx)
	      IF ( i .GT. 0 ) THEN
	         WRITE (out_buff((ii-1)*10+1:ii*10),'(A)') line_name(i)(:10)
	      ELSEIF ( i .EQ. 0 ) THEN
	         WRITE (out_buff((ii-1)*10+1:ii*10),'(A)') 'NORMAL    '
	      ELSE
	         out_buff((ii-1)*10+1:ii*10) = '(invalid) '
	      ENDIF
 500	   CONTINUE
	   WRITE ( risc_buff, 3060 ) doing, grid_name(cx), out_buff(:40)
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

	ELSE
	   CALL WARN( 'CORRUPTION!  unkn_act_cls' )

	ENDIF

	RETURN
	END


