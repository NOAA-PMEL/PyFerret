	SUBROUTINE GCF_IMPOSE_ARG_LIM_DFLTS( vname, arg,
     .					     cx_in, cx_out, uvar, status )

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
* This routine is used in setting the context used for **obtaining**
* the component arguments to a grid changing function.
* This is the so-called "upstream" passing of context.

* The input context, cx_in, is the limits requested for the result.
* For example, in
*	LET a = GCFCN( b )
*	LIST/I=1:5 a[J=5:10]
* the input context would contain I=1:5 and J=5:10.
* The decision of what defaults to set when requesting the argument ("b")
* is the business of this routine.

* V450: 1/97
* V500: 10/98 - *jc* We need to remove the @RDC transform from
*       contexts when the parent is also a grid changing variable.
* V490: 11/97 - bug fix: when wiping out context axis wipe ww lims, too
* V491:  3/98 - bug fix: crashed because pseudo-variables did not have
*		valid grids when this routine was called. Changed to use
*		**result** grid to convert world coords to subscripts when
*		offsets need to be applied -- argument grid not needed.
*		Hugely simplified this routine!
* v580 *acm* 11/04 Reset cx_cal_id, the calendar ID used when  
*                  computing world coordinates
* v581 *acm*  2/05 for axis_expand, if the requested offset is given as 
*                  unspecified_int4 then use the full axis range, or if 
*                  the axis is modulo, then one full modulo repetition

*	Note: the "offset" code in this routine is largely untested!

	include 'tmap_dims.parm'
	include	'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'grid_chg_fcns.parm'
	include	'xcontext.cmn'


* calling argument declarations
	INTEGER		arg, cx_in, cx_out, uvar, status
	CHARACTER*(*)	vname		! unused after 3/98

* local variable declarations
	INTEGER	TM_LENSTR1, ISUBSCRIPT, KNOWN_GRID,
     .		gcfcn, idim, arg_grd, mods_cx, tmp_var_cx,
     .		lo_off(4), hi_off(4), action(4), uvgrid,
     .          WHICH_GCFCN, lo, hi, ss_mid, cx_mid
       LOGICAL ss_fix

* initialize
	CALL TRANSFER_CONTEXT ( cx_in, cx_out )

* 10/98 fix
	DO 10 idim = 1, 4
           IF ( cx_trans(idim, cx_out) .EQ. trans_reduced ) THEN
              cx_trans(idim, cx_out) = trans_no_transform
           ENDIF
 10     CONTINUE

* which grid changing function does this uvar have?
	gcfcn = WHICH_GCFCN(uvar)

* get the grid of the result of this function (user variable)
	uvgrid = KNOWN_GRID(cx_data_set(cx_in), cat_user_var, uvar)

* What did the function creator specify about this argument?
	CALL GCF_GET_ARG_LIMS( gcfcn, arg, cx_out, uvgrid,
     .				lo_off, hi_off, action, status)
	IF ( status .NE. ferr_ok ) RETURN

* loop through the axes
	DO 500 idim = 1,4

	   IF ( action(idim) .EQ. pset_to_unspec ) THEN
* set the axis limits to unspecified
	      cx_lo_ss(cx_out,idim)      = unspecified_int4
	      cx_hi_ss(cx_out,idim)      = unspecified_int4
	      cx_by_ss(idim,cx_out)      = .TRUE.
	      cx_given(idim,cx_out)      = .FALSE.	! needed ?
	      cx_lo_ww(idim,cx_out)      = unspecified_val8	! 11/97
	      cx_hi_ww(idim,cx_out)      = unspecified_val8	! 11/97
	      cx_cal_id(cx_out)          = unspecified_int4
!	      cx_trans (idim,cx_out)     = trans_no_transform
!	      cx_trans_arg(idim,cx_out)  = bad_val4  
	      IF (idim .eq. t_dim) 
     .                cx_calendar(cx) = .FALSE.

	   ELSEIF (lo_off(idim).NE.0 .OR. hi_off(idim).NE.0) THEN
* pass along the axis limits 
                 
              IF (lo_off(idim) .EQ. unspecified_int4 .OR.
     .            hi_off(idim) .EQ. unspecified_int4)  
     .           CALL GRID_SUBSCRIPT_EXTREMES_NO_MOD
     .                             (lo, hi, uvgrid, idim)
                 ss_mid = (hi - lo + 1)/ 2.


	      IF ( cx_by_ss(idim,cx_out) ) THEN
* ... expand the subscript range by the offsets requested. If the
*     requested offset is unspecified_int4 then use the full axis range,
*     or if modulo, then use either 
*    1) one full modulo repitition, centered on the output range, or
*    2) the requested range, if it is larger than one modulo repetitionpetition

	         IF ( cx_hi_ss(cx_out,idim).NE.unspecified_int4 ) THEN

* special fix for modulo axes.
                    ss_fix = ( cx_lo_ss(cx_out,idim) .LT. lo .OR.
     .                      cx_hi_ss(cx_out,idim) .GT. hi )  ! special fix for modulo axes

                    cx_mid = (1 + cx_hi_ss(cx_out,idim) +
     .                            cx_lo_ss(cx_out,idim))/2.
                    IF (lo_off(idim) .EQ. unspecified_int4 )  THEN
                       IF (ss_fix) THEN
                          cx_lo_ss(cx_out,idim) =  MIN(cx_mid - ss_mid,
     .                                cx_lo_ss(cx_out,idim))
                       ELSE
                          cx_lo_ss(cx_out,idim) = lo
                       ENDIF
                    ELSE
	               cx_lo_ss(cx_out,idim) = cx_lo_ss(cx_out,idim)
     .					  + lo_off(idim)
                    ENDIF
                    IF (hi_off(idim) .EQ. unspecified_int4 )  THEN
                       IF (ss_fix) THEN
                          cx_hi_ss(cx_out,idim) = MAX(cx_mid + ss_mid, 
     .                                cx_hi_ss(cx_out,idim) )
                       ELSE
                          cx_hi_ss(cx_out,idim) = hi
                       ENDIF
                    ELSE
 	               cx_hi_ss(cx_out,idim) = cx_hi_ss(cx_out,idim)
     .					  + hi_off(idim)
                    ENDIF
		 ENDIF
	      ELSE
* ... convert the world coordinate range to subscript limits. Then
* ... expand the subscript range by the offsets requested
	         IF ( cx_hi_ww(idim,cx_out).NE.unspecified_val8 ) THEN
                 
	            cx_lo_ss(cx_out,idim) =
     .				ISUBSCRIPT(cx_lo_ww(idim,cx_out),
     .		 			   uvgrid, idim, round_up)
                    cx_hi_ss(cx_out,idim) = 
     .				ISUBSCRIPT(cx_hi_ww(idim,cx_out),
     .					   uvgrid, idim, round_dn)

* special fix for modulo axes. 
                    ss_fix = ( cx_lo_ss(cx_out,idim) .LT. lo .OR.
     .                      cx_hi_ss(cx_out,idim) .GT. hi )  
                    cx_mid = (1 + cx_hi_ss(cx_out,idim) +
     .                            cx_lo_ss(cx_out,idim))/2.

                    IF (lo_off(idim) .EQ. unspecified_int4 )  THEN
                       IF (ss_fix) THEN
                          cx_lo_ss(cx_out,idim) = MIN(cx_mid - ss_mid,
     .                                cx_lo_ss(cx_out,idim))
                       ELSE
                          cx_lo_ss(cx_out,idim) = lo
                       ENDIF
                    ELSE
	               cx_lo_ss(cx_out,idim) = cx_lo_ss(cx_out,idim)
     .					  + lo_off(idim)
                    ENDIF

                    IF (hi_off(idim) .EQ. unspecified_int4 )  THEN
                       IF (ss_fix) THEN
                          cx_hi_ss(cx_out,idim) = MAX(cx_mid + ss_mid, 
     .                                cx_hi_ss(cx_out,idim) )
                       ELSE
                          cx_hi_ss(cx_out,idim) = hi
                       ENDIF
                    ELSE
	               cx_hi_ss(cx_out,idim) = cx_hi_ss(cx_out,idim)
     .					  + hi_off(idim)
                    ENDIF

		 ENDIF
	         cx_by_ss(idim,cx_out) = .TRUE.
	      ENDIF
	   ENDIF

 500	CONTINUE

* successfully merged
	status = ferr_ok
	RETURN

* error exit
! 5000	RETURN
	END

