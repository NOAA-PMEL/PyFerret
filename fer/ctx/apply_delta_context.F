	SUBROUTINE APPLY_DELTA_CONTEXT ( cx, reg_name, status )

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
* apply delta coordinate limits of the FORM DX
* to the indicated context

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:   5/2/90 - extracted from GET_DELTA_CONTEXT
* Unix/RISC port - 1/91 - cannot use "//" with char*(*)

#ifdef unix
	include 'ferret_cmn/errmsg.parm'	
	include 'ferret_cmn/ferret.parm'	
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xdelta_context.cmn'
	include 'ferret_cmn/xtext_info.cmn'
	include 'ferret_cmn/xrisc.cmn'
#else
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'	
	INCLUDE 'FERRET_CMN:FERRET.PARM'	
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XDELTA_CONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
	INCLUDE 'FERRET_CMN:XRISC.CMN'
#endif

* calling arguments:
	INTEGER		cx, status
	CHARACTER*(*)	reg_name

* internal variables:
	INTEGER		idim
	LOGICAL		INVALID_R8_TO_I4

* initialize fixed length string for RISC
        risc_buff = reg_name
        len_rbuff = MIN( size_rbuff, LEN(reg_name) )

	DO 100 idim = 1, 4

	   IF ( .NOT.dcx_given(idim) ) GOTO 100

* apply delta limits from common
	   IF ( dcx_by_ss(idim) .NEQV. cx_by_ss(idim,cx) ) THEN
	      IF ( cx_lo_ss(cx,idim).EQ.unspecified_int4
     .	     .AND. cx_lo_ww(idim,cx).EQ.unspecified_val8 ) THEN
	         GOTO 5500
	      ELSE
	         GOTO 5300
	      ENDIF
	   ENDIF
	   IF ( dcx_by_ss(idim) ) THEN
	      IF ( cx_lo_ss(cx,idim) .EQ. unspecified_int4 ) GOTO 5500
	      IF ( INVALID_R8_TO_I4( dcx_lo(idim) )
     .        .OR. INVALID_R8_TO_I4( dcx_hi(idim) )) GOTO 5400
	      cx_lo_ss(cx,idim) = cx_lo_ss(cx,idim) + NINT( dcx_lo(idim) )
	      cx_hi_ss(cx,idim) = cx_hi_ss(cx,idim) + NINT( dcx_hi(idim) )
	   ELSE
	      IF ( cx_lo_ww(idim,cx) .EQ. unspecified_val8 ) GOTO 5500
	      cx_lo_ww(idim,cx) = cx_lo_ww(idim,cx) + dcx_lo(idim)
	      cx_hi_ww(idim,cx) = cx_hi_ww(idim,cx) + dcx_hi(idim)
	   ENDIF

* flag that this axis was specified
	   cx_given(idim,cx) = .TRUE.

 100	CONTINUE

* success
	status = ferr_ok
	RETURN

* error exits
 5000	RETURN
 5300	CALL ERRMSG( ferr_relative_coord, status, 
     .		     'must be a subscript or a world position'//pCR//
     .		     'as in the original '//ww_dim_name(idim)//
     .		     ' axis of region '//risc_buff(:len_rbuff), *5000)
 5400	CALL ERRMSG( ferr_out_of_range, status,
     .		     'delta value is too large to be a subscript', *5000)
 5500	CALL ERRMSG( ferr_relative_coord, status,
     .		     ww_dim_name(idim)//' axis coordinate is unspecified'//
     .		     pCR//'on region '//risc_buff(:len_rbuff), *5000)

	END
