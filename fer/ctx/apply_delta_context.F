	SUBROUTINE APPLY_DELTA_CONTEXT ( cx, reg_name, status )

* apply delta coordinate limits of the FORM DX=, DY=, etc.
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
