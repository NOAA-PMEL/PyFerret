	SUBROUTINE GET_NEW_CX ( dflt_cx, new_cx, status )

* using the default context supplied and the context modifiers given with the
* command produce a new context

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 9/22/87 - code stoled from GET_CMND_DATA
* revision 0.1 - 7/27/88 - added dflt_cx to get_context_mods call
* V200:  5/19/89 - transforms not permitted on command line

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations
	INTEGER		dflt_cx, new_cx, status

* local variable declarations
	INTEGER		idim, cmnd_mods_cx
	CHARACTER	ALG_TRANS_CODE*3

* de-allocate old context space
	cx_stack_ptr = cx_stack_ptr_base

* start with the default context for the evaluation of this expression
	CALL TRANSFER_CONTEXT ( dflt_cx, new_cx )

* create a clean stack slot to receive the command context modifications
	cmnd_mods_cx = cx_stack_ptr + 1
	CALL INIT_CONTEXT_MODS ( cmnd_mods_cx )

* translate the command context modifiers into the modifications buffer
	CALL GET_CONTEXT_MODS ( dflt_cx,
     .				cmnd_buff,
     .				cmnd_mods_cx,
     .				max_qual_list,
     .				num_qualifiers,
     .				qual_start,
     .				qual_end,
     .				unknown_qual_ok,
     .				status	)
	IF ( status .NE. ferr_ok ) GOTO 5000

* transformation given on command line ?
	DO 100 idim = 1, 4
 100	IF (cx_trans(idim,cmnd_mods_cx) .NE. trans_no_transform)
     .        GOTO 5010

* modify the command context with the given modifiers
	CALL APPLY_CONTEXT_MODS
     .			( cmnd_mods_cx, new_cx, replace_context, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

	RETURN

* error exit
 5000	RETURN
 5010	CALL ERRMSG( ferr_syntax, status,
     .	'transforms not permitted on command: @'//
     .		ALG_TRANS_CODE( cx_trans(idim,cmnd_mods_cx) ), *5000 )
	END
