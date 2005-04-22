	SUBROUTINE SHOW_1_UVAR_XML( lun, uvar, vname, first )

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
* output some lines describing a user variable in xml-style format
* 20-Oct-2004 acm  from show_1_uvar.F
*  7-Feb-2005 acm  (bug 1165) fix string vars so they are output with 
*                  surrounding &quot;

	include 'tmap_dims.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include	'ferret.parm'
	include	'xvariables.cmn'
	include	'xrisc.cmn'
	include 'xprog_state.cmn'
        include 'errmsg.parm'

* calling argument declarations:
	INTEGER	lun, uvar
	CHARACTER*(*) vname	! buffer -- 255 char
        LOGICAL first
	
* local variable declarations:
	INTEGER TM_LENSTR1, GRID_FROM_NAME_XML,
     .          slen, llen, grid, cx_expr, status
	CHARACTER FULL_UVAR_NAME*150, TM_FMT*12,
     .            buff*150
        LOGICAL save_mode_ignore, save_silent_err

 2020   FORMAT ('<var name="', A, '" def="', A, '">' )
 2030   FORMAT( '<units>', A, '</units>' )
 2040   FORMAT( '<long_name>', A, '</long_name>' )
 2070   FORMAT( '<missing_value>', A, '</missing_value>' )
 2080   FORMAT( '<grid>', A, '</grid>' )
 2090   FORMAT ('</var>')


* initialize
	IF (uvar_num_items(uvar) .EQ. uvar_deleted  ) RETURN

* See if user variable is relevant to the context
* turn off error reporting

        save_mode_ignore = mode_ignore_err 
        silent_err = save_silent_err
        mode_ignore_err = .TRUE.
        silent_err = .TRUE.

        vname = FULL_UVAR_NAME(uvar,slen)
        grid= GRID_FROM_NAME_XML(vname, cx_last, cx_expr, 
     .       status)

* Restore error reporting

        save_mode_ignore = mode_ignore_err 
        mode_ignore_err = save_mode_ignore
        silent_err = save_silent_err

        IF (status .NE. ferr_ok) RETURN

* Before first variable output the global tag.

        IF (first) THEN
           risc_buff = '<global>'
           CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
           first = .FALSE.
        ENDIF

* variable name and definition

* ... output variable code, units, grid, and type

        vname = FULL_UVAR_NAME(uvar,slen)
	buff = uvar_text(uvar)
        llen = TM_LENSTR1(buff)

        IF ( uvar_text(uvar)(1:1) .EQ.'"' .AND.
     .       uvar_text(uvar)(llen:llen) .EQ.'"' ) THEN
           buff = '&quot;' // uvar_text(uvar)(2:llen-1) // '&quot;'
           llen = TM_LENSTR1(buff)
        ENDIF

        WRITE ( risc_buff, 2020 ) vname(1:slen), buff(1:llen)
	
	CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	risc_buff = ' '


* ... title and units 
	IF ( uvar_units(uvar) .NE. ' ' ) THEN
           slen = TM_LENSTR1(uvar_units(uvar))
	   WRITE(risc_buff, 2030) uvar_units(uvar)(1:slen)
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
        ENDIF

	IF ( uvar_title(uvar) .NE. ' ' ) THEN
           slen = TM_LENSTR1(uvar_title(uvar))
	   WRITE(risc_buff, 2040) uvar_title(uvar)(1:slen)
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
        ENDIF

* ... bad data flag 

        buff = TM_FMT(uvar_bad_data(uvar), 16,16,slen)
        WRITE ( risc_buff, 2070 ) buff(1:slen)
	CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

* ... grid

        grid= GRID_FROM_NAME_XML(vname, cx_last, cx_expr, 
     .       status)

        IF (grid .NE. unspecified_int4) 
     .          CALL SHOW_GRID_XML (lun, grid, cx_expr)

        WRITE ( risc_buff, 2090 )
	CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

	RETURN
	END
