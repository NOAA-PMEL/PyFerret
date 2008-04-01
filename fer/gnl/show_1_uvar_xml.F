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
* V602 12/06 *acm* remove extra call to GRID_FROM_NAME_XML
* V606  8/07 *acm* when there are quotes within the var definition, as in
*                  a date specification change these to &quot; for valid xml.

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
     .          slen, llen, grid, cx_expr, i, j, status,
     .            varid, vtype, nvdims, vdims(8), nvatts, blen, 
     .            len_attbuff, attoutflag, all_outflag, attid, 
     .            attlen, attype
	CHARACTER FULL_UVAR_NAME*150, TM_FMT*12,
     .            buff*150, buff2*170, c*1, 
     .            attname*128, attbuff*2048
	REAL      vals(10)
        LOGICAL TM_HAS_STRING, save_mode_ignore, save_silent_err, new

 2020   FORMAT ('<var name="', A, '" def="', A, '">' )
 2030   FORMAT( '<attribute name="units" value="', A, '" />' )
 2040   FORMAT( '<attribute name="long_name" value="', A, '" />' )
 2070   FORMAT( '<attribute name="missing_value" value="', A, '" />' )
 2075   FORMAT( '<attribute name="', A, '" value="', A, '" />' )
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
        IF ( TM_HAS_STRING( buff, '"') ) THEN
           j = 1
           buff2(1:1) = buff(1:1)
           DO 100 i = 1, llen
              c = buff(i:i)
              IF (c .EQ.'"') THEN
                 buff2 = buff2(1:j) // '&quot;'
                 j = j + 6
              ELSE
                 buff2 = buff2(1:j) // c
                 IF (i.GT.1) j = j + 1
              ENDIF
 100       CONTINUE
           buff = buff2
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


* All attributes not already listed.

           DO 200 iatt = 1, nvatts
              CALL CD_GET_VAR_ATT_INFO (dset, varid, iatt, attname, 
     .              attype, attlen, attoutflag, status )

              new = .TRUE.
              new = (STR_SAME(attname, 'units') .NE. 0)         .AND. 
     .              (STR_SAME(attname, 'long_name') .NE. 0)     .AND. 
     .              (STR_SAME(attname, 'history') .NE. 0)       .AND. 
     .              (STR_SAME(attname, 'missing_value') .NE. 0) .AND. 
     .              (STR_SAME(attname, '_FillValue') .NE. 0)

              IF (new) THEN
                 blen = 2048  ! len of attbuff
                 got_it = NC_GET_ATTRIB (dset, varid, attname, 
     .                                .TRUE., uvarname, blen, attlen, 
     .                                attoutflag, attbuff, vals)

                 IF (attype .NE. NCCHAR) THEN
                    blen = 1
                    attbuff = ' '
                    DO 150 i = 1, attlen
                       buff = TM_FMT(vals(i), 7, 14, llen)
                       attbuff(blen:blen+llen+1) = buff//' '
                       blen = blen + llen + 1
 150                   CONTINUE
                 ELSE 
                    blen = TM_LENSTR1(attbuff)
                 ENDIF

                 slen = TM_LENSTR1(attname)
	         WRITE (risc_buff, 2075) attname(1:slen), 
     .                  attbuff(1:blen)
                 CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

              ENDIF

 200       CONTINUE

* ... grid

        IF (grid .NE. unspecified_int4) 
     .          CALL SHOW_GRID_XML (lun, grid, cx_expr)

        WRITE ( risc_buff, 2090 )
	CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

	RETURN
	END
