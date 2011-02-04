	SUBROUTINE LINE_FACTS_XML( lun, idim, grid, orient, status )

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

* display basic facts about the given line in xml-style format 

* programmer - Ansley Manke, based on line_facts.F
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* 6-Aug-2003

* 2/2005     bug 1171: the code checked line_regular instead
*            of line_modulo for modulo attribute!
* V581*acm*  6/05 - For fix to bug 1271, add flag line_shift_origin
*            and restore original t0 on output
* V600 *acm* 2/06 - Fix for bug 1394, send do_err flag to TM_DATE_REFORMAT
* V602 *acm* 1/07 - add <direction> output -- I J K L for the axis.
* V616  1/2009 *acm* Fixing LAS ticket 481, new style of xml output
* V631 11/09 *acm* For XML-formatted output check for and replace the strings < > and &
*                  with &lt; etc.
* V67  *acm* 2/11 XML fixes. New attribute formatting, ![CDATA] escape, upper/lowercase handling.

	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include 'xtext_info.cmn'
	include 'xrisc.cmn'          ! 12/94 SPLIT_LIST buffer
        include 'errmsg.parm'
	include 'xdset_info.cmn_text'
	include 'netcdf.inc'

* calling argument declarations
	INTEGER	  lun, idim, grid, orient

* local variable declarations:
	LOGICAL	BKWD_AXIS, GEOG_LABEL,
     .          calendar, new, got_it, coordvar
	INTEGER TM_LENSTR, TM_LENSTR1, TM_GET_CALENDAR_ID, 
     .          STR_SAME, line, slen, llen, ndec, cal_id, status,
     .            varid, vtype, nvdims, vdims(8), nvatts, blen, i, 
     .            len_attbuff, attoutflag, all_outflag, attid, 
     .            attlen, attype, iatt, iset, dset, alen
	CHARACTER TM_FMT*16, show_str*16, lo_str*20, hi_str*20, 
     .            title*32, outbuff*20, aname*128,
     .            attname*128, attbuff*2048, buff*20, outstring*2048,
     .            atype*10, axis_name*128

        CHARACTER*1 idir(4)
	REAL*8	TM_WORLD, lo_val8, hi_val8
	REAL      vals(100)

* local parameter declarations:
	INTEGER		sig_figs
	PARAMETER     ( sig_figs = 5 )
        DATA idir/'I', 'J', 'K', 'L'/

 2010	FORMAT ( '<axis name="', A8, 5x, A1, '">' )
 2020	FORMAT ( '<axis name="', A, '">' )
 2030	FORMAT ( '<attribute name="', A, '" type="char">' )
 2040	FORMAT ( '<attribute name="', A, '" type="short">' )
 2050	FORMAT ( '<attribute name="', A, '" type="double">' )
 2060   FORMAT ('   <value><![CDATA[', A, ' since ', A, ']]></value>')
 2070   FORMAT ('   <value>', A, '</value>')
 2075   FORMAT ('   <value><![CDATA[', A, ']]></value>')
 2080   FORMAT ('</attribute>')

* pull line from grid table
	line = grid_line( idim, grid )

* line ok ?
	risc_buff = ' '

	IF ( line .EQ. munknown ) THEN
	   title = 'unknown'
	   WRITE ( risc_buff, 2010 ) title, ww_dim_name( idim )
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
           status = 0
	ELSEIF ( line .EQ. mnormal ) THEN
	   title = 'normal'
	   WRITE ( risc_buff, 2010 ) title, ww_dim_name( idim )
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
           status = 0
	ELSE

	   axis_name = line_name_orig(line)

* header

           CALL ESCAPE_FOR_XML (axis_name, outstring, slen)
	   WRITE ( risc_buff, 2020 ) outstring(1:slen)
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

* axis units, including ... since start date if time axis

* direction I,J,K, or L

           buff = idir(idim)
           CALL ESCAPE_FOR_XML(buff, outstring, slen)
           WRITE ( risc_buff, 2030 ) 'direction'
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
           WRITE ( risc_buff, 2075 ) outstring(1:slen)
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
           WRITE ( risc_buff, 2080 )
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

* Units. Report since time origin for formatted time axis

           slen = TM_LENSTR(line_units(line))
           IF (line_units(line)(1:2) .NE. char_init 
     .           .AND. slen.GT.0) THEN
              
              WRITE ( risc_buff, 2030 ) 'units'
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
              CALL ESCAPE_FOR_XML (line_units(line), outstring, slen)
              WRITE (risc_buff, 2075) outstring(1:slen)

              IF ( line_direction(line) .EQ. 'TI' ) THEN
              
	         buff = line_cal_name (line)
	         cal_id = TM_GET_CALENDAR_ID (buff)

	         IF (line_shift_origin(line)) THEN
	            buff = '01-JAN-0001 00:00:00'
	         ELSE
	            buff = line_t0(line)
	         ENDIF

	         CALL TM_DATE_REFORMAT (buff, 1, cal_id, outbuff, 2, 
     .               .TRUE., status) 
                 llen = TM_LENSTR1(outbuff)
                 WRITE (risc_buff, 2060) outstring(1:slen), outbuff(:llen)

              ENDIF
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
              WRITE ( risc_buff, 2080 )
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

           ENDIF

* line dimension

           show_str = TM_FMT(FLOAT(line_dim(line)), 7,16,slen)
	   CALL ESCAPE_FOR_XML (show_str, outstring, slen)
           WRITE ( risc_buff, 2040 ) 'length'
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
           WRITE ( risc_buff, 2070 ) outstring(1:slen)
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
           WRITE ( risc_buff, 2080 )
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)


* start and end

	   calendar = ( idim .EQ. t_dim .AND. GEOG_LABEL(idim, grid) )

           lo_val8 = TM_WORLD( 1, grid, idim, box_middle )
	   IF ( calendar ) THEN
	      ndec = 5 		! accurate to minutes
 	      CALL TRANSLATE_TO_WORLD( lo_val8, idim, grid, ndec, lo_str )
              CALL TM_DATE_REFORMAT (lo_str, 1, cal_id, outbuff, 2,
     .                               .TRUE., status)
              lo_str = outbuff
	   ELSE
	      ndec = -sig_figs	! 5 sig places w/out extra zeros
	      fval = lo_val8
	      lo_str = TM_FMT(fval, 7,16,slen)
	   ENDIF

	   hi_val8 = TM_WORLD( line_dim(line), grid, idim, box_middle )
	   IF ( calendar ) THEN
	      ndec = 5 
	      CALL TRANSLATE_TO_WORLD( hi_val8, idim, grid, ndec, hi_str )
              CALL TM_DATE_REFORMAT (hi_str, 1, cal_id, outbuff, 2,
     .                               .TRUE., status) 
              hi_str = outbuff
	   ELSE
	      ndec = -sig_figs
	      fval = hi_val8
	      hi_str = TM_FMT(fval, 7, 16, slen)
	   ENDIF

* Write the start and end; if dates then format them as for time origin.

           WRITE ( risc_buff, 2050 ) 'start'
           CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
           CALL ESCAPE_FOR_XML (lo_str, outstring, slen)

           WRITE ( risc_buff, 2070 ) outstring(1:slen)
           CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
           WRITE ( risc_buff, 2080 )
           CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

           WRITE ( risc_buff, 2050 ) 'end' 
           CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
           CALL ESCAPE_FOR_XML (hi_str, outstring, slen)

           WRITE ( risc_buff, 2070 ) outstring(1:slen)
           CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
           WRITE ( risc_buff, 2080 )
           CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

* axis oriented backwards ?
	   IF ( BKWD_AXIS(idim, grid) ) THEN
              WRITE ( risc_buff, 2030 ) 'positive'
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
              WRITE ( risc_buff, 2075 ) 'down'
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
              WRITE ( risc_buff, 2080 )
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	   ENDIF

* regularly spaced ?
           WRITE ( risc_buff, 2030 ) 'point_spacing'
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	   IF ( line_regular( line ) ) THEN
              WRITE ( risc_buff, 2075 ) 'even'
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
           ELSE
              WRITE ( risc_buff, 2070 ) 'odd'
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
           ENDIF
           WRITE ( risc_buff, 2080 )
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

* modulo ?
           WRITE ( risc_buff, 2030 ) 'modulo'
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	   IF ( line_modulo( line ) ) THEN
              WRITE ( risc_buff, 2075 ) 'yes' 
           ELSE
              WRITE ( risc_buff, 2070 ) 'no'
           ENDIF
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
           WRITE ( risc_buff, 2080 )
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

* All attributes not already listed.

1000    CONTINUE
        status = ferr_ok
	ENDIF
	RETURN
	END
