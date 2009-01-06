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

	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include 'xtext_info.cmn'
	include 'xrisc.cmn'          ! 12/94 SPLIT_LIST buffer
        include 'errmsg.parm'
	include 'xdset_info.cmn_text'

* calling argument declarations
	INTEGER	  lun, idim, grid, orient

* local variable declarations:
	LOGICAL	BKWD_AXIS, GEOG_LABEL, NC_GET_ATTRIB,
     .          calendar, new, got_it
	INTEGER TM_LENSTR, TM_LENSTR1, TM_GET_CALENDAR_ID, 
     .          STR_SAME, line, slen, llen, ndec, cal_id, status,
     .            varid, vtype, nvdims, vdims(8), nvatts, blen, i, 
     .            len_attbuff, attoutflag, all_outflag, attid, 
     .            attlen, attype, iatt, ncchar, iset, dset
	CHARACTER TM_FMT*16, show_str*16, lo_str*20, hi_str*20, 
     .            title*32, outbuff*20, 
     .            attname*128, attbuff*2048, buff*14
        CHARACTER*1 idir(4)
	REAL*8	TM_WORLD, lo_val8, hi_val8
	REAL      vals(10)

* local parameter declarations:
	INTEGER		sig_figs
	PARAMETER     ( sig_figs = 5 )
        DATA idir/'I', 'J', 'K', 'L'/

C New form for attribute output, dont implement it yet. (Check the
C Write statement that uses number 2075 when we change to using these.)

 2010	FORMAT ( '<axis name="', A8, 5x, A1, '">' )
 2020	FORMAT ( '<axis name="', A, '">' )
 2030	FORMAT ( '<attribute name="direction" value="', A, '" />' )
 2040	FORMAT ( '<attribute name="units" value="', A, '" />')
 2045   FORMAT ( ' since ', A, ' "/>' )
 2050	FORMAT ( '<attribute name="length" value="', A, '" />' )
 2060	FORMAT ( '<attribute name="positive" value="down" />' )
 2070	FORMAT ( '<attribute name="point_spacing" value="even" />' )
 2080	FORMAT ( '<attribute name="point_spacing" value="uneven" />' )

 2090	FORMAT ( '<attribute name="modulo" value="yes" />' )
 2100	FORMAT ( '<attribute name="modulo" value="no" />' )
 2110	FORMAT ( '<attribute name="start" value="', A, '" />' )
 2120	FORMAT ( '<attribute name="end" value="', A, '" />' )
 2075   FORMAT( '<attribute name="', A, '" value="', A, '" />' )

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
  
* header

           slen = TM_LENSTR1 ( line_name(line) )
	   WRITE ( risc_buff, 2020 ) line_name(line)(1:slen)
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

* axis units, including ... since start date if time axis

* direction I,J,K, or L

           WRITE (risc_buff, 2030) idir(idim)
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

* report /T0= for formatted time axis

           slen = TM_LENSTR(line_units(line))
           IF (line_units(line)(1:2) .NE. char_init 
     .           .AND. slen.GT.0) THEN
              
              WRITE ( risc_buff, 2040 ) line_units(line)(1:slen)
              llen = slen
	      blen = llen + 8
              blen = TM_LENSTR1(risc_buff)

              IF ( line_direction(line) .EQ. 'TI' ) THEN
              
	         buff = line_cal_name (line)
	         cal_id = TM_GET_CALENDAR_ID (buff)

                 IF (line_shift_origin(line)) THEN
                    buff = '01-JAN-0001 00:00:00'
                 ELSE
 	            buff = line_t0(line)
                 ENDIF
                 
                 CALL TM_DATE_REFORMAT (buff, 1, cal_id, outbuff, 2,
     .                                  .TRUE., status) 
                 slen = TM_LENSTR1(outbuff)
		 blen = blen - 3
                 WRITE (risc_buff(blen:blen+slen+6+8), 2045) 
     .                outbuff(:slen)

              ENDIF

	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
           ENDIF

* line dimension

           show_str = TM_FMT(FLOAT(line_dim(line)), 16,16,slen)
           WRITE ( risc_buff, 2050 ) show_str(1:slen)
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

* start and end

	   calendar = ( idim .EQ. t_dim .AND. GEOG_LABEL(idim, grid) )

           lo_val8 = TM_WORLD( 1, grid, idim, box_middle )
	   IF ( calendar ) THEN
	      ndec = 5 		! accurate to minutes
	   ELSE
	      ndec = -sig_figs	! 5 sig places w/out extra zeros
	   ENDIF
	   CALL TRANSLATE_TO_WORLD( lo_val8, idim, grid, ndec, lo_str )

	   hi_val8 = TM_WORLD( line_dim(line), grid, idim, box_middle )
	   IF ( calendar ) THEN
	      ndec = 5 
	   ELSE
	      ndec = -sig_figs
	   ENDIF
	   CALL TRANSLATE_TO_WORLD( hi_val8, idim, grid, ndec, hi_str )

* Write the start and end; if dates then format them as for time origin.

           IF (calendar) THEN
              CALL TM_DATE_REFORMAT (lo_str, 1, cal_id, outbuff, 2,
     .                               .TRUE., status) 
              slen = TM_LENSTR1(outbuff)
              WRITE ( risc_buff, 2110 ) outbuff(1:slen)
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

              CALL TM_DATE_REFORMAT (hi_str, 1, cal_id, outbuff, 2,
     .                               .TRUE., status) 
              slen = TM_LENSTR1(outbuff)
              WRITE ( risc_buff, 2120 ) outbuff(1:slen)
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
           ELSE
              
              slen = TM_LENSTR1(lo_str)
              WRITE ( risc_buff, 2110 ) lo_str(1:slen)
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

              slen = TM_LENSTR1(hi_str)
              WRITE ( risc_buff, 2120 ) hi_str(1:slen)
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
           ENDIF

* axis oriented backwards ?
	   IF ( BKWD_AXIS(idim, grid) ) THEN
              WRITE ( risc_buff, 2060 ) 
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	   ENDIF

* regularly spaced ?
	   IF ( line_regular( line ) ) THEN
              WRITE ( risc_buff, 2070 ) 
           ELSE
              WRITE ( risc_buff, 2080 ) 
           ENDIF
           CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

* modulo ?
	   IF ( line_modulo( line ) ) THEN
              WRITE ( risc_buff, 2090 ) 
           ELSE
              WRITE ( risc_buff, 2100 ) 
           ENDIF
           CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

* All attributes not already listed.
 
           DO 100 iset = 1, maxdsets
	     IF ( ds_name(iset) .EQ. char_init1024) GOTO 1000
             CALL CD_GET_VAR_ID (iset, line_name(line), varid, status)
             IF (status .EQ. ferr_ok) GOTO 50
  100      CONTINUE
   50      dset = iset

           DO 200 iatt = 1, nvatts
              CALL CD_GET_VAR_ATT_INFO (dset, varid, iatt, attname, 
     .              attype, attlen, attoutflag, status )

              new = .TRUE.
              new = (STR_SAME(attname, 'direction') .NE. 0)     .AND. 
     .              (STR_SAME(attname, 'units') .NE. 0)         .AND. 
     .              (STR_SAME(attname, 'length') .NE. 0)        .AND. 
     .              (STR_SAME(attname, 'positive') .NE. 0)      .AND. 
     .              (STR_SAME(attname, 'point_spacing') .NE. 0) .AND. 
     .              (STR_SAME(attname, 'modulo') .NE. 0)        .AND. 
     .              (STR_SAME(attname, 'start') .NE. 0)         .AND. 
     .              (STR_SAME(attname, 'end') .NE. 0)

              IF (new) THEN
                 blen = 2048  ! len of attbuff
                 got_it = NC_GET_ATTRIB (dset, varid, attname, 
     .                        .TRUE., line_name(line), blen, attlen, 
     .                        attoutflag, attbuff, vals)

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
     .                  attbuff(1:blen), attname(1:slen)
                 CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

              ENDIF

 200       CONTINUE

1000    CONTINUE
        status = ferr_ok
	ENDIF
	RETURN
	END
