	SUBROUTINE LINE_FACTS_XML( lun, idim, grid, orient )

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

	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include 'xtext_info.cmn'
	include 'xrisc.cmn'          ! 12/94 SPLIT_LIST buffer

* calling argument declarations
	INTEGER	  lun, idim, grid, orient

* local variable declarations:
	LOGICAL	BKWD_AXIS, GEOG_LABEL, calendar
	INTEGER TM_LENSTR, TM_LENSTR1, TM_GET_CALENDAR_ID, 
     .          line, slen, llen, ndec, cal_id, status
	CHARACTER TM_FMT*16, show_str*16, lo_str*20, hi_str*20, 
     .            title*32, buff*20, outbuff*20
	REAL*8	TM_WORLD, lo_val8, hi_val8

* local parameter declarations:
	INTEGER		sig_figs
	PARAMETER     ( sig_figs = 5 )

 2010	FORMAT ( '<axis name="', A8, 5x, A1, '">' )
 2020	FORMAT ( '<axis name="', A, '">' )
 2030	FORMAT ( '<units>', A)
 2035   FORMAT ( ' since ', A, '</units>' )
 2036   FORMAT ( '</units>' )
 2040	FORMAT ( '<length>', A, '</length>' )
 2050	FORMAT ( '<positive>down</positive>' )
 2060	FORMAT ( '<point_spacing>even</point_spacing>' )
 2070	FORMAT ( '<point_spacing>uneven</point_spacing>' )

 2080	FORMAT ( '<modulo>yes</modulo>' )
 2090	FORMAT ( '<modulo>no</modulo>' )
 2100	FORMAT ( '<start>', A, '</start>' )
 2110	FORMAT ( '<end>', A, '</end>' )

* pull line from grid table
	line = grid_line( idim, grid )

* line ok ?
	risc_buff = ' '
	IF ( line .EQ. munknown ) THEN
	   title = 'unknown'
	   WRITE ( risc_buff, 2010 ) title, ww_dim_name( idim )
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	ELSEIF ( line .EQ. mnormal ) THEN
	   title = 'normal'
	   WRITE ( risc_buff, 2010 ) title, ww_dim_name( idim )
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	ELSE
  
* header

           slen = TM_LENSTR1 ( line_name(line) )
	   WRITE ( risc_buff, 2020 ) line_name(line)(1:slen)
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

* axis units, including ... since start date if time axis


* report /T0= for formatted time axis

           slen = TM_LENSTR(line_units(line))
           IF (line_units(line)(1:2) .NE. char_init 
     .           .AND. slen.GT.0) THEN
              
              WRITE ( risc_buff, 2030 ) line_units(line)(1:slen)
              llen = TM_LENSTR1(risc_buff)

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
                 llen = TM_LENSTR1(risc_buff)
                 WRITE (risc_buff(llen+1:), 2035) outbuff(:slen)
              ELSE
                 WRITE (risc_buff(llen+1:), 2036)
              ENDIF

	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
           ENDIF

* line dimension

           show_str = TM_FMT(FLOAT(line_dim(line)), 16,16,slen)
           WRITE ( risc_buff, 2040 ) show_str(1:slen)
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
              WRITE ( risc_buff, 2100 ) outbuff(1:slen)
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

              CALL TM_DATE_REFORMAT (hi_str, 1, cal_id, outbuff, 2,
     .                               .TRUE., status) 
              slen = TM_LENSTR1(outbuff)
              WRITE ( risc_buff, 2110 ) outbuff(1:slen)
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
           ELSE
              
              slen = TM_LENSTR1(lo_str)
              WRITE ( risc_buff, 2100 ) lo_str(1:slen)
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

              slen = TM_LENSTR1(hi_str)
              WRITE ( risc_buff, 2110 ) hi_str(1:slen)
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
           ENDIF


* axis oriented backwards ?
	   IF ( BKWD_AXIS(idim, grid) ) THEN
              WRITE ( risc_buff, 2050 ) 
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	   ENDIF

* regularly spaced ?
	   IF ( line_regular( line ) ) THEN
              WRITE ( risc_buff, 2060 ) 
           ELSE
              WRITE ( risc_buff, 2070 ) 
           ENDIF
           CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

* modulo ?
	   IF ( line_modulo( line ) ) THEN
              WRITE ( risc_buff, 2080 ) 
           ELSE
              WRITE ( risc_buff, 2090 ) 
           ENDIF
           CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

	ENDIF
	RETURN
	END
