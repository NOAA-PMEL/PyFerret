	SUBROUTINE SHOW_LINE_XML( lun, line)

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

* programmer - Ansley Manke, based on show_line.F
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* 6-Aug-2003
* V581 *acm* 6/05 - For fix to bug 1271, add flag line_shift_origin
*            and restore original t0 on output
* V600 *acm* 2/06 - Fix for bug 1394, send do_err flag to TM_DATE_REFORMAT
* V602 *acm*11/06 - Fix bug 1468, wrong string was written instead of the
*                   calendar name.
* V6.1  2/2008 *acm* Use attribute structure to get all attributes for the axis.
* V612  8/2008 *acm* Revert to old xml style output for infile_datatype and 
*                    ferret_datatype attributes
* V616  1/2009 *acm* Fixing LAS ticket 481, new style of xml output
* V631 11/09 *acm* For XML-formatted output check for and replace the strings < > and &
*                  with &lt; etc.

	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xunits.cmn_text'
	external xunits_data
	include	'ferret.parm'
	include 'xprog_state.cmn'
	include 'xtext_info.cmn'
	include 'calendar.decl'
	include 'calendar.cmn'
	include 'xrisc.cmn'
        include 'errmsg.parm'
        include 'netcdf.inc'
	include 'xdset_info.cmn_text'

* calling argument declarations
	INTEGER 	lun, line

* local variable declarations:
	LOGICAL		TM_ITS_SUBSPAN_MODULO, NC_GET_ATTRIB, got_it
	INTEGER		idim, orient, slen, iset, dset
	INTEGER		TM_LENSTR1, TM_GET_CALENDAR_ID, cal_id, status
	CHARACTER	TM_FMT*32, buff*80, outbuff*20

        LOGICAL         coordvar, new
        INTEGER         STR_SAME, 
     .                  ivar,  llen,
     .                  varid, vtype, nvdims, vdims(8), nvatts, blen, i, 
     .                  len_attbuff, attoutflag, all_outflag, attid, 
     .                  attlen, attype, iatt
        CHARACTER       vname*128, attname*128, attbuff*2048, 
     .                  outstring*2048
	REAL            vals(10)

C New form for attribute output, dont implement it yet. (Check the
C Write statement that uses number 2075 when we change to using these.)

 2020	FORMAT ( '<attribute name="time_origin" value="', A, '" />' )
 2030	FORMAT ( '<attribute name="calendar" value="', A, '" />' )
 2040	FORMAT ( '<attribute name="modulo_length" value="', A, '" />' )
 2050	FORMAT ( '</axis>' )
 2075   FORMAT( '<attribute name="', A, '" value="', A, '" />' )

* determine orientation

	DO 100 idim = 1, 5
	   IF ( line_direction(line) .EQ. axis_orients( idim ) ) THEN
	      orient = idim
	      GOTO 110
	   ENDIF
 100	CONTINUE
        IF (line_direction(line) .EQ. 'XX') THEN
           orient = 1
           idim = 1
           GOTO 110
        ENDIF
        IF (line_direction(line) .EQ. 'YY') THEN
           orient = 2
           idim = 2
           GOTO 110
        ENDIF
        IF (line_direction(line) .EQ. 'TT') THEN
           orient = 4
           idim = 4
           GOTO 110
        ENDIF

	orient = 0
	idim   = 1			! use X slot for non-oriented
 110	IF ( idim .EQ. 5 ) idim = 3	! DU --> UD on Z axis

* create a dummy grid with this axis
* (kludge to use existing code that works with grids)
	grid_line(idim, mgrid_buff) = line

* report basic facts using code from SHOW_GRID
	CALL LINE_FACTS_XML( ttout_lun, idim, mgrid_buff, orient, status )
        IF (status .NE. ferr_ok) GOTO 1000

* report /T0= for formatted time axis
	IF ( line_direction(line) .EQ. 'TI' ) THEN
        
	   buff = line_cal_name (line)
	   cal_id = TM_GET_CALENDAR_ID (buff)

           IF (line_shift_origin(line)) THEN
              buff = '01-JAN-0001 00:00:00'
           ELSE
              buff = line_t0(line)
           ENDIF

           CALL TM_DATE_REFORMAT (buff, 1, cal_id, outbuff, 2, 
     .        .TRUE., status) 
           CALL ESCAPE_FOR_XML(outbuff, outstring, slen)
           WRITE ( risc_buff, 2020 ) outstring(1:slen)
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

* report the calendar of a time axis if "interesting"

	   IF (cal_id .GT. gregorian .AND. cal_id .LE. max_calendars) THEN
              buff = line_cal_name(line)
              CALL ESCAPE_FOR_XML(buff, outstring, slen)
              WRITE ( risc_buff, 2030 ) outstring(1:slen)
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	   ENDIF
	ENDIF

* if line is subspan modulo, report its modulo cycle length
	IF ( line_modulo(line) .AND. TM_ITS_SUBSPAN_MODULO(line) ) THEN
	   buff = TM_FMT( SNGL(line_modulo_len(line)),7,32,slen)
           CALL ESCAPE_FOR_XML(buff, outstring, slen)
	   WRITE (risc_buff, 2040) outstring(1:slen)
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	ENDIF

        
* Get the dataset (show line/xml only used for axes in datasets)

        vname = line_name(line)
	DO 300 iset = 1, maxdsets
	   IF ( ds_name(iset) .EQ. char_init1024) GOTO 1000
           CALL CD_GET_VAR_ID (iset, vname, varid, status)
           IF (status .EQ. ferr_ok) GOTO 50
  300   CONTINUE
   50   dset = iset

* other attributes

        CALL CD_GET_VAR_INFO (dset, varid, vname, vtype, nvdims,
     .           vdims, nvatts, coordvar, all_outflag, status)

* Data type of the axis in the file...

        IF (vtype .EQ. ncbyte ) THEN
           risc_buff = 
     .         '<attribute name="infile_datatype" value="BYTE" />'
        ELSE IF (vtype .EQ. ncchar ) THEN
           risc_buff = 
     .         '<attribute name="infile_datatype" value="CHAR" />'
        ELSE IF (vtype .EQ. ncshort ) THEN
           risc_buff = 
     .         '<attribute name="infile_datatype" value="SHORT" />'
        ELSE IF (vtype .EQ. nclong ) THEN
           risc_buff = 
     .         '<attribute name="infile_datatype" value="LONG" />'
        ELSE IF (vtype .EQ. ncfloat ) THEN
           risc_buff = 
     .         '<attribute name="infile_datatype" value="FLOAT" />'
        ELSE IF (vtype .EQ. ncdouble ) THEN
           risc_buff = 
     .         '<attribute name="infile_datatype" value="DOUBLE" />'
        ENDIF
        
        CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

* All attributes not already listed.

        DO 200 iatt = 1, nvatts
           CALL CD_GET_VAR_ATT_INFO (dset, varid, iatt, attname, 
     .              attype, attlen, attoutflag, status )

           new = .TRUE.
           new = (STR_SAME(attname, 'direction') .NE. 0) .AND. 
     .              (STR_SAME(attname, 'units') .NE. 0) .AND. 
     .              (STR_SAME(attname, 'length') .NE. 0) .AND. 
     .              (STR_SAME(attname, 'positive') .NE. 0) .AND. 
     .              (STR_SAME(attname, 'modulo') .NE. 0) .AND. 
     .              (STR_SAME(attname, 'modulo_length') .NE. 0) .AND. 
     .              (STR_SAME(attname, 'start') .NE. 0) .AND. 
     .              (STR_SAME(attname, 'end') .NE. 0) .AND. 
     .              (STR_SAME(attname, 'time_origin') .NE. 0) .AND. 
     .              (STR_SAME(attname, 'calendar') .NE. 0) .AND. 
     .              (STR_SAME(attname, 'point_spacing') .NE. 0)

           IF (new) THEN
              blen = 2048  ! len of attbuff
              got_it = NC_GET_ATTRIB (dset, varid, attname, 
     .                                .TRUE., vname, blen, attlen, 
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

! Store the converted attname string in the variable buff, not now needed.

              CALL ESCAPE_FOR_XML(attname, buff, slen)
              CALL ESCAPE_FOR_XML (attbuff, outstring, len)
	      WRITE (risc_buff, 2075) buff(1:slen), 
     .                  outstring(1:blen)
              CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

           ENDIF

 200    CONTINUE
* closer tag

 1000   CONTINUE
        WRITE ( risc_buff, 2050 ) 
	CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

	RETURN
	END



