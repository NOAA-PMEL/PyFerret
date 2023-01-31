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
* V650  Write to show_lun which may be set to a file unit number rather than std out
* V67  *acm* 2/11 XML fixes. New attribute formatting, ![CDATA] escape, upper/lowercase handling.
*      *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
*      *acm* 3/13 Dont show axes with names in parentheses e.g. (AX004)
* V71 12/16 *acm* For ticket 2158: working with true monthly time axes.

        IMPLICIT NONE
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

	LOGICAL		TM_ITS_SUBSPAN_MODULO, NC_GET_ATTRIB, GEOG_LABEL, 
     .                  ITSA_TRUEMONTH_AXIS,
     .			got_it, calendar, bounds_done, do_bounds
	INTEGER		idim, orient, slen, iset, dset, alen, igrid
	INTEGER		TM_LENSTR1, TM_GET_CALENDAR_ID, cal_id, status
	CHARACTER	TM_FMT*48, buff*80, outbuff*20

        LOGICAL         new
        INTEGER         STR_SAME, 
     .                  ivar, llen, varid, vtype, nvatts, blen, i, len_attbuff, 
     .                  attoutflag, attid,  attlen, attype, iatt
        CHARACTER       vname*128, attname*128, attbuff*2048, 
     .                  outstring*2048, atype*10
	REAL            vals(100)

C New form for attribute output, list values separately.


 2020	FORMAT ('<attribute name="', A, '" type="char">' )
 2030	FORMAT ('<attribute name="', A, '" type="short">' )
 2040   FORMAT ('<attribute name="', A, '" type="', A,'">' )
 2050   FORMAT ('   <value>', A, '</value>')
 2055   FORMAT ('   <value><![CDATA[', A, ']]></value>')
 2060   FORMAT ('</attribute>')
 2080	FORMAT ( '</axis>' )

* determine orientation

	DO 100 idim = 1, nferdims+1
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
        IF (line_direction(line) .EQ. 'EE') THEN  !! ??
           orient = 5
           idim = 5
           GOTO 110
        ENDIF
        IF (line_direction(line) .EQ. 'FF') THEN
           orient = 6
           idim = 6
           GOTO 110
        ENDIF

	orient = 0
	idim   = 1			! use X slot for non-oriented
 110	IF ( idim .EQ. 7 ) idim = 3	! DU --> UD on Z axis

        ! Oriengation still not defined. Look for the axis in the grids
        IF (orient .EQ. 0) THEN
	   DO 130 igrid = 1, max_dyn_grids
	   DO 120 idim = 1, nferdims
	      IF ( line .EQ. grid_line(idim,igrid) ) THEN
	         orient = idim
	         GOTO 140
	      ENDIF
 120	   CONTINUE
 130	   CONTINUE
        ENDIF

 140	CONTINUE
        IF (orient .EQ. 0) RETURN  ! axis has no direction and is not used.

* create a dummy grid with this axis
* (kludge to use existing code that works with grids)
	grid_line(idim, mgrid_buff) = line

* report basic facts using code from SHOW_GRID
	CALL LINE_FACTS_XML( lun, idim, mgrid_buff, orient, status )
        IF (status .NE. ferr_ok) GOTO 1000

* report /T0= for formatted time axis
	IF ( line_direction(line) .EQ. 'TI' ) THEN
	   calendar = GEOG_LABEL(idim, igrid)
           IF (calendar) THEN
	      buff = line_cal_name (line)
	      cal_id = TM_GET_CALENDAR_ID (buff)

              IF (line_shift_origin(line)) THEN
                 buff = '01-JAN-0001 00:00:00'
              ELSE
                 buff = line_t0(line)
              ENDIF

* Make sure it is formatted as  2000-11-01 00:00:00
              CALL TM_DATE_REFORMAT (buff, 1, cal_id, outbuff, 2, 
     .        .TRUE., status) 
              CALL ESCAPE_FOR_XML(outbuff, outstring, slen)
              WRITE ( risc_buff, 2020 ) 'time_origin'
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	      WRITE (risc_buff, 2055) outstring(1:slen)
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
              WRITE ( risc_buff, 2060 ) 
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

* report the calendar of a time axis

              buff = line_cal_name(line)
              CALL ESCAPE_FOR_XML(buff, outstring, slen)
              WRITE ( risc_buff, 2020 ) 'calendar'
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
              WRITE ( risc_buff, 2055 ) outstring(1:slen)
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
              WRITE ( risc_buff, 2060 )
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	   ENDIF
	ENDIF

* if line is subspan modulo, report its modulo cycle length
	IF ( line_modulo(line) .AND. TM_ITS_SUBSPAN_MODULO(line) ) THEN
	   buff = TM_FMT( (line_modulo_len(line)),7,22,slen)
           CALL ESCAPE_FOR_XML(buff, outstring, slen)
	   WRITE (risc_buff, 2030) 'modulo_length'
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	   WRITE ( risc_buff, 2050 ) outstring(1:slen)
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	   WRITE ( risc_buff, 2060 )
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	ENDIF

* Get the dataset (show line/xml only used for axes in datasets)

        vname = line_name_orig(line)
	DO 300 iset = 1, maxdsets
	   IF ( ds_name(iset) .EQ. char_init2048) GOTO 400
           CALL CD_GET_VAR_ID (iset, vname, varid, status)
           IF (status .EQ. ferr_ok) GOTO 50
  300   CONTINUE

* Try user-defined axes, may want to get some attributes (F-TDS variables)
  400   CONTINUE
        iset = -2
        CALL CD_GET_VAR_ID (iset, vname, varid, status)
	IF (status .NE. ferr_ok) GOTO 1000

   50   dset = iset

* other attributes
        CALL CD_GET_VAR_TYPE (dset, varid, vname, vtype, status)
        CALL CD_GET_VAR_NATTS (dset, varid, vname, nvatts, status)

* Data type of the axis in the file...

        WRITE (risc_buff, 2020) 'infile_datatype'
        CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
        IF (vtype .EQ. ncbyte ) THEN
	   buff = 'BYTE'
        ELSE IF (vtype .EQ. ncchar ) THEN
           buff = 'CHAR'
        ELSE IF (vtype .EQ. ncshort ) THEN
           buff = 'SHORT'
        ELSE IF (vtype .EQ. nclong ) THEN
           buff = 'LONG'
        ELSE IF (vtype .EQ. ncfloat ) THEN
           buff = 'FLOAT'
        ELSE IF (vtype .EQ. ncdouble ) THEN
           buff = 'DOUBLE'
        ENDIF
        CALL ESCAPE_FOR_XML(buff, outstring, slen)
        WRITE ( risc_buff, 2055 ) outstring(1:slen)
        CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
        WRITE ( risc_buff, 2060 )
        CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

* All attributes not already listed.

	bounds_done = .FALSE.

        DO 200 iatt = 1, nvatts
           CALL CD_GET_VAR_ATT_INFO (dset, varid, iatt, attname, 
     .              attype, attlen, attoutflag, status )

           new = .TRUE.
           new = (STR_SAME(attname,    'direction')     .NE. 0) .AND. 
     .              (STR_SAME(attname, 'units')         .NE. 0) .AND. 
     .              (STR_SAME(attname, 'length')        .NE. 0) .AND. 
     .              (STR_SAME(attname, 'positive')      .NE. 0) .AND. 
     .              (STR_SAME(attname, 'modulo')        .NE. 0) .AND. 
     .              (STR_SAME(attname, 'modulo_length') .NE. 0) .AND. 
     .              (STR_SAME(attname, 'start')         .NE. 0) .AND. 
     .              (STR_SAME(attname, 'end')           .NE. 0) .AND. 
     .              (STR_SAME(attname, 'time_origin')   .NE. 0) .AND. 
     .              (STR_SAME(attname, 'calendar')      .NE. 0) .AND.  
     .              (STR_SAME(attname, 'point_spacing') .NE. 0) .AND. 
     .              (STR_SAME(attname, 'infile_datatype') .NE. 0)

	if (.NOT.new) new = (STR_SAME(attname, 'long_name') .EQ. 0)

           IF (new) THEN
              blen = 2048  ! len of attbuff
              got_it = NC_GET_ATTRIB (dset, varid, attname, 
     .                                .TRUE., vname, blen, attlen, 
     .                                attoutflag, attbuff, vals)
              IF (attlen .GT. 0) THEN

	      IF (STR_SAME(attname, 'edges')  .EQ. 0) THEN
	         attname = 'bounds'
	         slen = TM_LENSTR1(vname)
	         attbuff = vname(1:slen)//'_bnds'
              ENDIF

              CALL ESCAPE_FOR_XML(attname, outstring, slen)
	      IF (STR_SAME(attname, 'bounds') .EQ. 0) bounds_done = .TRUE.

* Original type of the attribute
              IF (attype .EQ. NCCHAR)   atype = 'char'
              IF (attype .EQ. NCBYTE)   atype = 'byte'
              IF (attype .EQ. NCSHORT)  atype = 'int'
              IF (attype .EQ. NCLONG)   atype = 'int'
              IF (attype .EQ. NCFLOAT)  atype = 'float'
              IF (attype .EQ. NCDOUBLE) atype = 'double'
              alen = TM_LENSTR1(atype)

	      WRITE (risc_buff, 2040) attname(1:slen), atype(1:alen)
              CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

              IF (attype .NE. NCCHAR) THEN
                 slen = 1
                 attbuff = ' '
                 DO 150 i = 1, attlen
                    buff = TM_FMT(vals(i), 7, 14, llen)
                    CALL ESCAPE_FOR_XML (buff, outstring, llen)
	            WRITE ( risc_buff, 2050 ) outstring(1:llen)  ! was slen
	            CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
 150             CONTINUE
              ELSE 
                 CALL ESCAPE_FOR_XML (attbuff, outstring, llen)
                 WRITE ( risc_buff, 2055 ) outstring(1:llen)
                 CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
              ENDIF

	      WRITE ( risc_buff, 2060 )
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

           ENDIF  ! attlen GT 0
           ENDIF

 200    CONTINUE

* Write a bounds attribute for unevenly-spaced axes even if the intput file
* didnt have a bounds or edges attribute.

	do_bounds = ITSA_TRUEMONTH_AXIS(line) .OR. .NOT.line_regular( line )
	IF ( (.NOT.bounds_done) .AND. do_bounds ) THEN
	   attname = 'bounds'
	   atype = 'char'
	   slen = TM_LENSTR1(attname)
	   alen = TM_LENSTR1(atype)

	   WRITE (risc_buff, 2040) attname(1:slen), atype(1:alen)
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

	   slen = TM_LENSTR1(vname)
	   attbuff = vname(1:slen)//'_bnds'

	   CALL ESCAPE_FOR_XML (attbuff, outstring, llen)
	   WRITE ( risc_buff, 2050 ) outstring(1:llen)
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)


	   WRITE ( risc_buff, 2060 )
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	ENDIF

* closer tag

 1000   CONTINUE
	IF (status .LT. 0) THEN
	   status = ferr_ok
	ELSE
           WRITE ( risc_buff, 2080 ) 
   	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	ENDIF

	RETURN
	END



