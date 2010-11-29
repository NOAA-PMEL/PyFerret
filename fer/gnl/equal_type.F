	SUBROUTINE EQUAL_TYPE( string, max_fields, nfields,
     .			       field_type, wk_start, wk_end, status )

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
* from a string of the form "TYPE=type1, type2, ..." determine the
* number of fields and their data types

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V530: 10/00 - for delimited reads

* calling argument declarations
	INTEGER		status
	INTEGER		max_fields, nfields
	INTEGER		field_type(max_fields),
     .			wk_start(max_fields), wk_end(max_fields)
	CHARACTER*(*)	string

* internal parameter declarations
	INTEGER	maxtypes
	PARAMETER (maxtypes = 8)

* internal variable declarations
	LOGICAL	MATCH4
	INTEGER	TM_LENSTR1, TM_LENSTR, iif, iit, equal_pos, slen,
     .          len_test, len_mchars
	CHARACTER*9 ftypes(maxtypes)

	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xprog_state.cmn'
	include 'xrisc.cmn'

* order of entries must match delimitedRead.h
	DATA ftypes(1)/'-'/,
     .	     ftypes(2)/'NUMERIC'/,
     .       ftypes(3)/'CHARACTER'/,
     .       ftypes(4)/'LATITUDE'/,
     .       ftypes(5)/'LONGITUDE'/,
     .       ftypes(6)/'DATE'/,
     .       ftypes(7)/'EURODATE'/,
     .       ftypes(8)/'TIME'/


* initialize
	slen      = TM_LENSTR1( string )
	equal_pos = INDEX( string, '=' )

	IF ( equal_pos .EQ. 0 ) THEN
	   nfields = 1
	   field_type(1) = 2  ! numeric as default for content-free /TYPE

	ELSEIF ( equal_pos .EQ. slen ) THEN
	  GOTO 5000                            ! "name ="

	ELSE

* parse the comma separated list into fields
	  CALL PARSE_COMMA_LIST ( string,
     .                            equal_pos+1,
     .                            slen,
     .                            max_fields,
     .                            nfields,
     .                            wk_start,
     .                            wk_end,
     .                            status          )
	  IF ( status .NE. ferr_ok ) GOTO 5000

* identify each field
	  DO 200 iif = 1, nfields
            len_test = TM_LENSTR(string(wk_start(iif):wk_end(iif)))
	    DO 100 iit = 1, maxtypes
              len_mchars = TM_LENSTR(ftypes(iit)(:4))
	      IF (MATCH4(string(wk_start(iif):wk_end(iif)),len_test,
     .			 ftypes(iit),len_mchars)) THEN
	        field_type(iif) = iit 
	        GOTO 200
	      ENDIF
 100	    CONTINUE
	    GOTO 5100
 200	  CONTINUE
	ENDIF

* success
	status = ferr_ok
	RETURN

* syntax error
 5000	RETURN

 5100	CALL ERRMSG(ferr_invalid_command, status, 'unknown data type',
     .		    string(wk_start(iif):wk_end(iif)), *5110)
 5110	risc_buff = ftypes(2)
	iif = 7
	DO 5120 iit = 3, maxtypes
	  risc_buff(iif:) = ','//ftypes(iit)
	  iif = iif + 1 + TM_LENSTR1(ftypes(iit))
 5120	CONTINUE
	CALL WARN('Valid types are: '//risc_buff(:iif))
	GOTO 5000

	END
