	INTEGER FUNCTION TM_LOC_STRING( string, substring, ioc )

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
* Case-independent search to find the "ioc"th occurrence of * "substring" 
* in "string".  If there are fewer than "ioc" occurrences, return 0.

* programmer - Ansley Manke, based on tm_has_string.F
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* revision 0.00 - 03/17/03

* calling argument declarations
        INTEGER ioc
	CHARACTER*(*)	string, substring

* local variable declarations
	INTEGER	STR_UPCASE, TM_LENSTR1,
     .		vms_stat, sublen, strlen, i, loc
	CHARACTER*512	up_string, up_sub, newstring

* length of substring
	sublen = TM_LENSTR1( substring )
	strlen = TM_LENSTR1( string )

        IF (ioc .LT. 1) ioc = 1

        IF (ioc .GT. strlen) THEN
	   TM_LOC_STRING = 0
	   RETURN
	ENDIF

	IF (sublen .GT. 512) THEN
	   TM_LOC_STRING = 0
	   RETURN
	ENDIF

* convert to upper case
	vms_stat = STR_UPCASE( up_string, string )
	vms_stat = STR_UPCASE( up_sub, substring(:sublen) )

* find the substring
	TM_LOC_STRING = INDEX( up_string, up_sub(:sublen) ) 
        IF (TM_LOC_STRING .GE. 1 .AND. ioc .GT. 1) THEN
           DO 100 i = 2, ioc
              loc = TM_LOC_STRING
              IF (loc+1 .LT. strlen) THEN
                 newstring = up_string(loc+1:strlen)
                 TM_LOC_STRING = INDEX( newstring, 
     .                               up_sub(:sublen) )
                 IF (TM_LOC_STRING .NE. 0) TM_LOC_STRING = 
     .                    TM_LOC_STRING + loc
              ENDIF
  100      CONTINUE
        ENDIF

	RETURN
	END
