	CHARACTER*13 FUNCTION TM_STRING ( dval )
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
*  Will take a double precision real and output it as a string. It will be
*  written as an integer if there is no fractional part and the value is
*  under 1,000,000 , otherwise it will be written out in G format.
*
*  written 3/87 by M. Verschell for PMEL/TMAP
*
* revision 0.00 - 03/15/87
* revision 0.10 - 02/15/89 - added temporary string
* revision 0.20 - 07/14/89 - changed format from 1PG12.6 to 1PG12.5 to catch
*			   - all real numbers
*
* argument
	REAL*8	dval
* local variable
	CHARACTER*13	temp

	temp(1:13) = ' '
	IF (ABS(dval) .LT. 1D6) THEN
	  IF (dval .EQ. DBLE(INT(dval))) THEN
	    WRITE (temp(1:12),'(I12)') INT(dval)
	    GOTO 999
	  ENDIF
	ENDIF

	WRITE (temp(1:12),'(1PG12.5)') dval

999	TM_STRING = temp
	RETURN
	END
