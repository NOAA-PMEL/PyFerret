	LOGICAL FUNCTION TM_HAS_STRING( string, substring )

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
* check in a case-independent search to see if "substring" is in "string"
* note: string and substring can be at most 80 characters

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 02/11/88
* revision 0.01 - 05/03/88 - made temp strings CHAR*255 (this is the max length
*			     that will work correctly (MAV)
* revision 0.10 - 01/20/91 - replaced VAX runtim library reference with INDEX
*                            FORTRAN intrinsic function
* V530 *sh* 11/00 - more graceful behavior if 255 max is exceeded

* calling argument declarations
	CHARACTER*(*)	string, substring

* local variable declarations
	INTEGER	STR_UPCASE, TM_LENSTR1,
     .		vms_stat, sublen, strlen
	CHARACTER*512	up_string, up_sub

* length of substring
	sublen = TM_LENSTR1( substring )
	strlen = TM_LENSTR1( string )
	IF (sublen .GT. 512) THEN
	   TM_HAS_STRING = .FALSE.
	   RETURN
	ENDIF

* convert to upper case
	vms_stat = STR_UPCASE( up_string, string )
	vms_stat = STR_UPCASE( up_sub, substring(:sublen) )

* find the substring
	TM_HAS_STRING = INDEX( up_string, up_sub(:sublen) ) .GT. 0

	RETURN
	END
