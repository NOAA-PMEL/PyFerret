	INTEGER FUNCTION TM_UNIT_ID( string )

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
* identify the string as a known unit or return 0

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 04/15/88
* revision 1.00 - 05/26/88 - to work with TMAP routines
* revision 1.01 - 11/21/88 - variable used check changed to specific char_initXX
* 7/96 *sh*: check that "deg" is not hiding "degrees C" or "deg K" ...
* V5.1 *sh* 12/99 - changed logic for matching un_name to insist on exact
*	match with possible final "s". It was matching "Kelvin" to "in"

* Include files
#ifdef unix
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
	include 'xunits.cmn_text'
	external xunits_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:XUNITS.CMN'
#endif

* calling argument declarations:
	CHARACTER*(*)	string

* internal parameter declarations:
	INTEGER		ndeg_temp
	PARAMETER     ( ndeg_temp = 6 )

* internal variable declarations:
	INTEGER		iun, ideg
	INTEGER         STR_CASE_BLIND_COMPARE, TM_LENSTR1, istat, slen
	LOGICAL		TM_HAS_STRING, match
	CHARACTER	deg_temp(ndeg_temp)*1

* initialize characters that signal "degrees" is really temperature
	DATA deg_temp / 'C', 'K', 'F', 'c', 'k', 'f' /	

* check through alternate (long) names first to avoid ambiguity
* eg. "m" is the un_name for "meters" but also exits in many other unit names
	DO 110 iun = 1, pun_num_alt

	   IF ( un_alt_name(iun) .EQ. char_init12 ) GOTO 200

	   IF ( TM_HAS_STRING( string, un_alt_name(iun) ) ) THEN
* ... check that "deg" is not hiding "degrees C" or "deg K" ...
	      IF ( iun .EQ. pun_degrees ) THEN
	         DO 105 ideg = 1, ndeg_temp
	            IF (TM_HAS_STRING(string,deg_temp(ideg))) GOTO 110
 105	         CONTINUE
	      ENDIF
	      TM_UNIT_ID = un_alt_subsc( iun )
	      RETURN
	   ENDIF

 110	CONTINUE

* now it's safe to search the list of short abbreviations
 200	slen = TM_LENSTR1(string)
	DO 210 iun = pun_1st_unit, pun_last_unit

	   IF ( un_name(iun) .EQ. char_init12 .OR.
     .	        un_name(iun) .EQ. ' '             ) GOTO 210

* ... check for string match or possibly plural
	   istat = STR_CASE_BLIND_COMPARE(string, un_name(iun))
	   match = istat .EQ. str_match
	   IF (.NOT.match .AND. slen.GT.1) THEN
	      istat = STR_CASE_BLIND_COMPARE(string(:slen-1), un_name(iun))
	      match = istat .EQ. str_match
	      IF ( match ) match = string(slen:slen).EQ.'s'
     .			      .OR. string(slen:slen).EQ.'S'
	   ENDIF

	   IF ( match ) THEN
* ... check that "deg" is not hiding "degrees C" or "deg K" ...
	      IF ( iun .EQ. pun_degrees ) THEN
	         DO 205 ideg = 1, ndeg_temp
	            IF (TM_HAS_STRING(string,deg_temp(ideg))) GOTO 210
 205	         CONTINUE
	      ENDIF
	      TM_UNIT_ID = iun
	      RETURN
	   ENDIF

 210	CONTINUE

* not found
	TM_UNIT_ID = 0
	RETURN

	END
