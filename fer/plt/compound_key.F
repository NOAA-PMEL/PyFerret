	CHARACTER*(*) FUNCTION COMPOUND_KEY( cx_list, ncx, flags, tlen )

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
* create a full title for a plot 
* - possibly with more than one variable documented

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:   1/4/89

* calling argument declarations:
	INTEGER	ncx, cx_list(ncx), flags(0:10), tlen

* internal variable declarations:
	CHARACTER*200	KEY_STRING
	INTEGER		slen, maxlen, i

#ifdef unix
	include 'ferret_cmn/ferret.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* maximum title length
	maxlen = LEN( COMPOUND_KEY )

* initialize to the first variable context
	COMPOUND_KEY = KEY_STRING( cx_list(1), flags, tlen )

* add any remaining variables separated by commas
	DO 100 i = 2, ncx
	   COMPOUND_KEY = COMPOUND_KEY(:tlen)//' , '//
     .			  KEY_STRING( cx_list(i), flags, slen )
	   tlen = MIN( maxlen, tlen+slen+3 )
 100	CONTINUE

* in case it was too long give an indication
	IF ( tlen .EQ. maxlen ) COMPOUND_KEY(tlen:tlen) = '*'

	RETURN
	END
