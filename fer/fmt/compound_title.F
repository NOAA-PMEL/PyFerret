	CHARACTER*(*) FUNCTION COMPOUND_TITLE( cx_list, ncx, tlen )

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
* - possibly with more than one variable documented

* support routines:
*	VAR_TITLE	   - gets gets a simple title string for variable
*			     eg. "TEMPERATURE"
*	VAR_UNITS	   - gets a string with the units for this variable
*	VAR_TITLE_MOD	   - gets a modification to the title string, if any
*			     eg. "( averaged in X )"
*	FULL_VAR_TITLE	   - VAR_TITLE//VAR_UNITS//VAR_TITLE_MOD with parens
*			     eg. "TEMPERATURE ( averaged in X )"

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:   1/4/89 - from COMPOUND_PLOT_TITLE: major re-write without colors

* calling argument declarations:
	INTEGER	ncx, cx_list(ncx), tlen

* internal variable declarations:
	CHARACTER*200	FULL_VAR_TITLE
	INTEGER		slen, maxlen, i

#ifdef unix
	include 'ferret_cmn/ferret.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* maximum title length
	maxlen = LEN( COMPOUND_TITLE )

* initialize to the first variable context
	COMPOUND_TITLE = FULL_VAR_TITLE( cx_list(1), tlen )

* add any remaining variables separated by commas
	DO 100 i = 2, ncx
	   COMPOUND_TITLE = COMPOUND_TITLE(:tlen)//' , '//
     .			    FULL_VAR_TITLE( cx_list(i), slen )
	   tlen = MIN( maxlen, tlen+slen+3 )
 100	CONTINUE

* in case it was too long give an indication
	IF ( tlen .EQ. maxlen ) COMPOUND_TITLE(tlen:tlen) = '*'

	RETURN
	END
