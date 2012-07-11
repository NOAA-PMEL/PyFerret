	LOGICAL FUNCTION TM_LEGAL_NAME_OP( name )
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
* determine whether a particular string is an operator or IF keyword.


* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Thermal Modeling and Analysis Program
* 5/2012

        include 'ferret.parm'
        include 'errmsg.parm'
        include 'xalgebra.cmn'

* calling argument declarations:
	CHARACTER*(*) name
*
* internal variable declarations:
	INTEGER STR_MATCH
	INTEGER	TM_LENSTR1, STR_SAME, slen
	
* string length
	slen = TM_LENSTR1( name )

* Is it an operator or IF or ELSE ? 
        imatch = STR_MATCH ( name(:slen),
     .                       alg_op, num_operators )
        IF ( imatch .NE. atom_not_found ) GOTO 5000

	IF ( STR_SAME (name, 'if') .EQ. 0) GOTO 5000	
	IF ( STR_SAME (name, 'else') .EQ. 0) GOTO 5000

* it's ok
	TM_LEGAL_NAME_OP = .TRUE.
	RETURN
*
* it's bad
 5000	TM_LEGAL_NAME_OP = .FALSE.
	RETURN

	END
