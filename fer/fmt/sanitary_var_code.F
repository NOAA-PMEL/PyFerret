	CHARACTER*(*) FUNCTION SANITARY_VAR_CODE ( category, variable )

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
* return name code for variable that is "sanitary" in the sense that it can
* be used as a part of a filename

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  12/4/89 - created to make more honest VAR_CODE for other purposes
* V230:  11/4/92 - expanded user variable name length
* V500 *kob* 3/99- up VAR_CODE and vcode decl's from 8 to 64 chars
* V552 *acm* 4/03- up VAR_CODE decl from 64 to 128 chars


	include	'ferret.parm'
	include	'xvariables.cmn'

* calling argument declarations:
	INTEGER category, variable

* internal variable declarations:
	INTEGER		HASH_NAME
	CHARACTER	VAR_CODE*128, vcode*128

* get basic variable code
	vcode = VAR_CODE( category, variable )

* sanitize it
	IF ( INDEX(vcode,'#') .GT. 0 ) WRITE( vcode, 3000 ) 'E', 
     .				HASH_NAME( uvar_text( variable ), 999 )
 3000	FORMAT(A1,I3.3,4X)

* done
	SANITARY_VAR_CODE = vcode

	RETURN
	END
