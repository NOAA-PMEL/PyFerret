	INTEGER FUNCTION HASH_CX( cx )

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
* generate a pseudo-random pile number between 1 and pmv_npiles based on
* information in the passed context

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/21/88
* V200:  7/12/89 - special treatment for cat_temp_var
* V510: 8/99 *sh* special treatment for cat_work_mr
* v580: 9/04 *acm* Fix Bug 993: constant variables and string variables
*                  use (1000*uvar + item) to pack extra information
*                  into what is stored in cx_variable(cx).  When these
*                  very large values go into computing nsum here,
*                  the expression 2107*nsum overflows and becomes
*                  a negative number, and so HASH_CX computed at the 
*                  end of this routine is 0 or > 0.  This messes up 
*                  the linked lists for variables in memory. Use uvar 
*                  for these types of vars so HASH_CX is always negative.

	include 'ferret.parm'
	include 'xvariables.cmn'	! for pmv_npiles
	include 'xcontext.cmn'

* calling argument declarations:
	INTEGER		cx

* internal variable declarations:
	INTEGER		idim, nsum, cat, uvar

* initialize
	cat = cx_category( cx )

* special Q&D sol'n for temporary variables, etc.
	IF ( cat .EQ. cat_constant 
     .  .OR. cat .EQ. cat_temp_var
     .  .OR. cat .EQ. cat_pseudo_var ) THEN
	   HASH_CX = -1
	   RETURN
	ELSEIF ( cat .EQ. cat_work_mr ) THEN
	   HASH_CX = pmv_work_var_pile
	   RETURN
	ENDIF	   
	
* arbitrary number generated from context information
        uvar = cx_variable( cx )
        IF (cat .EQ. cat_const_var .OR. cat .EQ. cat_string)  
     .       uvar = cx_variable(cx)/1000

	nsum = cx_data_set( cx )
     .	     + uvar * 7
     .	     + cat * 13
	DO 100 idim = 1, 4
	   IF ( cx_trans( idim, cx ) .NE. trans_no_transform )
     .							nsum = nsum + idim*idim	
 100	CONTINUE

* randomize the number and return a NEGATIVE pile number

	HASH_CX = -1 * ( MOD( 2107*nsum, pmv_npiles ) + 1 )
        IF (HASH_CX .GE. 0) HASH_CX = -1

	RETURN
	END
