	LOGICAL FUNCTION ITSA_PURE_SUM_VAR( uvar )

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
* is the indicated variable of the form a + b + c ?
* Variables of this type may return a string result if all of the summands
* are strings.


* V533 *sh* 7/01

* calling argument declarations
	INTEGER	uvar

* internal variable declarations
	INTEGER i, i1, ii, itype

	include 'ferret.parm'
	include 'xvariables.cmn'

* This routine should never be called for GC vars -- it is to handle "legacy"
* variables in Ferret.
* Expressions like
*      LET a = STRING_FUNCTION(c+d) + "b" + {"a","b","c"}
* are acceptable because INIT_UVAR will have decomposed them and removed GC
* function calls such as  STRING_FUNCTION(c+d).  The only function calls that
* could remain would be numeric functions (e.g. SIN(x)). A uvar containing
* a numeric function call is NOT "pure_sum"

* Further "subleties" (aka hacks) added to support IF cond THEN "a" ELSE "b"
* In this case the "cond" logic does not bear on the data type of the result
* so we do not consider it in our search

* initialize - start testinging after the "alg_then_marker" (if exists)
* (probably will be buggy for nested IFs)
	DO i1 =  1, uvar_num_items(uvar)
	  itype = uvar_item_type(i1,uvar)
	  IF ( itype .EQ. alg_then_marker ) GOTO 10
	ENDDO
	i1 = 1

* test the "purity" of the expression
 10	DO 100 i = i1, uvar_num_items(uvar)

	  itype = uvar_item_type(i,uvar)

* constants and pseudo-variables are known to be numeric
	  IF   ( itype .EQ. alg_constant   
     .	  .OR.   itype .EQ. alg_pseudo_var ) THEN
	     GOTO 1000

	  ELSEIF (
     .		 itype .EQ. alg_child_var   
     .	  .OR.   itype .EQ. alg_variable   
     .	  .OR.   itype .EQ. alg_grid_chg_fcn 
     .	  .OR.   itype .EQ. alg_const_var
     .	  .OR.   itype .EQ. alg_string      ) THEN
	     GOTO 100

	  ELSEIF ( itype .EQ. alg_if_marker
     .    .OR.     itype .EQ. alg_then_marker
     .	      .OR. itype .EQ. alg_log_struct) THEN
	     GOTO 100

	  ELSEIF ( itype .EQ. alg_function  ) THEN
	     GOTO 1000

	  ELSEIF ( itype .EQ. alg_operator  ) THEN
	     ii = uvar_item_start( i, uvar )
	     IF ( uvar_text(uvar)(ii:ii).EQ.'+' ) THEN
	       GOTO 100
	     ELSE
	       GOTO 1000
	     ENDIF

	  ELSE
	     STOP 'unknown_alg_cat'
	  ENDIF
 100	CONTINUE

* yes
	ITSA_PURE_SUM_VAR = .TRUE.
	RETURN

* no
 10000	ITSA_PURE_SUM_VAR = .FALSE.
	RETURN

	END
