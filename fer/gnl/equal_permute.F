	SUBROUTINE EQUAL_PERMUTE(string,fill,v_ok,neg_ok,perm,status )

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
* from a string of the form "perm=XYZTEF", return array perm with the ordering
* expressed as integers

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:   9/1/89
* V230: 4/92 (SUN port) - bug fix on loop: "fill in any missing dimensions"
*
* 7/15/98 *js* Added support for "XYZTV" permutations. 
*    5/99 *sh* Added booleans "fill" argument to control if permutation array
*	       is filled, var_ok arg to determine if "V" is allowable, and
*	       neg_ok for acceptable minus signs
* 2/03 *kob* - g77 port - g77 won't allow intrinsic functions in PARAMETER
*                         statements.  use an octal constant instead
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V693+ *sh* 2/15 - added support for _DQ_ quotations
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

      IMPLICIT NONE

* includes
      include 'tmap_dims.parm'

* calling argument declarations
	LOGICAL		fill, v_ok, neg_ok
	INTEGER		perm(*), status
	CHARACTER*(*)	string

* internal variable declarations
	INTEGER	TM_LENSTR1, STR_UPCASE, equal_pos, slen, pos, vax_code,
     .		idim, ndim, i, j, rev, pmax
	CHARACTER*1	ax

* internal parameter declarations:
	CHARACTER*1	tab



	PARAMETER     ( tab = CHAR(9))


	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xtext_info.cmn'

* initialize
	slen	  = TM_LENSTR1( string )
	equal_pos = INDEX( string, '=' )
	IF ( equal_pos .EQ. 0 .OR. equal_pos .EQ. slen ) GOTO 5100
	IF (v_ok) THEN
	   pmax = nferdims+1
	ELSE
	   pmax = nferdims
	ENDIF

* skip blanks and tabs
	DO  pos = equal_pos+1, slen
	   IF ( string(pos:pos) .NE. ' '
     .	  .AND. string(pos:pos) .NE. tab ) GOTO 100
	END DO
	GOTO 5100

* remove quotation marks
 100	   IF ( string(pos :pos  ) .EQ. '"'
     .	  .AND. string(slen:slen ) .EQ. '"' ) THEN
	      pos  = pos  + 1
	      slen = slen - 1
	   ELSEIF (string(pos :pos ) .EQ. '_'
     .	     .AND. string(slen:slen) .EQ. '_' 
     .	     .AND. slen-pos.GE.8            ) THEN
	      IF (string(pos:pos+3)   .EQ. '_DQ_'
     .	    .AND. string(slen-3:slen) .EQ. '_DQ_' ) THEN
	         pos  = pos  + 4
	         slen = slen - 4
	      ENDIF
	   ENDIF

* decode permutation string
	ndim = 0
	rev = 1	 ! reversed axis ?
	DO 250 pos = pos, slen
	   vax_code = STR_UPCASE( ax, string(pos:pos) )
	   IF ( v_ok .AND. ax .EQ. 'V') THEN   ! Variable name permutation
	      ndim = ndim + 1
	      perm(ndim) = nferdims+1
	   ELSEIF ( neg_ok .AND. ax .EQ. '-' ) THEN
	      rev = -1
	   ELSE
	      DO 220 idim = 1, nferdims
	         IF ( ax .EQ. ss_dim_name(idim)
     .	         .OR. ax .EQ. ww_dim_name(idim) ) THEN
	            IF (ndim .EQ. pmax) GOTO 5100
*   ... check for duplicates
	            DO 210 i = 1, ndim
 210		    IF ( idim .EQ. ABS(perm(i)) ) GOTO 5100
*  ... save the axis
	            ndim = ndim + 1
	            perm(ndim) = idim * rev
	            rev = 1
	            GOTO 250
	         ENDIF
 220	      CONTINUE
	      GOTO 5100
	   ENDIF
 250	CONTINUE

* fill in any missing dimensions (e.g. "perm=YX" is interpreted as YXZT)
	IF (fill) THEN
	  DO 350 j = ndim+1, nferdims+1
	     DO 320 idim = 1, nferdims+1
	        DO i = 1, j-1   ! slots already set (4/92 bug fix)
		   IF ( idim .EQ. ABS(perm(i)) ) GOTO 320
	        ENDDO
	        perm(j) = idim
	        GOTO 350
 320	     CONTINUE
 350	  CONTINUE
	ENDIF

* success
	status = ferr_ok
	RETURN

* syntax error
 5000	RETURN
 5100	CALL ERRMSG( ferr_syntax, status, string, *5000 )
	END
