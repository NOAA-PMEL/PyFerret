	INTEGER FUNCTION AUX_AXIS_FROM_ARG( cx, arg_num )

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
* given the argument number within a specified aux var regrid
* return the corresponding axis orientation (idim)

	IMPLICIT NONE

	include	'ferret.parm'
        include 'tmap_dims.parm'
	include	'xcontext.cmn'

* calling argument declarations:
	INTEGER	 cx, arg_num

* internal variable declarations:
	INTEGER	 i, idim


* which auxiliary var is the subject of interest?
* e.g. for "temp[gz(depth)=zax]" the argument "depth" is found in the z dim
	i = 1
	DO 100 idim = 1, nferdims
	   IF (cx_aux_stat(idim, cx) .NE. paux_stat_na ) THEN
! ??     .	 .AND. cx_aux_stat(idim, cx) .NE. paux_passed  ) THEN
	      IF (i .EQ. arg_num) THEN
	         GOTO 1000
	      ELSE
	         i = i + 1  ! not the right argument yet
	      ENDIF
	   ENDIF
 100	CONTINUE
* ... should never land here
	CALL WARN( 'YIKES: aux_var_crptn' )

 1000	AUX_AXIS_FROM_ARG = idim
	RETURN
	END
