	SUBROUTINE TM_AXIS_ORDER( lines4, user_perm, order, ndim, vname )

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
* based on 1 to 4 axes pointed to by lines4 try to determine how to place
* this object in 4 dimensional space
* e.g. if the axes are X,Y and T then slot 3 should probably be NORMAL
* suppose further that axis 2 has no units or clues to its orientation
* then our problem is to map  3 dimensions "XNT" to an ordering of 1240

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* revision 0.0 3/92 for TMAP library
* Linux port -kob - 3/97 - Modify include for tmap_dset.parm to be a
*			   preprocessor include for F90 - needed because
*			   tmap_dset.parm contains an ifdef
* V500 *sh*  6/99 - added user_perm (permutation) input argument allowing the
*		user to coach this routine on the ordering of the axes

#ifdef unix
        include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
        include 'xtm_grid.cmn_text'
        include 'xio.cmn_text'
        include 'xrisc_buff.cmn'
#else
        INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
        INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
        INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
        INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
        INCLUDE 'TMAP_FORMAT:XIO.CMN'
        INCLUDE 'TMAP_FORMAT:XRISC_BUFF.CMN'
#endif

        INTEGER tt  ! nice short name
        EQUIVALENCE (tt,lunit_errors)

* calling argument declarations:
      INTEGER lines4(4), user_perm(4), order(4), ndim
      CHARACTER*(*) vname

* internal variable declarations:
!      LOGICAL TM_FIRM_AXIS_ORIENT, present
      INTEGER TM_AXIS_DIRECTION, idim, lasdim, vlen, mdim, nuser,
     .	      ax1, firm_dim, orient, i, ii, perm(4)

* initialize
      DO 10 idim = 1, 4
 10   order(idim) = unspecified_int4
      mdim =  MIN(4, ndim)

* number of user-specified axes?
	nuser = 0
	DO 20 idim = 1, 4
	   IF (user_perm(idim) .NE.unspecified_int4) THEN
	      nuser = nuser + 1
	   ELSE
	      GOTO 30
	   ENDIF
 20	CONTINUE


* nothing happens here!
* arguably, one should permute the negatives so that they refer to
* the axis in the file, rather than the axis in memory. But it was too
* big a hassle (*sh*)
 30	DO 36 idim = 1, 4
 36	perm(idim) = user_perm(idim)


	IF (nuser .EQ. 0) THEN
	   ax1 = 0+1    ! user specified no axis orientations (typical)
	ELSE

****** user-specified axis order ******

	   IF (nuser .EQ. mdim) THEN
* ... all axes were specified by the user e.g. /ORDER=YXT for a 3D grid
	      DO 50 idim = 1, mdim
 50	      order(idim) = perm(idim)
	      ax1 = mdim + 1
	   ELSEIF (nuser .LT. mdim) THEN
* ... user gave list shorter than mdim -- apply list to initial axes
*     e.g. /ORDER=YX on an XYT grid will first process X and Y
	      DO 60 idim = 1, nuser
 60	      order(idim) = perm(idim)
* ... now map the remaining axes
* ... first map self-oriented axes if they match an unused orientation
	      DO 62 idim = nuser+1, mdim
	         orient = TM_AXIS_DIRECTION(lines4(idim))
	         IF (orient .NE. 0) THEN
	            DO 61 i = 1, mdim
 61		    IF (ABS(order(i)).EQ.orient) GOTO 62
	            nuser = nuser + 1
	            order(nuser) = orient
	         ENDIF
 62	      CONTINUE
* ... then map any remaining axes to any unused orientations
	      DO 66 idim = nuser+1, mdim
* ... find unused orientation
	         DO 64 orient = 1, 4
	            DO 63 i = 1, idim
 63		    IF (ABS(order(i)) .EQ. orient) GOTO 64 ! in use
		    GOTO 65   ! found a free one
 64		 CONTINUE
	         CONTINUE   ! should never get here !!!
* ... assign axis
 65		 order(idim) = orient
 66	      CONTINUE
	      ax1 = mdim + 1
	   ELSE
* ... user gave list longer than mdim ... how to use this info? (messy)     
*     e.g. /ORDER=YXT on a 2D grid of "?T" orientation should map to XT
* ==> just use as much of the users mapping as applies
	      DO 70 idim = 1, mdim
 70	      order(idim) = perm(idim)
	      ax1 = mdim + 1

!* this block of logic abandoned as making Ferret "too intellligent"
!* because it has imperfection. For example /ORDER=TXYZ given for a
!* firmly XYT grid maps to "XYZ" because Ferret identifies the "X" axis
!* as "firm" first off
!
!*     Step 1: find the first axis of the grid whose orientation is
!*     known with high certainty. Try to map this to a user-given axis
!	      DO 76 firm_dim = 1, mdim
!	         IF (TM_FIRM_AXIS_ORIENT(lines4(firm_dim), orient)) THEN
!*     ... check that this orientation is in the users list
!	            DO 72 idim = 1, nuser 
!	               IF (orient .EQ. ABS(perm(idim))) THEN
!*     ... check that this orientation is possible to work with
!*         e.g.  /ORDER=YXT with lines4(1) of 2D grid matching "T" is bad
!*	                or with lines4(2) of 2D grid matching "Y" is bad
!	                  IF ( (nuser-idim).GE.(mdim-firm_dim)
!     .			 .AND.  idim .GE. firm_dim ) GOTO 80
!	               ENDIF
! 72		    CONTINUE
!	         ENDIF
! 76	      CONTINUE
!*     ... nothing firmly known. Just start from first axis
!	      firm_dim = 1
!	      idim = 1
!*     Step 2: map the orientations to the left and right of firm_dim
! 80	      order(firm_dim) = perm(idim)
!	      DO 84 i = firm_dim-1,1,-1
! 84	      order(i) = perm(idim-(firm_dim-i))
!	      DO 86 i = firm_dim+1,mdim
! 86	      order(i) = perm(idim-(firm_dim-i))
!	      ax1 = mdim + 1

	   ENDIF
	ENDIF

****** automatic determination of axis order ******

* ... pass 1:  e.g. map to 1 0 4 -111
      DO 100 idim = ax1, mdim
 100  order(idim) = TM_AXIS_DIRECTION(lines4(idim))

* ..  pass 2: e.g. map "N" to "Y" to give 1 2 4 0
      lasdim = 0
      DO 200 idim = 1, 4
         IF ( order(idim).EQ.0 ) THEN
            lasdim = lasdim + 1
            order(idim) = lasdim
         ELSEIF (order(idim).NE.unspecified_int4 ) THEN
            lasdim = ABS(order(idim))
         ELSE
            order(idim) = 0
         ENDIF
 200  CONTINUE

* ... pass 3: if out of order (e.g. YXT given) a problem
*             ==> remap to most basic
* 6/99 - only REPEATED axes are a problem -- permutations OK
      lasdim = 0
      DO 310 idim = 1, 4
	 DO 300 i = 1, idim-1
            IF ( order(i) .NE. 0
     .     .AND. ABS(order(i)) .EQ. ABS(order(idim)) ) GOTO 500
 300	 CONTINUE
 310  CONTINUE

* remap the lines4 array to match TMAP 4-dimensional ordering instead of
* netCDF file ordering. For example, change an XYT grid from current
* line1, line2, line4, unspecified to line1, line2, unspecified, line4
!	i = mdim
!	DO 410 idim = 4, 1, -1
!* ... is this axis present in the grid
!	   present = .FALSE.
!	   DO 400 ii = 1, 4
!	      IF (ABS(order(ii)) .EQ. idim) present = .TRUE.
! 400	   CONTINUE
!	   IF (present) THEN
!	      lines4(idim) = lines4(i)
!	      i = i - 1
!	   ELSE
!	      lines4(idim) = unspecified_int4
!	   ENDIF
! 410	CONTINUE

* success
      RETURN

* failure - select most basic order
 500  DO 600 idim = 1, 4
         IF ( idim .LE. mdim ) THEN
            order(idim) = idim
         ELSE
            order(idim) = 0
         ENDIF
 600  CONTINUE

* sound a warning horn
      risc_buff = vname
      vlen = LEN(vname)
      CALL TM_NOTE( 'unsupported ordering of axes in variable '
     .               //risc_buff(:vlen), tt )
      CALL TM_NOTE( 'The default ordering will be used', tt )

      RETURN
      END
