        SUBROUTINE TM_CHECK_BNDS_ATTRIB (cdfid, iaxis, vname, vlen, 
     .                                   bname, bndid, npts, status)

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
* V570 5/2004 *acm*
* Check the bounds attribute, issue any warnings

	INCLUDE 'tmap_dims.parm'
        INCLUDE 'xtm_grid.cmn_text'
        INCLUDE 'xio.cmn_text'
        INCLUDE 'tmap_errors.parm'

* argument declarations
      INTEGER cdfid, iaxis, vlen, bndid, npts, status
      CHARACTER*(*) bname, vname

* Local variable declarations
      INTEGER TM_LENSTR1, NCVID
      INTEGER blen, cdfstat, ncnoerr, soft_err, vartype, 
     .        nvdims, vdims(8), nvatts
      CHARACTER*132 buff, vbuff

* does the bounds variable pointed to really exist ?
        blen = TM_LENSTR1(bname)
        bndid = NCVID( cdfid, bname(:blen), cdfstat )
        IF (cdfstat.NE.NCNOERR) THEN
           soft_err = 21
           GOTO 6000
        ENDIF

* get the vital statistics of the bounds variable
        CALL NCVINQ(cdfid, bndid, bname(:blen), vartype, nvdims,
     .               vdims, nvatts,cdfstat)
        IF ( nvdims .NE. 2 ) THEN
           soft_err = 22
           GOTO 6000
        ENDIF

* get axis length: 
        CALL NCDINQ( cdfid, vdims(1), buff, npts, cdfstat )
        IF (cdfstat.NE.NCNOERR) GOTO 5100   ! should never happen

* check for error conditions: bounds is 2 by N
        IF ( npts .NE. 2 ) THEN
           soft_err = 23
           GOTO 6000
        ENDIF
        CALL NCDINQ( cdfid, vdims(2), buff, npts, cdfstat )
        IF (cdfstat.NE.NCNOERR) GOTO 5100   ! should never happen

* check for error conditions: 2nd dimension of bounds is N
        IF ( npts .NE. line_dim(iaxis) ) THEN
           soft_err = 23
           GOTO 6000
        ENDIF

        status = merr_ok
        RETURN

* error messages

 6000   CALL TM_NOTE( 'netCDF bounds variable definition error', lunit_errors )
        buff = bname
        vbuff = vname
        IF ( soft_err .EQ. 21 ) THEN
           CALL TM_NOTE('Bounds definition "'//buff(:blen)//
     .                   '" points to no existing axis', lunit_errors )
        ELSEIF ( soft_err .EQ. 22 ) THEN
           CALL TM_NOTE('Bounds definition "'//buff(:blen)//
     .                   '" is not 2D', lunit_errors )
        ELSEIF ( soft_err .EQ. 23 ) THEN
           CALL TM_NOTE('Bounds "'//buff(:blen)//
     .          '" must be 2x dimension of '//vbuff(:vlen), lunit_errors )
        ENDIF
        CALL TM_NOTE( 'Ignoring BOUNDS attribute', lunit_errors )

        status = merr_ok + 1
        RETURN

 5100   status = pcdferr
        RETURN

        END
