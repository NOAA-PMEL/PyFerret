	SUBROUTINE COPY_AGG_INTO ( src, msrc, cxs, dst, mdst, mloc )

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
* copy data from source grid into dstination grid using limits of source
* context (i.e. destination grid is ASSUMED) big enough to hold data.
* The source grid does not have an ensemble axis (dimension 5) but the
* destination axis does.  Copy the ensemble member into e-axis location 
* pointed to by index mloc.
*
* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Thermal Modeling and Analysis Program
*
* V680 5/12 *acm* 6D Ferret, changes for DEFINE DATA/AGGREGATE
*
	include 'tmap_dims.parm'
	include 'ferret.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'

* calling argument declarations:
	INTEGER msrc, cxs, mdst, dcx, mloc
	REAL    src( * ), dst( * )

* internal variable declarations
	INTEGER idim, soffset, doffset, mdel,
     .		smlo(nferdims), smhi(nferdims),
     .		dmlo(nferdims), dmhi(nferdims),
     .		cxlo(nferdims), cxhi(nferdims)

* set-up modified limits wherever delta is significant
* cuz the FORTRAN dimensioned size can only have delta of 1 

	DO idim = 1,nferdims
	  IF (mr_delta(idim,msrc) .NE. unspecified_val8) THEN
            mdel = NINT(mr_delta(idim,msrc))

	    IF (mr_delta(idim,msrc).NE.mr_delta(idim,mdst)
     .	   .OR. mr_delta(idim,msrc).NE.cx_delta(idim,cxs) ) CALL WARN
     .		('Internl_delta_err')
	    soffset = (mr_lo_ss(msrc,idim)-cx_lo_ss(cxs,idim))/
     .			mdel  + 1
	    doffset = (mr_lo_ss(mdst,idim)-cx_lo_ss(cxs,idim))/
     .			mdel  + 1
	    cxlo(idim) = 1

	    cxhi(idim) = 1 + (cx_hi_ss(cxs,idim)-cx_lo_ss(cxs,idim))/
     .			     mdel
	    dmlo(idim) = doffset
	    dmhi(idim) = doffset +
     .			     (mr_hi_ss(mdst,idim)-mr_lo_ss(mdst,idim))/
     .			     mdel
	    smlo(idim) = soffset
	    smhi(idim) = soffset + 
     .			     (mr_hi_ss(msrc,idim)-mr_lo_ss(msrc,idim))/
     .			     mdel
	  ELSE
	    cxlo(idim) = cx_lo_ss(cxs, idim)
	    cxhi(idim) = cx_hi_ss(cxs, idim)
	    dmlo(idim) = mr_lo_ss(mdst,idim)
	    dmhi(idim) = mr_hi_ss(mdst,idim)
	    smlo(idim) = mr_lo_ss(msrc,idim)
	    smhi(idim) = mr_hi_ss(msrc,idim)
	  ENDIF
	ENDDO

	IF (mr_type(msrc) .EQ. ptype_string ) THEN
	   CALL COPY_AGG_PTR_INTO_SUB( mloc, cxlo, cxhi,
     .			    src, smlo(1), smhi(1),
     .				 smlo(2), smhi(2),
     .				 smlo(3), smhi(3),
     .				 smlo(4), smhi(4),
     .				 smlo(5), smhi(5),
     .				 smlo(6), smhi(6),
     .			    dst, dmlo(1), dmhi(1),
     .				 dmlo(2), dmhi(2),
     .				 dmlo(3), dmhi(3),
     .				 dmlo(4), dmhi(4),
     .				 dmlo(5), dmhi(5),
     .				 dmlo(6), dmhi(6) )
	ELSE
	   CALL COPY_AGG_INTO_SUB( mloc, cxlo, cxhi,
     .			    src, smlo(1), smhi(1),
     .				 smlo(2), smhi(2),
     .				 smlo(3), smhi(3),
     .				 smlo(4), smhi(4),
     .				 smlo(5), smhi(5),
     .				 smlo(6), smhi(6),
     .			    dst, dmlo(1), dmhi(1),
     .				 dmlo(2), dmhi(2),
     .				 dmlo(3), dmhi(3),
     .				 dmlo(4), dmhi(4),
     .				 dmlo(5), dmhi(5),
     .				 dmlo(6), dmhi(6),
     .				 cx_bad_data(cxs),
     .				 mr_bad_data(mdst) )
	ENDIF

	RETURN
	END 
