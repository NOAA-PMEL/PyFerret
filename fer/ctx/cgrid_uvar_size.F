	INTEGER*8 FUNCTION CGRID_UVAR_SIZE ( cx )

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
* determine the number of values in the given cx, recognizing that if it is
* a uvar, then in its pyramid of definitions there may be compressing
* transformations.

* For example in LET Vave = V[x=@ave]
* even though the X limits of the context may X=1:1000, the X size of Vave
* is only 1

* V702: *sh* 3/17
* v767 6/22 *acm* See issue 112: Make this an integer*8 routine.

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'

	INTEGER	CX_DIM_LEN, cx, uvar, idim, dimsize
	INTEGER*8 CGRID_SIZE8, isize


	IF ( cx_category(cx) .EQ. cat_user_var ) THEN
	   uvar = cx_variable(cx)
	   isize = 1
	   DO idim = 1, nferdims
	      IF (uvar_given(idim, uvar) .EQ. uvlim_need_xact) THEN
	         dimsize = 1
	      ELSE
	         dimsize = CX_DIM_LEN( idim, cx )
	      ENDIF
	      isize = isize * dimsize
	   ENDDO
	   CGRID_UVAR_SIZE = isize
	ELSE
	   CGRID_UVAR_SIZE = CGRID_SIZE8(cx)
	ENDIF

	RETURN
	END
