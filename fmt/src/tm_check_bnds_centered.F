	LOGICAL FUNCTION TM_CHECK_BNDS_CENTERED (nline)
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
* Checks whether data points are all centered within the bounds on an axis
* Do this in single precision, as in the check for regular axis in cd_get_1_axis.

* 4/2004 ACM 
*
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data

* Argument declaration
	INTEGER nline

* Local variable declarations
        LOGICAL TM_FPEQ
	INTEGER ipt1, npts, ibx1, i
        REAL diff_hi, diff_lo

	ipt1 = line_subsc1(nline)
	npts = line_dim(nline)
	ibx1 = ipt1 + npts

	DO 100 i = 0, npts-1
          diff_hi = line_mem(ipt1+i) - line_mem(ibx1+2*i+1)
          diff_lo = line_mem(ibx1+2*i) - line_mem(ipt1+i)

	  IF ( .NOT. TM_FPEQ(diff_lo, diff_hi) ) THEN
	    TM_CHECK_BNDS_CENTERED = .FALSE.
	    RETURN
	  ENDIF
  100	CONTINUE

	TM_CHECK_BNDS_CENTERED = .TRUE.
	RETURN
	END
