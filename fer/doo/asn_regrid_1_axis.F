	SUBROUTINE ASN_REGRID_1_AXIS( cx_lims, axis, dst_lo, dst_hi,
     .				      src, msrc, dst, mdst )

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
* perform a regrid by "association"  (i.e. copy the data)

* 	programmer - Steve Hankin
* 	NOAA/PMEL,Seattle,WA - Tropical Modeling and Analysis Program

* V230:  8/22/92
* V570:  5/ 5/04  remove VMS includes
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xvariables.cmn'
	include 'xmem_subsc.cmn'
	include 'xcontext.cmn'

* Calling argument declarations:
	INTEGER cx_lims, axis, dst_lo, dst_hi, msrc, mdst
	REAL    src( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .               m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .          dst( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .               m2lot:m2hit,m2loe:m2hie,m2lof:m2hif )

* Internal variable declarations:
	INTEGER i, j, k, l, m, n, idim,
     .          lo_lim(nferdims), hi_lim(nferdims)


* Initialize: limits for calculation
        DO idim = 1, nferdims
           lo_lim(idim) = cx_lo_ss(cx_lims, idim)
           hi_lim(idim) = cx_hi_ss(cx_lims, idim)
	ENDDO

        lo_lim(axis) = MAX( dst_lo, mr_lo_ss(msrc,axis) )
        hi_lim(axis) = MIN( dst_hi, mr_hi_ss(msrc,axis) )

* Copy the data to the destination grid
	DO n = lo_lim(f_dim), hi_lim(f_dim)
	DO m = lo_lim(e_dim), hi_lim(e_dim)
	DO l = lo_lim(t_dim), hi_lim(t_dim)
	DO k = lo_lim(z_dim), hi_lim(z_dim)
	DO j = lo_lim(y_dim), hi_lim(y_dim)
	DO i = lo_lim(x_dim), hi_lim(x_dim)
 	   dst(i,j,k,l,m,n) = src(i,j,k,l,m,n)
	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO

	RETURN
	END
