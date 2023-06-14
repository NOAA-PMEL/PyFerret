        SUBROUTINE WRITE_DODS_FMT ( data, 
     .           m1lox,m1hix, m1loy,m1hiy, m1loz,m1hiz, m1lot,m1hit, 
     .           m1loe,m1hie, m1lof,m1hif, 
     .           workspace, filename, clobber, status)

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
* write a gridded variable in DODS DAS binary format

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* Ferret v5.54 9/4/06
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

* change this comment, so that the checkin under branch FDSv1
* will occur.

        IMPLICIT NONE
* calling argument declarations:
        LOGICAL clobber
        INTEGER status,
     .          m1lox,m1hix, m1loy,m1hiy, m1loz,m1hiz, m1lot,m1hit, 
     .          m1loe,m1hie, m1lof,m1hif
	REAL data(m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit,m1loe:m1hie,m1lof:m1hif)
        REAL    workspace (*)  
        CHARACTER*(*) filename

* Local variable declarations
        INTEGER TM_LENSTR1, i, j, k, l, m, n, size, swap_flag,
     .          clobber_flag, slen

	include 'xmachine_int1.cmn'


* BYTEORDER - get byteorder of this architecture
 	IF (active_cpu .EQ. cptype_sun) THEN
	   swap_flag = 0
	ELSE
	   swap_flag = 1
	ENDIF

        size = 0

	DO n = m1lof,m1hif
	DO m = m1loe,m1hie
	DO l = m1lot,m1hit
	DO k = m1loz,m1hiz
	DO j = m1loy,m1hiy
	DO i = m1lox,m1hix

           size = size + 1
           workspace(size) = data(i,j,k,l,m,n)
	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO

         slen = TM_LENSTR1(filename)
         clobber_flag = 0
         IF (clobber) clobber_flag = 1
         CALL write_dods (filename, slen, clobber_flag, swap_flag, 
     .                    size,  workspace)

        RETURN

 5010   status = 0
        RETURN
        END
