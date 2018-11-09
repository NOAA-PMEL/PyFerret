	SUBROUTINE GCF_SET_ALT_FCN_FOR_DIR( gcfcn, idirs, num_dirs)

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
* Find the correct function name and update to the right function number
* for direction-changing functions which have arguments that specifies the
* direction of operation.

* V7.20 *acm* 4/2017

        IMPLICIT NONE
        include 'tmap_dims.parm'
        include 'ferret.parm'
	include 'grid_chg_fcns.parm' 
	include 'xgrid_chg_fcns.cmn'


* calling argument declarations
	INTEGER		gcfcn, idirs(*), num_dirs

* Local declarations
	INTEGER	TM_LENSTR1, GCF_FIND_FCN, nlen, isubst, idim
	CHARACTER name*40, buff*40, axdir(nferdims)*1
        DATA axdir / 'X', 'Y', 'Z', 'T', 'E', 'F' /

	IF (num_dirs .EQ. 0)  GOTO 5100

	name = gfcn_alt_name_base(gcfcn)
	nlen = TM_LENSTR1( name )
	isubst = INDEX(name, '%')
	IF (isubst .EQ. 0) GOTO 5100

* substitute axis direction names to get the new function

	DO idim = 1, nferdims
	   IF (idirs(idim) .NE. 0) THEN
	      buff = name
	      nlen = TM_LENSTR1( buff )
	      isubst = INDEX(buff, '%')
	      IF (isubst .EQ. 0) GOTO 5100
	      buff = buff(:isubst-1)//axdir(idirs(idim))//
     .          buff(isubst+1:nlen)
              name = buff
	   ENDIF
        ENDDO
	   
	nlen = TM_LENSTR1( name )
	gcfcn = GCF_FIND_FCN( name )

	RETURN
5100    gcfcn = unspecified_int4
	RETURN
	END
