	SUBROUTINE CD_TRANSLATE_ERROR (cdfstat, ebuff)
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
* Call NF_STRERROR to get the error translation from netCDF library.
* The length of the string is set in netcdf.inc at 512.
*
* include files
        include 'netcdf.decl'
#include "netcdf.inc"
	include 'tmap_errors.parm'

! Declare netcdf function names. 
! The ones without trailing underscore are declared in netcdf.inc
#include "netcdf_declareUnderscore.inc"

c Argument definitions

	INTEGER cdfstat
	CHARACTER*(*) ebuff

c Local declararations
        INTEGER	TM_LENSTR1, STR_SAME, STR_UPCASE, 
     .		dlen, i
	REAL	val
	CHARACTER*512 upstr
	CHARACTER*10 TM_FMT, errcode


#ifdef usingDODSf2cUnderscore 
        upstr =  NF_STRERROR_(cdfstat)
#else
        upstr =  NF_STRERROR(cdfstat)
#endif

        ebuff = upstr

        upstr = ' '
        errcode = ' '

        i = STR_UPCASE(upstr,ebuff)
        IF (STR_SAME(upstr(1:13), 'UNKNOWN ERROR') .EQ. 0) THEN
           ebuff = 'Unknown error reading from remote dataset '
        ENDIF

        dlen = TM_LENSTR1(ebuff)

        val = cdfstat
        errcode = TM_FMT(val, 0, 10, i)
        ebuff(dlen+2:)  = '(OPeNDAP/netCDF Error code '//
     .      errcode(1:TM_LENSTR1(errcode))//') '
        dlen = TM_LENSTR1(ebuff)

	RETURN
	END
