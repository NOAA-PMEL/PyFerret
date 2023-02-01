	SUBROUTINE CD_ENCODE_URL ( urlbuff, ebuff, status)
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

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP 

* 
* V6.85 2/13 
* argument definitions:
*       urlbuff - url to encode
*       ebuff   - result enocded url
*	status	- return status

* argument declarations:
	INTEGER	      status
	CHARACTER*(*) urlbuff, ebuff

	INTEGER url_encode


        include 'netcdf.inc'
        include 'tmap_errors.parm'


        INTEGER   TM_LENSTR1, llen, slen
        PARAMETER (slen=2048)
	INTEGER*1 fhol_url(slen)      ! c-type Hollerith string buffer
	INTEGER*1 fhol_eurl(slen)

* Save dataset info in GLOBAL_NCdsetList for attribute handling

        llen = TM_LENSTR1( urlbuff )
        CALL TM_FTOC_STRNG( urlbuff(1:llen), fhol_url, slen )

        status = url_encode(fhol_url, fhol_eurl, llen)         
        IF (status .EQ. 0) THEN
           CALL TM_CTOF_STRNG (fhol_eurl, ebuff, llen) 
	   status = merr_ok
	ENDIF

* successful completion
	
	RETURN
	END
