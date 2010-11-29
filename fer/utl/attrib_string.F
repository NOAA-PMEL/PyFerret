	SUBROUTINE ATTRIB_STRING (cx, mr, dset, varid, varname, attname,
     .			          status)

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

* get string attribute values

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP

* V600 8/18/05

* calling argument declarations
	INTEGER		cx, mr, status
	INTEGER		dset, varid
	CHARACTER*(*)	varname, attname

* internal variable declarations
	LOGICAL		NC_GET_ATTRIB, got_em, do_warn
	INTEGER 	TM_LENSTR1, attlen, n, vlen, alen, aflag
	CHARACTER	buff*2048
        REAL            vals

* internal parameter declarations
        INTEGER       maxlen  ! length of buff to store string value
        PARAMETER     ( maxlen = 2048)

	include 'ferret.parm'
	include 'errmsg.parm'
	include 'gfdl_vms.parm'
	include 'xvariables.cmn'

* initialize
	n      = 1
        do_warn = .FALSE.

* Get float values 
	got_em = NC_GET_ATTRIB ( dset, varid, attname,
     .                           do_warn, varname, maxlen,
     .                           attlen, aflag, buff, vals )
        IF (.NOT. got_em) GOTO 5000

c store in string variable
	CALL STORE_STRING(buff(1:attlen), mr, n-1, status)

* success
 1000	status = ferr_ok
	RETURN

* syntax error
 4999	RETURN

 5000	vlen = TM_LENSTR1(varname)
        alen = TM_LENSTR1(attname)
        buff = varname(:vlen)//"."//attname(:alen)
        vlen = vlen + alen + 1
        CALL ERRMSG( ferr_unknown_attribute, status, 
     .  'attribute not found:  '//buff(:vlen), *4999 )

	END
