	SUBROUTINE SHOW_ATTRIBUTE ( varname, coordvar, attname, attype, 
     .                    attlen, attstr, attvals, attoutflag, sho_out) 
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
* execute SHOW ATTRIBUTE for a single variable and attribute name

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP
*
* V600 6/05 *acm* Attribute control.

	include	'ferret.parm'
	include 'errmsg.parm'
	include 'xtext_info.cmn'
        include 'xprog_state.cmn'
        include 'xinterrupt.cmn'
        include 'netcdf.inc'

* calling argument declarations
        LOGICAL coordvar, sho_out
	INTEGER	attype, attlen, attoutflag
	REAL attvals(*)
	CHARACTER varname*(*), attname*(*), attstr*(*)

* local parameter declarations:
	LOGICAL	    full_precision
	PARAMETER ( full_precision = .TRUE. )

* local variable declarations:
	INTEGER  TM_LENSTR1, slen, llen, vlen, i, status
        CHARACTER TM_FMT*14, vbuff*14, buff*2048, punc*2

* local parameter declarations:
	INTEGER		sig_figs
	PARAMETER     ( sig_figs = 7 )   ! was 5 1/95

        slen = 2
        vlen = TM_LENSTR1(varname)
        llen = TM_LENSTR1(attname)
        buff = ' '//varname(:vlen)//'.'//attname(:llen)//' '
        slen = slen + llen + vlen + 1
        IF (coordvar) THEN
           buff = ' ('//varname(:vlen)//').'//attname(:llen)//' '
           slen = slen + 2
        ENDIF

* if character type, write the attribute name with its string value

        IF (attype .EQ. NCCHAR) THEN

           buff = buff(:slen)//'= '//attstr(:attlen)
           slen = slen + 3 + attlen

           IF (sho_out) THEN
              IF (attoutflag .EQ. 0) buff = buff(:slen)//'    0'
              IF (attoutflag .EQ. 1) buff = buff(:slen)//'    1'
              slen = slen + 5
           ENDIF
        
           CALL SPLIT_LIST(pttmode_explct, ttout_lun,
     .		           buff, slen)

* If numeric, write the value(s)

        ELSE
           
           punc = '= '
	   DO 100 i = 1, attlen
	      IF ( interrupted ) CALL ERRMSG
     .		   ( ferr_interrupt, status, ' ', *1000 )
	   
              vbuff = TM_FMT(attvals(i), sig_figs, 14, llen )
              buff = buff(:slen)//punc//vbuff(:llen)
              slen = slen + 2 + llen
              punc = ', '

 100	   CONTINUE
 
           IF (sho_out) THEN
              IF (attoutflag .EQ. 0) buff = buff(:slen)//'    0'
              IF (attoutflag .EQ. 1) buff = buff(:slen)//'    1'
              slen = slen + 5
           ENDIF

           CALL SPLIT_LIST(pttmode_explct, ttout_lun, buff, slen)

	ENDIF      

* successful completion
 1000	RETURN
	END
