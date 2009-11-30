	SUBROUTINE SHOW_DATA_SET_ATTRS( lun, dset, full )

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
* describe the attributes in the named nc data set

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* - based on SHOW_DATA_SET_VARS 
* V600 6/05 *acm* Attribute control.
* V600 4/08 SHOW DATA/ATT only for nc data so far; var info not in 
*           the linked list structure.
* V62  4/09 List the data type of variables as they are in the input
*           dataset.

	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'ez_lib.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xez_info.cmn_text'
	external xez_info_data
	include	'ferret.parm'
	include	'xvariables.cmn'
	include	'xtext_info.cmn'
	include	'xprog_state.cmn'
	include	'xrisc.cmn'
	include 'calendar.decl'
	include 'calendar.cmn'
        include 'netcdf.decl'
#include "netcdf.inc"

* calling argument declarations:
	LOGICAL	full
	INTEGER	lun, dset
	
* local variable declarations:
        LOGICAL         NC_GET_ATTRIB, coordvar, got_it
	INTEGER 	TM_LENSTR1, STR_SAME, NCF_INQ_DS, varid, ndims,
     .                  nvars, ngatts, recdim, iatt, attype, attlen,
     .                  slen,  llen, vlen, vtype, nvdims,
     .                  vdims(8),  nvatts, status, blen, i, len_attbuff,
     .                  attoutflag, all_outflag, attid
        REAL            vals(10)
	CHARACTER	TM_FMT*14, varname*512, attname*128, 
     .                  buff*14, attbuff*2048, oflag*1
        CHARACTER*6     typ(0:7)
        PARAMETER       (len_attbuff = 2048)
        
* See netcdf.h for the data types
	DATA typ/'NaN',          ! NAT = 'Not A Type' (c.f. NaN)
     .           'BYTE',         ! signed 1 byte integer
     .           'CHAR',         ! ISO/ASCII character
     .           'SHORT',        ! signed 2 byte integer
     .           'INT',          ! signed 4 byte integer
     .           'FLOAT',        ! single precision floating point number
     .           'DOUBLE',       ! double precision floating point number
     .           ' '/		 ! blank for the global attributes

* describe the variables in the set, only for datasets originating 
* as netcdf data.

        IF (dset .LE. 0) RETURN
        status = NCF_INQ_DS (dset, ndims, nvars, ngatts, recdim )

	risc_buff = ' '
	WRITE ( risc_buff, 3200 )
 3200	FORMAT(' Variable',T14,'VarType(in dset)',T34,'AttributeName',
     .         T50,'AttType',T62,'Size',T67,'OutFlag',T75,'AttValue')
 3201	FORMAT('---------------------------------------------',
     .         '---------------------------------------------')
 3300	FORMAT(' ',A,T24,A,T34,A,T50,A,T62,A,T67,A,T75,A)
 3400	FORMAT('(',A,')',T24,A,T34,A,T50,A,T62,A,T67,A,T75,A)
 3500	FORMAT(T34,A,T50,A,T62,A,T67,A,T75,A)

	CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 
        WRITE (lun, 3201) 
        WRITE (lun, *) ' '

* Write global atttributes

        IF ( ngatts .GE. 1) THEN
   
	      varname = '.'
	      CALL CD_GET_VAR_ID (dset, varname, varid, status)
              IF (varid .EQ. 0) vtype = 7  ! 
              
	      vlen = TM_LENSTR1(varname)
	      DO 100 iatt = 1, ngatts
                 CALL CD_GET_VAR_ATT_NAME (dset, varid, iatt, attname, 
     .              status )
                 CALL CD_GET_VAR_ATT_INFO (dset, varid, iatt, attname, 
     .              attype, attlen, attoutflag, status )
                 CALL CD_GET_VAR_ATT_INFO (dset, varid, iatt, attname, 
     .              attype, attlen, attoutflag, status )

	         blen = len_attbuff
	         got_it = NC_GET_ATTRIB (dset, varid, attname, 
     .                                .TRUE., varname, blen, attlen, 
     .                                attoutflag, attbuff, vals)

	         IF (got_it) THEN
	            IF (attype .NE. NCCHAR) THEN
                       blen = 1
                       attbuff = ' '
                       DO 50 i = 1, attlen
                          buff = TM_FMT(vals(i), 7, 14, llen)
                          attbuff(blen:blen+llen+1) = buff//' '
                          blen = blen + llen + 1
  50                  CONTINUE
                    ELSE 
                       blen = TM_LENSTR1(attbuff)
                    ENDIF
                    oflag = 'F'
                    IF (attoutflag .EQ. 1) oflag = 'T'
                 ELSE
                    attbuff = ' '
                    blen = 1
                    oflag = ' '
                 ENDIF

                 slen = TM_LENSTR1(attname)
                 buff = TM_FMT (FLOAT(attlen),7,14,llen) 
                 WRITE (risc_buff, 3300) varname(1:vlen), typ(vtype),
     .              attname(1:slen), typ(attype), buff(1:llen), oflag,
     .              attbuff(1:blen)

                 CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 
                 risc_buff = ' '
                 varname = ' '
 100         CONTINUE
             WRITE (lun, *) ' '

        ENDIF

* Write attributes for all variables
        IF (nvars .GE. 1) THEN

           DO 300 varid = 1, nvars
              CALL CD_GET_VAR_INFO (dset, varid, varname, vtype, nvdims,
     .           vdims, nvatts, coordvar, all_outflag, status)
              IF (varid .EQ. 0) vtype = 7  ! 
              vlen = TM_LENSTR1(varname)

              IF (nvatts .EQ. 0) THEN
                 IF (coordvar) THEN
                    WRITE (risc_buff, 3400) varname(1:vlen), typ(vtype)
                 ELSE
                    WRITE (risc_buff, 3300) varname(1:vlen), typ(vtype)
                 ENDIF
		 
		 CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 
              ENDIF

              DO 200 iatt = 1, nvatts
                 CALL CD_GET_VAR_ATT_INFO (dset, varid, iatt, attname, 
     .              attype, attlen, attoutflag, status )

                 blen = len_attbuff
                 got_it = NC_GET_ATTRIB (dset, varid, attname, 
     .                                .TRUE., varname, blen, attlen, 
     .                                attoutflag, attbuff, vals)

                 IF (got_it) THEN
                    IF (attype .NE. NCCHAR) THEN
                       blen = 1
                       attbuff = ' '
                       DO 150 i = 1, attlen
                          buff = TM_FMT(vals(i), 7, 14, llen)
                          attbuff(blen:blen+llen+1) = buff//' '
                          blen = blen + llen + 1
 150                   CONTINUE
                    ELSE 
                       blen = TM_LENSTR1(attbuff)
                    ENDIF
                    oflag = 'F'
                    IF (attoutflag .EQ. 1) oflag = 'T'
                 ELSE
                    attbuff = ' '
                    blen = 1
                    oflag = ' '
                 ENDIF

                 slen = TM_LENSTR1(attname)
                 buff = TM_FMT (FLOAT(attlen),7,14,llen) 
                
                 IF (iatt .EQ. 1) THEN
                    IF (coordvar) THEN
                       WRITE (risc_buff, 3400) varname(1:vlen), 
     .                    typ(vtype), attname(1:slen), typ(attype), 
     .                    buff(1:llen), oflag, attbuff(1:blen)
                    ELSE
                       WRITE (risc_buff, 3300) varname(1:vlen), 
     .                    typ(vtype), attname(1:slen), typ(attype), 
     .                    buff(1:llen), oflag, attbuff(1:blen)
                    ENDIF
                 ELSE 
                    WRITE (risc_buff, 3500) 
     .                  attname(1:slen), typ(attype), buff(1:llen),
     .                  oflag, attbuff(1:blen)
                 ENDIF

                 CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 

                 risc_buff = ' '
 200          CONTINUE
              WRITE (lun, *) ' '
 300       CONTINUE
        ENDIF

5100    RETURN

	END
