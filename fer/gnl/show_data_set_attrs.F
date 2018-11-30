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
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
* V698  2/16 *acm* For ticket 2352: variables defined with LET/D are
*                  added to that dataset in the attribute structure, 
*                  and are shown with their dataset on SHOW DAT/ATT
* V698  3/16 *acm* Ticket 2371: Wrap the output of SHOW DAT/ATT to 
*                  the next line when the variable name or attribute 
*                  name is long
* V7.44 11/18 *acm* Issue 1905: add int64 as a netcdf data type, recognize 
*                   other newer types


        IMPLICIT NONE
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
#include "netcdf.inc"

* calling argument declarations:
	LOGICAL	full
	INTEGER	lun, dset
	
* local variable declarations:
        LOGICAL         NC_GET_ATTRIB, coordvar, got_it, first_time
	INTEGER 	TM_LENSTR1, STR_SAME, NCF_INQ_DS, varid, ndims,
     .                  nvars, ngatts, recdim, iatt, attype, attlen,
     .                  slen,  llen, vlen, vtype, nvdims,
     .                  vdims(8),  nvatts, status, blen, i, len_attbuff,
     .                  max_vals, attoutflag, all_outflag, attid, uvflag
        REAL            vals(1000)
	CHARACTER	TM_FMT*14, GET_UNIQUE_DSET_NAME*1024, varname*512, 
     .                  attname*128, buff*14, dsname*1024, attbuff*10240, oflag*1
        CHARACTER*6     typ(0:13)
        PARAMETER       (len_attbuff = 10240, max_vals=1000)
        
* See netcdf.h for the data types
	DATA typ/'NaN',          ! NAT = 'Not A Type' (c.f. NaN)
     .           'BYTE',         ! signed 1 byte integer
     .           'CHAR',         ! ISO/ASCII character
     .           'SHORT',        ! signed 2 byte integer
     .           'INT',          ! signed 4 byte integer
     .           'FLOAT',        ! single precision floating point number
     .           'DOUBLE',       ! double precision floating point number
     .           'UBYTE',        ! Newer types as of 11/18:
     .           'USHORT',       ! 
     .           'UINT',         !  
     .           'INT64 ',       ! 
     .           'UINT64',       !  
     .           'STRING',       !  
     .           ' '/		 ! blank for the global attributes

* netcdf types
*      parameter (nf_byte = 1)
*      parameter (nf_char = 2)
*      parameter (nf_short = 3)
*      parameter (nf_int = 4)
*      parameter (nf_float = 5)
*      parameter (nf_double = 6)

*      parameter (nf_ubyte = 7)
*      parameter (nf_ushort = 8)
*      parameter (nf_uint = 9)
*      parameter (nf_int64 = 10)
*      parameter (nf_uint64 = 11)
*      parameter (nf_string = 12)



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
 4300	FORMAT(' ',T50,A,T62,A,T67,A,T75,A)
 5300	FORMAT(' ',T24,A,T34,A,T50,A,T62,A,T67,A,T75,A)
 3301	FORMAT(' ',A,T24,A,T34,A,T50,A,T62,A,T67,A,T75)
 5301	FORMAT(' ',T24,A,T34,A,T50,A,T62,A,T67,A,T75)
 3302	FORMAT(' ',A)
 3304	FORMAT(' ',A,T34,A,T50,A,T62,A,T67,A,T75,A)
 4304	FORMAT(' ',T50,A,T62,A,T67,A,T75,A)
 5304	FORMAT(' ',T34,A,T50,A,T62,A,T67,A,T75,A)
 3400	FORMAT('(',A,')',T24,A,T34,A,T50,A,T62,A,T67,A,T75,A)
 4400	FORMAT(T50,A,T62,A,T67,A,T75,A)
 5400	FORMAT(T24,A,T34,A,T50,A,T62,A,T67,A,T75,A)
 3500	FORMAT(T34,A,T50,A,T62,A,T67,A,T75,A)
 4500	FORMAT(T50,A,T62,A,T67,A,T75,A)


	CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 
        WRITE (lun, 3201) 
        WRITE (lun, *) ' '

* Write global atttributes

        IF ( ngatts .GE. 1) THEN
   
	      varname = '.'
	      CALL CD_GET_VAR_ID (dset, varname, varid, status)
              IF (varid .EQ. 0) vtype = 13  ! don't show a type for global atts
              
	      vlen = TM_LENSTR1(varname)
	      DO 100 iatt = 1, ngatts
                 CALL CD_GET_VAR_ATT_NAME (dset, varid, iatt, attname, 
     .              status )
                 CALL CD_GET_VAR_ATT_INFO (dset, varid, iatt, attname, 
     .              attype, attlen, attoutflag, status )
                 CALL CD_GET_VAR_ATT_INFO (dset, varid, iatt, attname, 
     .              attype, attlen, attoutflag, status )

	         blen = len_attbuff
		 IF (attype .NE. NCCHAR) attlen = MIN(attlen, max_vals)
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
		 IF (slen .GT. 15) THEN

                    WRITE (risc_buff, 3300) varname(1:vlen), typ(vtype),
     .                 attname(1:slen)

                    CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 
                    WRITE (risc_buff, 4300) typ(attype), buff(1:llen), oflag,
     .                 attbuff(1:blen)

		 ELSE
		    IF (vlen+slen+llen+blen+20 .LE. len_attbuff) THEN
                       WRITE (risc_buff, 3300) varname(1:vlen), typ(vtype),
     .                    attname(1:slen), typ(attype), buff(1:llen), oflag,
     .                    attbuff(1:blen)

                    ELSE
                       WRITE (risc_buff, 3301) varname(1:vlen), typ(vtype),
     .                    attname(1:slen), typ(attype), buff(1:llen), oflag
     
                       CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 
                       WRITE (risc_buff, 3302) attbuff(1:len_attbuff-2)
                    ENDIF
                 ENDIF
                 CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 
                 risc_buff = ' '
                 varname = ' '
 100         CONTINUE
             WRITE (lun, *) ' '

        ENDIF

* Write attributes for all variables
        IF (nvars .GE. 1) THEN

	   first_time = .TRUE.
           DO 300 varid = 1, nvars
              CALL CD_GET_VAR_INFO (dset, varid, varname, vtype, nvdims,
     .           vdims, nvatts, coordvar, all_outflag, status)
              IF (varid .EQ. 0) vtype = 7  ! 
              vlen = TM_LENSTR1(varname)

* Special listing for user-defined variables added to this dataset
	      CALL CD_GET_VAR_UVFLAG (dset, varid, uvflag, status)
	      
	      IF ( uvflag .EQ. 1) THEN
	         IF (first_time ) CALL SPLIT_LIST(pttmode_explct, lun,
     .			' ------------------------------', 31) 
	         first_time = .FALSE.
		 IF (vtype .EQ. 0) vtype = 7             ! user-var data type not set
		 dsname = GET_UNIQUE_DSET_NAME( dset )
		 llen = TM_LENSTR1(dsname)
		 varname = varname(1:vlen)//'[D='//dsname(:llen)//']' 
		 vlen = TM_LENSTR1(varname)
	      ENDIF

              IF (nvatts .EQ. 0) THEN
                 IF (coordvar) THEN
                    WRITE (risc_buff, 3400) varname(1:vlen), typ(vtype)
                    IF (vlen .GT. 21) THEN
		       WRITE (risc_buff, 3400) varname(1:vlen)
		       CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 
		       WRITE (risc_buff, 5400) typ(vtype)
                    ENDIF
                 ELSE
                    WRITE (risc_buff, 3300) varname(1:vlen), typ(vtype)
                    IF (vlen .GT. 21) THEN
                       WRITE (risc_buff, 3300) varname(1:vlen)
		       CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 
		       WRITE (risc_buff, 5300) typ(vtype)
		    ENDIF
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
                    IF (uvflag.EQ.0) THEN
                    IF (coordvar) THEN
		       IF (slen .GT. 15) THEN
                          WRITE (risc_buff, 3400) varname(1:vlen), 
     .                       typ(vtype), attname(1:slen)
     
                          IF (vlen .GT. 21) THEN
                             WRITE (risc_buff, 3400) varname(1:vlen)
                             CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 
                             WRITE (risc_buff, 5400) typ(vtype), attname(1:slen)
                          ENDIF
                          CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 

                          WRITE (risc_buff, 4400) typ(attype), 
     .                       buff(1:llen), oflag, attbuff(1:blen)

		       ELSE
                          WRITE (risc_buff, 3400) varname(1:vlen), 
     .                       typ(vtype), attname(1:slen), typ(attype), 
     .                       buff(1:llen), oflag, attbuff(1:blen)
                          IF (vlen .GT. 21) THEN
                             WRITE (risc_buff, 3400) varname(1:vlen)
                             CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 
                             WRITE (risc_buff, 5400) typ(vtype), attname(1:slen), 
     .                          typ(attype), buff(1:llen), oflag, attbuff(1:blen)
                          ENDIF
		       ENDIF

		    ELSE
		       IF (slen .GT. 15) THEN
                          WRITE (risc_buff, 3300) varname(1:vlen), typ(vtype),
     .                    attname(1:slen)

                          IF (vlen .GT. 21) THEN
                             WRITE (risc_buff, 3300) varname(1:vlen)
                             CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 
                             WRITE (risc_buff, 5300) typ(vtype), attname(1:slen)
                          ENDIF
                          CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 

                          WRITE (risc_buff, 4300) typ(attype), buff(1:llen), oflag,
     .                    attbuff(1:blen)

		       ELSE

                          IF (vlen+slen+llen+blen+20 .LE. len_attbuff) THEN
                             WRITE (risc_buff, 3300) varname(1:vlen), typ(vtype),
     .                         attname(1:slen), typ(attype), buff(1:llen), oflag,
     .                         attbuff(1:blen)
     
                             IF (vlen .GT. 21) THEN
                                WRITE (risc_buff, 3300) varname(1:vlen)
                                CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 
                                WRITE (risc_buff, 5300) typ(vtype),
     .                            attname(1:slen), typ(attype), buff(1:llen), oflag,
     .                            attbuff(1:blen)
                             ENDIF

                          ELSE
                             WRITE (risc_buff, 3301) varname(1:vlen), typ(vtype),
     .                         attname(1:slen), typ(attype), buff(1:llen), oflag

                             IF (vlen .GT. 21) THEN
                                WRITE (risc_buff, 3301) varname(1:vlen)
                                CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 
                                WRITE (risc_buff, 5301) typ(vtype),
     .                             attname(1:slen), typ(attype), buff(1:llen), oflag
                             ENDIF

                             CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 
                             WRITE (risc_buff, 3302) attbuff(1:len_attbuff-2)
                          ENDIF

		       ENDIF
                    ENDIF  ! coordvar or not
                    ENDIF  ! not uvflag

                    IF (uvflag .EQ. 1) THEN
		       IF (slen .GT. 15) THEN
                          WRITE (risc_buff, 3304) varname(1:vlen), 
     .                       attname(1:slen)
                          IF (vlen .GT. 32) THEN
                             WRITE (risc_buff, 3304) varname(1:vlen)
			     CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
                             WRITE (risc_buff, 5304) attname(1:slen)
			  ENDIF

			  CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 
 
                          WRITE (risc_buff, 4304) typ(attype), buff(1:llen), oflag,
     .                       attbuff(1:blen)
     
		       ELSE
                          WRITE (risc_buff, 3304) varname(1:vlen), 
     .                       attname(1:slen), typ(attype), buff(1:llen), oflag,
     .                       attbuff(1:blen)
                          IF (vlen .GT. 32) THEN
                             WRITE (risc_buff, 3304) varname(1:vlen)
			     CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
			     WRITE (risc_buff, 5304) 
     .                       attname(1:slen), typ(attype), buff(1:llen), oflag,
     .                       attbuff(1:blen)
                          ENDIF

		       ENDIF

                    ENDIF

                 ELSE 
                    IF (slen .GT. 15) THEN
                       WRITE (risc_buff, 3500) attname(1:slen)
                       CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 
                       WRITE (risc_buff, 4500) typ(attype), buff(1:llen),
     .                   oflag, attbuff(1:blen)
                    ELSE
                       WRITE (risc_buff, 3500) 
     .                     attname(1:slen), typ(attype), buff(1:llen),
     .                     oflag, attbuff(1:blen)
                    ENDIF
                 ENDIF

                 CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0) 

                 risc_buff = ' '
 200          CONTINUE
              WRITE (lun, *) ' '
 300       CONTINUE
        ENDIF

5100    RETURN

	END
