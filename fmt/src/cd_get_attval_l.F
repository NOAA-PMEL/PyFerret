	LOGICAL FUNCTION CD_GET_ATTVAL_L ( dset, varid, attrib,
     .                                     do_warn, vname, val )

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
* This routine is like cd_attvalc except it checks for a yes-or-no result.
* It can return only a single value, of type LOGICAL.

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* 12/02  *acm*
* V550 *acm* 1/03 - Check for value of attribute " ", return TRUE if so.
* 
* 1/03 - *kob* g77 port - need to use risc_buff and also local var 
*                         tmp_buff for arbitrary  string concats 
* V600 *ACM* attribute control. 
*             Replace NCAINQ with CD_GET_VAR_ATT_INFO
*             Change call to CD_GET_ATTRIB to NC_GET_ATTRIB - get attrib from 
*               linked list structure
*             CD_GET_ATTVAL replaced by NC_GET_ATTRIB, which now returns  
*               strings or real values according to attrib type
 
* argument definitions:
*	dset	 - dataset number for already opened CDF file
*       varid    - variable id in netCDF file
*       attrib   - attribute to search for
*	do_warn	 - logical flag to issue warning if invalid attribute
*       vname    - name of variable owning attribute (used in error msgs)
*       val      - logical result, if modulo = "T" or "NO", etc was set.

* argument declarations
      LOGICAL       do_warn
      INTEGER	    dset, varid
      CHARACTER*(*) attrib, vname
      LOGICAL       val

      include 'xio.parm'
      include 'netcdf.decl'
#include "netcdf.inc"
      include 'xrisc_buff.cmn'         !1/03 *kob*
      include 'tmap_errors.parm'

* internal variable declarations:
      LOGICAL NC_GET_ATTRIB
      INTEGER TM_LENSTR1, lena, attype, attlen, attoutflag,
     .        attid, status
      REAL tmp
      CHARACTER buff*132, upbuff*132, aname*128
      CHARACTER tmp_buff*2048          !1/03 *kob*	

* inquire about the attribute by name and id

      CALL CD_GET_VAR_ATT_ID (dset, varid, attrib, attid, status)
      IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid, attid,
     .                  aname, attype, attlen, attoutflag, status )

      lena = TM_LENSTR1( aname )

      IF ( status .EQ. merr_ok ) THEN

	IF (attype .EQ. NCCHAR) THEN

* get a text attribute

	   CD_GET_ATTVAL_L = NC_GET_ATTRIB(dset, varid, aname(:lena),
     .                            do_warn, vname, 132, attlen,
     .                            attoutflag, buff, tmp)

	   IF (.NOT.CD_GET_ATTVAL_L) THEN
	     RETURN

	   ELSE

             CALL STR_UPCASE (upbuff, buff)	     

             IF ( upbuff .EQ. 'Y' .OR. upbuff .EQ. 'YES' .OR.
     .            upbuff .EQ. 'T' .OR. upbuff .EQ. 'TRUE' .OR.
     .            upbuff .EQ. 'ON' .OR. upbuff .EQ. ' ') THEN
	       CD_GET_ATTVAL_L = .TRUE.
               val = .TRUE.

             ELSE IF ( upbuff .EQ. 'N' .OR. upbuff .EQ. 'NO' .OR.
     .                 upbuff .EQ. 'F' .OR. upbuff .EQ. 'FALSE'.OR.
     .                 upbuff .EQ. 'OFF' ) THEN
	       CD_GET_ATTVAL_L = .TRUE.
               val = .FALSE.

             ELSE
               GOTO 5000

             ENDIF
           ENDIF

	 ELSE  ! not a char type

	   CD_GET_ATTVAL_L = .FALSE.
	 ENDIF

      ELSE ! no attribute found

	 CD_GET_ATTVAL_L = .FALSE.
      ENDIF

      RETURN

* error exit
 5000 CONTINUE
 
* *kob* 1/03 - add risc_buff - needed for g77 port
      IF (do_warn) THEN
	risc_buff = attrib(:lena)
	tmp_buff = vname
	CALL TM_NOTE('Undecipherable value of netCDF attribute '//
     .		      risc_buff(:TM_LENSTR1(risc_buff))//' on variable '//
     .                tmp_buff, tooplu)
        CALL TM_NOTE('modulo = "'//buff(:TM_LENSTR1(buff))//'"', tooplu)

      ENDIF

      CD_GET_ATTVAL_L = .FALSE.

      RETURN
      END
