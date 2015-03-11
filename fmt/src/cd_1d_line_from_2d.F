	SUBROUTINE CD_1D_LINE_FROM_2D (dset, cdfid, varid, vname, ax1d, status)

*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*.
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
* include files
        include 'netcdf.inc'
	include 'tmap_errors.parm'
#include "gt_lib.parm"
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
	include 'xdset_info.cmn_text'

	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xbuild_grids.cmn'

	external xdset_info_data
        include 'xdset_info.cd_equiv'
        include 'xio.cmn_text'
        include 'xunits.cmn_text'
        external xunits_data
        include 'calendar.decl'
        include 'calendar.cmn'

	INTEGER dset, cdfid, varid, ax1d, status
	CHARACTER*(*) vname


	INTEGER bufflen
	PARAMETER (bufflen = 256)
	CHARACTER buff*256, dbuf*3, vupcase*128

	LOGICAL TM_DFPEQ_TOL, TM_FPEQ_EPS, NC_GET_ATTRIB, 
     .		coordvar, got_it, do_warn, ok_cal, yyyymmdd_time, got_torg
	INTEGER STR_UPCASE, TM_LENSTR1, TM_GET_CALENDAR_ID, TM_UNIT_ID,
     .          i, vartype, nvdim, vdims(8), nvatts, 
     .		npts, npts_cdf, ipt1, ipte, ipt, outflag, next_line_storage, 
     .		start(nferdims),count(nferdims), zero(nferdims),
     .          maxlen, attlen, attoutflag, cal_id, since_T0, slen, vlen, 
     .          units, istat
	REAL val, epsilon_23
	REAL*8 depsilon_46, depsilon, dlast_coord, dfirs_coord,
     .       dfirst_delta, ddelta

      INTEGER tt  ! nice short name
      EQUIVALENCE (tt,lunit_errors)

* initialize things
	do_warn = .FALSE.

* Define new axis

	CALL TM_ALLO_TMP_LINE (ax1d, status)
	CALL TM_NEW_LINE_NAME ( 'ftime1d', buff )
	line_dattype(ax1d) = nf_double
	line_name(ax1d) = buff

	CALL CD_GET_VAR_INFO (dset, varid, vname, vartype, 
     .            nvdim, vdims, nvatts, coordvar, outflag, status)

* Get units, calendar, origin

* Get the calendar name.

      line_cal_name(ax1d) = 'GREGORIAN'   ! default
      maxlen = bufflen
      got_it = NC_GET_ATTRIB( dset, varid, 'calendar', do_warn,
     .                        vname(:vlen), maxlen, attlen, attoutflag, 
     .                        buff, val)
      IF (got_it) then
         CALL CD_GET_CALENDAR_NAME(buff, ok_cal)
         IF (ok_cal) THEN
	    line_cal_name(ax1d) = buff
	 ELSe
            slen = TM_LENSTR1(buff)
            CALL TM_NOTE('calendar attribute for 2D Time variable "'//
     .               vname(:vlen)//'" is not recognized: '//buff(1:slen), tt)
            CALL TM_NOTE('Assuming Gregorian calendar', tt)
         ENDIF
      ENDIF
      cal_id = TM_GET_CALENDAR_ID (line_cal_name(ax1d))

* optional units for axis (may contain "tunits since T0_date" 2/95)
      line_units(ax1d) = ' '   ! default
      maxlen = bufflen
      got_it = NC_GET_ATTRIB( dset, varid, 'units', do_warn,
     .                        vname(:vlen), maxlen, attlen, attoutflag, 
     .                        buff, val)
      since_T0 = MAX( INDEX(buff,'since'), INDEX(buff,'SINCE') )
      yyyymmdd_time = buff(1:15) .EQ. 'yyyymmddhhmmss'

      IF ( since_T0 .GT. 2 ) THEN
         line_units(ax1d) = buff(:since_T0-1)  ! just the "tunits" part 

         istat = STR_UPCASE( dbuf, buff(1:3) )

         slen = TM_LENSTR1(buff)
         IF ( slen .GT. since_T0 + 5) THEN

            got_torg = .TRUE.
            CALL TM_DATE_REFORMAT(buff(since_T0+6:),pdate_numslashnum,
     .                            cal_id, line_t0(ax1d), 
     .                            pdate_vax, .FALSE., status )

c fix for bug 1394
c If the part after since is not a date/time, just keep the whole
c string as the units.  i.e., "days_since_event".

c           IF ( status .NE. merr_ok ) GOTO 1000        ! 3/99 soft error 

            IF ( status .NE. merr_ok ) THEN   
               line_units(ax1d) = buff(1:slen)
               got_torg = .FALSE.
               since_t0 = 0
               status = merr_ok
            ENDIF
         ELSE
            since_t0 = 0        ! flag that no T0 was present
         ENDIF
      ELSEIF ( yyyymmdd_time ) THEN
         line_units(ax1d) = 'days'
         line_t0(ax1d) = ' ' ! to be determined when coords are read
         got_torg = .TRUE.
      ELSE
         IF (got_it) line_units(ax1d) = buff
         got_torg = .FALSE.
      ENDIF

* ... decode the units

      units = TM_UNIT_ID( line_units(ax1d) )

* Named calendar with unit=year gets length of year in that calendar.

      IF (units .EQ. -6 .OR. units .EQ. -10) THEN
         IF (cal_id .EQ. d360) units = -9
         IF (cal_id .EQ. julian) units = -12
	 IF (cal_id .EQ. noleap) units = -11
	 IF (cal_id .EQ. all_leap) units = -8
      ENDIF

* Named calendar with unit=month gets length of month in that calendar.

      IF (units .EQ. -5) THEN
         IF (cal_id .EQ. d360) units = -14
         IF (cal_id .EQ. julian) units = -15
	 IF (cal_id .EQ. noleap) units = -13
      ENDIF

      line_unit_code(ax1d) = units
      IF  ( units.EQ.0 .AND. line_units(ax1d) .NE. ' ' ) THEN
         CALL TM_NOTE('Units on axis "'//vname(:vlen)//
     .                '" are not recognized: '//line_units(ax1d), tt)
         CALL TM_NOTE('They will not be convertible:', tt)
      ENDIF
      IF ( units .GE. 0 ) THEN
         line_tunit(ax1d) = real4_init
      ELSE
         line_tunit(ax1d) = un_convert(line_unit_code(ax1d))
      ENDIF

* The units must describe calendar time 

      ! check on line_tunit... else GOTO 5400 
      IF (line_tunit(ax1d) .NE. real4_init) THEN
         line_direction(ax1d) = 'TI'
      ELSE
         GOTO 5400
      ENDIF

* Get the coordinate values and stretch them out onto a single time axis.

*  first read the data from temp2d into line storage. Use it to compute the 1d line coords

	npts = 1
	DO 100 i = 1, nvdim
	   CALL CD_GET_DS_DIMS (dset, vdims(i), buff, npts_cdf, status)
	   start(i) = 1
	   count(i) = npts_cdf
	   zero(i) = 0
	   npts = npts* npts_cdf
 100	CONTINUE

	next_line_storage = next_line_mem_pos
c	IF ((next_line_mem_pos+ npts+1).GT.maxlinestore) GOTO 5300
	ipt1 = next_line_mem_pos
	ipte = ipt1 + npts - 1

	CALL CD_READ_SUB(cdfid, varid, nvdim, start, count,
     .		   zero, zero, line_mem(ipt1), 0, 0, status)


	ipt = ipt1
	npts = 0
	DO 200 i = ipt1, ipte-1
	   line_mem(i) = line_mem(ipt)
	   npts = npts + 1
	   IF ( line_mem(ipt+1) .GT. line_mem(ipt) ) THEN
	      ipt = ipt + 1
	   ELSE
	      DO WHILE (line_mem(ipt+1) .LE. line_mem(i) )
	         ipt = ipt + 1
              ENDDO
	      ipt = ipt + 1
	   ENDIF
	   IF (ipt .GT. ipte) GOTO 300
  200	CONTINUE
  300	CONTINUE
        
	line_dim(ax1d) = npts
	ipte = ipt1 + npts-1
	
* from cd_get_1_axis.
* check for regular spacing according to the data type of incoming data.
* the variables (depsilon, etc) are in double precision. The tolerance
* chec, in TM_FPEQ_EPS is done to an equivalent single precision tol.

	IF (vartype .EQ. ncdouble) THEN
	   dlast_coord = line_mem(ipte)
           dfirs_coord = line_mem(ipt1)
           dfirst_delta = line_mem(ipt1+1) - line_mem(ipt1)
           depsilon = depsilon_46 * 2.*(ABS(line_mem(ipt1)) / dfirst_delta )

	   DO 340 i = ipt1+2, ipte	
	      ddelta = line_mem(i) - line_mem(i-1)
	      IF (.NOT. TM_DFPEQ_TOL(dfirst_delta, ddelta, depsilon) ) GOTO 380
  340	   CONTINUE
	ELSE
           dlast_coord = line_mem(ipte)
           dfirs_coord = line_mem(ipt1)
           dfirst_delta = line_mem(ipt1+1) - line_mem(ipt1)
           depsilon = epsilon_23 * 2.*(ABS(line_mem(ipt1)) / dfirst_delta )

           DO 360 i=ipt1+2,ipte
              ddelta = line_mem(i) - line_mem(i-1)
              IF (.NOT. TM_FPEQ_EPS(depsilon, dfirst_delta, ddelta) ) GOTO 380
  360      CONTINUE
	ENDIF

* It's regular - save it as such
	line_regular(ax1d) = .TRUE.
	line_start  (ax1d) = line_mem(ipt1)
	line_delta  (ax1d) = 
     .              (line_mem(ipte)-line_mem(ipt1))/DBLE(npts-1)
	line_subsc1 (ax1d) = unspecified_int4

	GOTO 390

*  Irregular
  380	line_regular(ax1d) = .FALSE.
	IF ((next_line_mem_pos+2*npts+1).GT.maxlinestore) GOTO 5300
 
	CALL TM_IRREGAX_MID( ipt1, npts )
	

 390	CONTINUE
 
* Save the axis name (upper case) and memory storage.  For all irregular axes, 
* we have saved N+1 edges for the axis cells. Increment the pointer.   

	i = STR_UPCASE( vupcase, line_name(ax1d) )
        line_name(ax1d) = vupcase
        line_name_orig(ax1d) = vname
        IF ( .NOT.line_regular(ax1d) ) THEN
           line_subsc1(ax1d) = next_line_mem_pos
           next_line_mem_pos =  next_line_storage + 2*npts+1 
        ENDIF

	
* successful completion
        status = merr_ok

	RETURN

 5300	CONTINUE
c	error 'no more line storage'
 5400   CONTINUE
c   error 'variable does not have valid calendar time attributes'
	RETURN
	END
