	SUBROUTINE CD_MAKE_FMRC_FCAL_AXIS (dset, ens_ax, nF, axcalt, status)

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

* this routine synthesizes the F forecast initialization calendar axis,
* using initialization times stored temporarily in line_mem
* it does so by modifying the generic ensemble axis, "ens_ax"
* patterning after time axis info in axcalt

* V695+ *sh*
* v698  *acm* 4/16 remove unused function declarations

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

* calling argument declarations
	INTEGER dset, ens_ax, nF, axcalt, status

* internal variable declarations
	INTEGER bufflen
	PARAMETER (bufflen = 256)
	CHARACTER TM_STRING*13, buff*256, dbuf*3, vupcase*128, pCR*1

	LOGICAL TM_DFPEQ_TOL, TM_FPEQ_EPS, NC_GET_ATTRIB, 
     .		coordvar, got_it, do_warn, ok_cal, yyyymmdd_time, got_torg
	INTEGER STR_UPCASE, TM_LENSTR1, TM_FIND_LIKE_DYN_LINE,
     .          i, l, n,  nvdim, vdims(8), nvatts, 
     .		npts_cdf, ipt1, ipte, iptn, iptnp1,
     .		outflag, next_line_storage, pre_exist,
     .		start(nferdims),count(nferdims), zero(nferdims),
     .          maxlen, attlen, attoutflag, cal_id, since_T0, slen, vlen, 
     .          units, istat, nlag, noverlap, nleadtimes
	REAL val, t0NextForecast
	REAL*8 depsilon_46, depsilon, dlast_coord, dfirs_coord,
     .       dfirst_delta, ddelta, avg_lag

        INTEGER tt  ! nice short name
        EQUIVALENCE (tt,lunit_errors)


* initialize things
	ipt1 = next_line_mem_pos
	ipte = ipt1 + nF - 1
      depsilon_46 =  2.**(-46) 

* from cd_get_1_axis.
* check for regular spacing according to the data type of incoming data.
* the variables (depsilon, etc) are in double precision. The tolerance
* chec, in TM_FPEQ_EPS is done to an equivalent single precision tol.

	dlast_coord = line_mem(ipte)
        dfirs_coord = line_mem(ipt1)
        dfirst_delta = line_mem(ipt1+1) - line_mem(ipt1)
        depsilon = depsilon_46 * 2.*(ABS(line_mem(ipt1)) / dfirst_delta )

	DO i = ipt1+2, ipte	
	   ddelta = line_mem(i) - line_mem(i-1)
	   IF (.NOT. TM_DFPEQ_TOL(dfirst_delta, ddelta, depsilon) ) GOTO 100
	ENDDO

* It's regular - save it as such
!	line_regular(ens_ax) = .TRUE.		! already set
	line_start  (ens_ax) = line_mem(ipt1)
	line_delta  (ens_ax) = 
     .              (line_mem(ipte)-line_mem(ipt1))/DBLE(nF-1)
!	line_subsc1 (ens_ax) = unspecified_int4  ! already set
	GOTO 200

*  Irregular
 100	line_regular(ens_ax) = .FALSE.
	line_subsc1(ens_ax)  = next_line_mem_pos
	IF ((next_line_mem_pos+2*nF+1).GT.maxlinestore) GOTO 5300
!	CALL TM_IRREGAX_MID( ipt1, nF )  -- not needed.  have real edges
        next_line_mem_pos =  next_line_mem_pos + 2*nF + 1
	
 200	CONTINUE
 
* inherit units from the calendar time axis just created
	line_unit_code(ens_ax) = line_unit_code(axcalt)
	line_units(ens_ax)     = line_units(axcalt)  
	line_tunit(ens_ax)     = un_convert(line_unit_code(ens_ax))

	line_t0(ens_ax)        = line_t0(axcalt)  
	line_direction(ens_ax) = 'FI'

* Save the axis name (upper case) and memory storage.  For all irregular axes, 
* we have saved N+1 edges for the axis cells. Increment the pointer.   
	CALL TM_NEW_LINE_NAME ( 'TF_CAL_F', buff )
	line_dattype(ens_ax) = nf_double
	line_name_orig(ens_ax) = buff
	i = STR_UPCASE( line_name(ens_ax), line_name_orig(ens_ax))

* successful completion
        status = merr_ok

 5000	RETURN


 5300	CALL TM_ERRMSG (merr_linstorlim, status, 'CD_MAKE_FMRC_FCAL_AXIS',
     .                  no_descfile, no_stepfile,
     .                  'MAX='//TM_STRING(DBLE(maxlinestore)),
     .                  no_errstring, *5000)

	RETURN
	END
