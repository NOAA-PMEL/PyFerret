	SUBROUTINE GTVAR_LIST ( lun, gdata, mr, cx, lstep, recnum )

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
* write out a the records of data for a single variable in GT format
* note: all output files are ordered XYZT for sequence numbering

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 6/24/88
* revision 0.1 - 2/6/89  - kludge seq0 calculation for modulo axes
* V200:  5/24/89 - 4D symmetrical
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN
*	 2/20/90 - corrected sequence number with compressing transforms
* RISC/Unix port *sh* 5/91:  must use direct access for Unix (added recnum)

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xmem_subsc.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
#endif

* calling argument declarations:
	INTEGER		lun, mr, cx, lstep, recnum
	REAL    gdata( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit )

* internal variable declarations:
	LOGICAL		no_y_size, no_z_size
	INTEGER		CX_DIM_LEN, seq, i, j, k, l,
     .			grid, ndim, dim(4), lodim2, hidim2, ysize
	CHARACTER	VAR_CODE*8, vcode*4

* initialization
	grid    = mr_grid( mr )

* determine variable name - either standard "code" or "Ennn"
	vcode = VAR_CODE( cx_category(cx), cx_variable(cx) )

* grid possibly compressed by transforms
	no_y_size = cx_lo_s2(cx) .EQ. unspecified_int4
	no_z_size = cx_lo_s3(cx) .EQ. unspecified_int4

* determine size of xy plane
	IF ( no_y_size ) THEN
	   ysize = 1
	ELSE
	   CALL GRID_SUBSCRIPT_EXTREMES(lodim2,hidim2,grid,y_dim)
	   ysize = hidim2 - lodim2 + 1
	ENDIF

* write out data for one variable at one time step
	DO 100 k = cx_lo_s3(cx), cx_hi_s3(cx)
	DO 100 j = cx_lo_s2(cx), cx_hi_s2(cx)

* compute sequence number
	   IF ( no_y_size ) THEN
	      IF ( no_z_size ) THEN
	         seq = 1
	      ELSE
	         seq = k
	      ENDIF
	   ELSEIF ( no_z_size ) THEN
	      seq = j
	   ELSE
	      seq = (k-1)*ysize + j
	   ENDIF

# ifdef unix
	   WRITE ( UNIT=lun, REC=recnum ) vcode,
     .			                  seq,
     .		(gdata(i,j,k,lstep) ,i = cx_lo_s1(cx),cx_hi_s1(cx) )
           recnum = recnum + 1
#else
	   WRITE ( lun ) vcode,
     .			 seq,
     .		(gdata(i,j,k,lstep) ,i = cx_lo_s1(cx),cx_hi_s1(cx) )
#endif

 100	CONTINUE

	RETURN
	END
