	SUBROUTINE CD_WRITE_VAR ( cdfid, dset, vname, grid, lo,
     .                            hi, dat, edges_flag, do_bounds, 
     .                            mode_upcase_output, 
     .                            keepax_flag, user_var, fer_bad, 
     .                            do_coords, status )

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
* Write a hyperslab of data into the current output netCDF file

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/92
*  performance: 10/92 consolidated record axis data writes under 1 ncendf call
*                     and use CD_SET_MODE to change between data & define modes
*  12/92: coordinate variables -> double precision
*   1/95: use cd_get_generic_bounds to synchronize output in files without
*	  the slab_(min/max)_limits machinery 
* 10/96 *kob* - Linux port - had to move variable assignments off of
*		declaration line.
*	      - had to use F90 Intrinsic function TRANSFER to properly
*		pass single values to NCVPT as arrays of 1 element
* 6/97	*kob* - fix introduced bug in call to NCVPT
* V533 *sh* 6/01 - add support for string variables (NF_CHAR)
* V552 *acm* 5/03   increase vbuff to 128 chars 
* V570 *acm*  5/04 write bounds 2xN coordinates when record axis is irregular.
* V600 *acm*  6/05 Attribute control. Get info from linked-list structure.
* V601 *acm* 10/06 Fix bug 1461: bounds variable was created for output in cd_write_axis
*                  with an upcased variable name, e.g. TIME_bnds. Here we need to upcase
*                  as well when finding bndid (if mode upcase_output is set).
* V602 *acm* 11/06 - Fix bug 1471; new routine CD_AXIS_OUTNAME gets correct 
*                    output axis name.
* V63  *acm   9/09 Fix bug 1692: issue an error message if we try to append irreg timesteps
*                  to a previously-regular axis that didn't have a bounds attribute.
* V65  *acm* 1/10  Change all netcdf2-style calls and parameter names to 
*                  netcdf3-style calls, which are also consistent with netcdf4.
* V65  *acm* 1/10  Add #ifdef usingDODSf2cUnderscore, for 32-bit linux build
*                  linked with OPeNDAP library. 
* V641 *acm* 1/10  Fix bug 1715 and output of packed data.
* V674 2/12 *acm* 6D Ferret: use nferdims rather than 4 for dimension indexing
*      4/12 *acm* fix for finding the record axis, never axes 5 or 6.
* V685 6/13 *acm* new qualifier SAVE/NOBOUNDS
* V685 6/13 *acm* Dont add SAVE/NOCOORDS
* V687 3/14 *acm* Fix ticket 1750. Write Ferret-internal axis names to cdf files, unless
*                 the user has said /KEEP_AXISNAMES
* V69+ 7/14 *acm* Further fix to 1750. Ticket 2182: Keep case of axis name consistent when
*                 creating and writing axis bounds.
* V69+ *acm*  9/14 Ticket 1758. Memory request uses number of blocks not words
* V694 *acm*  3/15 Ticket 2240. Get scale, offset, and bad per variable from the dataset 
*                  being written.
* V71 12/16 *acm* Ticket 2158: working with true monthly time axes. Write these axes
*                  as irregularly spaced daily axes. For true-month axes, call TM_WORLD 
*                  to write any true-month time coordinates as days.
* V702 1/17 *sh* for trac enhancement #2369 -- dynamic memory management 
*	         including bug fix in computation of npts, work mem size
*                also: bug fix of unitialized "count" variable
* v73+ *acm* 12/2017 Issue # 1846, writing user-variables with scale_factor, add_offset
* V74  *acm*  2/2018 Issue # 1861, handling missing- and _FillValue with scale and offset
* V74  *acm*  7/2018 Issue # 1882, save/append with bounds and irreg data
* V751 *acm* 11/19 *acm* new qualifier SAVE/NOCOORDS
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

* argument definitions:
*     cdfid   - netCDF id number for already opened CDF file
*     dset    - dset (if netcdf file, for getting name and attribute 
*               info from linked-list structure  
*     vname   - name of variable
*     grid    - defining grid for variable
*     lo      - low indices of variable hyperslab
*     hi      - high indices of variable hyperslab
*     dat     - data values to write
*     status  - return status

* include files
        include 'netcdf.inc'
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
        include 'cdf_tmap.parm'
        include 'ferret.parm'
        include 'xvariables.cmn'
	include 'xdset_info.cmn_text'
	external xdset_info_data
        include 'xunits.cmn_text'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data

! Declare netcdf function names. 
! The ones without trailing underscore are declared in netcdf.inc
#include "netcdf_declareUnderscore.inc"

* argument declarations
      LOGICAL mode_upcase_output, do_coords
      INTEGER cdfid, dset, grid, lo(nferdims), hi(nferdims), 
     .        edges_flag, do_bounds, keepax_flag, status
      CHARACTER*(*) vname
      REAL dat(*), fer_bad

* local variable declarations:
* *kob* explicit definitions not allowed on decl line in linux...
*       moved it to parameter line 10/96
      INTEGER TM_LENSTR1, STR_SAME,
     .        start(nferdims), count(nferdims), idim, iaxis, cdfstat, vlen,
     .        slab_min(nferdims), slab_max(nferdims), tmrecax, cdrecax,
     .	      dims, varid, wrkspc,
     .        hilim, i100, i, j, k, l, m, n, recaxvar, lorec, hirec, rectyp,
     .	      vartyp, nvdim, vdims(8), nvatts, bndstart(2), bndcount(2),
     .        mblk1, nblks, npts, savecount, llen, grid_blocks, wunits
      INTEGER*8 rqst_size
      LOGICAL CD_GET_BOUNDS, TM_ABSTRACT_AXIS, TM_DFPEQ, CD_GET_ATTVAL, 
     .        MATCH_NAME, ITSA_TRUEMONTH_AXIS,
     .        got_it, scaled, writebounds, do_warn, true_month,user_var,  
     .        ldo(nferdims)
      CHARACTER*13 TM_STRING
      CHARACTER*1  dimnam(nferdims)
      DATA  	   dimnam/'I','J','K','L','M','N'/
      CHARACTER*128 vbuff     ! needed because of fussy f772.1 DEC compiler
      REAL   bad, r_scale, r_offset, secspmon, secs_p_unit, unit_p_mon
      REAL*8 scalefac, addoff
      REAL*8 r100_8(200)
      REAL*8 TM_WORLD
      
      INTEGER bndid, nlen, elen, reset, numvals, nv 
      CHARACTER name*128, buff*128, CD_AXIS_OUTNAME*128, CD_AXIS_NAME*128

      REAL*4 r100_4(200)
      REAL*8 new_box_lo, old_box_hi, reset_bnd_lo, missflag
c      EQUIVALENCE (r100_8, r100_4)

* For error message
      CHARACTER*9 typnam(12)
      DATA typnam/'NC_BYTE', 'NC_CHAR', 'NC_SHORT', 'NC_INT', 
     .            'NC_FLOAT', 'NC_DOUBLE', 'NC_UBYTE', 'NC_USHORT',
     .            'NC_UINT', 'NC_INT64', 'NC_UINT64', 'NC_STRING'/

* initialize
      vlen = TM_LENSTR1( vname )
      writebounds = do_bounds .GT. 0
      IF (.NOT.do_coords) writebounds = .FALSE.

      DO idim = 1, nferdims
         count(idim) = 0
      ENDDO

* get the variable id
      cdfstat = NF_INQ_VARID(cdfid, vname(:vlen), varid)
      IF ( cdfstat .NE. NF_NOERR ) GOTO 5100

* get the vital statistics for this variable
      cdfstat = NF_INQ_VAR (cdfid, varid, vbuff, vartyp, nvdim,
     .             vdims, nvatts)

* Get values for scalefac, addoff and bad-flag set in cdf_list for this variable

      scalefac = 1.
      addoff = 0.
      numvals = 1
      do_warn = .FALSE.
      got_it = CD_GET_ATTVAL ( cdfid, varid, "scale_factor",
     .                        do_warn, vname,
     .                        scalefac, numvals, nv )
      got_it = CD_GET_ATTVAL ( cdfid, varid, "add_offset",
     .                        do_warn, vname,
     .                        addoff, numvals, nv )
      got_it = CD_GET_ATTVAL ( cdfid, varid, "missing_value",
     .                        do_warn, vname,
     .                        bad, numvals, nv )
      IF (.NOT.got_it) got_it = 
     .         CD_GET_ATTVAL ( cdfid, varid, "_FillValue",
     .                        do_warn, vname,
     .                        bad, numvals, nv )

* enter data mode so data can be read and written
* (CD_GET_GENERIC_BOUNDS needs to read coordinate values)
      CALL CD_SET_MODE(cdfid, pcd_mode_data, status)
      IF ( status .NE. merr_ok ) RETURN

* get the hyperslab limits of this variable as stored in the file
      got_it = CD_GET_BOUNDS( cdfid, varid, slab_min, slab_max,
     .                        tmrecax, status )
      IF ( status .NE. merr_ok ) GOTO 5900


* Generic bounds -- if not writing coordinates at all this will be just 1 thru npts

      IF (.NOT. do_coords) THEN
         DO idim = 1, nferdims
	    ldo(nferdims) = line_dim_only(grid_line(idim,grid))
	    line_dim_only(grid_line(idim,grid)) = .TRUE.
	 ENDDO
      ENDIF


      IF (.NOT.got_it) CALL CD_GET_GENERIC_BOUNDS
     . ( cdfid, varid, grid, lo, slab_min, slab_max, tmrecax, status )

      IF ( status .NE. merr_ok ) GOTO 5900
      
* restore line_dim_only

      IF (.NOT. do_coords) THEN
         DO idim = 1, nferdims
	    line_dim_only(grid_line(idim,grid)) = ldo(nferdims) 
	 ENDDO
      ENDIF

* special check for abstract axes (kludge!) - length, only
* if i=5:15 overwrites i=1:10 it will be detected but
* if i=6:15 overwrite i=1:10 it still will not be detected resulting in an
* unhelpful error message, below

* issue 1871: Dont reset the slab range on an abstract axis. 
* Can't make an example of what was trying to be handled previously.
	 
         DO 10 idim = 1, nferdims
            IF ( idim .EQ. tmrecax ) GOTO 10
            iaxis = grid_line(idim, grid)
            IF ( TM_ABSTRACT_AXIS(iaxis)
     .     .AND. lo(idim).NE.unspecified_int4) THEN
               IF ( hi(idim)-lo(idim)
     .         .GT. slab_max(idim)-slab_min(idim) ) GOTO 5250
c               slab_min(idim) = lo(idim)
c               slab_max(idim) = hi(idim)
            ENDIF
 10      CONTINUE
* ... possible unnecessary record axis check
         IF ( lo(t_dim).EQ.unspecified_int4 ) tmrecax = 0

* check for valid limits (must be consistent with first time write)
* When reading a 4D file with 6D Ferret, skip checks when slab_min, 
* slab_max have been set to  unspecified_int4

      DO 100 idim = 1, nferdims
         IF ( idim .EQ. tmrecax ) GOTO 100
	 IF (slab_min(idim) .EQ. unspecified_int4 ) GOTO 100
         IF ( lo(idim) .EQ. unspecified_int4 ) THEN
            IF ( slab_min(idim) .NE. 1 ) GOTO 5200
         ELSE
            IF ( lo(idim) .LT. slab_min(idim)
     .      .OR. hi(idim) .GT. slab_max(idim) ) GOTO 5200
         ENDIF
 100  CONTINUE

* determine the start and count as netCDF will want them
* and determine which axis (cdrecax = 1-4) of THIS (possibly less than 4D)
* variable is the record axis.  
* 4/2012 6D Ferret: Note that dimensions 5 and 6 will NOT
* be the record axis. 

* (the complex logic for this is to allow future record axes not to be T)
      cdrecax = tmrecax
      dims = 0
      DO 200 idim = 1, nferdims
         iaxis = grid_line(idim, grid)
         IF ( iaxis.EQ.mnormal .OR. lo(idim).EQ.unspecified_int4) THEN
* TODO: this logic is NOT ok for 6D, it assumes rec ax # is larger than other axes. 
*       Wind up with cdrecax = 0...
            IF ( dims.LE.tmrecax .AND. idim.LE.4 ) cdrecax = cdrecax - 1   ! logic ok ??
         ELSE
            dims = dims + 1
            start(dims) = lo(idim) - slab_min(idim) + 1
            count(dims) = hi(idim) - lo(idim) + 1
         ENDIF
 200  CONTINUE

* synchronize with data already in the file along the record dimension
* then write the coordinate values to the file

* is record axis involved in this variable (e.g. time-independent) ?
      IF ( tmrecax .NE. 0 ) THEN
         lorec = lo(tmrecax)
         hirec = hi(tmrecax)
      ELSE
         lorec = unspecified_int4
      ENDIF
      IF ( lorec .NE. unspecified_int4 ) THEN

         CALL CD_REC_SYNCH( cdfid, lorec, hirec, tmrecax,
     .               grid, start(cdrecax), recaxvar, rectyp, 
     .               reset, old_box_hi, new_box_lo, writebounds,
     .               status )

         IF ( status .NE. merr_ok ) GOTO 5900
	 IF (.NOT. do_coords) GOTO 400

	 
         iaxis = grid_line(tmrecax, grid)
	 IF (line_dim_only(iaxis)) GOTO 400

* RESET flag: need to adjust box limits and possibly add a void point between the 
* last cell of the existing axis and the lower bound of the axis being appended

        IF (reset .NE. 0) THEN    
            IF (reset .EQ. 2) THEN  ! just need to adjust low bound of new axis.
               reset_bnd_lo = new_box_lo

            ELSE IF (reset .EQ. 1) THEN  ! write a void point

               savecount = count(cdrecax)
               count(cdrecax) = 1

**************************************

* Write out  an extra point on the axis; NC library 
* writes fill-missflagged data for the extra point?  


               r100_8(1) = (old_box_hi + new_box_lo)/2.D0

#ifdef FORTRAN_90
            cdfstat = NF_PUT_VARA_DOUBLE(cdfid, recaxvar, 
     .                 TRANSFER(start(cdrecax),start,1),
     .                 TRANSFER(1,count,1), 
     .		       TRANSFER(r100_8,r100_4))
#else
	    cdfstat = NF_PUT_VARA_DOUBLE(cdfid, recaxvar,
     .                 start(cdrecax), 1, r100_8)
#endif
            IF ( cdfstat .NE. NF_NOERR ) GOTO 5300


* Now write the bounds for this point

            r100_8(1) = old_box_hi
            r100_8(2) = new_box_lo

            bndstart(1) = 1
            bndcount(1) = 2
            bndstart(2) = start(cdrecax)
            bndcount(2) = 1

            cdfstat = NF_PUT_VARA_DOUBLE(cdfid, bndid, bndstart,
     .		           bndcount, r100_8)
            IF ( cdfstat .NE. NF_NOERR ) GOTO 5300

* Write a time-step slab containing missing data
            npts = 1
            DO 220  idim = 1, nferdims
! *sh* commented out 1/17 -- why was this line ever here?
! npts is udes to compute the work memory size.  Excluding recdim is
! inconsistent with the loop a few lines below where the work memory is filled
!               IF ( idim .EQ. tmrecax ) GOTO 220
               IF ( count(idim) .GT. 0 ) npts = npts* count(idim)
 220        CONTINUE

* allocate memory for missing-data
! Note: This code belongs in the Ferret libraries, rather than
!       in the TMAP libs, as it calls upon Ferret COMMON and memory mgmt

!DYNMEM_FIXME
* COMPUTE # BLOCKS NEEDED
!            grid_blocks = ( npts*4 + mem_blk_size - 1 ) / mem_blk_size
!
!            CALL GET_MEMORY( npts*4, grid_blocks, mblk1, nblks, status )
!            IF ( status .NE. merr_ok ) GOTO 5900

	    wrkspc = 4  ! arbitrarily slot 4 of Ferret tempry work storage
            rqst_size = npts * 4
	    CALL GET_WS_DYNMEM( rqst_size, wrkspc, status )
	    IF ( status .NE. merr_ok ) GOTO 5900

            CALL CD_GET_MISSING_FLAG (cdfid, varid, vname, .TRUE., 
     .           missflag, status)
            IF (status .EQ. 0) missflag = unspecified_val4

            npts = 1
	    workmem(wrkspc)%ptr(npts) = missflag
! 1/17 *sh* note that recdim is not excluded from this look
            DO i = 1, MAX(count(1),1)
            DO j = 1, MAX(count(2),1)
            DO k = 1, MAX(count(3),1)
            DO l = 1, MAX(count(4),1)
            DO m = 1, MAX(count(5),1)
            DO n = 1, MAX(count(6),1)
               workmem(wrkspc)%ptr(npts) = missflag
               npts = npts + 1
            ENDDO
            ENDDO
            ENDDO
            ENDDO
            ENDDO
            ENDDO

            CALL CD_WRITE_VAR_SUB( cdfid, varid, vartyp, dims, start,
     .             count, vdims(1), workmem(wrkspc)%ptr, cdfstat )
 
            CALL FREE_WS_DYNMEM( wrkspc )
            start(cdrecax) = start(cdrecax) + 1
            count(cdrecax) = savecount
         ENDIF

      ENDIF  ! Reset
**************************************

* write the record axis coords (buffer 100 at a time) (12/92 REAL*8 option)
* (code lifted from CD_REC_SYNCH 10/92)

         DO 310 i100 = lorec, hirec, 100
            hilim = MIN(i100+99, hirec)
            DO 300 i = i100, hilim
               r100_8(i-i100+1) = TM_WORLD(i,grid,tmrecax,box_middle)
               IF ( rectyp .EQ. NF_FLOAT ) r100_4(i-i100+1)
     .                                  = r100_8(i-i100+1)
 300        CONTINUE

#ifdef FORTRAN_90
	    cdfstat = NF_PUT_VARA_DOUBLE(cdfid, recaxvar, 
     .                 TRANSFER(i100-lorec+start(cdrecax),start,1),
     .                 TRANSFER(hilim-i100+1,count,1), 
     .		       TRANSFER(r100_8,r100_4) )
#else
	    cdfstat = NF_PUT_VARA_DOUBLE(cdfid, recaxvar, 
     .                 i100-lorec+start(cdrecax),
     .                 hilim-i100+1, r100_8)
#endif
            IF ( cdfstat .NE. NF_NOERR ) GOTO 5300
 310     CONTINUE

* Now write the bounds, if record axis is irregular, or if /BOUNDS was requested.

         iaxis = grid_line(tmrecax, grid)


	 IF (do_bounds .GE. 0) THEN
         IF ( (.NOT.line_regular(iaxis) .OR. writebounds .OR. 
     .          ITSA_TRUEMONTH_AXIS(iaxis) ) .AND. 
     .        (edges_flag .EQ. 0) ) THEN

            name = CD_AXIS_NAME( dset, grid, tmrecax, nlen )
            IF (keepax_flag .EQ. 1) name = CD_AXIS_OUTNAME( dset, grid, tmrecax, nlen )
    
* If mode upcase_output is set, then upcase the name

            IF (mode_upcase_output) THEN
               CALL STR_UPCASE( buff, name)
               name = buff
            
* If not mode upcase_output, then keep the incoming case. 
            ELSE 
               buff = CD_AXIS_OUTNAME( dset, grid, tmrecax, llen )
               IF (STR_SAME(name, buff) .EQ. 0) THEN 
	          name = buff
	          nlen = llen
               ENDIF
            ENDIF

            name = name(:nlen)//'_bnds'
            elen = nlen + 5
	    cdfstat = NF_INQ_VARID(cdfid, name(:elen),  bndid)
	    IF (cdfstat .NE. NF_NOERR ) THEN

* If bounds variable not found in file we're appending to, and the user 
* didn't specificially ask for /BOUNDS, just don't write bounds here either
	       IF (do_bounds .EQ. 0) THEN
	          writebounds = .FALSE.
	          GOTO 340
	       ELSE
	          GOTO 5400
	       ENDIF
	       
	    ENDIF
	       

            DO 330 i100 = lorec, hirec, 100
               hilim = MIN(i100+99, hirec)
               r100_8(1) = TM_WORLD(i100,grid,tmrecax,box_lo_lim)
               DO 320 i = i100, hilim
                  j = 2*(i-i100+1)-1
                  r100_8(2*(i-i100+1)-1) = TM_WORLD(i,grid,tmrecax,box_lo_lim)
                  r100_8(2*(i-i100+1)) = TM_WORLD(i,grid,tmrecax,box_hi_lim)
                  IF ( rectyp .EQ. NF_FLOAT ) r100_4(i-i100+1)
     .                                  = r100_8(i-i100+1)
 320           CONTINUE

               bndstart(1) = 1
               bndstart(2) = i100-lorec+start(cdrecax)
               bndcount(1) = 2
               bndcount(2) =  hilim-i100+1

               IF (reset_bnd_lo .NE. 0.D0) THEN
                  r100_8(i100-lorec+start(cdrecax)) = reset_bnd_lo
                  reset_bnd_lo = 0.D0
               ENDIF

	       cdfstat = NF_PUT_VARA_DOUBLE(cdfid, bndid, bndstart,
     .		           bndcount, r100_8)
               IF ( cdfstat .NE. NF_NOERR ) GOTO 5300
 330        CONTINUE
         ENDIF
         ENDIF
 340	 CONTINUE

      ENDIF ! IF ( lorec

*xxxx   Never scale the missing flag: instead we read data, insert the default missing-flag
*xxxx   for internal use in computing etc, and then on output restore the
*xxxx   packing, and the incoming missing-data flag.
*XXXXXXXXXXXXXXXX OLD BEHAVIOR was XXXXXXXXXXXXXXXX
* Scale the bad data too... on input to netCDF we scale all the data,
* compute the missing flag for internal use as indat*inscale + inoff
* and insert that internal missing flag.
*
* To write scaled data, scale all of it by ferretdat(1./inscale) - (inoff/inscale)
* the scaling to be used is sent into this routine.

*XXXXXXXXXXXXXXXX OLD BEHAVIOR above XXXXXXXXXXXXXXXX

 400  CONTINUE

      scaled = .FALSE.
      r_offset = 0.
      r_scale = 1.

      IF ( .NOT. TM_DFPEQ(scalefac, 1.D0) .OR. 
     .     .NOT. TM_DFPEQ(addoff, 0.D0) ) THEN

         IF (.NOT. TM_DFPEQ(scalefac, 0.D0) ) THEN
	    r_scale = 1. /scalefac
            r_offset = -1.D0* addoff* r_scale
         ENDIF

         npts = 1
         DO 250  idim = 1, nferdims
            IF ( count(idim) .GT. 0 ) npts = npts* count(idim)
 250     CONTINUE

	 IF (user_var) THEN
	    CALL TM_SCALE_NOBAD( r_scale, r_offset, dat, npts, bad, vartyp )
	 ELSE

c file var: rescale, skipping missing and restoring the _FillValue from the input file.
            CALL TM_SCALE_SWITCH_BAD ( r_scale, r_offset, fer_bad, bad, dat, npts )
	 ENDIF

         scaled = .TRUE.
      ENDIF

* write the slab

      CALL CD_WRITE_VAR_SUB( cdfid, varid, vartyp, dims, start, count, 
     .	vdims(1), dat, cdfstat )

* If rescaled above, restore it to previous values so it can be reused 

      IF ( scaled ) THEN
         r_scale = scalefac
         r_offset = addoff
	 
	 IF (user_var) THEN
	    CALL TM_SCALE_NOBAD( r_scale, r_offset, dat, npts, bad, vartyp )
	 ELSE

c unpack again, skipping missing and going back to Ferret internal bad-flag.
            CALL TM_SCALE_SWITCH_BAD ( r_scale, r_offset, bad, fer_bad, dat, npts )

	 ENDIF
      ENDIF

      IF ( cdfstat .EQ. -60) GOTO 5260
      IF ( cdfstat .NE. NF_NOERR ) GOTO 5300

* successful completion
 1000 status = merr_ok
      RETURN

* error messages
 5100 CALL TM_ERRMSG
     . ( merr_unkvar, status, 'CD_WRITE_VAR', unspecified_int4,
     .   unspecified_int4,
     .   'variable was never created in CDF file',
     .   vname(:vlen), *5900 )

 5200 vbuff = vname   ! needed because of fussy DEC f77 v2.1 compiler
      CALL TM_ERRMSG (merr_outofgrid, status, 'CD_WRITE_VAR',
     .     no_descfile, no_stepfile,
     .     vbuff(:vlen)//'['//dimnam(idim)//'='//
     .     TM_STRING(DBLE(lo(idim)))//':'//
     .     TM_STRING(DBLE(hi(idim)))//']',
     .     vbuff(:vlen)//'['//dimnam(idim)//'='//
     .     TM_STRING(DBLE(slab_min(idim)))//':'//
     .     TM_STRING(DBLE(slab_max(idim)))//']',
     .     *5900)

 5250 vbuff = vname 
      CALL TM_ERRMSG (merr_outofgrid, status, 'CD_WRITE_VAR',
     .     no_descfile, no_stepfile, 
     .    'inconsistent '//dimnam(idim)//' axis length for '//
     .     vbuff(:vlen), no_errstring, *5900 )
     

 5260 vbuff = vname 
      CALL TM_ERRMSG (cdfstat+pcdferr, status, 'CD_WRITE_VAR',
     .     no_descfile, no_stepfile, 
     .    'data in variable '//vbuff(:vlen)//
     .    ' not representable in output type '//typnam(vartyp),
     .     no_errstring, *5900 )

 5300 CALL TM_ERRMSG
     . ( cdfstat+pcdferr, status, 'CD_WRITE_VAR', unspecified_int4,
     . no_varid, 'failure writing to CDF output file',
     . no_errstring, *5900 )

 5400  CALL TM_ERRMSG
     . ( cdfstat+pcdferr, status, 'CD_WRITE_VAR', 
     .   no_descfile, no_stepfile, name(:elen)//' not found. '//
     . 'Attempt to append to NetCDF axis having no bounds attribute. '//
     . 'Use /NOBOUNDS or rewrite original data with bounds.',
     . no_errstring, *5900 )

* error exit
 5900   RETURN
	END

