	SUBROUTINE CD_WRITE_GRID ( cdfid, grid, lo, hi, recdim, 
     .                             itsa_slab, gname, status )

* Write a grid definition out to the current output netCDF file

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/92
*               10/92 - put into netCDF into define mode before writing attribs
*		1/95  - change in "parent" argument to cd_write_axis (now int)
*   11/95 - *sh* use CD_AXIS_NAME rather than line_name for dynamic lines
*    2/96 - *sh* same deal for dynamic grids ... a quick fix right here

* argument definitions:
*     cdfid   - netCDF id number for already opened CDF file
*     grid    - TMAP grid id
*     lo      - low indices of variable hyperslab
*     hi      - high indices of variable hyperslab
*     recdim  - which axis (1-4) is to be the CDF record axis
*     itsa_slab- does the underlying variable need a formal grid defn
*     gname   - the name of the parent grid created
*     status  - return status

* include files
#ifdef unix
        include 'tmap_format/netcdf.decl'
        include 'tmap_format/netcdf.inc'
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/tmap_dims.parm'
        include 'tmap_format/cdf_tmap.parm'
        include 'tmap_format/gt_lib.parm'
        include 'tmap_format/cd_lib.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
#else
xx        include 'xxtmap_format/netcdf.decl'  ! untested
xx        include 'xxtmap_format/netcdf.inc'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
        INCLUDE 'TMAP_FORMAT:CDF_TMAP.PARM'
        INCLUDE 'TMAP_FORMAT/GT_LIB.PARM'
        INCLUDE 'TMAP_FORMAT/CD_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif

* argument declarations
      LOGICAL itsa_slab
      INTEGER cdfid, grid, lo(4), hi(4), recdim, status
      CHARACTER*(*) gname

* local variable declarations:
      LOGICAL gmod
      INTEGER TM_LENSTR1, NCDID, NCVID, NCVDEF, NCDDEF,
     .        defid, defdim, nlen, slen, cdfstat, iaxis, idim, glen, i
      CHARACTER CD_CHILDAX_NAME*24, CD_AXIS_NAME*24,
     .		names*132, lname*24
      CHARACTER*1 ww_dim_name(4)
      DATA ww_dim_name/'X', 'Y', 'Z', 'T'/

* are the axes, alone, enough of a grid definition
      IF ( .NOT.itsa_slab ) GOTO 100

* put CDF into define mode so attributes can be written
      CALL CD_SET_MODE(cdfid, pcd_mode_define, status)
      IF ( status .NE. merr_ok ) RETURN

* create "grid_definition" dimension name if not already in the file
      defdim = NCDID( cdfid, 'grid_definition', cdfstat )
      IF ( cdfstat .NE. NCNOERR ) defdim =
     .     NCDDEF( cdfid, 'grid_definition', 1, cdfstat )

* create an output grid name that reflects axes which have been averaged, etc.
* and a string with a list of the axis names (starting w/ 2 blanks)
      gname = grid_name(grid)
      glen = TM_LENSTR1( gname )
! 2/96 - strip parentheses from dynamic grid name
      IF ( gname(1:1).EQ.'(' .AND. glen.GT.2 ) THEN
	gname = gname(2:glen-1)
	glen = glen - 2
      ENDIF
! 2/96 - end of insertion
      gmod = .FALSE.
      names = ' '
      nlen = 1
      DO 50 idim = 1, 4
         iaxis = grid_line(idim, grid)
         lname = CD_AXIS_NAME(grid, idim, slen )
         IF ( iaxis .EQ. mpsnorm ) THEN
            names = names(:nlen)//' '//'NORMAL'
         ELSEIF ( lo(idim) .EQ. unspecified_int4 ) THEN
            names = names(:nlen)//' '//'NORMAL'
            IF ( gmod ) THEN
               gname = gname(:glen)//ww_dim_name(idim)
               glen = glen + 1
            ELSE
               gname = gname(:glen)//'_N'//ww_dim_name(idim)
               gmod = .TRUE.
               glen = glen + 3
            ENDIF
         ELSEIF ( lname .EQ. 'ABSTRACT' ) THEN
            names = names(:nlen)//' '//
     .           CD_CHILDAX_NAME( ww_dim_name(idim)//'AX',
     .                            lo(idim), hi(idim), i )
         ELSE
            names = names(:nlen)//' '//lname
         ENDIF
         nlen = TM_LENSTR1(names)
 50   CONTINUE

* does this grid already exist ? (no consistency check if so)
      defid = NCVID( cdfid, gname(:glen), cdfstat )
      IF ( cdfstat .EQ. NCNOERR ) GOTO 1000

* create a dummy variable for this grid definition
      defid = NCVDEF( cdfid, gname(:glen), NCCHAR, 1,
     .                defdim, cdfstat )

* write the axis list as an attribute of the grid
      CALL NCAPTC( cdfid, defid, 'axes', NCCHAR, nlen-2,
     .             names(3:), cdfstat )

* create the parent axes
 100  DO 200 idim = 1, 4
         IF ( grid_line(idim, grid) .NE. mpsnorm
     .  .AND. lo(idim) .NE. unspecified_int4     ) THEN
            CALL CD_WRITE_AXIS( cdfid, grid, idim, idim.EQ.recdim,
     .           lo(idim), hi(idim), pcdf_full_span, status ) 
            IF ( status .NE. merr_ok ) GOTO 5900
         ENDIF
 200  CONTINUE

* successful completion
 1000 status = merr_ok
      RETURN

* error exit
 5900   RETURN
	END
