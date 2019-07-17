        SUBROUTINE README_DSG_IN_FERRET

* V720 *sh* 6/2017
* V720 *acm 9/2017.  On merging the trunk back to this branch, I found that
*                    both the dynamic coordinate-data changes on the trunk
*                    and this dsg work use pointer storage named "linemem".
*		     Changing linemem in the dsg branch to dsg_linemem.
*
        RETURN 
        END

* This is a dummy subroutine -- not intended ever to be called by Ferret.
* It exists as a vehicle for inserting descriptive DSG documentation
* into the code in a place that will be discovered using grep and ls

*     DSG grids, datasets and contexts
*     ================================

*** Background:

* See CF convention documentation at http://cfconventions.org/
* Chapter 9 - Discrete Sampling Geometries (DSGs)

* A netCDF DSG file contains one or more "features".
* All features in the file must be of the same type.
* Supported DSG feature type in Ferret at the time of this writing include
* only collections of time series, however it is expected that collections
* of profiles and trajectories (XY, XYT, XYZT) will shortly be supported, too.
* The 2D feature types - trajectoryProfile and timeSeriesProfile are less
* commonly seen.  The level of support they will get in Ferret is TBD.

* The challenge of supporting DSG features is that the individual features
* may be of different lengths.  This means that they cannot be efficiently
* represented as a multi-dimensional grid.  Their inherent structure is that
* of a 'ragged array'.

* In the Ferret code all DSG-specific routines have 'dsg' in their filename.
* Thus 'ls *dsg*' is a quick way to locate the key DSG logic. The associated
* variables have 'dsg' in their variable names, so 'grep dsg' should pretty
* reliably locate all of the DSG implementation details.

*** File encodings:

* CF standard defines four different storage encodings.  The two "orthogonal"
* encodings are close enough to Ferret grids that no special support
* code was deemed to be necessary. The Ferret DSG enhancements as-of 6/2017
* support only the 'contiguous ragged array' encoding.  The 'indexed ragged
* array' encoding is intended for real-time capture, and we have never (yet)
* encountered its usage. Indexed contiguous array encodings, however, could be
* added to the current code quite easily -- simply as a re-ordering of the
* observarions at the end of cd_dsg_read.F, if the encoding type is indexed.

*** Instance variables versus observation variables:

* A DSG collection is a group of sampling features.  Each individual feature
* has its own distinct length, flen.
* There are two fundamental types of variables in a DSG dataset. 'Instance' 
* variables have one value per feature -- e.g. the longitude position of
* a time series station.  'Observation' variables have flen values per feature.

* Add a command option SET DATA/FEATURE= valid only for DSG datasets, but which
* will change the view of the data, particularly for treating Trajectory datasets 
* as timeseries datasets.  Possibly there will be other use cases.

*** DSG coordinate values and ranges:

* When Ferret receives a SET DATA command to open a DSG file it identifies
* the variables containing geo-location coordinates (XYZT==lon/lat/depth/time)
* and reads them into dynamic line memory (cd_dsg_scan_vars.F).
* For as long as the dataset remains open these variable occupy memory.
* (Note: dsg_linemem is NOT included in the managed memory totals of SET MEMORY.)
* (NOTE: dsg_linemem is also distinct from the linemem and edgmem dynamic memory 
* for coordinate data. The coordinate linemem structures use the same memory 
* allocation methods and could potentially be used for dsg_linemem storage, but 
* for now at least the separate structures to make development and debugging cleaner.)
* Dataset COMMON variables (xdset_info.cmn_text: dsg_*) record which variables
* contain coordinate information and where they are stored.  So DSG coordinate
* information is available whenever it is needed: for commands LIST, PLOT, etc;
* for transformations @AVE, etc; for functions (including external fcns); and
* for pseudo-variable values (see pseudo-variable section below). When
* one of these coordinte variables is reverenced **as a Ferret variable**
* e.g. 'PLOT/VS longitude, latitude', Ferret does not read the coordinates
* from the file.  Rather, in find_mem_var.F, it creates a 'mr_table_entry_only'
* mvar that uses the pointer value from dsg_linemem.  The critical DSG 'rowSize'
* variable is handled very similarly to the coordinate variables.
* At the time of SET DATA Ferret also determines the min/max range of each
* coordinate variable.  If these are not provided with the attribute\
* 'actual_range, then Ferret will compute them and create the 'actual_range'
* attribute (cd_get_dsg_coord_extremes.F)

*** IO and Subsetting (incl. future logic):

* When Ferret reads a variable from a DSG file (cd_dsg_read.F) it always
* reads the entire variable.  It does no subsetting.  The appearance of
* subsetting is maintained by the DSG context (see section below).
* As of 6/2017 no true DSG subsetting capabilities (i.e. only reading the 
* smaller subset data volume) exist.  The future plan is to implement 
* a new command 'DEFINE DATA/SUBSET/PARENT=xxx subset=<constraint expression>'
* When this new dataset is created new, subsetted rowSize and coordinate
* variables will be stored in dsg_linemem.  When a DSG variable is thereafter
* read through this new dataset, true subsetting will occur at the IO level,
* (reduced-size memory allocated), but the variable will behave exactly as
* current DSG variables do inside of Ferret.
 
*** DSG grids:

* A DSG dataset will generate 2 grids:  the instance grid and the observations
* grid. The instance grid is a straightforward 1D grid along the E axis.  Its
* only special feature is that the associated E axis sets line_dsg_dset to
* point to the dset number of the DSG dset.  Thus the grid is tied to a
* particular dataset.  The observations grid appears to be 2D, where its 2
* axes are the E axis and the orientation axis of the feature type (e.g. T for
* a time axis collection). See the 'DSG contexts' section for more discussion.

*** DSG contexts (**important!!**):

* In a context (cx) representing a DSG variable, the normal strict association
* between the subscript values (cx_lo/hi_ss) and the world coordinate values
* (cx_lo/hi_ww) that we are accustomed to with gridded variables is broken.
* For a DSG context the subscript values always represent the size and shape
* of the data as stored in memory; and the world limits represent constraints.
* The size and shape of a stored instance variable is always a 1D array
* lying along the E axis of the same size as the E axis.  The size and shape
* of an observations variable is always a 1D array lying along the orientation
* axis of the feature type, of length equal to the length of the observation
* dimension of the netCDf file.  The world coordinate constraints may mask
* which features are relevant.  For example, if X=lo:hi has been issued to
* constrain the longitude range of a time series collection, then cx_lo/hi_ww
* on the X axis will effectively mask out any time series whose longitude
* coordinates lie outside of this longitude range.  In recognizing cache
* hits (find_mem_var.F) the WW ranges of a variable are consulted.
* In the interest of computation efficiency, the implied masking of features
* by world coordinate limits is used to reduce the amount of calculation that
* is done by some operations (transformations, regridding, etc.). Computation
* results on features that are masked out are pre-filled with the value
* dsg_masked_out_flag = -12345. If the value "-12345" were ever to show up
* mysteriously in a visible result, it would be an indication of a bug in
* application of the cx_lo/hi_ww constraints.  World constraints implicitly
* mask values at the observations level as well as at the feature instance
* level.  For example a LIST command will not show observations that lie
* outside of a requested X=lo:hi range.  However, observation-level constraints
* are not normally used to reduce the amount of computation done.  This is
* because the results of some calculations depend upon the values of
* neighboring points, even though they may be masked out.  There may be
* subtleties in this issue that have not yet been uncovered.


*** The F axis (future logic):

* The F axis of a DSG dataset lies outside the scope of the DSG concepts.  
* If an F axis were added to a DSG instance or observations grid it could
* in principle be regarded as orthogonal to the entire DSG collection.
* As such it could be used in creating, say, conceptual ensembles of the 
* DSG collection, where a variation on some analysis is different for each F
* index.  F would in effect be the lone 'gridded' axis of a DSG variable.
* Placeholders have been inserted into the code to permit this interpretation
* of an orthogonal F axis, but it has never been tested, and code additions
* would be needed. 

*** Pseudo-variables:

* When pseudo-variables X, Y, Z, or T are referenced, the values of them
* are draw from the coordinate variables stored in dsg_linemem (see section above).
* A given pseuo-variable may have the structure of an instance variable or an
* observation variable, depending upon how it is represented in the DSG file.
* For example, in a time series collection X, and Y will typically be instance
* variables, whereas T will be an observation variable. The distinction between
* instance vs observation has special implications for the pseudo-variables
* I, J, K, L, M.  Namely, the index value for observation pseudo-variables
* will restart at 1 at the start of each feature.  Thus, the L pseudo-variable
* of a time series collection will show L=1 at the start of each time series.
* The behavior of XBOX, YBOX, ZBOX, TBOX has not been fully implemented as of
* 5/2017 (see make_pseudo_var.F).  Ditto for XBOXLO, XBOXHI, etc.

