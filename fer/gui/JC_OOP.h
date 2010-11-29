/*
 * JC_OOP.h
 *
 * Jonathan Callahan
 * Nov 30th 1995
 *
 * This file contains function declarations for the methods which interact 
 * with the new  structures I have defined for the Ferret GUI.
 *
 * I have endeavored to stay very close to true Object-Oriented-Programming
 * principles and have in general followed the C++ guidelines as I learned
 * them from "Weiskamp & Flamig:  The complete C++ primer".
 *
 * NB_ This file is included by main.h so the functions are NOT declared "extern".
 */


/* .................... JC_Axis methods .................... */


void JC_Axis_Clear( JC_Axis *this );

/*
 * Input
 *    
 * Output    zeroes the memory associated with 'this' axis
 */

void JC_Axis_PrintNoValues( JC_Axis *this, FILE *File_ptr );

/*
 * Input     File_ptr: File which will receive output (eg. stdout, stderr)
 *    
 * Output    Prints out the values of 'this' JC_Axis structure.
 */


void JC_Axis_Printssww( JC_Axis *this, FILE *File_ptr );

/*
 * Input     File_ptr: File which will receive output (eg. stdout, stderr)
 *    
 * Output    Prints out the ss[] and ww[] arrays of 'this' JC_Axis structure.
 */


int JC_Axis_ReturnIndex( JC_Axis *this, float val );

/*
 * Input     val: value in ww coordinates
 *    
 * Output    returns an index to the closest value to val in 'this'->value[] array
 */


float JC_Axis_ReturnNearestValue( JC_Axis *this, float val );

/*
 * Input     val: value in ww coordinates
 *    
 * Output    returns the closest value to val in 'this'->value[] array
 */


float JC_Axis_ReturnNearestMidpoint( JC_Axis *this, float val );

/*
 * Input     val: value in ww coordinates
 *    
 * Output    returns the closest midpoint to val in 'this'->value[] array
 *           (a midpoint is exactly half way between two values in 'this'->value[] array)
 */


/* .................... JC_Span methods .................... */


void JC_Span_Clear( JC_Span *this );

/*
 * Input
 *    
 * Output    zeroes the memory associated with 'this' span
 */


void JC_Span_NewAxis( JC_Span *this, JC_Axis *A_ptr );

/*
 * Input     A_ptr: pointer to the new axis
 *    
 * Output    adjusts the span (if necessary) to fit the new axis
 */


void JC_Span_NewTransform( JC_Span *this, JC_Transform *T_ptr );

/*
 * Input     T_ptr: pointer to the new transform
 *    
 * Output    adjusts the span (if necessary) for the new transform
 */


void JC_Span_Print( JC_Span *this, FILE *File_ptr );

/*
 * Input     File_ptr: File which will receive output (eg. stdout, stderr)
 *    
 * Output    Prints out the values of 'this' JC_Span structure.
 */


/* .................... JC_Transform methods .................... */


void JC_Transform_Clear( JC_Transform *this );

/*
 * Input
 *    
 * Output    zeroes the memory associated with 'this' transform
 */


/* .................... JC_Regridding methods .................... */


void JC_Regridding_Initialize( JC_Regridding *this );

/*
 * Input
 *    
 * Output    initializes the regridding structure to var="", dset="", trans="LIN", type=UNIFORM
 */


/* .................... JC_Variable methods .................... */


void JC_Variable_Clear( JC_Variable *this );

/*
 * Input
 *    
 * Output    zeroes the memory associated with 'this' variable
 */


void JC_Variable_New( JC_Variable *this, char *name, char *dset );

/*
 * Input     name: variable name to be used
 *           dset: dataset from which the variable came
 *    
 * Output    queries Ferret and fills 'this' variable with all the information
 */

void JC_Variable_SetokGeoms( JC_Variable *this );

/*
 * Input
 *
 * Output    sets okGeoms[] in 'this' variable based on axes[] in 'this' variable
 */


/* .................... JC_Region methods .................... */


void JC_Region_Clear( JC_Region *this );

/*
 * Input
 *    
 * Output    zeroes the memory associated with 'this' region
 */


void JC_Region_Initialize( JC_Region *this );

/*
 * Input
 *    
 * Output    zeroes the memory associated with 'this' region
 *           sets all the spans to IRRELEVANT_AXIS
 */


void JC_Region_NewFramer( JC_Region *this, float lat[3], float lon[3] );

/*
 * Input     ~LO/HI:  LO and HI values for the X and Y axes in 'this' region
 *
 * Output    adjusts the spans of 'this' region after the Framer has been adjusted
 */


void JC_Region_NewGeometry( JC_Region *this, int geometry );

/*
 * Input     geometry:  new geometry for the region
 *
 * Output    adjusts the spans and transforms of 'this' region to represent the new geometry
 */


void JC_Region_NewTransform( JC_Region *this, int xyzt, JC_Transform *T_ptr);

/*
 * Input     T_ptr:     new transform which demands recalculation of the spans
 *
 * Output    adjusts the spans and geometry to be compatible with the new transform
 */


void JC_Region_NewVariable( JC_Region *this, JC_Variable *V_ptr, Boolean first_ever );

/*
 * Input     V_ptr:     new variable which demands a recalculated region
 *           first_ever:TRUE if this is the first variable accessed in this session
 *
 * Output    Adjusts the geometry and span of 'this' region to be compatible with V_ptr axes
 */


void JC_Region_SetGeometryFromSpan( JC_Region *this );

/*
 * Input     
 *
 * Output    Adjusts 'this' geometry according to 'this'->span
 */


/* .................... JC_Object methods (aka. ClonedVariable) .................... */


void JC_Clone_New( JC_Object *this, char *name, char *dset );

/*
 * Input     name: name of the cloned variable
 *           dset: dataset to which the cloned variable belongs
 *
 * Output    Copies the objects variable, region and regridding to GLOBAL_~
 *           'this' is unused
 */


void JC_Clone_Print( JC_Object *this, FILE *File_ptr );

/*
 * Input     File_ptr: File which will receive output (eg. stdout, stderr)
 *    
 * Output    Prints out the values of 'this' JC_Object structure.
 */


JC_Object *JC_Clone_ReturnPointer( char *name, char *dset );

/*
 * Input     name: name of the cloned variable
 *           dset: dataset to which the cloned variable belongs
 *
 * Output    Searches GLOBAL_DatasetList for the 'dset', then searches DE_ptr->cvarList for the 'name'
 *           Returns a pointer to the object or NULL if none is found.
 */


/* .................... JC_DatasetElement methods .................... */


void JC_DatasetElement_QueryFerret( JC_DatasetElement *this, Boolean new_dataset );

/*
 * Input     new_dataset: TRUE implies the need to initialize var, dvar and cvar lists
 *
 * Output    Initializes and fills in the contents of 'this' element's lists:
 *           LIST  *var is filled with the results of QUERY_VARIABLE
 *           LIST *dvar is filled with the results of QUERY_DVARS
 *           LIST *cvar is left empty 
 */


int JC_DatasetElement_VarnameExists( JC_DatasetElement *this, char *name );

/*
 * Input     name:    name which is searched for in each of 'this' element's lists.
 *
 * Output    Retruns TRUE if the name is already in use, FALSE otherwise.
 */


void JC_DatasetList_Print( LIST *this, FILE *File_ptr );

/*
 * Input     File_ptr:  file to which all output is sent.
 *
 * Output    Prints the name of the Dataset and the names of the variables in each list.
 */


void JC_DatasetList_Free( LIST *this );

/*
 * Input     
 *
 * Output    Frees the all the lists in each dataset list element in the list.
 */


/* ~~~~~~~~~~~~~~~~~~~~ END OF JC_OOP.h ~~~~~~~~~~~~~~~~~~~~ */
