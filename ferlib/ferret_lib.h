#ifndef FERRET_LIB_H_
#define FERRET_LIB_H_

extern float *memory;
extern float *ppl_memory;

#define MAX_FERRET_NDIM 4

/* Enumerated type to assist in creating cdms2.axis objects */
typedef enum AXISTYPE_ {
    AXISTYPE_LONGITUDE = 1,
    AXISTYPE_LATITUDE = 2,
    AXISTYPE_LEVEL = 3,
    AXISTYPE_TIME = 4,
    AXISTYPE_CUSTOM = 5,
    AXISTYPE_ABSTRACT = 6,
    AXISTYPE_NORMAL = 7,
} AXISTYPE;

/* Indices of a time integer array to assist in create cdtime objects */
typedef enum TIMEARRAY_INDEX_ {
    TIMEARRAY_DAYINDEX  = 0,
    TIMEARRAY_MONTHINDEX = 1,
    TIMEARRAY_YEARINDEX = 2,
    TIMEARRAY_HOURINDEX = 3,
    TIMEARRAY_MINUTEINDEX = 4,
    TIMEARRAY_SECONDINDEX = 5,
} TIMEARRAY_INDEX;

/* Enumerated type to assist in creating time cdms2.axis objects */
typedef enum CALTYPE_ {
    CALTYPE_NONE = -1,
    CALTYPE_360DAY = 0,
    CALTYPE_NOLEAP = 50000,
    CALTYPE_GREGORIAN = 52425,
    CALTYPE_JULIAN = 52500,
    CALTYPE_ALLLEAP = 60000,
} CALTYPE;

/* Prototypes for library C functions */
void set_fer_memory(float *mem, int mem_size);
void set_ppl_memory(float *mem, int mem_size);
void set_shared_buffer(void);
void decref_pyobj_(void *pyobj_ptr_ptr);

/* Prototypes for library Fortan functions accessed from C routines */
void add_pystat_var_(void *data_ndarray_ptr_ptr, char codename[], char title[], char units[],
                     float *bdfval, char dset[], int axis_nums[MAX_FERRET_NDIM],
                     int axis_starts[MAX_FERRET_NDIM], int axis_ends[MAX_FERRET_NDIM],
                     char errmsg[], int *len_errmsg, int len_codename, int len_title,
                     int len_units, int len_dset, int maxlenerrmsg);
void clear_fer_last_error_info_(void);
void ef_get_single_axis_info_(int *id, int *argnum, int *axisnum,
                              char axis_name[], char axis_unit[],
                              int *backwards_axis, int *modulo_axis, int *regular_axis,
                              int axis_name_maxlen, int axis_unit_maxlen);
void finalize_(void);
void get_data_array_params_(char dataname[], int *lendataname, float *memory, int *arraystart,
                            int memlo[MAX_FERRET_NDIM], int memhi[MAX_FERRET_NDIM],
                            int steplo[MAX_FERRET_NDIM], int stephi[MAX_FERRET_NDIM],
                            int incr[MAX_FERRET_NDIM], char dataunit[], int *lendataunit,
                            AXISTYPE axtypes[MAX_FERRET_NDIM], float *badval, char errmsg[],
                            int *lenerrmsg, int maxdatanamelen, int maxdataunitlen, int maxerrmsglen);
void get_data_array_coordinates_(double axiscoords[], char axisunits[], char axisname[],
                                 int *axisnum, int *numcoords, char errmsg[], int *lenerrmsg,
                                 int maxaxisunitslen, int maxaxisnamelen, int maxerrmsglen);
void get_data_array_time_coords_(int timecoords[][6], CALTYPE *caltype, char axisname[],
                                 int *axisnum, int *numcoords, char errmsg[], int *lenerrmsg,
                                 int maxaxisnamelen, int maxerrmsglen);
void get_fer_last_error_info_(int *errval, char errmsg[], int maxerrmsglen);
void get_ferret_params_(char errnames[][32], int errvals[], int *numvals);
void init_journal_(int *status);
void initialize_(void);
void no_journal_(void);
void proclaim_c_(int *ttoutLun, char *leader);
void set_one_cmnd_mode_(int *one_cmnd_mode_int);
void turnoff_verify_(int *status);

#endif
