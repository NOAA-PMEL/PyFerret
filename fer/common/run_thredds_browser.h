#ifndef RUNTHREDDSBROWSER_H_
#define RUNTHREDDSBROWSER_H_

#include <sys/types.h>
#include <stdio.h>
#include "ferret.h" /* for FORTRAN */

/* 
 * Interface to the FORTRAN function:
 * INTEGER FUNCTION RUN_THREDDS_BROWSER(DATASET_NAME, ERR_WARN_MSG)
 *         CHARACTER*(*) DATASET_NAME, ERR_WARN_MSG
 */
int FORTRAN(run_thredds_browser)(char dataset_name[], char err_warn_msg[], int max_len_data_set, int max_len_err_warn_msg);

/* Standard C functions */
int   runThreddsBrowser(char datasetName[], char errWarn[]);
int   getJavaVersion(char javaExeName[], char errMsg[]);
FILE *executableOutput(char *exeArgv[], pid_t *childPidPtr, char errMsg[]);

#endif /* RUNTHREDDSBROWSER_H_ */
