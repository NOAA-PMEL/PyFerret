/*
 * $__Header$
 *
 * $__copyright$
 *
 * This module initializes all global variables in the Fortran XGKS
 * interface.  Having a separate module permits a shared-library
 * implementation under the SunOS operating system.
 */

#include <stdio.h>
#include <xgks.h>

int	currfortpoints		= 0;	/* current number of points */
int	currforttext		= 0;	/* current amount of text */
int	currfortint		= 0;	/* current space for integers */
int	*fortint		= 0;
int	NUMWTYPES		= 0;
int	error_lun		= -1;	/* error file LUN */
char	*xgks_connection	= 0;
char	*forttext		= 0;
char	**wtypes		= 0;
FILE	*errfp			= stderr;	/* error file */
Gpoint	*fortpoints		= 0;
