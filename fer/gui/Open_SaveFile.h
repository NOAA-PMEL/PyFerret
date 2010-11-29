
/*******************************************************************************
       Open_SaveFile.h
       This header file is included by Open_SaveFile.c

*******************************************************************************/

#ifndef	_OPEN_SAVEFILE_INCLUDED
#define	_OPEN_SAVEFILE_INCLUDED


#include <stdio.h>
#include "UxLib.h"
#include "UxFsBox.h"
#include "UxTopSh.h"

extern swidget	Open_Save_dset;
extern swidget	fileSelectionBox1;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

swidget	create_Open_Save_dset( swidget _UxUxParent );

#endif	/* _OPEN_SAVEFILE_INCLUDED */
