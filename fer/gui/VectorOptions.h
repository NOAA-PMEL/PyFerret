
/*******************************************************************************
       VectorOptions.h
       This header file is included by VectorOptions.c

*******************************************************************************/

#ifndef	_VECTOROPTIONS_INCLUDED
#define	_VECTOROPTIONS_INCLUDED


#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/PushB.h>
#include <Xm/RowColumn.h>
#include <Xm/TextF.h>
#include <Xm/Label.h>
#include <Xm/Scale.h>
#include <Xm/Form.h>
#include <X11/Shell.h>

extern Widget	VectorOptions;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_VectorOptions( swidget _UxUxParent );

#endif	/* _VECTOROPTIONS_INCLUDED */
