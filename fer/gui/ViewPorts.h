
/*******************************************************************************
       ViewPorts.h
       This header file is included by ViewPorts.c

*******************************************************************************/

#ifndef	_VIEWPORTS_INCLUDED
#define	_VIEWPORTS_INCLUDED


#include <stdio.h>
#include "UxLib.h"
#include "UxTogB.h"
#include "UxRowCol.h"
#include "UxBboard.h"
#include "UxFrame.h"
#include "UxSep.h"
#include "UxLabel.h"
#include "UxForm.h"
#include "UxTopSh.h"

extern swidget	ViewPorts;
extern swidget	frame9;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

swidget	create_ViewPorts( swidget _UxUxParent );

#endif	/* _VIEWPORTS_INCLUDED */
