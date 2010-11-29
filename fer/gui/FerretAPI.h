void Ferret_GetStatus(int *flags);

void Ferret_GetMessage(int *nLines, char *msgs);

void Ferret_GetAllDsets(int *numOpen, char *dsets);

void Ferret_GetCurrentDset(char *dset);

void Ferret_GetCurrentRegion(Boolean ssOrww[4], double lo[4], double hi[4]);

void Ferret_GetDsetVars(char *dsetName, int *numVars, char *vars);

void Ferret_GetAllGridNames(int *numGrids, char *grids);

void Ferret_GetAllAxesNames(int *numAxes, char *axes);

void Ferret_GetGrid4Axes(char *gridName, char *axesNames[4]);

void Ferret_GetAxis(char *axisName, int *numPts, char *units, Boolean *regular,
	int *orientation, double *axisMin, double *axisMaxOrDelta, 
	Boolean *formatted);

void Ferret_GetIrregularCoordinates(char *axisName, int *loSS, int *hiSS, 
	double *coords);

void Ferret_GetAllFunctions(int *numFcns, int *numArgs, char *fcns);

void Ferret_GetAllTransforms(int *numTrans, Boolean needsArg[], int argType[],
	Boolean *compresses, char *trans);

void Ferret_GetAllUserVars(int *numUserVars, char *userVars);

void Ferret_GetUserVar(char *userVar, char* definition, char *units, 
	char *title, char *titleModifier, char *gridName);

void Ferret_GetFileVariable(char *dataSet, char *varName, char *units, 
	char *title, char *titleModifier, char *gridName, int loSS[4],
	int hiSS[4]);

void Ferret_GetDsetBackground(char *dset, char *title, char *titleModifier);

void Ferret_GetAllWindows(int *numWindows, int windowIDs[]);

void Ferret_GetCurrentWindow(int *openWindowID);

void Ferret_GetAllViewPorts(int *numViewPorts, char *viewPorts);

void Ferret_GetCurrentViewPort(char *viewPort);

void Ferret_GetViewPort(char *viewPort, float *xLo, float *xHi, float *yLo,
	float *yHi, float *textProminence, Boolean *clipFlag);

void Ferret_GetAllModes(int *numModes, char *modeNames);

void Ferret_GetMode(char *modeName, Boolean *modeSet, int *modeArgType,
	char *modeArg);






