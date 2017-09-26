# simple csh script to restrict the ferret environment when running benchmark tests
# must be sourced using 'source' into the current shell - do not make executable

setenv FER_GO ". ./v4jnls ./v5jnls ./v6jnls ./v7jnls ./genjnls $FER_GO"
setenv FER_DATA ". ./data"
setenv FER_DESCR ". ./data"
setenv FER_DSETS ". ./data"
setenv FER_GRIDS ". ./data"
setenv FER_DIR "."

