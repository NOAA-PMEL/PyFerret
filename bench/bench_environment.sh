# simple sh script to restrict the ferret environment when running benchmark tests
# must be sourced using '.' into the current shell - do not make executable

export FER_GO=". ./v4jnls ./v5jnls ./v6jnls ./v7jnls ./genjnls $FER_GO"
export FER_DATA=". ./data"
export FER_DESCR=". ./data"
export FER_DSETS=". ./data"
export FER_GRIDS=". ./data"
export FER_DIR="."

