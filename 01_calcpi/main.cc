#include <mpi.h>
#include <stdio.h>
#include <assert.h>

#define C_BLACK   "\e[0;30m"
#define C_RED     "\e[0;31m"
#define C_GREEN   "\e[0;32m"
#define C_YELLOW  "\e[0;33m"
#define C_BLUE    "\e[0;34m"
#define C_MAGENTA "\e[0;35m"
#define C_CYAN    "\e[0;36m"
#define C_WHITE   "\e[0;37m"
#define C_RESET   "\e[0m"

#ifndef NDEBUG
#define DEBUG_PRINT(FMT__, ...)                                                                                 \
    do                                                                                                          \
    {                                                                                                           \
        int mpi_proc_count__;                                                                                   \
        MPI_Comm_size( MPI_COMM_WORLD, &mpi_proc_count__);                                                      \
        int mpi_rank__;                                                                                         \
        MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank__);                                                            \
                                                                                                                \
        fprintf( stdout, C_BLUE "[%d/%d] " C_RESET FMT__ "\n", mpi_rank__, mpi_proc_count__, ##__VA_ARGS__);    \
    } while( 0)
#else
#define DEBUG_PRINT(...)
#endif

double evalPartialSum( int begin, int n_terms)
{
    /**
     * a_i = 4 * (-1)^i / (2i+1)
     */
    double sum = 0.0;
    double sign = begin % 2 ? -1.0 : 1.0;
    DEBUG_PRINT( "[%d; %d)", begin, begin + n_terms);
    for ( int i = begin; i < begin + n_terms; i++ )
    {
        double term = 4 * sign / (2.0 * i + 1);
        sum += term;
        sign *= -1.0;
    }

    return sum;
}

/**
 * Distributes nodes among processors in this pattern:
 * | x | x | x | x | x |x+1|x+1|x+1|
 */
int getNodeCount( int rank, int n_proc, int n_nodes)
{
    int nodes_per_proc = n_nodes / n_proc;
    int n_tail = n_nodes % n_proc;

    int count = nodes_per_proc;
    if ( rank >= (n_proc - n_tail) )
        count++;

    return count;
}

int getBeginNode( int rank, int n_proc, int n_nodes)
{
    int nodes_per_proc = n_nodes / n_proc;
    int n_tail = n_nodes % n_proc;

    if ( rank < (n_proc - n_tail) )
    {
        return rank * nodes_per_proc;
    }
    else
    {
        return rank * nodes_per_proc + (rank - (n_proc - n_tail));
    }
}

int main()
{
    const int n_terms = 1'000'000'000;

    MPI_Init( NULL, NULL);

    int n_proc;
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc);

    int proc_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &proc_rank);

    if ( proc_rank == 0 )
    {
        double start = MPI_Wtime();
        double reference = evalPartialSum(0, n_terms);
        double end = MPI_Wtime();
        fprintf( stderr, "Reference: %.12lf. Evaluation time: %lf s\n", reference, end - start);
    }

    MPI_Barrier( MPI_COMM_WORLD);

    double start = MPI_Wtime();
    int begin = getBeginNode( proc_rank, n_proc, n_terms);
    int count = getNodeCount( proc_rank, n_proc, n_terms);
    double result = evalPartialSum( begin, count);

    if ( proc_rank == 0 )
    {
        for ( int i = 1; i < n_proc; i++ )
        {
            double tmp = 0.0;
            MPI_Recv( &tmp, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            result += tmp;
        }
        double end = MPI_Wtime();

        printf( "Result: %.12lf. Evaluation time: %lf s\n", result, end - start);
    }
    else
    {
        DEBUG_PRINT( "Sending: %.12lf", result);
        MPI_Send( &result, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
}

