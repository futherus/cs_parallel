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

int main()
{
    int n_comms = 10'000'000;

    MPI_Init( NULL, NULL);

    int n_proc;
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc);
    int proc_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &proc_rank);

    if ( n_proc != 2 && proc_rank == 0 )
    {
        fprintf( stderr, "Invalid number of processors. Terminating.\n");
        MPI_Abort( MPI_COMM_WORLD, MPI_ERR_ASSERT);
    }

    MPI_Barrier( MPI_COMM_WORLD);

    if ( proc_rank == 0 )
    {
        double start = MPI_Wtime();
        for ( int i = 0; i < n_comms; i++ )
        {
            if ( i % 1000000 == 0)
                printf( "i = %d\n", i);
            double tmp = 0.0;
            MPI_Recv( &tmp, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send( &tmp, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        }

        double end = MPI_Wtime();

        printf( "Result time: %lf s\n", end - start);
    }
    else
    {
        for ( int i = 0; i < n_comms; i++ )
        {
            double tmp = 1.0;
            MPI_Send( &tmp, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            MPI_Recv( &tmp, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    MPI_Finalize();
}

