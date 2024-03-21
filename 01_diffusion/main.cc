#include <cmath>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <limits>

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
        fprintf( stderr, C_BLUE "[%d/%d] " C_RESET FMT__ "\n", mpi_rank__, mpi_proc_count__, ##__VA_ARGS__);    \
    } while( 0)
#else
#define DEBUG_PRINT(...)
#endif

class Grid
{
public:
    Grid()
    {
        _buffer = nullptr;
    }

    ~Grid()
    {
        free( _buffer);
    }

    void init( int t_size, int x_size)
    {
        assert( _buffer == nullptr );

        _t_size = t_size;
        _x_size = x_size;

        _buffer = static_cast< float*>( calloc( _t_size * _x_size, sizeof( float)));
        assert( _buffer);

        setFull( std::numeric_limits< float>::signaling_NaN());
    }

    void init( float* buffer, int t_size, int x_size)
    {
        assert( _buffer == nullptr );

        _t_size = t_size;
        _x_size = x_size;

        _buffer = buffer;
        assert( _buffer);
    }

    bool operator==( Grid& other)
    {
        if ( _t_size != other._t_size
             || _x_size != other._x_size )
        {
            return false;
        }

        for ( int t = 0; t < _t_size; t++ )
        {
            for ( int x = 0; x < _x_size; x++ )
            {
                float val = getUnsafe( t, x);
                float other_val = other.getUnsafe( t, x);
                if ( val != other_val
                     && (!std::isnan( val) || !std::isnan( other_val)) )
                {
                    return false;
                }
            }
        }

        return true;
    }

    void set( int t, int x, float val) { _buffer[t * _x_size + x] = val; }

    void setFull( float val)
    {
        for ( int t = 0; t < _t_size; t++ )
            for ( int x = 0; x < _x_size; x++ )
                set( t, x, val);
    }

    float get( int t, int x) const
    {
        float val = getUnsafe( t, x);
        if ( std::isnan( val) )
        {
            DEBUG_PRINT( "Access: (%d, %d) == nan", t, x);
            assert( 0);
        }
        return val;
    }

    float*       getRow( int t)       { return &_buffer[t * _x_size]; }
    const float* getRow( int t) const { return &_buffer[t * _x_size]; }

    int getTSize() const { return _t_size; }
    int getXSize() const { return _x_size; }

    void dump( FILE* outfile) const
    {
        for ( int t = _t_size - 1; t >= 0; t-- )
        {
            for ( int x = 0; x < _x_size; x++ )
            {
                fprintf( outfile, "|%5.2f", getUnsafe(t, x));
            }
            fprintf( outfile, "|\n");
        }
    }

    float*       getUnderlyingBuffer()       { return _buffer; }
    const float* getUnderlyingBuffer() const { return _buffer; }

    float* detachUnderlyingBuffer()
    {
        float* tmp = _buffer;
        _buffer = nullptr;
        return tmp;
    }

private:
    float getUnsafe( int t, int x) const
    {
        return _buffer[t * _x_size + x];
    }

private:
    int   _t_size;
    int   _x_size;

    float* _buffer;
};

using Function = float (*)( float t, float x);

float identicalZero( float, float)
{
    return 0.f;
}

/**
 * Sequential solver with 4-point explicit method.
 */
void solveGrid4PE( Grid* grid, Function func, float t_step, float x_step)
{
    assert( t_step <= x_step && "Stability condition is not satisfied.");

    for ( int t = 0; t < grid->getTSize() - 1; t++ )
    {
        int x;
        for ( x = 1; x < grid->getXSize() - 1; x++ )
        {
            float diff = t_step/x_step * (grid->get(t, x+1) - 2.f*grid->get(t, x) + grid->get(t, x-1))
                         - (grid->get(t, x+1) - grid->get(t, x-1));

            grid->set( t+1, x, grid->get(t, x) + func(t*t_step, x*x_step)*t_step + 0.5f*t_step/x_step * diff);
        }

        float diff = t_step/x_step * (grid->get(t, x) - grid->get(t, x-1));
        grid->set( t+1, x, grid->get(t, x) + func(t*t_step, x*x_step)*t_step - diff);
    }
}

/**
 * Distributes nodes among processors in this pattern:
 * |x+1|x+1|x+1| x | x | x | x | x |
 */
int getNodeCount( int rank, int n_proc, int n_nodes)
{
    int nodes_per_proc = n_nodes / n_proc;
    int n_tail = n_nodes % n_proc;

    int count = nodes_per_proc;
    if ( rank < n_tail )
        count++;

    return count;
}

/**
 * Distributes nodes among processors in this pattern:
 * |x+1|x+1|x+1| x | x | x | x | x |
 */
int getBeginNode( int rank, int n_proc, int n_nodes)
{
    int nodes_per_proc = n_nodes / n_proc;
    int n_tail = n_nodes % n_proc;

    if ( rank < n_tail )
    {
        return rank * (nodes_per_proc + 1);
    }
    else
    {
        return rank * nodes_per_proc + n_tail;
    }
}

/**
 * MPI worker for transport (diffusion) equation solving.
 */
void worker( int t_count, int x_count,          // Amount of nodes to be CALCULATED. Additional left and right boundary nodes are not counted in x_count.
             float t_step, float x_step,        // Time and coordinate steps of the grid.
             float* taxis_boundary,             // Array of t-axis boundary conditions.
             float* xaxis_boundary,             // Array of x-axis boundary conditions. xaxis_boundary[0] must be under first CALCULATED node.
             Function func,                     // Function in free part of the equation.
             int proc_rank, int n_proc,         // Rank of current proc and amount of procs.
             Grid* grid)
{
    assert( t_step <= x_step && "Stability condition is not satisfied.");

    const bool is_last_proc = (proc_rank == (n_proc - 1));
    const bool is_zero_proc = (proc_rank == 0);

    /**
     * We allocate additional left and right node for all procs,
     * except the last one.
     *
     * E.g. for n_proc = 4:
     *  proc0     proc2
     * _-----_   _-----_
     * _-----_   _-----_
     * _-----_   _-----_
     * _-----_   _-----_
     *      _-----_   _-----
     *      _-----_   _-----
     *      _-----_   _-----
     *      _-----_   _-----
     *       proc1     proc3
     */
    grid->init( t_count, x_count + 1 + !is_last_proc);

    /**
     * Applying boundary condition on x axis.
     *
     * _----_
     * _----_
     * _----_
     * _----_
     * xxxxxx
     *
     * NOTE: Left boundary condition is accessed at index = -1.
     * NOTE: For last proc right boundary condition is not accessed, therefore doesn't go out-of-bounds.
     */
    for ( int x = 0; x < grid->getXSize(); x++ )
    {
        grid->set( 0, x, xaxis_boundary[x - 1]);
    }

    /**
     * Zero proc must set boundary coniditions on time axis.
     *
     * t----_
     * t----_
     * t----_
     * t----_
     * txxxxx
     */
    if ( is_zero_proc )
    {
        assert( taxis_boundary[0] == xaxis_boundary[-1] && "Boundary conditions disagree in (0;0).");
        for ( int t = 0; t < t_count; t++ )
            grid->set( t, 0, taxis_boundary[t]);
    }

    /**
     * Deduce neighbours.
     */
    int left_proc  = is_zero_proc ? MPI_UNDEFINED : proc_rank - 1;
    int right_proc = is_last_proc ? MPI_UNDEFINED : proc_rank + 1;

    /**
     * Apply 4-point explicit method.
     */
    for ( int t = 0; t < grid->getTSize() - 1; t++ )
    {
        int x;
        for ( x = 1; x < grid->getXSize() - 1; x++ )
        {
            float diff = t_step/x_step * (grid->get(t, x+1) - 2.f*grid->get(t, x) + grid->get(t, x-1))
                         - (grid->get(t, x+1) - grid->get(t, x-1));

            grid->set( t+1, x, grid->get(t, x) + func(t*t_step, x*x_step)*t_step + 0.5f*t_step/x_step * diff);
        }

        /**
         * Last proc uses explicit left angle for last point.
         */
        if ( is_last_proc )
        {
            float diff = t_step/x_step * (grid->get(t, x) - grid->get(t, x-1));
            grid->set( t+1, x, grid->get(t, x) + func(t*t_step, x*x_step)*t_step - diff);
        }

        const bool is_even = !(proc_rank % 2);

        if ( is_even )
        {
            /**
             * Send/receive to the right.
             */
            if ( !is_last_proc )
            {
                float recv = 0.f;
                float send = grid->get( t + 1, grid->getXSize() - 2);
                MPI_Send( &send, 1, MPI_FLOAT, right_proc, 0, MPI_COMM_WORLD);
                MPI_Recv( &recv, 1, MPI_FLOAT, right_proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                grid->set( t + 1, grid->getXSize() - 1, recv);
            }

            /**
             * Send/receive to the left.
             */
            if ( !is_zero_proc )
            {
                float recv = 0.f;
                float send = grid->get( t + 1, 1);
                MPI_Send( &send, 1, MPI_FLOAT, left_proc, 0, MPI_COMM_WORLD);
                MPI_Recv( &recv, 1, MPI_FLOAT, left_proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                grid->set( t + 1, 0, recv);
            }
        }
        else /* is odd */
        {
            /**
             * Receive/send to the left.
             */
            if ( !is_zero_proc )
            {
                float recv = 0.f;
                float send = grid->get( t + 1, 1);
                MPI_Recv( &recv, 1, MPI_FLOAT, left_proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send( &send, 1, MPI_FLOAT, left_proc, 0, MPI_COMM_WORLD);
                grid->set( t + 1, 0, recv);
            }

            /**
             * Receive/send to the right.
             */
            if ( !is_last_proc )
            {
                float recv = 0.f;
                float send = grid->get( t + 1, grid->getXSize() - 2);
                MPI_Recv( &recv, 1, MPI_FLOAT, right_proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send( &send, 1, MPI_FLOAT, right_proc, 0, MPI_COMM_WORLD);
                grid->set( t + 1, grid->getXSize() - 1, recv);
            }
        }
    }
}

/**
 * Append src_buffer to dst_buffer row-wise.
 *
 *     /----- dst_buffer
 *     |
 * ____ssss
 * ____ssss
 * ____ssss
 * ____ssss
 *     <-->   src_width
 * <------>   dst_width
 */
void appendBuffer( float* dst_buffer, float* src_buffer, int dst_width, int src_width, int height)
{
    for ( int h = 0; h < height; h++ )
        memcpy( dst_buffer + h*dst_width, src_buffer + h*src_width, src_width * sizeof( float));
}

int main()
{
    const int t_total = 10000;
    const int x_total = 10000;
    const float t_step = 1e-4f;
    const float x_step = 1e-4f;

    MPI_Init( NULL, NULL);

    int n_proc;
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc);
    int proc_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &proc_rank);

    Grid ref_grid;
    if ( proc_rank == 0 )
    {
        double start = MPI_Wtime();

        ref_grid.init( t_total, x_total);
        for ( int t = 0; t < t_total; t++ )
        {
            ref_grid.set( t, 0, powf( 2.71, t_step * t));
        }
        for ( int x = 0; x < x_total; x++ )
        {
            ref_grid.set( 0, x, powf( 2.71, -x_step * x));
        }

#if 0
        fprintf( stderr, "Initial:\n");
        ref_grid.dump( stdout);
#endif

        solveGrid4PE( &ref_grid, &identicalZero, t_step, x_step);

        double end = MPI_Wtime();
        fprintf( stderr, "Time elapsed: %lf s\n", end - start);
#if 0
        fprintf( stderr, "Ref solution:\n");
        ref_grid.dump( stdout);
#endif
    }

    MPI_Barrier( MPI_COMM_WORLD);

    double start = MPI_Wtime();
    float taxis_boundary[t_total];
    float xaxis_boundary[x_total];
    for ( int t = 0; t < t_total; t++ )
    {
        taxis_boundary[t] = powf( 2.71, t_step * t);
    }
    for ( int x = 0; x < x_total; x++ )
    {
        xaxis_boundary[x] = powf( 2.71, -x_step * x);
    }

    Grid grid;

    int x_begin = getBeginNode( proc_rank, n_proc, x_total - 1) + 1;
    int x_part  = getNodeCount( proc_rank, n_proc, x_total - 1);
//     DEBUG_PRINT( "[%d; %d)", x_begin, x_begin + x_count);

    worker( t_total, x_part, t_step, x_step,
            taxis_boundary, xaxis_boundary + x_begin,
            &identicalZero, proc_rank, n_proc, &grid);

    if ( proc_rank == 0 )
    {
        // Allocate buffer for result.
        int merged_buf_sz = t_total * x_total;
        float* merged_buf = static_cast< float*>( calloc( merged_buf_sz, sizeof( float)));
        assert( merged_buf);

        int merged_buf_pos = 0;

        // Append proc == 0 grid to merged.
        appendBuffer( merged_buf + merged_buf_pos, grid.getUnderlyingBuffer(), x_total, grid.getXSize(), t_total);
        // Move position to LEFT NODE of the next proc's grid.
        merged_buf_pos += x_part;

        // Reuse grid buffer.
        float* tmp_buf = grid.detachUnderlyingBuffer();

        // Receive results from proc != 0 and append to merged.
        for ( int i = 1; i < n_proc; i++ )
        {
            bool is_last_proc = (i == n_proc-1);

            int x_part_other = getNodeCount( i, n_proc, x_total - 1);

            // Size of tmp_buf must be at least as other proc's buffer.
            assert( x_part >= x_part_other );

            // Evaluate size of proc's buffer, including left and right node.
            int x_size_other = x_part_other + 1 + !is_last_proc;

            MPI_Recv( tmp_buf, x_size_other * t_total, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            appendBuffer( merged_buf + merged_buf_pos, tmp_buf, x_total, x_size_other, t_total);
            // Move position to LEFT NODE of the next proc's grid.
            merged_buf_pos += x_part_other;
        }

        assert( merged_buf_pos == x_total - 1);

        Grid merged;
        merged.init( merged_buf, t_total, x_total);

        double end = MPI_Wtime();
        DEBUG_PRINT( "Time elapsed: %lf s", end - start);

#if 0
        DEBUG_PRINT( "Merged:");
        merged.dump( stderr);
#endif
        assert( merged == ref_grid);
    }
    else
    {
        // Send whole grid to proc == 0.
        MPI_Send( grid.getUnderlyingBuffer(), grid.getXSize() * grid.getTSize(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

        double end = MPI_Wtime();
        DEBUG_PRINT( "Time elapsed: %lf s", end - start);
    }

    MPI_Finalize();
}

