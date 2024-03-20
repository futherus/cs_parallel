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

    void init( int t_nodes, int x_nodes, float t_step, float x_step)
    {
        assert( _buffer == nullptr );

        _t_nodes = t_nodes;
        _x_nodes = x_nodes;
        _t_step = t_step;
        _x_step = x_step;

        _buffer = static_cast< float*>( calloc( _t_nodes * _x_nodes, sizeof( float)));
        assert( _buffer);

        setFull( std::numeric_limits< float>::signaling_NaN());
    }

    void init( float* buffer, int t_nodes, int x_nodes, float t_step, float x_step)
    {
        assert( _buffer == nullptr );

        _t_nodes = t_nodes;
        _x_nodes = x_nodes;
        _t_step = t_step;
        _x_step = x_step;

        _buffer = buffer;
        assert( _buffer);
    }

    ~Grid()
    {
        free( _buffer);
    }

    bool operator==( Grid& other)
    {
        if ( _t_nodes != other._t_nodes
             || _x_nodes != other._x_nodes )
        {
            return false;
        }

        for ( int t = 0; t < _t_nodes; t++ )
        {
            for ( int x = 0; x < _x_nodes; x++ )
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

    float get( int t, int x) const
    {
        float val = getUnsafe( t, x);
        if ( std::isnan( val) )
        {
            DEBUG_PRINT( "Access: (%d, %d) == nan", t, x);
            fprintf( stderr, "Access: (%d, %d) == nan\n", t, x);
            assert( 0);
        }
        return val;
    }

    void set( int t, int x, float val)
    {
        _buffer[t * _x_nodes + x] = val;
    }

    void setColumn( int x, float val)
    {
        for ( int t = 0; t < _t_nodes; t++ )
            set( t, x, val);
    }

    void setRow( int t, float val)
    {
        for ( int x = 0; x < _x_nodes; x++ )
            set( t, x, val);
    }

    float* getRow( int t)
    {
        return &_buffer[t * _x_nodes];
    }

    const float* getRow( int t) const
    {
        return &_buffer[t * _x_nodes];
    }

    void setFull( float val)
    {
        for ( int t = 0; t < _t_nodes; t++ )
            setRow( t, val);
    }

    int getTNodesCount() const { return _t_nodes; }
    int getXNodesCount() const { return _x_nodes; }

    double getTStep() const { return _t_step; }
    double getXStep() const { return _x_step; }

    void dump( FILE* outfile) const
    {
        for ( int t = _t_nodes - 1; t >= 0; t-- )
        {
            for ( int x = 0; x < _x_nodes; x++ )
            {
                fprintf( outfile, "|%5.2f", getUnsafe(t, x));
            }
            fprintf( outfile, "|\n");
        }
    }

private:
    float getUnsafe( int t, int x) const
    {
        return _buffer[t * _x_nodes + x];
    }

private:
    int   _t_nodes;
    int   _x_nodes;
    float _t_step;
    float _x_step;

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
void solveGrid4PE( Grid* grid, Function func)
{
    float t_step = grid->getTStep();
    float x_step = grid->getXStep();
    assert( t_step <= x_step && "Stability condition is not satisfied.");

    for ( int t = 0; t < grid->getTNodesCount() - 1; t++ )
    {
        int x;
        for ( x = 1; x < grid->getXNodesCount() - 1; x++ )
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
     */
    grid->init( t_count, x_count + 1 + !is_last_proc, t_step, x_step);

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
    for ( int x = 0; x < grid->getXNodesCount(); x++ )
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

    int left_proc  = is_zero_proc ? MPI_UNDEFINED : proc_rank - 1;
    int right_proc = is_last_proc ? MPI_UNDEFINED : proc_rank + 1;

    /**
     * Apply 4-point explicit method.
     */
    for ( int t = 0; t < grid->getTNodesCount() - 1; t++ )
    {
        int x;
        for ( x = 1; x < grid->getXNodesCount() - 1; x++ )
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
                float send = grid->get( t + 1, grid->getXNodesCount() - 2);
                MPI_Send( &send, 1, MPI_FLOAT, right_proc, 0, MPI_COMM_WORLD);
                MPI_Recv( &recv, 1, MPI_FLOAT, right_proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                grid->set( t + 1, grid->getXNodesCount() - 1, recv);
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
                float send = grid->get( t + 1, grid->getXNodesCount() - 2);
                MPI_Recv( &recv, 1, MPI_FLOAT, right_proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send( &send, 1, MPI_FLOAT, right_proc, 0, MPI_COMM_WORLD);
                grid->set( t + 1, grid->getXNodesCount() - 1, recv);
            }
//            DEBUG_PRINT( "Sol:");
//            grid->dump( stderr);
        }
    }
}

void appendBuffer( float* dst_buffer, float* src_buffer, int dst_width, int src_width, int height)
{
    for ( int h = 0; h < height; h++ )
        memcpy( dst_buffer + h*dst_width, src_buffer + h*src_width, src_width * sizeof( float));
}

int main()
{
    const int t_nodes = 10;
    const int x_nodes = 10;
    const float t_step = 0.1f;
    const float x_step = 0.1f;

    MPI_Init( NULL, NULL);

    int n_proc;
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc);
    int proc_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &proc_rank);

    Grid grid;
    if ( proc_rank == 0 )
    {
        grid.init( t_nodes, x_nodes, t_step, x_step);

        for ( int t = 0; t < t_nodes; t++ )
        {
            grid.set( t, 0, powf( 2.71, t_step * t));
        }
        for ( int x = 0; x < x_nodes; x++ )
        {
            grid.set( 0, x, powf( 2.71, -x_step * x));
        }

        printf( "Initial:\n");
        grid.dump( stdout);

        solveGrid4PE( &grid, &identicalZero);

        printf( "Solution:\n");
        grid.dump( stdout);
    }

    MPI_Barrier( MPI_COMM_WORLD);

    float t_bound[t_nodes];
    float x_bound[x_nodes];
    for ( int t = 0; t < t_nodes; t++ )
    {
        t_bound[t] = powf( 2.71, t_step * t);
    }
    for ( int x = 0; x < x_nodes; x++ )
    {
        x_bound[x] = powf( 2.71, -x_step * x);
    }

    Grid grid1;

    int x_begin = getBeginNode( proc_rank, n_proc, x_nodes - 1) + 1;
    int x_count = getNodeCount( proc_rank, n_proc, x_nodes - 1);
//     DEBUG_PRINT( "[%d; %d)", x_begin, x_begin + x_count);

    worker( t_nodes, x_count, t_step, x_step,
            t_bound, x_bound + x_begin,
            &identicalZero, proc_rank, n_proc, &grid1);

#if 0
    if ( proc_rank == 0) grid1.dump( stderr);
    MPI_Barrier( MPI_COMM_WORLD);

    if ( proc_rank == 1) grid1.dump( stderr);
    MPI_Barrier( MPI_COMM_WORLD);

    if ( proc_rank == 2) grid1.dump( stderr);
    MPI_Barrier( MPI_COMM_WORLD);
#endif

    /**
     * Extract required subgrid on every proc.
     *
     * FIXME: SKIPS COLUMN x=0.
     */
    int extract_buf_sz = t_nodes * x_count;
    float* extract_buf = static_cast< float*>( calloc( extract_buf_sz, sizeof( float)));
    assert( extract_buf);

    for ( int t = 0; t < t_nodes; t++ )
        memcpy( extract_buf + t * x_count, grid1.getRow( t) + 1, x_count * sizeof( float));

    if ( proc_rank == 0 )
    {
        // Allocate merged.
        int merged_buf_sz = t_nodes * x_nodes;
        float* merged_buf = static_cast< float*>( calloc( merged_buf_sz, sizeof( float)));
        assert( merged_buf);

        // t-axis boundary append to merged.
        int merged_buf_pos = 0;
        appendBuffer( merged_buf + merged_buf_pos, t_bound, x_nodes, 1, t_nodes);
        merged_buf_pos += 1;

        // proc == 0 append to merged.
        appendBuffer( merged_buf + merged_buf_pos, extract_buf, x_nodes, x_count, t_nodes);
        merged_buf_pos += x_count;

        // Receive from proc != 0 and append to merged.
        for ( int i = 1; i < n_proc; i++ )
        {
            int x_count_other = getNodeCount( i, n_proc, x_nodes - 1);
            assert( x_count >= x_count_other);

            int buf_sz_other = t_nodes * x_count_other;
            MPI_Recv( extract_buf, buf_sz_other, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            appendBuffer( merged_buf + merged_buf_pos, extract_buf, x_nodes, x_count_other, t_nodes);
            merged_buf_pos += x_count_other;
        }

        Grid merged;
        merged.init( merged_buf, t_nodes, x_nodes, t_step, x_step);
        DEBUG_PRINT( "Merged:");
        merged.dump( stderr);

        assert( merged == grid);
    }
    else
    {
        MPI_Send( extract_buf, extract_buf_sz, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }

    free( extract_buf);

    MPI_Finalize();
}

