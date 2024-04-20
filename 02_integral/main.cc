#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <limits>
#include <vector>
#include <mutex>
#include <thread>

#ifdef COLORS
#define C_BLACK   "\e[0;30m"
#define C_RED     "\e[0;31m"
#define C_GREEN   "\e[0;32m"
#define C_YELLOW  "\e[0;33m"
#define C_BLUE    "\e[0;34m"
#define C_MAGENTA "\e[0;35m"
#define C_CYAN    "\e[0;36m"
#define C_WHITE   "\e[0;37m"
#define C_RESET   "\e[0m"
#else
#define C_BLACK
#define C_RED
#define C_GREEN
#define C_YELLOW
#define C_BLUE
#define C_MAGENTA
#define C_CYAN
#define C_WHITE
#define C_RESET
#endif

#ifndef NDEBUG
#define DEBUG_PRINT(FMT__, ...)                                                                                 \
    do                                                                                                          \
    {                                                                                                           \
        fprintf( stderr, C_BLUE "[%d/%d] " C_RESET FMT__ "\n", rank, n_proc, ##__VA_ARGS__);                    \
    } while( 0)
#else
#define DEBUG_PRINT(...)
#endif

struct Part
{
    double a;
    double b;
    double f_a;
    double f_b;
    double s_ab;
};

using Function = double (*)( double x);

template< Function F>
double trapezoid( const double x_left,
                  const double x_right,
                  const double epsilon)
{
    assert( x_right > x_left);

    double a = x_left;
    double b = x_right;
    double result = 0;
    std::vector<Part> stk;
    // This estimation for depth of recursion is enough for any
    // precision up to machine epsilon.
    stk.reserve( 1000);

    double f_a = F( a);
    double f_b = F( b);
    double s_ab = (f_a + f_b) * (b - a) / 2;

    while ( true )
    {
        double c = (a + b) / 2;
        double f_c = F( c);

        double s_ac = (f_a + f_c) * (c - a) / 2;
        double s_cb = (f_c + f_b) * (b - c) / 2;
        double s_acb = s_ac + s_cb;

        if ( std::abs( s_ab - s_acb) > epsilon * std::abs( s_acb) )
        {
            stk.push_back( Part{ a, c, f_a, f_c, s_ac});
            a = c;
            f_a = f_c;
            s_ab = s_cb;
        }
        else
        {
            result += s_acb;

            if ( stk.empty() )
                break;

            Part tmp = stk.back();
            stk.pop_back();
            a    = tmp.a;
            b    = tmp.b;
            f_a  = tmp.f_a;
            f_b  = tmp.f_b;
            s_ab = tmp.s_ab;
        }
    }

    return result;
}

template< Function F>
void trapezoidJob( double epsilon,
                   int rank,
                   int n_proc,
                   int* global_nactive,
                   double* global_result,
                   std::vector<Part>* global_stk,
                   std::mutex* global_result_mtx,
                   std::mutex* global_stk_mtx,
                   std::mutex* global_has_task_mtx)
{
    DEBUG_PRINT( "Entered job.");

    std::vector<Part> stk;
    // This estimation for depth of recursion is enough for any
    // precision up to machine epsilon.
    stk.reserve( 1000);

    while ( 1 )
    {
        assert( stk.empty() );

        Part p;
        DEBUG_PRINT( "Trying to lock has_task.");
        global_has_task_mtx->lock();

        // Get part from global stack.
        {
            DEBUG_PRINT( "Trying to lock global stack.");
            const std::lock_guard< std::mutex> lock( *global_stk_mtx);

            p = global_stk->back();
            global_stk->pop_back();
            DEBUG_PRINT( "Got part (%lf; %lf) from global stack.", p.a, p.b);

            // If more parts are present, allow other threads to get them.
            if ( global_stk->size() != 0 )
            {
                DEBUG_PRINT( "%zu more parts are present.", global_stk->size());
                global_has_task_mtx->unlock();
            }

            // If part is not terminating, add this thread to active.
            if ( p.a <= p.b )
            {
                DEBUG_PRINT( "Adding to active.");
                (*global_nactive)++;
            }
        }

        // If part is terminating, terminate.
        if ( p.a > p.b )
        {
            assert( p.a == 2.0 && p.b == 1.0 );
            DEBUG_PRINT( "Got terminating part (%lf; %lf). Terminating.", p.a, p.b);
            break;
        }

        // Perform local stack algorithm.
        double result = 0;
        DEBUG_PRINT( "Local stack algorithm.");
        while ( 1 )
        {
            double c = (p.a + p.b) / 2;
            double f_c = F( c);

            double s_ac = (p.f_a + f_c) * (c - p.a) / 2;
            double s_cb = (f_c + p.f_b) * (p.b - c) / 2;
            double s_acb = s_ac + s_cb;

            if ( std::abs( p.s_ab - s_acb) > epsilon * std::abs( s_acb) )
            {
                assert( stk.size() < 1000);
                stk.push_back( Part{ p.a, c, p.f_a, f_c, s_ac});
                p.a = c;
                p.f_a = f_c;
                p.s_ab = s_cb;
            }
            else
            {
                DEBUG_PRINT( "Part (%lf:%lf) done.", p.a, p.b);
                result += s_acb;

                if ( stk.empty() )
                    break;

                p = stk.back();
                stk.pop_back();
            }

            // NOTE: Need to lock global stack before checking size?
            //         Presume that in worst case:
            //         1) Multiple threads will offload to global stack simulatneously.
            //            This will not break anything, because load will be distributed between threads anyway.
            //         2) Thread will not offload, but stack size was becoming zero at this moment.
            //            In this case thread will offload after next iteration, which is negligible.
            //         On the other hand, taking lock only to check global stack size
            //         on _every_ iteration is very long.
            const size_t kStackOffloadSize = 8;
            if ( stk.size() > kStackOffloadSize && global_stk->size() == 0 )
            {
                const std::lock_guard< std::mutex> lock( *global_stk_mtx);
                DEBUG_PRINT( "Offloading %zu parts to global stack (size: %zu).", stk.size() - 1, global_stk->size());

                // Unload local stack to global.
                while ( stk.size() > 1 )
                {
                    DEBUG_PRINT( "p: (%lf; %lf)", stk.back().a, stk.back().b);
                    assert( global_stk->size() < 1000);
                    global_stk->push_back( stk.back());
                    stk.pop_back();
                }

                global_has_task_mtx->unlock();
            }
        }

        // Add partial result to global.
        {

            DEBUG_PRINT( "Adding part to global result.");
            const std::lock_guard< std::mutex> lock( *global_result_mtx);
            *global_result += result;
        }

        // Finalize part processing.
        {
            DEBUG_PRINT( "Finalizing part.");
            const std::lock_guard< std::mutex> lock( *global_stk_mtx);
            // Remove this thread from active.
            (*global_nactive)--;

            // Fill global stack with terminating parts.
            if ( *global_nactive == 0 && global_stk->size() == 0 )
            {
                DEBUG_PRINT( "Filling terminating parts.");
                for ( int i = 0; i < n_proc; i++ )
                    global_stk->push_back( Part{ 2.0, 1.0, 0.0, 0.0, 0.0});

                global_has_task_mtx->unlock();
            }
        }
    }
}

template< Function F>
double trapezoidParallel( const double x_left,
                          const double x_right,
                          const double epsilon)
{
    assert( x_right > x_left);

    double a = x_left;
    double b = x_right;
    double f_a = F( a);
    double f_b = F( b);
    double s_ab = (f_a + f_b) * (b - a) / 2;

    int global_nactive = 0;
    double global_result = 0;
    std::vector<Part> global_stk;
    // This estimation for depth of recursion is enough for any
    // precision up to machine epsilon.
    global_stk.reserve( 1000);
    global_stk.push_back( Part{ a, b, f_a, f_b, s_ab});

    std::mutex global_result_mtx;
    std::mutex global_stk_mtx;
    std::mutex global_has_task_mtx;

    const int n_proc = 4;

    std::vector<std::thread> threads;
    for ( int rank = 0; rank < n_proc; rank++ )
    {
        std::thread t( trapezoidJob<F>, epsilon, rank, n_proc,
                       &global_nactive, &global_result, &global_stk,
                       &global_result_mtx, &global_stk_mtx, &global_has_task_mtx);

        threads.push_back( std::move( t));
    }

    for ( int rank = 0; rank < n_proc; rank++ )
    {
        threads.back().join();
        threads.pop_back();
    }

    return global_result;
}

double function( double x)
{
    return std::sin( 1 / x);
}

int main()
{
    double x_left = 0.001;
    double x_right = 1.0;
    double eps = 1e-10;

#if 0
    double result_seq = trapezoid<function>(x_left, x_right, eps);
    fprintf( stdout, "ResultSeq: %.11lf\n", result_seq);
#else
    double result_par = trapezoidParallel<function>(x_left, x_right, eps);
    fprintf( stdout, "ResultPar: %.11lf\n", result_par);
#endif
    return 0;
}
