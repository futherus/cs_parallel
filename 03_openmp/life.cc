#include <SFML/Config.hpp>
#include <SFML/System/Time.hpp>
#include <SFML/System/Vector2.hpp>
#include <cstdio>
#include <cassert>
#include <omp.h>
#include <vector>
#include <memory>
#include <cstring>
#include <format>
#include <SFML/Graphics.hpp>

#define NDEBUG

const int GRID_SZ_X = 5000;
const int GRID_SZ_Y = 1000;

const sf::Vector2i MAP_SZ{ 80, 60};
const int TILE_SZ = 16;

const sf::Color ALIVE_COLOR  = sf::Color::White;
const sf::Color DEAD_COLOR   = sf::Color::Black;
const sf::Color BORDER_COLOR = sf::Color::Magenta;

class Grid
{
private:
    std::unique_ptr<bool[]> data_;
    int x_sz_;
    int y_sz_;

    static int getMod( int val, int mod)
    {
        return ((val % mod) + mod) % mod;
    }

    int getRawIndex( int x, int y) const
    {
        x = getMod( x, x_sz_);
        y = getMod( y, y_sz_);
        assert( 0 <= x && x < x_sz_);
        assert( 0 <= y && y < y_sz_);
        return y * y_sz_ + x;
    }

public:
    Grid( int x_sz, int y_sz)
        : data_{ new bool[x_sz * y_sz]},
          x_sz_{ x_sz}, y_sz_{ y_sz}
    {
        std::memset( data_.get(), 0, x_sz * y_sz);
    }

    int getSizeX() const { return x_sz_; }
    int getSizeY() const { return y_sz_; }

    bool get( int x, int y) const
    {
         return data_[ getRawIndex( x, y)];
    }

    void set( int x, int y, bool val)
    {
        data_[ getRawIndex( x, y)] = val;
    }

    void dumpCli() const
    {
        for ( int y = -1; y < y_sz_ + 1; y++ )
        {
            for ( int x = -1; x < x_sz_ + 1; x++ )
            {
                if ( y == -1 || y == y_sz_ || x == -1 || x == x_sz_ )
                {
                    std::putchar( 'b');
                }
                else
                {
                    std::putchar( get( x, y) ? '*' : ' ');
                }
            }
            printf( "\n");
        }
    }
};

void iteration( Grid& current, Grid& previous)
{
    #pragma omp parallel for schedule( static) collapse( 2)
    for ( int y = 0; y < current.getSizeY(); y++ )
    {
        for ( int x = 0; x < current.getSizeX(); x++ )
        {
            int count = 0;

            count += previous.get( x-1, y-1);
            count += previous.get( x  , y-1);
            count += previous.get( x+1, y-1);

            count += previous.get( x+1, y  );
            count += previous.get( x  , y  );
            count += previous.get( x-1, y  );

            count += previous.get( x-1, y+1);
            count += previous.get( x  , y+1);
            count += previous.get( x+1, y+1);

            switch ( count )
            {
                case 3:
                    current.set( x, y, true);
                    break;
                case 4:
                    current.set( x, y, previous.get( x, y));
                    break;
                default:
                    current.set( x, y, false);
                    break;
            }
        }
    }
}

class TileMap : public sf::Drawable, public sf::Transformable
{
private:
    sf::VertexArray verticies_;
    int x_sz_;
    int y_sz_;
    int tile_sz_;

    void draw( sf::RenderTarget& target, sf::RenderStates states) const override
    {
        states.transform *= getTransform();
        target.draw( verticies_, states);
    }

    void init()
    {
        verticies_.setPrimitiveType( sf::Quads);
        verticies_.resize( x_sz_ * y_sz_ * 4);

        for ( int i = 0; i < x_sz_; ++i )
        {
            for ( int j = 0; j < y_sz_; ++j )
            {
                sf::Vertex* quad = &verticies_[(i + j * x_sz_) * 4];

                quad[0].position = sf::Vector2f( i * tile_sz_, j * tile_sz_);
                quad[1].position = sf::Vector2f( (i + 1) * tile_sz_, j * tile_sz_);
                quad[2].position = sf::Vector2f( (i + 1) * tile_sz_, (j + 1) * tile_sz_);
                quad[3].position = sf::Vector2f( i * tile_sz_, (j + 1) * tile_sz_);
                quad[0].color = sf::Color::Magenta;
                quad[1].color = sf::Color::Magenta;
                quad[2].color = sf::Color::Magenta;
                quad[3].color = sf::Color::Magenta;
            }
        }

    }

public:
    TileMap( int tile_sz, sf::Vector2i map_sz)
        : verticies_{}, x_sz_{ map_sz.x}, y_sz_{ map_sz.y}, tile_sz_{ tile_sz}
    {
        init();
    }

    bool zoomOut()
    {
        if ( tile_sz_ == 1 )
        {
            return false;
        }

        assert( tile_sz_ % 2 == 0);

        tile_sz_ /= 2;
        x_sz_ *= 2;
        y_sz_ *= 2;
        init();
        return true;
    }

    bool zoomIn()
    {
        if ( x_sz_ % 2 != 0 || y_sz_ % 2 != 0 )
        {
            return false;
        }

        tile_sz_ *= 2;
        x_sz_ /= 2;
        y_sz_ /= 2;
        init();
        return true;
    }

    void readGrid( const Grid& grid, sf::Vector2i map_pos)
    {
        for ( int y = 0; y < y_sz_; y++ )
        {
            for ( int x = 0; x < x_sz_; x++ )
            {
                sf::Vertex* quad = &verticies_[ (y*x_sz_ + x) * 4];
                sf::Color col = grid.get( map_pos.x + x, map_pos.y + y)
                                ? ALIVE_COLOR
                                : DEAD_COLOR;
                quad[0].color = col;
                quad[1].color = col;
                quad[2].color = col;
                quad[3].color = col;
            }
        }
    }
};

bool readPlaintext( Grid& grid, FILE* infile)
{
    int c = 0;
    int x = 0;
    int y = 0;
    bool fail_to_load = false;
    while ( (c = fgetc( infile)) != EOF && !fail_to_load)
    {
        switch ( c )
        {
            case 'O':
                grid.set( x, y, true);
                x++;
                if ( x >= grid.getSizeX() )
                {
                    fprintf( stderr, "Init file [%d,%d] is larger than grid [%d, %d].", x, y, grid.getSizeX(), grid.getSizeY());
                    fail_to_load = true;
                }
                break;
            case '.':
                x++;
                break;
            case '\n':
                x = 0;
                y++;
                if ( y > grid.getSizeY() )
                {
                    fprintf( stderr, "Init file [%d,%d] is larger than grid [%d, %d].", x, y, grid.getSizeX(), grid.getSizeY());
                    fail_to_load = true;
                }
                break;
            default:
                break;
        }
    }

    return !fail_to_load;
}

int main( int argc, char* argv[])
{
    assert( argc == 2);

    FILE* infile = fopen( argv[1], "r");
    assert( infile);

    sf::Font font;
    assert( font.loadFromFile("font.otf"));

    sf::Text text{ "????", font, 20};
    text.setFillColor( sf::Color::Red);

    Grid previous( GRID_SZ_X, GRID_SZ_Y);
    Grid current( GRID_SZ_X, GRID_SZ_Y);

    if ( !readPlaintext( previous, infile) )
    {
        return -1;
    }

    TileMap map{ TILE_SZ, MAP_SZ};
    sf::RenderWindow window( sf::VideoMode( MAP_SZ.x*TILE_SZ, MAP_SZ.y*TILE_SZ), "Window");
    text.setPosition( 10, window.getSize().y - text.getGlobalBounds().height - text.getGlobalBounds().top - 10);

    sf::Vector2i map_pos = {0, 0};

    sf::Clock clock;
    sf::Clock loop_clock;
    int delay = 100;
    bool is_pause = true;
    bool is_shift_pressed = false;

    sf::Time gui_time;
    sf::Time calc_time;
    bool need_read_grid = true;

    while ( window.isOpen())
    {
        sf::Event event;
        while ( window.pollEvent( event))
        {
            if ( event.type == sf::Event::Closed)
                window.close();

            if ( event.type == sf::Event::KeyPressed )
            {
                int shift = is_shift_pressed ? 20 : 1;

                switch ( event.key.scancode )
                {
                    case sf::Keyboard::Scan::Right:
                    {
                        map_pos.x += shift;
                        need_read_grid = true;
                        break;
                    }
                    case sf::Keyboard::Scan::Left:
                    {
                        map_pos.x -= shift;
                        need_read_grid = true;
                        break;
                    }
                    case sf::Keyboard::Scan::Up:
                    {
                        map_pos.y -= shift;
                        need_read_grid = true;
                        break;
                    }
                    case sf::Keyboard::Scan::Down:
                    {
                        map_pos.y += shift;
                        need_read_grid = true;
                        break;
                    }
                    case sf::Keyboard::Scan::F:
                    {
                        delay = (delay >= 2) ? delay / 2 : 1;
                        break;
                    }
                    case sf::Keyboard::Scan::S:
                    {
                        delay *= 2;
                        break;
                    }
                    case sf::Keyboard::Scan::P:
                    {
                        is_pause = !is_pause;
                        break;
                    }
                    case sf::Keyboard::Scan::Hyphen:
                    {
                        need_read_grid = map.zoomOut();
                        break;
                    }
                    case sf::Keyboard::Scan::Equal:
                    {
                        need_read_grid = map.zoomIn();
                        break;
                    }
                    case sf::Keyboard::Scan::LShift:
                    case sf::Keyboard::Scan::RShift:
                    {
                        is_shift_pressed = true;
                        break;
                    }
                    default:
                    {
                        break;
                    }
                }
            }

            if ( event.type == sf::Event::KeyReleased )
            {
                switch ( event.key.scancode )
                {
                    case sf::Keyboard::Scan::LShift:
                    case sf::Keyboard::Scan::RShift:
                    {
                        is_shift_pressed = false;
                        break;
                    }
                    default:
                    {
                        break;
                    }
                }
            }
        }

        window.clear( sf::Color::Blue);
        if ( need_read_grid )
        {
            map.readGrid( previous, map_pos);
            need_read_grid = false;
        }

        sf::Time tmp;
        if ( !is_pause && clock.getElapsedTime() >= sf::milliseconds( delay) )
        {
            sf::Clock iter_clock;
            iteration( current, previous);
            tmp = calc_time = iter_clock.getElapsedTime();

            std::swap( current, previous);
            need_read_grid = true;
            clock.restart();
        }

        text.setString( std::format( "[{:3},{:3}]  calc: {:3} ms, gui: {:3} ms",
                                     map_pos.x, map_pos.y, calc_time.asMilliseconds(), gui_time.asMilliseconds()));

        window.draw( map);
        window.draw( text);
        window.display();

        gui_time = loop_clock.getElapsedTime() - tmp;
        loop_clock.restart();
    }

    return 0;
}
